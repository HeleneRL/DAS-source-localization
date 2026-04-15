from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pymap3d import geodetic2enu

from common import load_toml, ensure_dir, path_from_cfg


def load_csvs(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    arrivals = pd.read_csv(path_from_cfg(cfg, "raw_detection_output_dir") / "all_locations_detections.csv")
    tx = pd.read_csv(path_from_cfg(cfg, "transmitter_output_dir") / "transmission_times_with_tx_positions.csv")
    trust = pd.read_csv(path_from_cfg(cfg, "trust_output_dir") / "location_channel_trust_summary.csv")
    global_trust = pd.read_csv(path_from_cfg(cfg, "trust_output_dir") / "overall_channel_trust_summary.csv")
    prior = pd.read_csv(path_from_cfg(cfg, "prior_output_dir") / "prior_cable_by_channel.csv")
    return arrivals, tx, trust, global_trust, prior


def compute_relative_arrival(arrivals: pd.DataFrame, channel_min: int, channel_max: int) -> pd.DataFrame:
    df = arrivals.copy()
    df = df[(df["channel"] >= channel_min) & (df["channel"] <= channel_max)].copy()
    df["observed_t_s"] = pd.to_numeric(df["peak_time_s_from_sequence_start"], errors="coerce")
    df["observed_dt_ref_s"] = np.nan

    for (loc, anchor), g in df.groupby(["location", "anchor_index"]):
        ref_ch = int(g["reference_channel"].iloc[0])
        ref_rows = g[g["channel"] == ref_ch]
        if ref_rows.empty:
            continue
        t_ref = float(ref_rows["observed_t_s"].iloc[0])
        idx = (df["location"] == loc) & (df["anchor_index"] == anchor)
        df.loc[idx, "observed_dt_ref_s"] = df.loc[idx, "observed_t_s"] - t_ref

    return df


def prepare_tx_table(tx: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in tx.iterrows():
        rows.append(
            {
                "location": r["location"],
                "anchor_index": 1,
                "anchor_label": "lfm35_45_rep1",
                "reference_channel": int(r["reference_channel"]),
                "tx_lat": float(r["tx_lat_peak1"]),
                "tx_lon": float(r["tx_lon_peak1"]),
                "tx_z_m": float(r["tx_depth_m"]),
            }
        )
        rows.append(
            {
                "location": r["location"],
                "anchor_index": 2,
                "anchor_label": "lfm35_45_rep2",
                "reference_channel": int(r["reference_channel"]),
                "tx_lat": float(r["tx_lat_peak2"]),
                "tx_lon": float(r["tx_lon_peak2"]),
                "tx_z_m": float(r["tx_depth_m"]),
            }
        )
    return pd.DataFrame(rows)


def merge_all(arrivals_rel, tx_long, trust, global_trust, prior):
    df = arrivals_rel.merge(
        tx_long,
        on=["location", "anchor_index", "anchor_label", "reference_channel"],
        how="left",
        validate="many_to_one",
    )

    trust_small = trust[
        [
            "location",
            "channel",
            "channel_trust_score",
            "recommended_channel",
            "median_smooth_offset_ms",
            "anchor_disagreement_ms",
            "median_abs_residual_ms",
            "valid_fraction",
            "stable_fraction",
            "median_pick_quality_score",
        ]
    ].copy()
    df = df.merge(trust_small, on=["location", "channel"], how="left", validate="many_to_one")

    global_small = global_trust[
        [
            "channel",
            "mean_channel_trust_score",
            "recommended_fraction",
            "recommended_global",
            "mean_median_pick_quality_score",
        ]
    ].copy()
    df = df.merge(global_small, on="channel", how="left", validate="many_to_one")

    prior_small = prior.rename(columns={"lat": "prior_lat", "lon": "prior_lon", "depth": "prior_z_m"})[
        ["channel", "prior_lat", "prior_lon", "prior_z_m"]
    ].copy()
    df = df.merge(prior_small, on="channel", how="left", validate="many_to_one")

    passed = df["passed_snr_threshold"].astype(str).str.upper().eq("TRUE")
    near_edge = df["near_window_edge"].astype(str).str.upper().eq("TRUE")
    df["base_valid"] = passed & (~near_edge)

    return df


def add_local_enu_coordinates(df, lat0_deg: float, lon0_deg: float, h0_m: float = 0.0):
    out = df.copy()

    tx_e, tx_n, tx_u = geodetic2enu(
        out["tx_lat"].to_numpy(dtype=float),
        out["tx_lon"].to_numpy(dtype=float),
        out["tx_z_m"].to_numpy(dtype=float),
        lat0_deg,
        lon0_deg,
        h0_m,
    )
    prior_e, prior_n, prior_u = geodetic2enu(
        out["prior_lat"].to_numpy(dtype=float),
        out["prior_lon"].to_numpy(dtype=float),
        out["prior_z_m"].to_numpy(dtype=float),
        lat0_deg,
        lon0_deg,
        h0_m,
    )

    out["tx_x_m"], out["tx_y_m"], out["tx_u_m"] = tx_e, tx_n, tx_u
    out["prior_x_m"], out["prior_y_m"], out["prior_u_m"] = prior_e, prior_n, prior_u
    return out


def make_weight(df: pd.DataFrame, wcfg: dict) -> pd.DataFrame:
    out = df.copy()

    pick_q = pd.to_numeric(out["pick_quality_score"], errors="coerce").fillna(float(wcfg["pick_quality_floor"]))
    pick_q = pick_q.clip(lower=float(wcfg["pick_quality_floor"]), upper=1.0)

    local_trust = pd.to_numeric(out["channel_trust_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    global_trust = pd.to_numeric(out["mean_channel_trust_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    local_rec = np.where(out["recommended_channel"].astype(str).str.upper().eq("TRUE"), 1.0, float(wcfg["not_recommended_local_factor"]))
    global_rec = np.where(out["recommended_global"].astype(str).str.upper().eq("TRUE"), 1.0, float(wcfg["not_recommended_global_factor"]))
    base_valid_factor = np.where(out["base_valid"], 1.0, float(wcfg["base_invalid_factor"]))

    # Encourage detections that are both individually clean and consistently good in trust summaries.
    w = np.ones(len(out), dtype=float)
    w *= pick_q
    w *= np.sqrt(np.maximum(local_trust, 0.0))
    w *= np.sqrt(np.maximum(global_trust, 0.0))
    w *= local_rec
    w *= global_rec
    w *= base_valid_factor

    out["weight"] = np.clip(w, 0.0, 1.0)
    out["use_observation"] = (
        out["observed_dt_ref_s"].notna()
        & out["tx_lat"].notna()
        & out["tx_lon"].notna()
        & out["prior_lat"].notna()
        & out["prior_lon"].notna()
        & out["tx_x_m"].notna()
        & out["tx_y_m"].notna()
        & out["prior_x_m"].notna()
        & out["prior_y_m"].notna()
        & (out["weight"] > float(wcfg["use_observation_min_weight"]))
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build inversion_observations.csv from all upstream products.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_toml(args.config)
    ocfg = cfg["inversion_dataset"]
    outdir = ensure_dir(path_from_cfg(cfg, "inversion_dataset_output_dir"))

    arrivals, tx, trust, global_trust, prior = load_csvs(cfg)
    arrivals_rel = compute_relative_arrival(arrivals, int(ocfg["channel_min"]), int(ocfg["channel_max"]))
    tx_long = prepare_tx_table(tx)
    merged = merge_all(arrivals_rel, tx_long, trust, global_trust, prior)
    merged = add_local_enu_coordinates(
        merged,
        lat0_deg=float(ocfg["enu_lat0_deg"]),
        lon0_deg=float(ocfg["enu_lon0_deg"]),
        h0_m=float(ocfg["enu_h0_m"]),
    )
    merged = make_weight(merged, ocfg)

    merged["enu_origin_lat_deg"] = float(ocfg["enu_lat0_deg"])
    merged["enu_origin_lon_deg"] = float(ocfg["enu_lon0_deg"])
    merged["enu_origin_h_m"] = float(ocfg["enu_h0_m"])

    out_csv = outdir / "inversion_observations.csv"
    merged.to_csv(out_csv, index=False)

    print(f"Saved: {out_csv}")
    print(f"Rows total: {len(merged)}")
    print(f"Rows usable: {int(merged['use_observation'].sum())}")


if __name__ == "__main__":
    main()
