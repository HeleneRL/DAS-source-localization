from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

try:
    from pymap3d import geodetic2enu
except ImportError as exc:
    raise ImportError(
        "This script requires pymap3d. Install it with: pip install pymap3d"
    ) from exc


CHANNEL_MIN = 348
CHANNEL_MAX = 2267
SOUND_SPEED_MPS = 1500.0

# A fixed local ENU origin near the cable. This does NOT affect physics,
# only the local x/y coordinates used internally.
# Chosen near the southwest side of the area so most cable/source points
# end up with positive easting/northing.
ENU_LAT0_DEG = 1.2160
ENU_LON0_DEG = 103.8518
ENU_H0_M = 0.0


def load_csvs(
    arrivals_csv: Path,
    tx_csv: Path,
    trust_csv: Path,
    global_trust_csv: Path,
    prior_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    arrivals = pd.read_csv(arrivals_csv)
    tx = pd.read_csv(tx_csv)
    trust = pd.read_csv(trust_csv)
    global_trust = pd.read_csv(global_trust_csv)
    prior = pd.read_csv(prior_csv)
    return arrivals, tx, trust, global_trust, prior


def compute_relative_arrival(arrivals: pd.DataFrame) -> pd.DataFrame:
    df = arrivals.copy()

    df = df[(df["channel"] >= CHANNEL_MIN) & (df["channel"] <= CHANNEL_MAX)].copy()

    df["observed_t_s"] = df["peak_time_s_from_sequence_start"]
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


def merge_all(
    arrivals_rel: pd.DataFrame,
    tx_long: pd.DataFrame,
    trust: pd.DataFrame,
    global_trust: pd.DataFrame,
    prior: pd.DataFrame,
) -> pd.DataFrame:
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
        ]
    ].copy()

    df = df.merge(
        trust_small,
        on=["location", "channel"],
        how="left",
        validate="many_to_one",
    )

    global_small = global_trust[
        [
            "channel",
            "mean_channel_trust_score",
            "recommended_fraction",
            "recommended_global",
        ]
    ].copy()

    df = df.merge(
        global_small,
        on="channel",
        how="left",
        validate="many_to_one",
    )

    prior_small = prior.rename(
        columns={
            "lat": "prior_lat",
            "lon": "prior_lon",
            "depth": "prior_z_m",
        }
    )[["channel", "prior_lat", "prior_lon", "prior_z_m"]].copy()

    df = df.merge(
        prior_small,
        on="channel",
        how="left",
        validate="many_to_one",
    )

    passed = df["passed_snr_threshold"].astype(str).str.upper().eq("TRUE")
    near_edge = df["near_window_edge"].astype(str).str.upper().eq("TRUE")
    df["base_valid"] = passed & (~near_edge)

    return df


def add_local_enu_coordinates(
    df: pd.DataFrame,
    lat0_deg: float,
    lon0_deg: float,
    h0_m: float = 0.0,
) -> pd.DataFrame:
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

    out["tx_x_m"] = tx_e
    out["tx_y_m"] = tx_n
    out["tx_u_m"] = tx_u

    out["prior_x_m"] = prior_e
    out["prior_y_m"] = prior_n
    out["prior_u_m"] = prior_u

    return out


def make_weight(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    w = np.ones(len(out), dtype=float)
    w *= out["channel_trust_score"].fillna(0.0).clip(0.0, 1.0)
    w *= out["mean_channel_trust_score"].fillna(0.0).clip(0.0, 1.0)

    w *= np.where(out["recommended_channel"].astype(str).str.upper().eq("TRUE"), 1.0, 0.35)
    w *= np.where(out["recommended_global"].astype(str).str.upper().eq("TRUE"), 1.0, 0.60)
    w *= np.where(out["base_valid"], 1.0, 0.15)

    out["weight"] = w

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
        & (out["weight"] > 0.05)
    )

    return out


def main() -> None:
    root = Path(r"D:\Singapore Data")

    arrivals_csv = root / "processed_outputs" / "lfm_35_45_bulk" / "all_locations_bulk_lfm35_45_results.csv"
    tx_csv = root / "transmission_times_sweeps_with_tx_positions.csv"
    trust_csv = root / "processed_outputs" / "lfm_35_45_bulk" / "trust_map_outputs" / "location_channel_trust_summary.csv"
    global_trust_csv = root / "processed_outputs" / "lfm_35_45_bulk" / "trust_map_outputs" / "overall_channel_trust_summary.csv"
    prior_csv = root / "Cable" / "interpolated_channels_from_boattrack.csv"

    arrivals, tx, trust, global_trust, prior = load_csvs(
        arrivals_csv, tx_csv, trust_csv, global_trust_csv, prior_csv
    )

    arrivals_rel = compute_relative_arrival(arrivals)
    tx_long = prepare_tx_table(tx)
    merged = merge_all(arrivals_rel, tx_long, trust, global_trust, prior)
    merged = add_local_enu_coordinates(
        merged,
        lat0_deg=ENU_LAT0_DEG,
        lon0_deg=ENU_LON0_DEG,
        h0_m=ENU_H0_M,
    )
    merged = make_weight(merged)

    merged["enu_origin_lat_deg"] = ENU_LAT0_DEG
    merged["enu_origin_lon_deg"] = ENU_LON0_DEG
    merged["enu_origin_h_m"] = ENU_H0_M

    out_csv = root / "Cable" / "inversion_observations.csv"
    merged.to_csv(out_csv, index=False)

    print(f"Saved: {out_csv}")
    print(f"Rows total: {len(merged)}")
    print(f"Rows usable: {int(merged['use_observation'].sum())}")
    print(
        f"ENU origin: lat={ENU_LAT0_DEG:.6f}, lon={ENU_LON0_DEG:.6f}, h={ENU_H0_M:.2f} m"
    )
    print(
        "Prior coordinate ranges: "
        f"x=[{merged['prior_x_m'].min():.2f}, {merged['prior_x_m'].max():.2f}] m, "
        f"y=[{merged['prior_y_m'].min():.2f}, {merged['prior_y_m'].max():.2f}] m"
    )
    print(
        "TX coordinate ranges: "
        f"x=[{merged['tx_x_m'].min():.2f}, {merged['tx_x_m'].max():.2f}] m, "
        f"y=[{merged['tx_y_m'].min():.2f}, {merged['tx_y_m'].max():.2f}] m"
    )


if __name__ == "__main__":
    main()