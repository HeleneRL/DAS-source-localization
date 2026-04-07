from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CHANNEL_MIN = 348
CHANNEL_MAX = 2267
DEFAULT_SOUND_SPEEDS = [1470.0, 1490.0, 1500.0, 1510.0, 1530.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Robustness scan for global channel offset under different sound speeds, subsets, and scoring metrics."
    )
    p.add_argument(
        "--inversion-csv",
        type=Path,
        default=Path(r"D:\Singapore Data\Cable\inversion_observations.csv"),
    )
    p.add_argument(
        "--prior-geometry-csv",
        type=Path,
        default=Path(r"D:\Singapore Data\Cable\prior_geometry.csv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"D:\Singapore Data\Cable\global_offset_robustness_outputs"),
    )
    p.add_argument("--offset-min", type=int, default=-150)
    p.add_argument("--offset-max", type=int, default=150)
    p.add_argument("--offset-step", type=int, default=2)
    p.add_argument(
        "--sound-speeds",
        type=float,
        nargs="*",
        default=DEFAULT_SOUND_SPEEDS,
        help="List of sound speeds to test, e.g. --sound-speeds 1470 1490 1500 1510 1530",
    )
    p.add_argument(
        "--include-raw",
        action="store_true",
        help="Also test raw observed_dt_ref_s in addition to smoothed offsets.",
    )
    p.add_argument(
        "--include-all-usable",
        action="store_true",
        help="Also test all use_observation rows in addition to trusted subset.",
    )
    p.add_argument("--min-weight", type=float, default=0.15)
    p.add_argument("--min-stable-fraction", type=float, default=0.50)
    return p.parse_args()


def load_data(inversion_csv: Path, prior_geometry_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    inv = pd.read_csv(inversion_csv)
    prior = pd.read_csv(prior_geometry_csv).sort_values("channel").reset_index(drop=True)
    return inv, prior


def make_fit_subset(df: pd.DataFrame, use_raw: bool, all_usable: bool, min_weight: float, min_stable_fraction: float) -> pd.DataFrame:
    out = df.copy()
    out = out[(out["channel"] >= CHANNEL_MIN) & (out["channel"] <= CHANNEL_MAX)].copy()

    if use_raw:
        out["obs_dt_s_fit"] = pd.to_numeric(out["observed_dt_ref_s"], errors="coerce")
        subset_name = "raw"
    else:
        out["obs_dt_s_fit"] = pd.to_numeric(out["median_smooth_offset_ms"], errors="coerce") / 1000.0
        subset_name = "smooth"

    out["weight_fit"] = pd.to_numeric(out["weight"], errors="coerce")

    if all_usable:
        mask = out["use_observation"].astype(bool)
        subset_name += "_allusable"
    else:
        rec_ch = out["recommended_channel"].astype(str).str.upper().eq("TRUE")
        rec_glob = out["recommended_global"].astype(str).str.upper().eq("TRUE")
        stable_ok = pd.to_numeric(out["stable_fraction"], errors="coerce").fillna(0.0) >= min_stable_fraction
        mask = (
            out["use_observation"].astype(bool)
            & (out["weight_fit"] >= min_weight)
            & rec_ch
            & rec_glob
            & stable_ok
        )
        subset_name += "_trusted"

    out = out[mask].copy()
    finite_cols = ["obs_dt_s_fit", "tx_x_m", "tx_y_m", "tx_u_m", "weight_fit", "reference_channel", "channel"]
    for col in finite_cols:
        out = out[np.isfinite(pd.to_numeric(out[col], errors="coerce"))].copy()

    out["subset_name"] = subset_name
    return out


def prior_xyz_at_channels(prior: pd.DataFrame, mapped_channels: np.ndarray) -> np.ndarray:
    ch = prior["channel"].to_numpy(dtype=float)
    x = prior["prior_x_m"].to_numpy(dtype=float)
    y = prior["prior_y_m"].to_numpy(dtype=float)
    z = prior["prior_u_m"].to_numpy(dtype=float)
    return np.column_stack([
        np.interp(mapped_channels, ch, x),
        np.interp(mapped_channels, ch, y),
        np.interp(mapped_channels, ch, z),
    ])


def predict_rows_with_offset(df: pd.DataFrame, prior: pd.DataFrame, offset_ch: float, sound_speed: float) -> pd.DataFrame:
    out_rows = []
    prior_min = float(prior["channel"].min())
    prior_max = float(prior["channel"].max())

    for (location, anchor_index, anchor_label), g in df.groupby(["location", "anchor_index", "anchor_label"], sort=False):
        g = g.sort_values("channel").copy()
        ref_ch = float(g["reference_channel"].iloc[0])

        mapped_ch = g["channel"].to_numpy(dtype=float) + offset_ch
        mapped_ref = ref_ch + offset_ch

        valid = (mapped_ch >= prior_min) & (mapped_ch <= prior_max) & (mapped_ref >= prior_min) & (mapped_ref <= prior_max)
        if not np.any(valid):
            continue

        g_valid = g.loc[valid].copy()
        mapped_valid = mapped_ch[valid]

        cable_xyz = prior_xyz_at_channels(prior, mapped_valid)
        ref_xyz = prior_xyz_at_channels(prior, np.array([mapped_ref], dtype=float))[0]

        tx_xyz = np.array([
            float(g_valid["tx_x_m"].iloc[0]),
            float(g_valid["tx_y_m"].iloc[0]),
            float(g_valid["tx_u_m"].iloc[0]),
        ])

        pred_dt = (np.linalg.norm(cable_xyz - tx_xyz[None, :], axis=1) - np.linalg.norm(ref_xyz - tx_xyz)) / sound_speed

        tmp = g_valid.copy()
        tmp["mapped_channel"] = mapped_valid
        tmp["pred_dt_s"] = pred_dt
        tmp["residual_s"] = tmp["pred_dt_s"] - tmp["obs_dt_s_fit"]
        tmp["residual_ms"] = 1000.0 * tmp["residual_s"]
        out_rows.append(tmp)

    if not out_rows:
        return pd.DataFrame()

    return pd.concat(out_rows, ignore_index=True)


def weighted_rmse_ms(df: pd.DataFrame) -> float:
    r = df["residual_ms"].to_numpy(dtype=float)
    w = df["weight_fit"].to_numpy(dtype=float)
    return float(np.sqrt(np.average(r**2, weights=w)))


def weighted_mae_ms(df: pd.DataFrame) -> float:
    r = np.abs(df["residual_ms"].to_numpy(dtype=float))
    w = df["weight_fit"].to_numpy(dtype=float)
    return float(np.average(r, weights=w))


def median_abs_residual_ms(df: pd.DataFrame) -> float:
    return float(np.median(np.abs(df["residual_ms"].to_numpy(dtype=float))))


def scan_one_condition(df: pd.DataFrame, prior: pd.DataFrame, offsets: np.ndarray, sound_speed: float) -> pd.DataFrame:
    rows = []
    for off in offsets:
        pred = predict_rows_with_offset(df, prior, float(off), sound_speed)
        if pred.empty:
            continue
        rows.append({
            "offset_ch": float(off),
            "weighted_rmse_ms": weighted_rmse_ms(pred),
            "weighted_mae_ms": weighted_mae_ms(pred),
            "median_abs_residual_ms": median_abs_residual_ms(pred),
            "n_rows": len(pred),
        })
    return pd.DataFrame(rows).sort_values("offset_ch").reset_index(drop=True)


def build_conditions(inv: pd.DataFrame, include_raw: bool, include_all_usable: bool, min_weight: float, min_stable_fraction: float) -> list[pd.DataFrame]:
    conditions = []

    conditions.append(make_fit_subset(inv, use_raw=False, all_usable=False, min_weight=min_weight, min_stable_fraction=min_stable_fraction))

    if include_all_usable:
        conditions.append(make_fit_subset(inv, use_raw=False, all_usable=True, min_weight=min_weight, min_stable_fraction=min_stable_fraction))

    if include_raw:
        conditions.append(make_fit_subset(inv, use_raw=True, all_usable=False, min_weight=min_weight, min_stable_fraction=min_stable_fraction))
        if include_all_usable:
            conditions.append(make_fit_subset(inv, use_raw=True, all_usable=True, min_weight=min_weight, min_stable_fraction=min_stable_fraction))

    unique = {}
    for df in conditions:
        unique[df["subset_name"].iloc[0]] = df
    return list(unique.values())


def plot_scan_families(scan_all: pd.DataFrame, metric_name: str, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for (subset_name, sound_speed), g in scan_all.groupby(["subset_name", "sound_speed"], sort=False):
        ax.plot(g["offset_ch"], g[metric_name], label=f"{subset_name}, c={sound_speed:.0f}")

    ax.set_xlabel("Global channel offset")
    ylabel = {
        "weighted_rmse_ms": "Weighted RMSE (ms)",
        "weighted_mae_ms": "Weighted MAE (ms)",
        "median_abs_residual_ms": "Median |residual| (ms)",
    }[metric_name]
    ax.set_ylabel(ylabel)
    ax.set_title(f"Global channel-offset robustness scan: {ylabel}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_best_offset_vs_soundspeed(best_df_long: pd.DataFrame, metric_name: str, outpath: Path) -> None:
    metric_df = best_df_long[best_df_long["metric_name"] == metric_name].copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    for subset_name, g in metric_df.groupby("subset_name", sort=False):
        g = g.sort_values("sound_speed")
        ax.plot(g["sound_speed"], g["best_offset_ch"], marker="o", label=subset_name)

    ax.set_xlabel("Sound speed (m/s)")
    ax.set_ylabel("Best offset (channels)")
    title = {
        "weighted_rmse_ms": "Best offset by sound speed (RMSE criterion)",
        "weighted_mae_ms": "Best offset by sound speed (MAE criterion)",
        "median_abs_residual_ms": "Best offset by sound speed (median |residual| criterion)",
    }[metric_name]
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_best_metric_value_vs_soundspeed(best_df_long: pd.DataFrame, metric_name: str, outpath: Path) -> None:
    metric_df = best_df_long[best_df_long["metric_name"] == metric_name].copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    for subset_name, g in metric_df.groupby("subset_name", sort=False):
        g = g.sort_values("sound_speed")
        ax.plot(g["sound_speed"], g["best_metric_value_ms"], marker="o", label=subset_name)

    ax.set_xlabel("Sound speed (m/s)")
    ylabel = {
        "weighted_rmse_ms": "Best weighted RMSE (ms)",
        "weighted_mae_ms": "Best weighted MAE (ms)",
        "median_abs_residual_ms": "Best median |residual| (ms)",
    }[metric_name]
    ax.set_ylabel(ylabel)
    ax.set_title(f"Best achievable misfit by sound speed: {ylabel}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inv, prior = load_data(args.inversion_csv, args.prior_geometry_csv)
    offsets = np.arange(args.offset_min, args.offset_max + args.offset_step, args.offset_step, dtype=float)
    conditions = build_conditions(
        inv,
        include_raw=args.include_raw,
        include_all_usable=args.include_all_usable,
        min_weight=args.min_weight,
        min_stable_fraction=args.min_stable_fraction,
    )

    all_scan_rows = []
    best_rows = []

    for cond_df in conditions:
        subset_name = cond_df["subset_name"].iloc[0]
        for c in args.sound_speeds:
            scan_df = scan_one_condition(cond_df, prior, offsets, float(c))
            if scan_df.empty:
                continue

            scan_df["subset_name"] = subset_name
            scan_df["sound_speed"] = float(c)
            all_scan_rows.append(scan_df)

            for metric_name in ["weighted_rmse_ms", "weighted_mae_ms", "median_abs_residual_ms"]:
                idx = scan_df[metric_name].idxmin()
                best_rows.append({
                    "subset_name": subset_name,
                    "sound_speed": float(c),
                    "metric_name": metric_name,
                    "best_offset_ch": float(scan_df.loc[idx, "offset_ch"]),
                    "best_metric_value_ms": float(scan_df.loc[idx, metric_name]),
                    "n_rows": int(scan_df.loc[idx, "n_rows"]),
                })

    if not all_scan_rows:
        raise RuntimeError("No valid scan results were produced.")

    scan_all = pd.concat(all_scan_rows, ignore_index=True)
    best_df_long = pd.DataFrame(best_rows)

    best_df = (
        best_df_long
        .pivot_table(
            index=["subset_name", "sound_speed"],
            columns="metric_name",
            values=["best_offset_ch", "best_metric_value_ms"],
            aggfunc="first",
        )
    )
    best_df.columns = [f"{lvl0}_{lvl1}" for lvl0, lvl1 in best_df.columns]
    best_df = best_df.reset_index()

    scan_all.to_csv(args.output_dir / "robustness_scan_all_rows.csv", index=False)
    best_df_long.to_csv(args.output_dir / "robustness_best_offsets_long.csv", index=False)
    best_df.to_csv(args.output_dir / "robustness_best_offsets_summary.csv", index=False)

    metadata = {
        "sound_speeds_tested": [float(x) for x in args.sound_speeds],
        "offset_min": int(args.offset_min),
        "offset_max": int(args.offset_max),
        "offset_step": int(args.offset_step),
        "include_raw": bool(args.include_raw),
        "include_all_usable": bool(args.include_all_usable),
        "min_weight": float(args.min_weight),
        "min_stable_fraction": float(args.min_stable_fraction),
        "subset_names": sorted(scan_all["subset_name"].unique().tolist()),
    }
    with open(args.output_dir / "robustness_scan_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    for metric_name in ["weighted_rmse_ms", "weighted_mae_ms", "median_abs_residual_ms"]:
        plot_scan_families(scan_all, metric_name, args.output_dir / f"scan_families_{metric_name}.png")
        plot_best_offset_vs_soundspeed(best_df_long, metric_name, args.output_dir / f"best_offset_vs_soundspeed_{metric_name}.png")
        plot_best_metric_value_vs_soundspeed(best_df_long, metric_name, args.output_dir / f"best_value_vs_soundspeed_{metric_name}.png")

    print(f"Saved outputs to: {args.output_dir}")
    print("Key summary:")
    print(best_df.to_string(index=False))


if __name__ == "__main__":
    main()
