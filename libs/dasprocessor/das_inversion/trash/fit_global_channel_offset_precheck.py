from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CHANNEL_MIN = 348
CHANNEL_MAX = 2267
DEFAULT_SOUND_SPEED = 1500.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scan a global channel offset between observed channels and prior-path channels."
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
        default=Path(r"D:\Singapore Data\Cable\global_channel_offset_precheck_outputs"),
    )
    p.add_argument("--offset-min", type=int, default=-150)
    p.add_argument("--offset-max", type=int, default=150)
    p.add_argument("--offset-step", type=int, default=1)
    p.add_argument("--sound-speed", type=float, default=DEFAULT_SOUND_SPEED)
    p.add_argument(
        "--use-raw",
        action="store_true",
        help="Use observed_dt_ref_s instead of median_smooth_offset_ms.",
    )
    p.add_argument(
        "--all-usable",
        action="store_true",
        help="Use all use_observation rows instead of stricter trusted subset.",
    )
    p.add_argument("--min-weight", type=float, default=0.15)
    p.add_argument("--min-stable-fraction", type=float, default=0.50)
    return p.parse_args()


def make_fit_subset(df: pd.DataFrame, use_raw: bool, all_usable: bool, min_weight: float, min_stable_fraction: float) -> pd.DataFrame:
    out = df.copy()
    out = out[(out["channel"] >= CHANNEL_MIN) & (out["channel"] <= CHANNEL_MAX)].copy()

    if use_raw:
        out["obs_dt_s_fit"] = out["observed_dt_ref_s"].astype(float)
    else:
        out["obs_dt_s_fit"] = out["median_smooth_offset_ms"].astype(float) / 1000.0

    out["weight_fit"] = out["weight"].astype(float)

    if all_usable:
        mask = out["use_observation"].astype(bool)
    else:
        rec_ch = out["recommended_channel"].astype(str).str.upper().eq("TRUE")
        rec_glob = out["recommended_global"].astype(str).str.upper().eq("TRUE")
        stable_ok = out["stable_fraction"].fillna(0.0) >= min_stable_fraction
        mask = (
            out["use_observation"].astype(bool)
            & (out["weight_fit"] >= min_weight)
            & rec_ch
            & rec_glob
            & stable_ok
        )

    out = out[mask].copy()
    out = out[np.isfinite(out["obs_dt_s_fit"])].copy()
    out = out[np.isfinite(out["tx_x_m"]) & np.isfinite(out["tx_y_m"]) & np.isfinite(out["tx_u_m"])].copy()
    return out


def load_data(inversion_csv: Path, prior_geometry_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    inv = pd.read_csv(inversion_csv)
    prior = pd.read_csv(prior_geometry_csv)
    prior = prior.sort_values("channel").reset_index(drop=True)
    return inv, prior


def prior_xyz_at_channels(prior: pd.DataFrame, mapped_channels: np.ndarray) -> np.ndarray:
    ch = prior["channel"].to_numpy(dtype=float)
    x = prior["prior_x_m"].to_numpy(dtype=float)
    y = prior["prior_y_m"].to_numpy(dtype=float)
    z = prior["prior_u_m"].to_numpy(dtype=float)

    xyz = np.column_stack([
        np.interp(mapped_channels, ch, x),
        np.interp(mapped_channels, ch, y),
        np.interp(mapped_channels, ch, z),
    ])
    return xyz


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


def summarize_solution(pred_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grp_rows = []
    for (location, anchor_index, anchor_label), g in pred_df.groupby(["location", "anchor_index", "anchor_label"], sort=False):
        grp_rows.append({
            "location": location,
            "anchor_index": int(anchor_index),
            "anchor_label": anchor_label,
            "weighted_rmse_ms": weighted_rmse_ms(g),
            "weighted_mae_ms": weighted_mae_ms(g),
            "median_abs_residual_ms": median_abs_residual_ms(g),
            "n_rows": len(g),
        })
    grp = pd.DataFrame(grp_rows)

    ch_rows = []
    for channel, g in pred_df.groupby("channel", sort=True):
        ch_rows.append({
            "channel": int(channel),
            "weighted_rmse_ms": weighted_rmse_ms(g),
            "weighted_mae_ms": weighted_mae_ms(g),
            "median_abs_residual_ms": median_abs_residual_ms(g),
            "n_rows": len(g),
        })
    chs = pd.DataFrame(ch_rows)
    return grp, chs


def scan_offsets(df: pd.DataFrame, prior: pd.DataFrame, offsets: np.ndarray, sound_speed: float) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    scan_rows = []
    best_offset = None
    best_score = np.inf
    best_pred = pd.DataFrame()

    for off in offsets:
        pred = predict_rows_with_offset(df, prior, float(off), sound_speed)
        if pred.empty:
            continue

        wrmse = weighted_rmse_ms(pred)
        wmae = weighted_mae_ms(pred)
        med = median_abs_residual_ms(pred)

        scan_rows.append({
            "offset_ch": float(off),
            "weighted_rmse_ms": wrmse,
            "weighted_mae_ms": wmae,
            "median_abs_residual_ms": med,
            "n_rows": len(pred),
        })

        if wrmse < best_score:
            best_score = wrmse
            best_offset = float(off)
            best_pred = pred.copy()

    scan_df = pd.DataFrame(scan_rows).sort_values("offset_ch").reset_index(drop=True)
    if best_offset is None:
        raise RuntimeError("No valid offsets produced predictions. Check scan range and data.")
    return scan_df, best_offset, best_pred


def plot_scan_curve(scan_df: pd.DataFrame, best_offset: float, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(scan_df["offset_ch"], scan_df["weighted_rmse_ms"], label="Weighted RMSE (ms)")
    ax.plot(scan_df["offset_ch"], scan_df["median_abs_residual_ms"], label="Median |residual| (ms)")
    ax.axvline(best_offset, linestyle="--", label=f"Best offset = {best_offset:.2f} ch")
    ax.set_xlabel("Global channel offset")
    ax.set_ylabel("Misfit (ms)")
    ax.set_title("Global channel-offset scan")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_observed_vs_pred(pred_df: pd.DataFrame, outpath: Path) -> None:
    groups = list(pred_df.groupby(["location", "anchor_index", "anchor_label"], sort=False))
    n = len(groups)
    fig, axes = plt.subplots(n, 1, figsize=(9, max(2.5*n, 6)), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, ((location, anchor_index, anchor_label), g) in zip(axes, groups):
        g = g.sort_values("channel")
        ax.plot(g["channel"], 1000.0 * g["obs_dt_s_fit"], label="Observed for fit")
        ax.plot(g["channel"], 1000.0 * g["pred_dt_s"], label="Predicted")
        ax.set_ylabel("dt to ref (ms)")
        ax.set_title(f"{location} | anchor {anchor_index} | {anchor_label}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Channel")
    fig.suptitle("Observed vs predicted after global channel-offset precheck", y=0.995, fontsize=14)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_residual_by_channel(ch_summary: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(ch_summary["channel"], ch_summary["weighted_rmse_ms"], label="Weighted RMSE")
    ax.plot(ch_summary["channel"], ch_summary["median_abs_residual_ms"], label="Median |residual|")
    ax.plot(ch_summary["channel"], ch_summary["weighted_mae_ms"], label="Weighted MAE")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Residual (ms)")
    ax.set_title("Timing misfit by channel after global channel-offset precheck")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_prior_vs_offset_path(prior: pd.DataFrame, best_offset: float, outpath: Path) -> None:
    ch = prior["channel"].to_numpy(dtype=float)
    x = prior["prior_x_m"].to_numpy(dtype=float)
    y = prior["prior_y_m"].to_numpy(dtype=float)
    shifted_ch = ch + best_offset
    valid = (shifted_ch >= ch.min()) & (shifted_ch <= ch.max())
    x_shift = np.interp(shifted_ch[valid], ch, x)
    y_shift = np.interp(shifted_ch[valid], ch, y)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(x, y, label="Prior path", linewidth=2.5)
    ax.plot(x_shift, y_shift, label=f"Offset-applied prior (Δch={best_offset:.2f})", linewidth=2.5)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title("Prior path vs globally offset prior")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inv, prior = load_data(args.inversion_csv, args.prior_geometry_csv)
    fit_df = make_fit_subset(inv, args.use_raw, args.all_usable, args.min_weight, args.min_stable_fraction)

    offsets = np.arange(args.offset_min, args.offset_max + args.offset_step, args.offset_step, dtype=float)
    scan_df, best_offset, best_pred = scan_offsets(fit_df, prior, offsets, args.sound_speed)
    grp, ch_summary = summarize_solution(best_pred)

    scan_df.to_csv(args.output_dir / "offset_scan_summary.csv", index=False)
    best_pred.to_csv(args.output_dir / "predicted_vs_observed_rows.csv", index=False)
    grp.to_csv(args.output_dir / "group_misfit_summary.csv", index=False)
    ch_summary.to_csv(args.output_dir / "channel_misfit_summary.csv", index=False)

    metrics = {
        "best_global_channel_offset_ch": best_offset,
        "n_rows_used": int(len(best_pred)),
        "weighted_rmse_ms_mean": weighted_rmse_ms(best_pred),
        "weighted_mae_ms_mean": weighted_mae_ms(best_pred),
        "median_abs_residual_ms_mean": median_abs_residual_ms(best_pred),
        "use_raw": bool(args.use_raw),
        "all_usable": bool(args.all_usable),
        "min_weight": float(args.min_weight),
        "min_stable_fraction": float(args.min_stable_fraction),
    }
    with open(args.output_dir / "fit_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_scan_curve(scan_df, best_offset, args.output_dir / "offset_scan_curve.png")
    plot_observed_vs_pred(best_pred, args.output_dir / "observed_vs_predicted_by_location_anchor.png")
    plot_residual_by_channel(ch_summary, args.output_dir / "residual_by_channel.png")
    plot_prior_vs_offset_path(prior, best_offset, args.output_dir / "prior_vs_offset_path.png")

    print(f"Best global channel offset: {best_offset:.2f} ch")
    print(f"Weighted RMSE: {metrics['weighted_rmse_ms_mean']:.3f} ms")
    print(f"Median |residual|: {metrics['median_abs_residual_ms_mean']:.3f} ms")
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
