from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


CHANNEL_MIN = 348
CHANNEL_MAX = 2267
DEFAULT_FIXED_SHIFT_CH = 61.26
DEFAULT_SOUND_SPEED_MPS = 1500.0


def str2bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().isin(["TRUE", "1", "YES"])


def load_inputs(inversion_csv: Path, prior_geometry_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = pd.read_csv(inversion_csv)
    geom = pd.read_csv(prior_geometry_csv)
    obs = obs[(obs["channel"] >= CHANNEL_MIN) & (obs["channel"] <= CHANNEL_MAX)].copy()
    geom = geom[(geom["channel"] >= CHANNEL_MIN) & (geom["channel"] <= CHANNEL_MAX)].copy()
    return obs, geom


def build_fit_table(obs: pd.DataFrame, use_raw: bool, all_usable: bool, min_weight: float) -> pd.DataFrame:
    df = obs.copy()

    if "base_valid" in df.columns:
        df["base_valid"] = str2bool_series(df["base_valid"])
    else:
        df["base_valid"] = (
            str2bool_series(df["passed_snr_threshold"]) & ~str2bool_series(df["near_window_edge"])
        )

    if "recommended_channel" in df.columns:
        df["recommended_channel"] = str2bool_series(df["recommended_channel"])
    else:
        df["recommended_channel"] = True

    if "recommended_global" in df.columns:
        df["recommended_global"] = str2bool_series(df["recommended_global"])
    else:
        df["recommended_global"] = True

    if "use_observation" in df.columns:
        df["use_observation"] = str2bool_series(df["use_observation"])
    else:
        df["use_observation"] = True

    if "weight" not in df.columns:
        df["weight"] = 1.0

    if use_raw:
        df["fit_dt_s"] = df["observed_dt_ref_s"]
    else:
        if "median_smooth_offset_ms" not in df.columns:
            raise ValueError("median_smooth_offset_ms not found. Rebuild inversion dataset or use --use-raw")
        df["fit_dt_s"] = 1e-3 * df["median_smooth_offset_ms"].astype(float)

    if all_usable:
        mask = df["use_observation"] & df["fit_dt_s"].notna() & (df["weight"] >= min_weight)
    else:
        mask = (
            df["use_observation"]
            & df["fit_dt_s"].notna()
            & df["recommended_channel"]
            & df["recommended_global"]
            & df["base_valid"]
            & (df["weight"] >= min_weight)
        )
        if "stable_fraction" in df.columns:
            mask &= (df["stable_fraction"].fillna(0.0) >= 0.5)

    out = df[mask].copy()
    if out.empty:
        raise ValueError("No rows available for fitting after masks. Try --all-usable or smaller --min-weight.")
    return out


def make_shifted_geometry(geom: pd.DataFrame, shift_ch: float) -> pd.DataFrame:
    g = geom.sort_values("channel").copy()
    ch = g["channel"].to_numpy(dtype=float)

    cols = [
        "prior_x_m",
        "prior_y_m",
        "prior_u_m",
        "tangent_x",
        "tangent_y",
        "normal_x",
        "normal_y",
        "cum_dist_horizontal_m",
        "cum_dist_3d_m",
    ]
    missing = [c for c in cols if c not in g.columns]
    if missing:
        raise ValueError(f"prior_geometry.csv missing columns: {missing}")

    ch_query = ch + shift_ch
    out = pd.DataFrame({"channel": ch})

    for c in cols:
        f = interp1d(ch, g[c].to_numpy(dtype=float), kind="linear", bounds_error=False, fill_value="extrapolate")
        out[c] = f(ch_query)

    # re-normalize tangent/normal after interpolation
    tnorm = np.hypot(out["tangent_x"], out["tangent_y"]).to_numpy()
    tnorm = np.where(tnorm <= 1e-12, 1.0, tnorm)
    out["tangent_x"] = out["tangent_x"] / tnorm
    out["tangent_y"] = out["tangent_y"] / tnorm

    nnorm = np.hypot(out["normal_x"], out["normal_y"]).to_numpy()
    nnorm = np.where(nnorm <= 1e-12, 1.0, nnorm)
    out["normal_x"] = out["normal_x"] / nnorm
    out["normal_y"] = out["normal_y"] / nnorm

    out["source_channel_for_shift"] = ch_query
    return out


def predict_relative_times(cable_xyz: np.ndarray, tx_xyz: np.ndarray, ref_idx: int, c: float) -> np.ndarray:
    ranges = np.linalg.norm(cable_xyz - tx_xyz[None, :], axis=1)
    return (ranges - ranges[ref_idx]) / c


def evaluate_translation(
    dx: float,
    dy: float,
    fit_df: pd.DataFrame,
    shifted_geom: pd.DataFrame,
    sound_speed: float,
) -> dict:
    g = shifted_geom.sort_values("channel").copy()
    channels = g["channel"].astype(int).to_numpy()
    ch_to_idx = {int(ch): i for i, ch in enumerate(channels)}

    cable_xyz = np.column_stack([
        g["prior_x_m"].to_numpy(dtype=float) + dx,
        g["prior_y_m"].to_numpy(dtype=float) + dy,
        g["prior_u_m"].to_numpy(dtype=float),
    ])

    pred_rows = []
    resid_all = []
    w_all = []

    for (location, anchor_index), gg in fit_df.groupby(["location", "anchor_index"]):
        gg = gg.sort_values("channel").copy()
        ref_ch = int(gg["reference_channel"].iloc[0])
        if ref_ch not in ch_to_idx:
            continue
        ref_idx = ch_to_idx[ref_ch]

        tx_xyz = np.array([
            float(gg["tx_x_m"].iloc[0]),
            float(gg["tx_y_m"].iloc[0]),
            float(gg.get("tx_u_m", gg["tx_z_m"]).iloc[0]),
        ])
        pred_dt = predict_relative_times(cable_xyz, tx_xyz, ref_idx, sound_speed)

        idxs = gg["channel"].astype(int).map(ch_to_idx).to_numpy()
        pred = pred_dt[idxs]
        obs = gg["fit_dt_s"].to_numpy(dtype=float)
        w = gg["weight"].to_numpy(dtype=float)
        resid = pred - obs

        tmp = gg[["location", "anchor_index", "anchor_label", "channel", "reference_channel", "fit_dt_s", "weight"]].copy()
        tmp["pred_dt_s"] = pred
        tmp["residual_s"] = resid
        pred_rows.append(tmp)
        resid_all.append(resid)
        w_all.append(w)

    pred_df = pd.concat(pred_rows, ignore_index=True)
    resid = np.concatenate(resid_all)
    w = np.concatenate(w_all)
    wsum = np.sum(w) + 1e-12

    wrmse_s = float(np.sqrt(np.sum(w * resid**2) / wsum))
    wmae_s = float(np.sum(w * np.abs(resid)) / wsum)
    med_abs_s = float(np.median(np.abs(resid)))

    return {
        "dx_m": dx,
        "dy_m": dy,
        "weighted_rmse_s": wrmse_s,
        "weighted_mae_s": wmae_s,
        "median_abs_residual_s": med_abs_s,
        "pred_df": pred_df,
        "channels": channels,
        "cable_xyz": cable_xyz,
        "shifted_geom": g,
    }


def grid_search_translation(
    fit_df: pd.DataFrame,
    shifted_geom: pd.DataFrame,
    sound_speed: float,
    dx_min: float,
    dx_max: float,
    dy_min: float,
    dy_max: float,
    step: float,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    best = None
    best_score = np.inf

    dx_values = np.arange(dx_min, dx_max + 0.5 * step, step)
    dy_values = np.arange(dy_min, dy_max + 0.5 * step, step)

    for dy in dy_values:
        for dx in dx_values:
            res = evaluate_translation(dx, dy, fit_df, shifted_geom, sound_speed)
            rows.append({
                "dx_m": dx,
                "dy_m": dy,
                "weighted_rmse_ms": 1e3 * res["weighted_rmse_s"],
                "weighted_mae_ms": 1e3 * res["weighted_mae_s"],
                "median_abs_residual_ms": 1e3 * res["median_abs_residual_s"],
            })
            score = res["weighted_rmse_s"]
            if score < best_score:
                best_score = score
                best = res

    return pd.DataFrame(rows), best


def summarize_groups(pred_df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (loc, anchor), g in pred_df.groupby(["location", "anchor_index"]):
        resid = g["residual_s"].to_numpy(dtype=float)
        w = g["weight"].to_numpy(dtype=float)
        wsum = np.sum(w) + 1e-12
        out.append({
            "location": loc,
            "anchor_index": anchor,
            "anchor_label": g["anchor_label"].iloc[0],
            "n_rows": len(g),
            "weighted_rmse_ms": 1e3 * np.sqrt(np.sum(w * resid**2) / wsum),
            "weighted_mae_ms": 1e3 * np.sum(w * np.abs(resid)) / wsum,
            "median_abs_residual_ms": 1e3 * np.median(np.abs(resid)),
        })
    return pd.DataFrame(out)


def summarize_channels(pred_df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for ch, g in pred_df.groupby("channel"):
        resid = g["residual_s"].to_numpy(dtype=float)
        w = g["weight"].to_numpy(dtype=float)
        wsum = np.sum(w) + 1e-12
        out.append({
            "channel": int(ch),
            "weighted_rmse_ms": 1e3 * np.sqrt(np.sum(w * resid**2) / wsum),
            "weighted_mae_ms": 1e3 * np.sum(w * np.abs(resid)) / wsum,
            "median_abs_residual_ms": 1e3 * np.median(np.abs(resid)),
            "n_rows": len(g),
        })
    return pd.DataFrame(out).sort_values("channel")


def add_translated_latlon_rows(shifted_geom: pd.DataFrame, dx: float, dy: float, obs: pd.DataFrame) -> pd.DataFrame:
    g = shifted_geom.copy()
    g["fit_x_m"] = g["prior_x_m"] + dx
    g["fit_y_m"] = g["prior_y_m"] + dy
    g["fit_u_m"] = g["prior_u_m"]
    # preserve origin info if available from inversion obs
    for c in ["enu_origin_lat_deg", "enu_origin_lon_deg", "enu_origin_h_m"]:
        if c in obs.columns:
            g[c] = obs[c].iloc[0]
    return g


def make_plots(grid_df: pd.DataFrame, best: dict, channel_summary: pd.DataFrame, pred_df: pd.DataFrame, shifted_geom: pd.DataFrame, outdir: Path):
    # 1) heatmap-like scatter for translation scan
    fig, ax = plt.subplots(figsize=(8, 6))
    pivot = grid_df.pivot(index="dy_m", columns="dx_m", values="weighted_rmse_ms")
    im = ax.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()],
    )
    ax.scatter([best["dx_m"]], [best["dy_m"]], c="red", marker="x", s=100, label=f"Best ({best['dx_m']:.1f}, {best['dy_m']:.1f}) m")
    ax.set_xlabel("Global x translation dx (m)")
    ax.set_ylabel("Global y translation dy (m)")
    ax.set_title("Global XY translation scan after fixed tangential shift")
    ax.legend(loc="upper right")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Weighted RMSE (ms)")
    fig.tight_layout()
    fig.savefig(outdir / "translation_scan_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 2) path plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(shifted_geom["prior_x_m"], shifted_geom["prior_y_m"], label="Shifted prior", lw=2)
    ax.plot(best["cable_xyz"][:, 0], best["cable_xyz"][:, 1], label=f"Shifted + translated fit (dx={best['dx_m']:.1f}, dy={best['dy_m']:.1f})", lw=2)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title("Shifted prior vs globally translated fitted path")
    ax.axis("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "shifted_prior_vs_translated_fit_path.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 3) observed vs predicted by group
    groups = list(pred_df.groupby(["location", "anchor_index", "anchor_label"]))
    n = len(groups)
    fig, axes = plt.subplots(n, 1, figsize=(9, max(2.4 * n, 8)), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, ((loc, anchor, label), g) in zip(axes, groups):
        gg = g.sort_values("channel")
        ax.plot(gg["channel"], 1e3 * gg["fit_dt_s"], label="Observed for fit")
        ax.plot(gg["channel"], 1e3 * gg["pred_dt_s"], label="Predicted")
        ax.set_ylabel("dt to ref (ms)")
        ax.set_title(f"{loc} | anchor {anchor} | {label}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Channel")
    fig.tight_layout()
    fig.savefig(outdir / "observed_vs_predicted_by_location_anchor.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 4) channel residual summary
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(channel_summary["channel"], channel_summary["weighted_rmse_ms"], label="Weighted RMSE")
    ax.plot(channel_summary["channel"], channel_summary["median_abs_residual_ms"], label="Median |residual|")
    ax.plot(channel_summary["channel"], channel_summary["weighted_mae_ms"], label="Weighted MAE")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Residual (ms)")
    ax.set_title("Timing misfit by channel after global XY translation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "residual_by_channel.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 5) path displacement vector quick-look
    fig, ax = plt.subplots(figsize=(8, 4))
    channels = shifted_geom["channel"].to_numpy()
    dx_arr = best["cable_xyz"][:, 0] - shifted_geom["prior_x_m"].to_numpy()
    dy_arr = best["cable_xyz"][:, 1] - shifted_geom["prior_y_m"].to_numpy()
    ax.plot(channels, dx_arr, label="dx(ch)")
    ax.plot(channels, dy_arr, label="dy(ch)")
    ax.axhline(0.0, ls="--")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Translation component (m)")
    ax.set_title("Applied global XY translation components vs channel")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "translation_components_vs_channel.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit global XY translation after fixed tangential shift.")
    root = Path(r"D:\Singapore Data\Cable")
    p.add_argument("--inversion-csv", type=Path, default=root / "inversion_observations.csv")
    p.add_argument("--prior-geometry-csv", type=Path, default=root / "prior_geometry.csv")
    p.add_argument("--output-dir", type=Path, default=root / "global_xy_translation_with_fixed_shift_outputs")
    p.add_argument("--fixed-shift", type=float, default=DEFAULT_FIXED_SHIFT_CH)
    p.add_argument("--sound-speed", type=float, default=DEFAULT_SOUND_SPEED_MPS)
    p.add_argument("--dx-min", type=float, default=-50.0)
    p.add_argument("--dx-max", type=float, default=50.0)
    p.add_argument("--dy-min", type=float, default=-50.0)
    p.add_argument("--dy-max", type=float, default=50.0)
    p.add_argument("--step", type=float, default=1.0)
    p.add_argument("--use-raw", action="store_true")
    p.add_argument("--all-usable", action="store_true")
    p.add_argument("--min-weight", type=float, default=0.05)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    obs, geom = load_inputs(args.inversion_csv, args.prior_geometry_csv)
    fit_df = build_fit_table(obs, use_raw=args.use_raw, all_usable=args.all_usable, min_weight=args.min_weight)
    shifted_geom = make_shifted_geometry(geom, args.fixed_shift)

    grid_df, best = grid_search_translation(
        fit_df,
        shifted_geom,
        sound_speed=args.sound_speed,
        dx_min=args.dx_min,
        dx_max=args.dx_max,
        dy_min=args.dy_min,
        dy_max=args.dy_max,
        step=args.step,
    )

    pred_df = best["pred_df"].copy()
    group_summary = summarize_groups(pred_df)
    channel_summary = summarize_channels(pred_df)
    fit_geom = add_translated_latlon_rows(best["shifted_geom"], best["dx_m"], best["dy_m"], obs)

    grid_df.to_csv(args.output_dir / "translation_scan_summary.csv", index=False)
    pred_df.to_csv(args.output_dir / "predicted_vs_observed_rows.csv", index=False)
    group_summary.to_csv(args.output_dir / "group_misfit_summary.csv", index=False)
    channel_summary.to_csv(args.output_dir / "channel_misfit_summary.csv", index=False)
    fit_geom.to_csv(args.output_dir / "translated_shifted_geometry.csv", index=False)

    metrics = {
        "fixed_shift_channels": args.fixed_shift,
        "best_dx_m": best["dx_m"],
        "best_dy_m": best["dy_m"],
        "weighted_rmse_ms": 1e3 * best["weighted_rmse_s"],
        "weighted_mae_ms": 1e3 * best["weighted_mae_s"],
        "median_abs_residual_ms": 1e3 * best["median_abs_residual_s"],
        "n_fit_rows": int(len(pred_df)),
        "use_raw": bool(args.use_raw),
        "all_usable": bool(args.all_usable),
    }
    with open(args.output_dir / "fit_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    make_plots(grid_df, best, channel_summary, pred_df, shifted_geom, args.output_dir)

    print(f"Saved outputs to: {args.output_dir}")
    print(f"Rows used for fit: {len(pred_df)}")
    print(f"Fixed tangential shift: {args.fixed_shift:.2f} ch")
    print(f"Best translation: dx={best['dx_m']:.2f} m, dy={best['dy_m']:.2f} m")
    print(f"Weighted RMSE: {1e3 * best['weighted_rmse_s']:.2f} ms")
    print(f"Weighted MAE : {1e3 * best['weighted_mae_s']:.2f} ms")
    print(f"Median |res| : {1e3 * best['median_abs_residual_s']:.2f} ms")


if __name__ == "__main__":
    main()
