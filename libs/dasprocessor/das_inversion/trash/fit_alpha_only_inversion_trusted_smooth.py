from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize


CHANNEL_MIN_DEFAULT = 348
CHANNEL_MAX_DEFAULT = 2267
SOUND_SPEED_DEFAULT = 1500.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Alpha-only cable inversion using trusted/smoothed arrival curves."
    )
    parser.add_argument(
        "--obs-csv",
        type=Path,
        default=Path(r"D:\Singapore Data\Cable\inversion_observations.csv"),
    )
    parser.add_argument(
        "--prior-csv",
        type=Path,
        default=Path(r"D:\Singapore Data\Cable\prior_geometry.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"D:\Singapore Data\Cable\alpha_only_inversion_trusted_smooth_outputs"),
    )
    parser.add_argument("--channel-min", type=int, default=CHANNEL_MIN_DEFAULT)
    parser.add_argument("--channel-max", type=int, default=CHANNEL_MAX_DEFAULT)
    parser.add_argument("--sound-speed", type=float, default=SOUND_SPEED_DEFAULT)
    parser.add_argument("--n-control", type=int, default=28)
    parser.add_argument("--lambda-smooth", type=float, default=50.0)
    parser.add_argument("--lambda-amp", type=float, default=0.01)
    parser.add_argument("--max-alpha-m", type=float, default=80.0)
    parser.add_argument("--global-weight-floor", type=float, default=0.67)
    parser.add_argument("--location-weight-floor", type=float, default=0.55)
    parser.add_argument("--min-stable-fraction", type=float, default=0.50)
    parser.add_argument("--min-weight", type=float, default=0.25)
    parser.add_argument("--use-smoothed-offset", action="store_true", default=True)
    parser.add_argument("--no-use-smoothed-offset", dest="use_smoothed_offset", action="store_false")
    parser.add_argument("--plot-top-groups", type=int, default=12)
    return parser.parse_args()


def ensure_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    vals = s.astype(str).str.strip().str.upper()
    return vals.isin(["TRUE", "1", "YES", "Y"])


def load_tables(obs_csv: Path, prior_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = pd.read_csv(obs_csv)
    prior = pd.read_csv(prior_csv)
    return obs, prior


def prepare_observations(obs: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    df = obs.copy()
    df = df[(df["channel"] >= args.channel_min) & (df["channel"] <= args.channel_max)].copy()

    # Normalize boolean-like columns
    for col in ["recommended_channel", "recommended_global", "use_observation", "base_valid"]:
        if col in df.columns:
            df[col] = ensure_bool_series(df[col])

    # Build the modeled observation. Use smoothed relative arrival when available.
    if args.use_smoothed_offset and "median_smooth_offset_ms" in df.columns:
        df["obs_dt_s"] = df["median_smooth_offset_ms"] / 1000.0
        df["obs_source"] = "median_smooth_offset_ms"
    else:
        df["obs_dt_s"] = df["observed_dt_ref_s"]
        df["obs_source"] = "observed_dt_ref_s"

    # Weighting: start from existing weight and strengthen trusted channels.
    if "weight" not in df.columns:
        df["weight"] = 1.0

    w = df["weight"].fillna(0.0).astype(float).copy()

    if "recommended_channel" in df.columns:
        w *= np.where(df["recommended_channel"], 1.0, 0.2)
    if "recommended_global" in df.columns:
        w *= np.where(df["recommended_global"], 1.0, 0.5)
    if "stable_fraction" in df.columns:
        w *= np.clip(df["stable_fraction"].fillna(0.0).to_numpy(dtype=float), 0.1, 1.0)
    if "mean_channel_trust_score" in df.columns:
        w *= np.clip(df["mean_channel_trust_score"].fillna(0.0).to_numpy(dtype=float), 0.1, 1.0)

    df["fit_weight"] = w

    # Stronger mask than before: use stable/trusted channels only.
    mask = pd.Series(True, index=df.index)
    if "use_observation" in df.columns:
        mask &= df["use_observation"]
    mask &= df["obs_dt_s"].notna()
    mask &= df["tx_x_m"].notna() & df["tx_y_m"].notna() & df["tx_u_m"].notna()

    if "recommended_global" in df.columns:
        mask &= (df["recommended_global"]) | (df["recommended_fraction"].fillna(0.0) >= args.global_weight_floor)
    if "recommended_channel" in df.columns:
        mask &= (df["recommended_channel"]) | (df["channel_trust_score"].fillna(0.0) >= args.location_weight_floor)
    if "stable_fraction" in df.columns:
        mask &= df["stable_fraction"].fillna(0.0) >= args.min_stable_fraction

    mask &= df["fit_weight"].fillna(0.0) >= args.min_weight

    df["use_for_fit"] = mask
    return df


def prepare_prior(prior: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    g = prior.copy()
    g = g[(g["channel"] >= args.channel_min) & (g["channel"] <= args.channel_max)].copy()
    g = g.sort_values("channel").reset_index(drop=True)
    required = [
        "channel", "prior_x_m", "prior_y_m", "prior_u_m",
        "normal_x", "normal_y", "tangent_x", "tangent_y"
    ]
    missing = [c for c in required if c not in g.columns]
    if missing:
        raise ValueError(f"prior_geometry.csv missing required columns: {missing}")
    return g


def build_candidate_curve(prior_df: pd.DataFrame, channels_ctrl: np.ndarray, alpha_ctrl: np.ndarray) -> pd.DataFrame:
    channels_full = prior_df["channel"].to_numpy(dtype=float)
    cs = CubicSpline(channels_ctrl, alpha_ctrl, bc_type="natural")
    alpha = cs(channels_full)

    out = prior_df.copy()
    out["alpha_m"] = alpha
    out["fit_x_m"] = out["prior_x_m"] + alpha * out["normal_x"]
    out["fit_y_m"] = out["prior_y_m"] + alpha * out["normal_y"]
    out["fit_u_m"] = out["prior_u_m"]
    return out


def predict_relative_times(cable_xyz: np.ndarray, tx_xyz: np.ndarray, ref_idx: int, c: float) -> np.ndarray:
    ranges = np.linalg.norm(cable_xyz - tx_xyz[None, :], axis=1)
    return (ranges - ranges[ref_idx]) / c


def objective_alpha_only(
    alpha_ctrl: np.ndarray,
    obs_fit: pd.DataFrame,
    prior_df: pd.DataFrame,
    channels_ctrl: np.ndarray,
    ch_to_idx: dict[int, int],
    sound_speed: float,
    lambda_smooth: float,
    lambda_amp: float,
) -> float:
    fit_df = build_candidate_curve(prior_df, channels_ctrl, alpha_ctrl)
    cable_xyz = fit_df[["fit_x_m", "fit_y_m", "fit_u_m"]].to_numpy(dtype=float)

    resid_chunks = []
    weight_chunks = []

    for (location, anchor_index), g in obs_fit.groupby(["location", "anchor_index"], sort=False):
        if g.empty:
            continue
        ref_ch = int(g["reference_channel"].iloc[0])
        if ref_ch not in ch_to_idx:
            continue
        ref_idx = ch_to_idx[ref_ch]
        tx_xyz = np.array([
            float(g["tx_x_m"].iloc[0]),
            float(g["tx_y_m"].iloc[0]),
            float(g["tx_u_m"].iloc[0]),
        ])
        pred_dt = predict_relative_times(cable_xyz, tx_xyz, ref_idx, sound_speed)
        idxs = g["channel"].map(ch_to_idx).to_numpy(dtype=int)
        pred = pred_dt[idxs]
        obs = g["obs_dt_s"].to_numpy(dtype=float)
        w = g["fit_weight"].to_numpy(dtype=float)
        resid_chunks.append(pred - obs)
        weight_chunks.append(w)

    if not resid_chunks:
        return 1e15

    resid = np.concatenate(resid_chunks)
    w = np.concatenate(weight_chunks)
    data_term = np.sum(w * resid**2)

    d2 = np.diff(alpha_ctrl, n=2)
    smooth_term = np.sum(d2**2)
    amp_term = np.sum(alpha_ctrl**2)

    return data_term + lambda_smooth * smooth_term + lambda_amp * amp_term


def build_predicted_rows(
    obs_df: pd.DataFrame,
    fit_df: pd.DataFrame,
    sound_speed: float,
) -> pd.DataFrame:
    ch_to_idx = {int(ch): i for i, ch in enumerate(fit_df["channel"].tolist())}
    cable_xyz = fit_df[["fit_x_m", "fit_y_m", "fit_u_m"]].to_numpy(dtype=float)

    rows = []
    for (location, anchor_index), g in obs_df.groupby(["location", "anchor_index"], sort=False):
        ref_ch = int(g["reference_channel"].iloc[0])
        if ref_ch not in ch_to_idx:
            continue
        ref_idx = ch_to_idx[ref_ch]
        tx_xyz = np.array([
            float(g["tx_x_m"].iloc[0]),
            float(g["tx_y_m"].iloc[0]),
            float(g["tx_u_m"].iloc[0]),
        ])
        pred_dt = predict_relative_times(cable_xyz, tx_xyz, ref_idx, sound_speed)

        for _, r in g.iterrows():
            ch = int(r["channel"])
            if ch not in ch_to_idx:
                continue
            pred = float(pred_dt[ch_to_idx[ch]])
            obs = float(r["obs_dt_s"]) if pd.notna(r["obs_dt_s"]) else np.nan
            resid = pred - obs if np.isfinite(obs) else np.nan
            row = dict(r)
            row["pred_dt_s"] = pred
            row["pred_dt_ms"] = 1000.0 * pred
            row["obs_dt_ms"] = 1000.0 * obs if np.isfinite(obs) else np.nan
            row["residual_s"] = resid
            row["residual_ms"] = 1000.0 * resid if np.isfinite(resid) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(pred_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    fit_only = pred_rows[pred_rows["use_for_fit"]].copy()

    group_summary = (
        fit_only.groupby(["location", "anchor_index", "anchor_label"], as_index=False)
        .agg(
            n_rows=("channel", "size"),
            weighted_rmse_ms=("residual_ms", lambda x: float(np.sqrt(np.mean(np.square(x))))),
            median_abs_residual_ms=("residual_ms", lambda x: float(np.nanmedian(np.abs(x)))),
            mean_abs_residual_ms=("residual_ms", lambda x: float(np.nanmean(np.abs(x)))),
        )
    )

    channel_summary = (
        fit_only.groupby("channel", as_index=False)
        .agg(
            rmse_ms=("residual_ms", lambda x: float(np.sqrt(np.mean(np.square(x))))),
            median_abs_residual_ms=("residual_ms", lambda x: float(np.nanmedian(np.abs(x)))),
            n_rows=("residual_ms", "size"),
        )
    )

    metrics = {
        "n_fit_rows": int(len(fit_only)),
        "global_rmse_ms": float(np.sqrt(np.nanmean(np.square(fit_only["residual_ms"].to_numpy(dtype=float))))) if len(fit_only) else np.nan,
        "global_median_abs_residual_ms": float(np.nanmedian(np.abs(fit_only["residual_ms"].to_numpy(dtype=float)))) if len(fit_only) else np.nan,
        "global_mean_abs_residual_ms": float(np.nanmean(np.abs(fit_only["residual_ms"].to_numpy(dtype=float)))) if len(fit_only) else np.nan,
    }
    return group_summary, channel_summary, metrics


def plot_path(prior_df: pd.DataFrame, fit_df: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(prior_df["prior_x_m"], prior_df["prior_y_m"], label="Prior path", linewidth=2.5)
    ax.plot(fit_df["fit_x_m"], fit_df["fit_y_m"], label="Alpha-only fit", linewidth=2.5)
    ax.set_title("Prior path vs alpha-only fitted path (trusted/smoothed inversion)")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_alpha(fit_df: pd.DataFrame, ctrl_df: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(fit_df["channel"], fit_df["alpha_m"], label="Interpolated alpha(channel)", linewidth=2.5)
    ax.scatter(ctrl_df["channel"], ctrl_df["alpha_m"], label="Control points", s=60)
    ax.axhline(0.0, linestyle="--", linewidth=1.5)
    ax.set_title("Optimized lateral correction relative to prior")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Lateral correction alpha (m)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_residual_by_channel(channel_summary: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(channel_summary["channel"], channel_summary["rmse_ms"], label="RMSE", linewidth=2.0)
    ax.plot(channel_summary["channel"], channel_summary["median_abs_residual_ms"], label="Median |residual|", linewidth=2.0)
    ax.set_title("Timing misfit by channel after alpha-only inversion")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Residual (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_observed_vs_predicted(pred_rows: pd.DataFrame, outpath: Path, n_groups: int = 12) -> None:
    fit_only = pred_rows[pred_rows["use_for_fit"]].copy()
    groups = list(fit_only.groupby(["location", "anchor_index", "anchor_label"], sort=False))
    groups = groups[:n_groups]
    n = len(groups)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(14, max(3 * n, 8)), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, ((location, anchor_index, anchor_label), g) in zip(axes, groups):
        g = g.sort_values("channel")
        ax.plot(g["channel"], g["obs_dt_ms"], label="Observed (smoothed/trusted)", linewidth=1.6)
        ax.plot(g["channel"], g["pred_dt_ms"], label="Predicted", linewidth=1.8)
        ax.set_title(f"{location} | anchor {anchor_index} | {anchor_label}")
        ax.set_ylabel("dt to ref (ms)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Channel")
    fig.suptitle("Observed vs predicted relative arrivals (fit subset)", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    obs, prior = load_tables(args.obs_csv, args.prior_csv)
    obs = prepare_observations(obs, args)
    prior = prepare_prior(prior, args)

    fit_obs = obs[obs["use_for_fit"]].copy()
    if fit_obs.empty:
        raise RuntimeError("No observations left after trusted/smoothed filtering. Relax thresholds.")

    channels_full = prior["channel"].to_numpy(dtype=float)
    ctrl_channels = np.linspace(channels_full.min(), channels_full.max(), args.n_control)
    alpha0 = np.zeros(args.n_control, dtype=float)

    ch_to_idx = {int(ch): i for i, ch in enumerate(prior["channel"].tolist())}

    bounds = [(-args.max_alpha_m, args.max_alpha_m)] * args.n_control
    res = minimize(
        objective_alpha_only,
        alpha0,
        args=(
            fit_obs,
            prior,
            ctrl_channels,
            ch_to_idx,
            args.sound_speed,
            args.lambda_smooth,
            args.lambda_amp,
        ),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 1000},
    )

    fit_df = build_candidate_curve(prior, ctrl_channels, res.x)
    ctrl_df = pd.DataFrame({"channel": ctrl_channels, "alpha_m": res.x})
    pred_rows = build_predicted_rows(obs, fit_df, args.sound_speed)
    group_summary, channel_summary, metrics = summarize(pred_rows)

    fit_df.to_csv(args.output_dir / "fitted_curve_alpha_only.csv", index=False)
    ctrl_df.to_csv(args.output_dir / "alpha_control_points.csv", index=False)
    pred_rows.to_csv(args.output_dir / "predicted_vs_observed_rows.csv", index=False)
    group_summary.to_csv(args.output_dir / "group_misfit_summary.csv", index=False)
    channel_summary.to_csv(args.output_dir / "channel_misfit_summary.csv", index=False)

    fit_metrics = {
        "success": bool(res.success),
        "message": str(res.message),
        "objective_value": float(res.fun),
        "n_fit_rows": int(fit_obs.shape[0]),
        "n_total_rows": int(obs.shape[0]),
        "alpha_min_m": float(np.min(fit_df["alpha_m"])),
        "alpha_max_m": float(np.max(fit_df["alpha_m"])),
        "alpha_std_m": float(np.std(fit_df["alpha_m"])),
        **metrics,
        "settings": {
            "sound_speed": args.sound_speed,
            "n_control": args.n_control,
            "lambda_smooth": args.lambda_smooth,
            "lambda_amp": args.lambda_amp,
            "max_alpha_m": args.max_alpha_m,
            "global_weight_floor": args.global_weight_floor,
            "location_weight_floor": args.location_weight_floor,
            "min_stable_fraction": args.min_stable_fraction,
            "min_weight": args.min_weight,
            "use_smoothed_offset": args.use_smoothed_offset,
        },
    }
    with open(args.output_dir / "fit_metrics.json", "w", encoding="utf-8") as f:
        json.dump(fit_metrics, f, indent=2)

    plot_path(prior, fit_df, args.output_dir / "path_prior_vs_fit.png")
    plot_alpha(fit_df, ctrl_df, args.output_dir / "alpha_vs_channel.png")
    plot_residual_by_channel(channel_summary, args.output_dir / "residual_by_channel.png")
    plot_observed_vs_predicted(pred_rows, args.output_dir / "observed_vs_predicted_by_location_anchor.png", args.plot_top_groups)

    print(f"Saved outputs to: {args.output_dir}")
    print(f"Optimizer success: {res.success}")
    print(f"Message: {res.message}")
    print(f"Rows used for fit: {len(fit_obs)} / {len(obs)}")
    print(f"Alpha range (m): {fit_df['alpha_m'].min():.3f} .. {fit_df['alpha_m'].max():.3f}")
    print(f"Global median |residual| (ms): {fit_metrics['global_median_abs_residual_ms']:.3f}")
    print(f"Global RMSE (ms): {fit_metrics['global_rmse_ms']:.3f}")


if __name__ == "__main__":
    main()
