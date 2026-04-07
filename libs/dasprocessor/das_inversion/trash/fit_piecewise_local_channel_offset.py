from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import least_squares


CHANNEL_MIN = 348
CHANNEL_MAX = 2267
DEFAULT_SOUND_SPEED_MPS = 1500.0
DEFAULT_FIXED_GLOBAL_OFFSET_CH = 61.255


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.upper().eq("TRUE")


def load_inputs(obs_csv: Path, prior_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = pd.read_csv(obs_csv)
    prior = pd.read_csv(prior_csv)
    return obs, prior


def prepare_prior(prior: pd.DataFrame) -> pd.DataFrame:
    req = ["channel", "prior_x_m", "prior_y_m", "prior_u_m"]
    missing = [c for c in req if c not in prior.columns]
    if missing:
        raise ValueError(f"prior_geometry.csv missing columns: {missing}")

    out = prior.copy()
    out = out[(out["channel"] >= CHANNEL_MIN) & (out["channel"] <= CHANNEL_MAX)].copy()
    out = out.sort_values("channel").reset_index(drop=True)
    return out


def prepare_observations(
    obs: pd.DataFrame,
    min_weight: float,
    min_stable_fraction: float,
    use_only_recommended: bool,
    use_smoothed: bool,
) -> pd.DataFrame:
    df = obs.copy()
    df = df[(df["channel"] >= CHANNEL_MIN) & (df["channel"] <= CHANNEL_MAX)].copy()

    required = [
        "location", "anchor_index", "anchor_label", "channel", "reference_channel",
        "tx_x_m", "tx_y_m", "tx_u_m", "weight", "use_observation",
        "recommended_channel", "recommended_global", "stable_fraction",
        "channel_trust_score", "mean_channel_trust_score", "observed_dt_ref_s",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"inversion_observations.csv missing columns: {missing}")

    df["use_observation"] = as_bool(df["use_observation"])
    df["recommended_channel"] = as_bool(df["recommended_channel"])
    df["recommended_global"] = as_bool(df["recommended_global"])

    if use_smoothed and "median_smooth_offset_ms" in df.columns:
        df["fit_dt_s"] = pd.to_numeric(df["median_smooth_offset_ms"], errors="coerce") / 1000.0
    else:
        df["fit_dt_s"] = pd.to_numeric(df["observed_dt_ref_s"], errors="coerce")

    mask = df["use_observation"] & df["fit_dt_s"].notna() & (pd.to_numeric(df["weight"], errors="coerce") >= min_weight)
    mask &= pd.to_numeric(df["stable_fraction"], errors="coerce").fillna(0.0) >= min_stable_fraction

    if use_only_recommended:
        mask &= df["recommended_channel"] & df["recommended_global"]

    df = df[mask].copy()
    if df.empty:
        raise ValueError("No observations left after filtering. Relax thresholds.")

    w = pd.to_numeric(df["weight"], errors="coerce").to_numpy(dtype=float)
    w *= np.clip(pd.to_numeric(df["channel_trust_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float), 0.0, 1.0)
    w *= np.clip(pd.to_numeric(df["mean_channel_trust_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float), 0.0, 1.0)
    w *= np.where(df["recommended_channel"].to_numpy(), 1.0, 0.5)
    w *= np.where(df["recommended_global"].to_numpy(), 1.0, 0.7)

    df["fit_weight"] = w
    df = df[df["fit_weight"] > 0].copy()
    return df


def make_prior_interpolators(prior: pd.DataFrame) -> dict[str, interp1d]:
    ch = prior["channel"].to_numpy(dtype=float)
    cols = ["prior_x_m", "prior_y_m", "prior_u_m"]
    return {
        c: interp1d(ch, prior[c].to_numpy(dtype=float), kind="linear", bounds_error=False, fill_value="extrapolate")
        for c in cols
    }


def evaluate_prior_at_mapped_channels(mapped_channels: np.ndarray, interp_map: dict[str, interp1d]) -> dict[str, np.ndarray]:
    return {name: fn(mapped_channels.astype(float)) for name, fn in interp_map.items()}


def predict_relative_times(cable_xyz: np.ndarray, tx_xyz: np.ndarray, ref_idx: int, sound_speed: float) -> np.ndarray:
    ranges = np.linalg.norm(cable_xyz - tx_xyz[None, :], axis=1)
    return (ranges - ranges[ref_idx]) / sound_speed


def weighted_metrics(resid_s: np.ndarray, w: np.ndarray) -> dict[str, float]:
    resid_ms = 1000.0 * resid_s
    wsum = float(np.sum(w))
    if wsum <= 0:
        return {
            "weighted_rmse_ms": np.nan,
            "weighted_mae_ms": np.nan,
            "weighted_bias_ms": np.nan,
            "median_abs_ms": np.nan,
        }
    return {
        "weighted_rmse_ms": float(np.sqrt(np.sum(w * resid_ms ** 2) / wsum)),
        "weighted_mae_ms": float(np.sum(w * np.abs(resid_ms)) / wsum),
        "weighted_bias_ms": float(np.sum(w * resid_ms) / wsum),
        "median_abs_ms": float(np.median(np.abs(resid_ms))),
    }


def build_control_points(ch_min: int, ch_max: int, spacing: float) -> np.ndarray:
    cps = np.arange(ch_min, ch_max + 0.5 * spacing, spacing, dtype=float)
    if cps[-1] < ch_max:
        cps = np.append(cps, float(ch_max))
    cps[0] = float(ch_min)
    cps[-1] = float(ch_max)
    return cps


def beta_curve_from_control_points(channels: np.ndarray, cp_channels: np.ndarray, cp_beta: np.ndarray) -> np.ndarray:
    return np.interp(channels.astype(float), cp_channels.astype(float), cp_beta.astype(float))


def make_mapped_channels(
    channels: np.ndarray,
    cp_channels: np.ndarray,
    cp_beta: np.ndarray,
    fixed_global_offset_ch: float,
) -> tuple[np.ndarray, np.ndarray]:
    beta = beta_curve_from_control_points(channels, cp_channels, cp_beta)
    mapped = channels.astype(float) + fixed_global_offset_ch + beta
    return mapped, beta


def prediction_rows_for_curve(
    obs: pd.DataFrame,
    prior_channels: np.ndarray,
    interp_map: dict[str, interp1d],
    fixed_global_offset_ch: float,
    cp_channels: np.ndarray,
    cp_beta: np.ndarray,
    sound_speed: float,
) -> pd.DataFrame:
    mapped_all, beta_all = make_mapped_channels(prior_channels, cp_channels, cp_beta, fixed_global_offset_ch)
    prior_xyz = evaluate_prior_at_mapped_channels(mapped_all, interp_map)
    cable_xyz = np.column_stack([prior_xyz["prior_x_m"], prior_xyz["prior_y_m"], prior_xyz["prior_u_m"]])
    ch_to_idx = {int(ch): i for i, ch in enumerate(prior_channels)}

    rows = []
    for (location, anchor_index), g in obs.groupby(["location", "anchor_index"], sort=True):
        g = g.sort_values("channel").copy()
        ref_ch = int(g["reference_channel"].iloc[0])
        if ref_ch not in ch_to_idx:
            continue

        ref_idx = ch_to_idx[ref_ch]
        idx = g["channel"].map(ch_to_idx).to_numpy(dtype=int)

        tx_xyz = np.array(
            [float(g["tx_x_m"].iloc[0]), float(g["tx_y_m"].iloc[0]), float(g["tx_u_m"].iloc[0])],
            dtype=float,
        )

        pred_dt = predict_relative_times(cable_xyz, tx_xyz, ref_idx, sound_speed)
        pred = pred_dt[idx]
        obs_dt = g["fit_dt_s"].to_numpy(dtype=float)
        resid = pred - obs_dt

        tmp = g[["location", "anchor_index", "anchor_label", "channel", "reference_channel", "fit_dt_s", "fit_weight"]].copy()
        tmp["pred_dt_s"] = pred
        tmp["residual_s"] = resid
        tmp["beta_local_ch"] = beta_all[idx]
        tmp["mapped_channel_effective"] = mapped_all[idx]
        rows.append(tmp)

    return pd.concat(rows, ignore_index=True)


def objective_residual_vector(
    cp_beta: np.ndarray,
    obs: pd.DataFrame,
    prior: pd.DataFrame,
    interp_map: dict[str, interp1d],
    cp_channels: np.ndarray,
    fixed_global_offset_ch: float,
    sound_speed: float,
    lambda_value: float,
    lambda_slope: float,
    lambda_anchor: float,
    slope_soft_limit_abs: float,
    monotonic_margin: float,
) -> np.ndarray:
    prior_channels = prior["channel"].to_numpy(dtype=int)
    pred_rows = prediction_rows_for_curve(
        obs=obs,
        prior_channels=prior_channels,
        interp_map=interp_map,
        fixed_global_offset_ch=fixed_global_offset_ch,
        cp_channels=cp_channels,
        cp_beta=cp_beta,
        sound_speed=sound_speed,
    )

    data_resid = pred_rows["residual_s"].to_numpy(dtype=float) * np.sqrt(pred_rows["fit_weight"].to_numpy(dtype=float))
    out = [data_resid]

    if len(cp_beta) >= 3 and lambda_value > 0:
        d2 = cp_beta[:-2] - 2.0 * cp_beta[1:-1] + cp_beta[2:]
        out.append(np.sqrt(lambda_value) * d2)

    if len(cp_beta) >= 2 and lambda_slope > 0:
        d1 = np.diff(cp_beta)
        out.append(np.sqrt(lambda_slope) * d1)

    if lambda_anchor > 0:
        out.append(np.sqrt(lambda_anchor) * cp_beta)

    if len(cp_beta) >= 2:
        dch = np.diff(cp_channels)
        slope = np.diff(cp_beta) / dch
        excess = np.clip(np.abs(slope) - slope_soft_limit_abs, 0.0, None)
        out.append(10.0 * excess)

        mono_violation = np.clip(monotonic_margin - (1.0 + slope), 0.0, None)
        out.append(20.0 * mono_violation)

    return np.concatenate(out)


def fit_piecewise_local_offset(
    obs: pd.DataFrame,
    prior: pd.DataFrame,
    interp_map: dict[str, interp1d],
    cp_channels: np.ndarray,
    fixed_global_offset_ch: float,
    sound_speed: float,
    lambda_value: float,
    lambda_slope: float,
    lambda_anchor: float,
    slope_soft_limit_abs: float,
    monotonic_margin: float,
    beta_bound_abs: float,
) -> tuple[np.ndarray, object, pd.DataFrame]:
    x0 = np.zeros(len(cp_channels), dtype=float)
    lower = -beta_bound_abs * np.ones_like(x0)
    upper = beta_bound_abs * np.ones_like(x0)

    result = least_squares(
        objective_residual_vector,
        x0=x0,
        bounds=(lower, upper),
        kwargs=dict(
            obs=obs,
            prior=prior,
            interp_map=interp_map,
            cp_channels=cp_channels,
            fixed_global_offset_ch=fixed_global_offset_ch,
            sound_speed=sound_speed,
            lambda_value=lambda_value,
            lambda_slope=lambda_slope,
            lambda_anchor=lambda_anchor,
            slope_soft_limit_abs=slope_soft_limit_abs,
            monotonic_margin=monotonic_margin,
        ),
        max_nfev=300,
        verbose=2,
    )

    best_beta = result.x.copy()
    pred_rows = prediction_rows_for_curve(
        obs=obs,
        prior_channels=prior["channel"].to_numpy(dtype=int),
        interp_map=interp_map,
        fixed_global_offset_ch=fixed_global_offset_ch,
        cp_channels=cp_channels,
        cp_beta=best_beta,
        sound_speed=sound_speed,
    )
    return best_beta, result, pred_rows


def channel_residual_summary(pred_rows: pd.DataFrame) -> pd.DataFrame:
    g = pred_rows.copy()
    g["residual_ms"] = 1000.0 * g["residual_s"]

    def w_rmse(df: pd.DataFrame) -> float:
        w = df["fit_weight"].to_numpy(dtype=float)
        r = df["residual_ms"].to_numpy(dtype=float)
        return float(np.sqrt(np.sum(w * r ** 2) / max(np.sum(w), 1e-12)))

    def w_mae(df: pd.DataFrame) -> float:
        w = df["fit_weight"].to_numpy(dtype=float)
        r = np.abs(df["residual_ms"].to_numpy(dtype=float))
        return float(np.sum(w * r) / max(np.sum(w), 1e-12))

    return (
        g.groupby("channel", sort=True)
        .apply(
            lambda df: pd.Series(
                {
                    "rmse_ms": w_rmse(df),
                    "mae_ms": w_mae(df),
                    "median_abs_ms": float(np.median(np.abs(df["residual_ms"]))),
                    "n_rows": int(len(df)),
                    "median_beta_local_ch": float(np.median(df["beta_local_ch"])),
                    "median_mapped_channel": float(np.median(df["mapped_channel_effective"])),
                }
            )
        )
        .reset_index()
    )


def group_summary(pred_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (location, anchor_index, anchor_label), g in pred_rows.groupby(["location", "anchor_index", "anchor_label"], sort=True):
        m = weighted_metrics(g["residual_s"].to_numpy(dtype=float), g["fit_weight"].to_numpy(dtype=float))
        m.update({"location": location, "anchor_index": int(anchor_index), "anchor_label": anchor_label, "n_rows": int(len(g))})
        rows.append(m)
    return pd.DataFrame(rows)


def plot_beta_curve(prior: pd.DataFrame, cp_channels: np.ndarray, cp_beta: np.ndarray, out_path: Path) -> None:
    ch = prior["channel"].to_numpy(dtype=float)
    beta = beta_curve_from_control_points(ch, cp_channels, cp_beta)

    plt.figure(figsize=(12, 5))
    plt.plot(ch, beta, linewidth=2, label="Interpolated local offset β(ch)")
    plt.scatter(cp_channels, cp_beta, s=60, label="Control points")
    plt.axhline(0.0, linestyle="--", linewidth=1.2)
    plt.xlabel("Channel")
    plt.ylabel("Local tangential correction β (channels)")
    plt.title("Piecewise local channel-offset correction after fixed global shift")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_effective_mapping(prior: pd.DataFrame, cp_channels: np.ndarray, cp_beta: np.ndarray, fixed_global_offset_ch: float, out_path: Path) -> None:
    ch = prior["channel"].to_numpy(dtype=float)
    mapped, _ = make_mapped_channels(ch, cp_channels, cp_beta, fixed_global_offset_ch)

    plt.figure(figsize=(12, 5))
    plt.plot(ch, mapped, linewidth=2, label="Effective mapped channel")
    plt.plot(ch, ch + fixed_global_offset_ch, linewidth=2, label="Channel + fixed global offset")
    plt.xlabel("Channel")
    plt.ylabel("Mapped prior channel")
    plt.title("Effective channel remapping")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_obs_pred(pred_rows: pd.DataFrame, out_path: Path) -> None:
    groups = list(pred_rows.groupby(["location", "anchor_index", "anchor_label"], sort=True))
    n = len(groups)
    fig, axes = plt.subplots(n, 1, figsize=(12, max(3 * n, 7)), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, ((loc, anchor, label), g) in zip(axes, groups):
        g = g.sort_values("channel")
        ax.plot(g["channel"], 1000.0 * g["fit_dt_s"], label="Observed fit target", linewidth=1.4)
        ax.plot(g["channel"], 1000.0 * g["pred_dt_s"], label="Predicted", linewidth=1.4)
        ax.set_ylabel("dt to ref (ms)")
        ax.set_title(f"{loc} | anchor {anchor} | {label}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Channel")
    fig.suptitle("Observed vs predicted after fixed-shift + piecewise local-offset inversion", y=0.995, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_residual_by_channel(ch_summary: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(ch_summary["channel"], ch_summary["rmse_ms"], label="Weighted RMSE")
    plt.plot(ch_summary["channel"], ch_summary["median_abs_ms"], label="Median |residual|")
    plt.plot(ch_summary["channel"], ch_summary["mae_ms"], label="Weighted MAE")
    plt.xlabel("Channel")
    plt.ylabel("Residual (ms)")
    plt.title("Timing misfit by channel after fixed-shift + piecewise local-offset fit")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_path_compare(prior: pd.DataFrame, interp_map: dict[str, interp1d], cp_channels: np.ndarray, cp_beta: np.ndarray, fixed_global_offset_ch: float, out_path: Path) -> None:
    ch = prior["channel"].to_numpy(dtype=float)
    mapped, _ = make_mapped_channels(ch, cp_channels, cp_beta, fixed_global_offset_ch)
    shifted_xyz = evaluate_prior_at_mapped_channels(mapped, interp_map)

    plt.figure(figsize=(8, 8))
    plt.plot(prior["prior_x_m"], prior["prior_y_m"], linewidth=2, label="Prior")
    plt.plot(shifted_xyz["prior_x_m"], shifted_xyz["prior_y_m"], linewidth=2, label="Fixed shift + local offset fit")
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.title("Prior vs fixed-shift + piecewise local-offset fitted path")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_depth_compare(prior: pd.DataFrame, interp_map: dict[str, interp1d], cp_channels: np.ndarray, cp_beta: np.ndarray, fixed_global_offset_ch: float, out_path: Path) -> None:
    ch = prior["channel"].to_numpy(dtype=float)
    mapped, _ = make_mapped_channels(ch, cp_channels, cp_beta, fixed_global_offset_ch)
    shifted_xyz = evaluate_prior_at_mapped_channels(mapped, interp_map)

    if "cum_dist_3d_m" in prior.columns:
        xaxis = prior["cum_dist_3d_m"].to_numpy(dtype=float)
        xlabel = "Cumulative 3D distance (m)"
    elif "cum_dist_horizontal_m" in prior.columns:
        xaxis = prior["cum_dist_horizontal_m"].to_numpy(dtype=float)
        xlabel = "Cumulative horizontal distance (m)"
    else:
        xaxis = ch
        xlabel = "Channel"

    plt.figure(figsize=(12, 5))
    plt.plot(xaxis, prior["prior_u_m"], linewidth=2, label="Prior depth/u")
    plt.plot(xaxis, shifted_xyz["prior_u_m"], linewidth=2, label="Fit depth/u")
    plt.xlabel(xlabel)
    plt.ylabel("Up / depth-like coordinate (m)")
    plt.title("Depth profile after fixed global shift + piecewise local-offset fit")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a piecewise local channel-offset correction after a fixed global offset.")
    parser.add_argument("--obs-csv", type=Path, default=Path(r"D:\Singapore Data\Cable\inversion_observations.csv"))
    parser.add_argument("--prior-csv", type=Path, default=Path(r"D:\Singapore Data\Cable\prior_geometry.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path(r"D:\Singapore Data\Cable\local_piecewise_offset_outputs"))
    parser.add_argument("--sound-speed", type=float, default=DEFAULT_SOUND_SPEED_MPS)
    parser.add_argument("--fixed-global-offset", type=float, default=DEFAULT_FIXED_GLOBAL_OFFSET_CH)
    parser.add_argument("--control-spacing-ch", type=float, default=120.0)
    parser.add_argument("--lambda-smooth", type=float, default=40.0)
    parser.add_argument("--lambda-slope", type=float, default=2.0)
    parser.add_argument("--lambda-anchor", type=float, default=0.02)
    parser.add_argument("--slope-soft-limit-abs", type=float, default=0.15)
    parser.add_argument("--monotonic-margin", type=float, default=0.15)
    parser.add_argument("--beta-bound-abs", type=float, default=80.0)
    parser.add_argument("--min-weight", type=float, default=0.15)
    parser.add_argument("--min-stable-fraction", type=float, default=0.5)
    parser.add_argument("--all-usable", action="store_true", help="Use all usable rows instead of only recommended channels.")
    parser.add_argument("--use-raw", action="store_true", help="Use raw observed_dt_ref_s instead of median_smooth_offset_ms.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    obs, prior = load_inputs(args.obs_csv, args.prior_csv)
    prior = prepare_prior(prior)
    obs = prepare_observations(
        obs=obs,
        min_weight=args.min_weight,
        min_stable_fraction=args.min_stable_fraction,
        use_only_recommended=not args.all_usable,
        use_smoothed=not args.use_raw,
    )

    interp_map = make_prior_interpolators(prior)
    cp_channels = build_control_points(CHANNEL_MIN, CHANNEL_MAX, args.control_spacing_ch)

    best_beta, result, pred_rows = fit_piecewise_local_offset(
        obs=obs,
        prior=prior,
        interp_map=interp_map,
        cp_channels=cp_channels,
        fixed_global_offset_ch=args.fixed_global_offset,
        sound_speed=args.sound_speed,
        lambda_value=args.lambda_smooth,
        lambda_slope=args.lambda_slope,
        lambda_anchor=args.lambda_anchor,
        slope_soft_limit_abs=args.slope_soft_limit_abs,
        monotonic_margin=args.monotonic_margin,
        beta_bound_abs=args.beta_bound_abs,
    )

    ch = prior["channel"].to_numpy(dtype=float)
    mapped, beta = make_mapped_channels(ch, cp_channels, best_beta, args.fixed_global_offset)
    shifted_xyz = evaluate_prior_at_mapped_channels(mapped, interp_map)

    prior_fit_df = prior.copy()
    prior_fit_df["fixed_global_offset_ch"] = args.fixed_global_offset
    prior_fit_df["beta_local_ch"] = beta
    prior_fit_df["mapped_channel_effective"] = mapped
    prior_fit_df["prior_x_m_fitted"] = shifted_xyz["prior_x_m"]
    prior_fit_df["prior_y_m_fitted"] = shifted_xyz["prior_y_m"]
    prior_fit_df["prior_u_m_fitted"] = shifted_xyz["prior_u_m"]

    cp_df = pd.DataFrame({
        "control_channel": cp_channels,
        "beta_local_ch": best_beta,
        "fixed_global_offset_ch": args.fixed_global_offset,
        "total_effective_offset_ch": args.fixed_global_offset + best_beta,
    })

    ch_summary = channel_residual_summary(pred_rows)
    grp_summary = group_summary(pred_rows)
    overall = weighted_metrics(pred_rows["residual_s"].to_numpy(dtype=float), pred_rows["fit_weight"].to_numpy(dtype=float))

    prior_fit_df.to_csv(args.output_dir / "prior_geometry_with_piecewise_local_offset.csv", index=False)
    cp_df.to_csv(args.output_dir / "piecewise_local_offset_control_points.csv", index=False)
    pred_rows.to_csv(args.output_dir / "predicted_vs_observed_rows.csv", index=False)
    ch_summary.to_csv(args.output_dir / "residual_by_channel_summary.csv", index=False)
    grp_summary.to_csv(args.output_dir / "group_misfit_summary.csv", index=False)

    fit_metrics = {
        "fixed_global_offset_ch": float(args.fixed_global_offset),
        "control_spacing_ch": float(args.control_spacing_ch),
        "n_control_points": int(len(cp_channels)),
        "lambda_smooth": float(args.lambda_smooth),
        "lambda_slope": float(args.lambda_slope),
        "lambda_anchor": float(args.lambda_anchor),
        "slope_soft_limit_abs": float(args.slope_soft_limit_abs),
        "monotonic_margin": float(args.monotonic_margin),
        "beta_bound_abs": float(args.beta_bound_abs),
        "n_fit_rows": int(len(obs)),
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "optimizer_cost": float(result.cost),
        "max_abs_beta_local_ch": float(np.max(np.abs(best_beta))),
        "min_beta_local_ch": float(np.min(best_beta)),
        "max_beta_local_ch": float(np.max(best_beta)),
        **overall,
    }

    with open(args.output_dir / "fit_metrics.json", "w", encoding="utf-8") as f:
        json.dump(fit_metrics, f, indent=2)

    plot_beta_curve(prior, cp_channels, best_beta, args.output_dir / "piecewise_local_offset_curve.png")
    plot_effective_mapping(prior, cp_channels, best_beta, args.fixed_global_offset, args.output_dir / "effective_channel_mapping.png")
    plot_obs_pred(pred_rows, args.output_dir / "observed_vs_predicted_by_location_anchor.png")
    plot_residual_by_channel(ch_summary, args.output_dir / "residual_by_channel.png")
    plot_path_compare(prior, interp_map, cp_channels, best_beta, args.fixed_global_offset, args.output_dir / "prior_vs_piecewise_local_offset_fit.png")
    plot_depth_compare(prior, interp_map, cp_channels, best_beta, args.fixed_global_offset, args.output_dir / "depth_profile_piecewise_local_offset_fit.png")

    print(f"Saved outputs to: {args.output_dir}")
    print(f"Rows used in fit: {len(obs)}")
    print(f"Fixed global offset: {args.fixed_global_offset:.3f} ch")
    print(f"Control points: {len(cp_channels)}")
    print(f"Weighted RMSE: {overall['weighted_rmse_ms']:.3f} ms")
    print(f"Weighted MAE: {overall['weighted_mae_ms']:.3f} ms")
    print(f"Median |residual|: {overall['median_abs_ms']:.3f} ms")
    print(f"Max |beta_local| at control points: {np.max(np.abs(best_beta)):.3f} ch")


if __name__ == "__main__":
    main()
