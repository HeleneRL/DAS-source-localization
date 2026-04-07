from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize


DEFAULT_ROOT = Path(r"D:\Singapore Data")
DEFAULT_INV_CSV = DEFAULT_ROOT / "Cable" / "inversion_observations.csv"
DEFAULT_PRIOR_CSV = DEFAULT_ROOT / "Cable" / "prior_geometry.csv"
DEFAULT_OUTDIR = DEFAULT_ROOT / "Cable" / "alpha_with_fixed_shift_outputs"

CHANNEL_MIN = 348
CHANNEL_MAX = 2267
DEFAULT_SOUND_SPEED = 1500.0
DEFAULT_FIXED_SHIFT_CH = 61.26


# -----------------------------
# Helpers
# -----------------------------

def _as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.upper().eq("TRUE")


def shift_prior_geometry(prior: pd.DataFrame, delta_ch: float) -> pd.DataFrame:
    """
    Evaluate prior geometry and derived quantities at shifted channel coordinates.
    Returns one row per original channel, but geometry sampled at channel + delta_ch.
    """
    g = prior.sort_values("channel").copy()
    base_ch = g["channel"].to_numpy(dtype=float)
    target_ch = base_ch + float(delta_ch)

    out = pd.DataFrame({"channel": g["channel"].to_numpy(dtype=int)})
    out["shifted_channel_coordinate"] = target_ch

    interp_cols = [
        "prior_x_m",
        "prior_y_m",
        "prior_u_m",
        "prior_z_smooth_m",
        "tangent_x",
        "tangent_y",
        "normal_x",
        "normal_y",
        "curvature_proxy_per_channel",
        "cum_dist_horizontal_m",
        "cum_dist_3d_m",
    ]

    for col in interp_cols:
        out[col] = np.interp(
            target_ch,
            base_ch,
            g[col].to_numpy(dtype=float),
            left=np.nan,
            right=np.nan,
        )

    ok = np.isfinite(out["tangent_x"]) & np.isfinite(out["tangent_y"])
    tan = out.loc[ok, ["tangent_x", "tangent_y"]].to_numpy(dtype=float)
    tan_norm = np.linalg.norm(tan, axis=1, keepdims=True)
    tan_norm[tan_norm == 0.0] = 1.0
    tan = tan / tan_norm
    out.loc[ok, "tangent_x"] = tan[:, 0]
    out.loc[ok, "tangent_y"] = tan[:, 1]

    okn = np.isfinite(out["normal_x"]) & np.isfinite(out["normal_y"])
    nor = out.loc[okn, ["normal_x", "normal_y"]].to_numpy(dtype=float)
    nor_norm = np.linalg.norm(nor, axis=1, keepdims=True)
    nor_norm[nor_norm == 0.0] = 1.0
    nor = nor / nor_norm
    out.loc[okn, "normal_x"] = nor[:, 0]
    out.loc[okn, "normal_y"] = nor[:, 1]

    out["in_shift_domain"] = np.isfinite(out["prior_x_m"]) & np.isfinite(out["prior_y_m"]) & np.isfinite(out["prior_u_m"])
    return out


def choose_offset_column(df: pd.DataFrame, prefer_smooth: bool = True) -> tuple[str, float]:
    if prefer_smooth and "median_smooth_offset_ms" in df.columns:
        return "median_smooth_offset_ms", 1e-3
    if "observed_dt_ref_s" in df.columns:
        return "observed_dt_ref_s", 1.0
    raise KeyError("Could not find either 'median_smooth_offset_ms' or 'observed_dt_ref_s'.")


# -----------------------------
# Observation preparation
# -----------------------------

def load_data(inv_csv: Path, prior_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    inv = pd.read_csv(inv_csv)
    prior = pd.read_csv(prior_csv)
    return inv, prior


def prepare_observations(
    inv: pd.DataFrame,
    prefer_smooth: bool = True,
    trusted_only: bool = True,
    all_usable: bool = False,
    min_weight: float = 0.10,
    min_stable_fraction: float = 0.50,
    min_channel_trust: float = 0.50,
    min_global_trust: float = 0.50,
) -> pd.DataFrame:
    df = inv.copy()
    df = df[(df["channel"] >= CHANNEL_MIN) & (df["channel"] <= CHANNEL_MAX)].copy()

    offset_col, scale = choose_offset_column(df, prefer_smooth=prefer_smooth)
    df["fit_offset_s"] = df[offset_col].astype(float) * scale

    # If using smoothed location+channel summaries merged into inversion_observations,
    # there will be duplicate rows per location/channel for anchor 1 and anchor 2 with identical
    # smoothed offsets. That is fine because tx differs by anchor.
    df["recommended_channel_bool"] = _as_bool(df["recommended_channel"]) if "recommended_channel" in df.columns else True
    df["recommended_global_bool"] = _as_bool(df["recommended_global"]) if "recommended_global" in df.columns else True
    df["base_valid_bool"] = _as_bool(df["base_valid"]) if "base_valid" in df.columns else True

    if "weight" not in df.columns:
        df["weight"] = 1.0

    if all_usable:
        mask = (
            df["fit_offset_s"].notna()
            & df["tx_x_m"].notna()
            & df["tx_y_m"].notna()
            & df["tx_u_m"].notna()
        )
    else:
        stable_frac = df["stable_fraction"] if "stable_fraction" in df.columns else 1.0
        ch_trust = df["channel_trust_score"] if "channel_trust_score" in df.columns else 1.0
        g_trust = df["mean_channel_trust_score"] if "mean_channel_trust_score" in df.columns else 1.0

        mask = (
            df["fit_offset_s"].notna()
            & df["tx_x_m"].notna()
            & df["tx_y_m"].notna()
            & df["tx_u_m"].notna()
            & (df["weight"].astype(float) >= min_weight)
            & (pd.to_numeric(stable_frac, errors="coerce").fillna(0.0) >= min_stable_fraction)
            & (pd.to_numeric(ch_trust, errors="coerce").fillna(0.0) >= min_channel_trust)
            & (pd.to_numeric(g_trust, errors="coerce").fillna(0.0) >= min_global_trust)
        )
        if trusted_only:
            mask &= df["recommended_channel_bool"] & df["recommended_global_bool"] & df["base_valid_bool"]

    out = df.loc[mask].copy()
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
    out = out[out["weight"] > 0.0].copy()
    return out


# -----------------------------
# Forward model and objective
# -----------------------------

def build_candidate_curve(shifted_prior: pd.DataFrame, channels: np.ndarray, ctrl_channels: np.ndarray, alpha_ctrl: np.ndarray) -> pd.DataFrame:
    cs = CubicSpline(ctrl_channels, alpha_ctrl, bc_type="natural")
    alpha = cs(channels)

    out = shifted_prior.copy()
    out["alpha_m"] = alpha
    out["fit_x_m"] = out["prior_x_m"] + alpha * out["normal_x"]
    out["fit_y_m"] = out["prior_y_m"] + alpha * out["normal_y"]
    out["fit_u_m"] = out["prior_u_m"]
    return out


def predict_relative_times(cable_xyz: np.ndarray, tx_xyz: np.ndarray, ref_idx: int, sound_speed: float) -> np.ndarray:
    ranges = np.linalg.norm(cable_xyz - tx_xyz[None, :], axis=1)
    return (ranges - ranges[ref_idx]) / sound_speed


def objective_alpha_only(
    alpha_ctrl: np.ndarray,
    obs: pd.DataFrame,
    shifted_prior: pd.DataFrame,
    channels: np.ndarray,
    ctrl_channels: np.ndarray,
    ch_to_idx: dict[int, int],
    sound_speed: float,
    lambda_smooth: float,
    lambda_amp: float,
) -> float:
    candidate = build_candidate_curve(shifted_prior, channels, ctrl_channels, alpha_ctrl)
    cable_xyz = candidate[["fit_x_m", "fit_y_m", "fit_u_m"]].to_numpy(dtype=float)

    resid_list = []
    w_list = []

    for (location, anchor_index), g in obs.groupby(["location", "anchor_index"]):
        ref_ch = int(g["reference_channel"].iloc[0])
        if ref_ch not in ch_to_idx:
            continue

        idxs = g["channel"].map(ch_to_idx)
        valid = idxs.notna()
        if not valid.any():
            continue
        g = g.loc[valid].copy()
        idx = idxs.loc[valid].astype(int).to_numpy()

        ref_idx = ch_to_idx[ref_ch]
        tx_xyz = np.array([
            float(g["tx_x_m"].iloc[0]),
            float(g["tx_y_m"].iloc[0]),
            float(g["tx_u_m"].iloc[0]),
        ])

        pred_dt = predict_relative_times(cable_xyz, tx_xyz, ref_idx, sound_speed)
        pred = pred_dt[idx]
        obs_dt = g["fit_offset_s"].to_numpy(dtype=float)
        w = g["weight"].to_numpy(dtype=float)

        resid_list.append(pred - obs_dt)
        w_list.append(w)

    if not resid_list:
        return 1e12

    resid = np.concatenate(resid_list)
    w = np.concatenate(w_list)

    data_term = np.sum(w * resid**2)
    d2 = np.diff(alpha_ctrl, n=2)
    smooth_term = np.sum(d2**2)
    amp_term = np.sum(alpha_ctrl**2)

    return data_term + lambda_smooth * smooth_term + lambda_amp * amp_term


# -----------------------------
# Evaluation and outputs
# -----------------------------

def evaluate_fit(
    obs: pd.DataFrame,
    shifted_prior: pd.DataFrame,
    channels: np.ndarray,
    ctrl_channels: np.ndarray,
    alpha_ctrl: np.ndarray,
    ch_to_idx: dict[int, int],
    sound_speed: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate = build_candidate_curve(shifted_prior, channels, ctrl_channels, alpha_ctrl)
    cable_xyz = candidate[["fit_x_m", "fit_y_m", "fit_u_m"]].to_numpy(dtype=float)

    rows = []

    for (location, anchor_index), g in obs.groupby(["location", "anchor_index"]):
        ref_ch = int(g["reference_channel"].iloc[0])
        if ref_ch not in ch_to_idx:
            continue

        idxs = g["channel"].map(ch_to_idx)
        valid = idxs.notna()
        if not valid.any():
            continue
        g = g.loc[valid].copy()
        idx = idxs.loc[valid].astype(int).to_numpy()

        ref_idx = ch_to_idx[ref_ch]
        tx_xyz = np.array([
            float(g["tx_x_m"].iloc[0]),
            float(g["tx_y_m"].iloc[0]),
            float(g["tx_u_m"].iloc[0]),
        ])
        pred_dt = predict_relative_times(cable_xyz, tx_xyz, ref_idx, sound_speed)
        pred = pred_dt[idx]
        obs_dt = g["fit_offset_s"].to_numpy(dtype=float)
        resid = pred - obs_dt

        tmp = g.copy()
        tmp["predicted_dt_ref_s"] = pred
        tmp["residual_s"] = resid
        tmp["predicted_dt_ref_ms"] = 1e3 * pred
        tmp["observed_dt_ref_ms_for_fit"] = 1e3 * obs_dt
        tmp["residual_ms"] = 1e3 * resid
        rows.append(tmp)

    pred_rows = pd.concat(rows, ignore_index=True)

    def _wrmse(x: pd.DataFrame) -> float:
        w = x["weight"].to_numpy(dtype=float)
        r = x["residual_s"].to_numpy(dtype=float)
        return float(np.sqrt(np.sum(w * r**2) / max(np.sum(w), 1e-12)))

    def _wmae(x: pd.DataFrame) -> float:
        w = x["weight"].to_numpy(dtype=float)
        r = np.abs(x["residual_s"].to_numpy(dtype=float))
        return float(np.sum(w * r) / max(np.sum(w), 1e-12))

    group_summary = (
        pred_rows.groupby(["location", "anchor_index", "anchor_label"], dropna=False)
        .apply(lambda g: pd.Series({
            "n_rows": len(g),
            "weighted_rmse_ms": 1e3 * _wrmse(g),
            "weighted_mae_ms": 1e3 * _wmae(g),
            "median_abs_residual_ms": float(np.median(np.abs(g["residual_ms"].to_numpy(dtype=float)))),
            "mean_abs_residual_ms": float(np.mean(np.abs(g["residual_ms"].to_numpy(dtype=float)))),
        }))
        .reset_index()
    )

    channel_summary = (
        pred_rows.groupby("channel", dropna=False)
        .apply(lambda g: pd.Series({
            "n_rows": len(g),
            "weighted_rmse_ms": 1e3 * _wrmse(g),
            "weighted_mae_ms": 1e3 * _wmae(g),
            "median_abs_residual_ms": float(np.median(np.abs(g["residual_ms"].to_numpy(dtype=float)))),
        }))
        .reset_index()
    )

    alpha_ctrl_df = pd.DataFrame({
        "control_channel": ctrl_channels,
        "alpha_m": alpha_ctrl,
    })

    return candidate, pred_rows, group_summary, channel_summary, alpha_ctrl_df


# -----------------------------
# Plots
# -----------------------------

def plot_path(shifted_prior: pd.DataFrame, candidate: pd.DataFrame, outpath: Path, delta_ch: float):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(shifted_prior["prior_x_m"], shifted_prior["prior_y_m"], label=f"Shifted prior (δ={delta_ch:.2f} ch)")
    ax.plot(candidate["fit_x_m"], candidate["fit_y_m"], label="Shift + alpha fit")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title("Shifted prior vs alpha-corrected fitted path")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_alpha(channels: np.ndarray, candidate: pd.DataFrame, alpha_ctrl_df: pd.DataFrame, outpath: Path):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(channels, candidate["alpha_m"], label="Interpolated alpha(channel)")
    ax.scatter(alpha_ctrl_df["control_channel"], alpha_ctrl_df["alpha_m"], s=35, label="Control points")
    ax.axhline(0.0, linestyle="--")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Lateral correction alpha (m)")
    ax.set_title("Optimized lateral correction after fixed tangential shift")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_residual_by_channel(channel_summary: pd.DataFrame, outpath: Path):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(channel_summary["channel"], channel_summary["weighted_rmse_ms"], label="Weighted RMSE")
    ax.plot(channel_summary["channel"], channel_summary["median_abs_residual_ms"], label="Median |residual|")
    ax.plot(channel_summary["channel"], channel_summary["weighted_mae_ms"], label="Weighted MAE")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Residual (ms)")
    ax.set_title("Timing misfit by channel after fixed-shift alpha inversion")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_observed_vs_pred(pred_rows: pd.DataFrame, outpath: Path):
    groups = list(pred_rows.groupby(["location", "anchor_index", "anchor_label"], dropna=False))
    n = len(groups)
    fig, axes = plt.subplots(n, 1, figsize=(14, max(3 * n, 10)), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, ((loc, anchor_idx, anchor_label), g) in zip(axes, groups):
        gg = g.sort_values("channel")
        ax.plot(gg["channel"], gg["observed_dt_ref_ms_for_fit"], label="Observed for fit")
        ax.plot(gg["channel"], gg["predicted_dt_ref_ms"], label="Predicted")
        ax.set_ylabel("dt to ref (ms)")
        ax.set_title(f"{loc} | anchor {anchor_idx} | {anchor_label}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Channel")
    fig.suptitle("Observed vs predicted relative arrivals after fixed-shift alpha inversion", y=0.995)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_depth(shifted_prior: pd.DataFrame, candidate: pd.DataFrame, outpath: Path):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(shifted_prior["channel"], shifted_prior["prior_u_m"], label="Shifted prior depth/u")
    ax.plot(candidate["channel"], candidate["fit_u_m"], label="Fit depth/u")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Up / depth-like coordinate (m)")
    ax.set_title("Depth profile after fixed tangential shift (alpha does not change depth)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit smooth lateral correction alpha(channel) after fixed global tangential shift.")
    p.add_argument("--inversion-csv", type=Path, default=DEFAULT_INV_CSV)
    p.add_argument("--prior-csv", type=Path, default=DEFAULT_PRIOR_CSV)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--fixed-shift", type=float, default=DEFAULT_FIXED_SHIFT_CH)
    p.add_argument("--sound-speed", type=float, default=DEFAULT_SOUND_SPEED)
    p.add_argument("--n-control", type=int, default=28)
    p.add_argument("--lambda-smooth", type=float, default=30.0)
    p.add_argument("--lambda-amp", type=float, default=0.005)
    p.add_argument("--prefer-smooth", action="store_true", default=True)
    p.add_argument("--use-raw", action="store_true")
    p.add_argument("--trusted-only", action="store_true", default=True)
    p.add_argument("--all-usable", action="store_true")
    p.add_argument("--min-weight", type=float, default=0.10)
    p.add_argument("--min-stable-fraction", type=float, default=0.50)
    p.add_argument("--min-channel-trust", type=float, default=0.50)
    p.add_argument("--min-global-trust", type=float, default=0.50)
    return p.parse_args()


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    inv, prior = load_data(args.inversion_csv, args.prior_csv)
    obs = prepare_observations(
        inv,
        prefer_smooth=not args.use_raw,
        trusted_only=args.trusted_only,
        all_usable=args.all_usable,
        min_weight=args.min_weight,
        min_stable_fraction=args.min_stable_fraction,
        min_channel_trust=args.min_channel_trust,
        min_global_trust=args.min_global_trust,
    )

    shifted_prior = shift_prior_geometry(prior, args.fixed_shift)
    shifted_prior = shifted_prior[(shifted_prior["channel"] >= CHANNEL_MIN) & (shifted_prior["channel"] <= CHANNEL_MAX)].copy()
    shifted_prior = shifted_prior[shifted_prior["in_shift_domain"]].copy()

    valid_channels = shifted_prior["channel"].to_numpy(dtype=int)
    channels_f = valid_channels.astype(float)
    ch_to_idx = {int(ch): i for i, ch in enumerate(valid_channels)}

    obs = obs[obs["channel"].isin(valid_channels)].copy()
    obs = obs[obs["reference_channel"].isin(valid_channels)].copy()

    if obs.empty:
        raise RuntimeError("No usable observations remain after filtering and fixed-shift domain restriction.")

    n_control = max(6, min(args.n_control, len(valid_channels)))
    ctrl_channels = np.linspace(valid_channels.min(), valid_channels.max(), n_control)
    alpha0 = np.zeros(n_control, dtype=float)

    res = minimize(
        objective_alpha_only,
        alpha0,
        args=(
            obs,
            shifted_prior,
            channels_f,
            ctrl_channels,
            ch_to_idx,
            args.sound_speed,
            args.lambda_smooth,
            args.lambda_amp,
        ),
        method="L-BFGS-B",
    )

    alpha_opt = res.x

    candidate, pred_rows, group_summary, channel_summary, alpha_ctrl_df = evaluate_fit(
        obs,
        shifted_prior,
        channels_f,
        ctrl_channels,
        alpha_opt,
        ch_to_idx,
        args.sound_speed,
    )

    # Save CSVs
    shifted_prior.to_csv(args.outdir / "shifted_prior_geometry.csv", index=False)
    candidate.to_csv(args.outdir / "fitted_curve_fixed_shift_alpha.csv", index=False)
    pred_rows.to_csv(args.outdir / "predicted_vs_observed_rows.csv", index=False)
    group_summary.to_csv(args.outdir / "group_misfit_summary.csv", index=False)
    channel_summary.to_csv(args.outdir / "channel_misfit_summary.csv", index=False)
    alpha_ctrl_df.to_csv(args.outdir / "alpha_control_points.csv", index=False)

    # Save metrics
    fit_metrics = {
        "success": bool(res.success),
        "message": str(res.message),
        "n_iterations": int(getattr(res, "nit", -1)),
        "objective_value": float(res.fun),
        "fixed_shift_channels": float(args.fixed_shift),
        "sound_speed_mps": float(args.sound_speed),
        "n_control": int(n_control),
        "lambda_smooth": float(args.lambda_smooth),
        "lambda_amp": float(args.lambda_amp),
        "n_observation_rows_used": int(len(obs)),
        "n_groups_used": int(obs.groupby(["location", "anchor_index"]).ngroups),
        "alpha_min_m": float(np.min(candidate["alpha_m"])),
        "alpha_max_m": float(np.max(candidate["alpha_m"])),
        "alpha_mean_abs_m": float(np.mean(np.abs(candidate["alpha_m"]))),
        "overall_weighted_rmse_ms": float(
            np.sqrt(np.sum(pred_rows["weight"] * pred_rows["residual_s"]**2) / max(np.sum(pred_rows["weight"]), 1e-12)) * 1e3
        ),
        "overall_weighted_mae_ms": float(
            np.sum(pred_rows["weight"] * np.abs(pred_rows["residual_s"])) / max(np.sum(pred_rows["weight"]), 1e-12) * 1e3
        ),
        "overall_median_abs_residual_ms": float(np.median(np.abs(pred_rows["residual_ms"]))),
    }
    with open(args.outdir / "fit_metrics.json", "w", encoding="utf-8") as f:
        json.dump(fit_metrics, f, indent=2)

    # Plots
    plot_path(shifted_prior, candidate, args.outdir / "shifted_prior_vs_fit_path.png", args.fixed_shift)
    plot_alpha(channels_f, candidate, alpha_ctrl_df, args.outdir / "alpha_vs_channel.png")
    plot_residual_by_channel(channel_summary, args.outdir / "residual_by_channel.png")
    plot_observed_vs_pred(pred_rows, args.outdir / "observed_vs_predicted_by_location_anchor.png")
    plot_depth(shifted_prior, candidate, args.outdir / "depth_profile_shifted_prior_vs_fit.png")

    print(f"Saved outputs to: {args.outdir}")
    print(f"Rows used: {len(obs)}")
    print(f"Groups used: {obs.groupby(['location', 'anchor_index']).ngroups}")
    print(f"Fixed tangential shift: {args.fixed_shift:.3f} channels")
    print(f"Optimization success: {res.success}")
    print(f"Message: {res.message}")
    print(f"Final objective: {res.fun:.6f}")
    print(f"Alpha range: [{fit_metrics['alpha_min_m']:.3f}, {fit_metrics['alpha_max_m']:.3f}] m")
    print(f"Alpha mean abs: {fit_metrics['alpha_mean_abs_m']:.3f} m")
    print(f"Overall weighted RMSE: {fit_metrics['overall_weighted_rmse_ms']:.3f} ms")
    print(f"Overall weighted MAE : {fit_metrics['overall_weighted_mae_ms']:.3f} ms")
    print(f"Overall median |resid|: {fit_metrics['overall_median_abs_residual_ms']:.3f} ms")


if __name__ == "__main__":
    main()
