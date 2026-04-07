from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline
from scipy.optimize import minimize

CHANNEL_MIN = 348
CHANNEL_MAX = 2267
SOUND_SPEED_MPS = 1500.0
DEFAULT_FIXED_SHIFT_CH = 61.26


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit smooth local tangential remapping beta(ch) after fixed global shift.")
    p.add_argument("--root", type=str, default=r"D:\Singapore Data")
    p.add_argument("--obs-csv", type=str, default=None)
    p.add_argument("--prior-csv", type=str, default=None)
    p.add_argument("--fixed-shift", type=float, default=DEFAULT_FIXED_SHIFT_CH)
    p.add_argument("--n-control", type=int, default=18)
    p.add_argument("--lambda-smooth", type=float, default=50.0)
    p.add_argument("--lambda-amp", type=float, default=0.5)
    p.add_argument("--lambda-slope", type=float, default=30.0)
    p.add_argument("--beta-bound", type=float, default=40.0, help="Bounds on local beta(ch) in channels; effective mapping is ch+fixed_shift+beta(ch)")
    p.add_argument("--min-weight", type=float, default=0.15)
    p.add_argument("--min-stable-fraction", type=float, default=0.50)
    p.add_argument("--all-usable", action="store_true")
    p.add_argument("--use-raw", action="store_true", help="Use observed_dt_ref_s instead of smooth offsets")
    return p.parse_args()


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    root = Path(args.root)
    obs_csv = Path(args.obs_csv) if args.obs_csv else root / "Cable" / "inversion_observations.csv"
    prior_csv = Path(args.prior_csv) if args.prior_csv else root / "Cable" / "prior_geometry.csv"
    outdir = root / "Cable" / "local_tangential_remap_outputs"
    outdir.mkdir(parents=True, exist_ok=True)

    obs = pd.read_csv(obs_csv)
    prior = pd.read_csv(prior_csv)
    return obs, prior, outdir


def prepare_fit_rows(obs: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    df = obs.copy()
    df = df[(df["channel"] >= CHANNEL_MIN) & (df["channel"] <= CHANNEL_MAX)].copy()

    if args.use_raw:
        df["fit_dt_s"] = df["observed_dt_ref_s"]
    else:
        if "median_smooth_offset_ms" not in df.columns:
            raise ValueError("median_smooth_offset_ms not found in inversion_observations.csv")
        df["fit_dt_s"] = df["median_smooth_offset_ms"] / 1000.0

    boolish = lambda s: s.astype(str).str.upper().eq("TRUE")
    if args.all_usable:
        mask = df["use_observation"].fillna(False)
    else:
        mask = (
            df["use_observation"].fillna(False)
            & boolish(df["recommended_channel"])
            & boolish(df["recommended_global"])
            & (df["weight"].fillna(0.0) >= args.min_weight)
            & (df["stable_fraction"].fillna(0.0) >= args.min_stable_fraction)
        )
    df = df[mask].copy()
    df = df[df["fit_dt_s"].notna()].copy()
    return df


def build_prior_interpolators(prior: pd.DataFrame):
    p = prior.copy().sort_values("channel")
    ch = p["channel"].to_numpy(dtype=float)
    x = p["prior_x_m"].to_numpy(dtype=float)
    y = p["prior_y_m"].to_numpy(dtype=float)
    u = p["prior_u_m"].to_numpy(dtype=float)

    fx = interp1d(ch, x, kind="linear", bounds_error=False, fill_value="extrapolate")
    fy = interp1d(ch, y, kind="linear", bounds_error=False, fill_value="extrapolate")
    fu = interp1d(ch, u, kind="linear", bounds_error=False, fill_value="extrapolate")
    return ch, fx, fy, fu


def build_beta(channels_full: np.ndarray, ctrl_channels: np.ndarray, beta_ctrl: np.ndarray) -> np.ndarray:
    cs = CubicSpline(ctrl_channels, beta_ctrl, bc_type="natural")
    return cs(channels_full)


def shifted_curve_from_beta(
    channels_full: np.ndarray,
    fx,
    fy,
    fu,
    fixed_shift: float,
    ctrl_channels: np.ndarray,
    beta_ctrl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    beta = build_beta(channels_full, ctrl_channels, beta_ctrl)
    mapped_ch = channels_full + fixed_shift + beta
    x = fx(mapped_ch)
    y = fy(mapped_ch)
    u = fu(mapped_ch)
    return mapped_ch, x, y, u


def predict_group_dt(channels_group: np.ndarray, ref_ch: int, tx_xyz: np.ndarray, all_channels: np.ndarray, x: np.ndarray, y: np.ndarray, u: np.ndarray) -> np.ndarray:
    ch_to_idx = {int(ch): i for i, ch in enumerate(all_channels)}
    ref_idx = ch_to_idx[int(ref_ch)]
    cable = np.column_stack([x, y, u])
    ranges = np.linalg.norm(cable - tx_xyz[None, :], axis=1)
    rel = (ranges - ranges[ref_idx]) / SOUND_SPEED_MPS
    idxs = np.array([ch_to_idx[int(c)] for c in channels_group], dtype=int)
    return rel[idxs]


def objective(beta_ctrl, fit_df, all_channels, fx, fy, fu, fixed_shift, ctrl_channels, args):
    mapped_ch, x, y, u = shifted_curve_from_beta(all_channels, fx, fy, fu, fixed_shift, ctrl_channels, beta_ctrl)

    # discourage mapping too far outside prior channel range
    over = np.maximum(CHANNEL_MIN - mapped_ch, 0.0) + np.maximum(mapped_ch - CHANNEL_MAX, 0.0)
    map_penalty = 5e3 * np.sum(over**2)

    data_terms = []
    weights = []
    for (loc, anchor_idx), g in fit_df.groupby(["location", "anchor_index"]):
        ref_ch = int(g["reference_channel"].iloc[0])
        tx_xyz = np.array([g["tx_x_m"].iloc[0], g["tx_y_m"].iloc[0], g["tx_u_m"].iloc[0]], dtype=float)
        pred = predict_group_dt(g["channel"].to_numpy(int), ref_ch, tx_xyz, all_channels, x, y, u)
        obs = g["fit_dt_s"].to_numpy(dtype=float)
        w = g["weight"].to_numpy(dtype=float)
        data_terms.append(pred - obs)
        weights.append(w)

    resid = np.concatenate(data_terms)
    w = np.concatenate(weights)
    data_term = np.sum(w * resid**2)

    d1 = np.diff(beta_ctrl)
    d2 = np.diff(beta_ctrl, n=2)
    smooth_term = args.lambda_smooth * np.sum(d2**2)
    amp_term = args.lambda_amp * np.sum(beta_ctrl**2)
    slope_term = args.lambda_slope * np.sum(d1**2)
    return data_term + smooth_term + amp_term + slope_term + map_penalty


def summarize_solution(fit_df, all_channels, fx, fy, fu, fixed_shift, ctrl_channels, beta_ctrl):
    mapped_ch, x, y, u = shifted_curve_from_beta(all_channels, fx, fy, fu, fixed_shift, ctrl_channels, beta_ctrl)
    rows = []
    for (loc, anchor_idx), g in fit_df.groupby(["location", "anchor_index"]):
        ref_ch = int(g["reference_channel"].iloc[0])
        tx_xyz = np.array([g["tx_x_m"].iloc[0], g["tx_y_m"].iloc[0], g["tx_u_m"].iloc[0]], dtype=float)
        pred = predict_group_dt(g["channel"].to_numpy(int), ref_ch, tx_xyz, all_channels, x, y, u)
        obs = g["fit_dt_s"].to_numpy(dtype=float)
        resid = pred - obs
        for ch, o, p, r, wt in zip(g["channel"], obs, pred, resid, g["weight"]):
            rows.append({
                "location": loc,
                "anchor_index": anchor_idx,
                "anchor_label": g["anchor_label"].iloc[0],
                "channel": int(ch),
                "observed_dt_ms": 1000.0 * o,
                "predicted_dt_ms": 1000.0 * p,
                "residual_ms": 1000.0 * r,
                "weight": float(wt),
            })
    pred_df = pd.DataFrame(rows)

    grp = pred_df.groupby(["location", "anchor_index", "anchor_label"], as_index=False).apply(
        lambda gg: pd.Series({
            "weighted_rmse_ms": float(np.sqrt(np.average(gg["residual_ms"]**2, weights=gg["weight"]))),
            "weighted_mae_ms": float(np.average(np.abs(gg["residual_ms"]), weights=gg["weight"])),
            "median_abs_residual_ms": float(np.median(np.abs(gg["residual_ms"]))),
            "n_rows": int(len(gg)),
        })
    ).reset_index(drop=True)

    chs = pred_df.groupby("channel", as_index=False).apply(
        lambda gg: pd.Series({
            "weighted_rmse_ms": float(np.sqrt(np.average(gg["residual_ms"]**2, weights=gg["weight"]))),
            "weighted_mae_ms": float(np.average(np.abs(gg["residual_ms"]), weights=gg["weight"])),
            "median_abs_residual_ms": float(np.median(np.abs(gg["residual_ms"]))),
            "n_rows": int(len(gg)),
        })
    ).reset_index(drop=True)

    curve = pd.DataFrame({
        "channel": all_channels.astype(int),
        "mapped_channel": mapped_ch,
        "beta_ch": build_beta(all_channels, ctrl_channels, beta_ctrl),
        "fit_x_m": x,
        "fit_y_m": y,
        "fit_u_m": u,
    })
    return pred_df, grp, chs, curve


def plot_outputs(prior, curve, pred_df, ch_summary, ctrl_channels, beta_ctrl, outdir: Path, fixed_shift: float):
    prior = prior.sort_values("channel")
    curve = curve.sort_values("channel")

    # path
    plt.figure(figsize=(8, 8))
    plt.plot(prior["prior_x_m"], prior["prior_y_m"], label="Prior", lw=2)
    plt.plot(curve["fit_x_m"], curve["fit_y_m"], label=f"Fixed shift + beta fit (δ={fixed_shift:.2f})", lw=2)
    plt.axis("equal")
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.title("Prior vs fixed-shift + local-beta fitted path")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "prior_vs_beta_fit_path.png", dpi=180)
    plt.close()

    # beta
    plt.figure(figsize=(12, 5))
    plt.plot(curve["channel"], curve["beta_ch"], label="Interpolated beta(ch)")
    plt.scatter(ctrl_channels, beta_ctrl, s=45, label="Control points")
    plt.axhline(0.0, ls="--")
    plt.xlabel("Channel")
    plt.ylabel("Local tangential correction beta (channels)")
    plt.title("Optimized local tangential correction after fixed global shift")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "beta_vs_channel.png", dpi=180)
    plt.close()

    # effective mapped channel
    plt.figure(figsize=(12, 5))
    plt.plot(curve["channel"], curve["mapped_channel"], label="effective mapped channel")
    plt.plot(curve["channel"], curve["channel"] + fixed_shift, label="channel + fixed shift", alpha=0.8)
    plt.xlabel("Channel")
    plt.ylabel("Mapped prior channel")
    plt.title("Effective channel remapping")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "mapped_channel_vs_channel.png", dpi=180)
    plt.close()

    # residual by channel
    plt.figure(figsize=(14, 6))
    plt.plot(ch_summary["channel"], ch_summary["weighted_rmse_ms"], label="Weighted RMSE")
    plt.plot(ch_summary["channel"], ch_summary["median_abs_residual_ms"], label="Median |residual|")
    plt.plot(ch_summary["channel"], ch_summary["weighted_mae_ms"], label="Weighted MAE")
    plt.xlabel("Channel")
    plt.ylabel("Residual (ms)")
    plt.title("Timing misfit by channel after fixed-shift + local-beta fit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "residual_by_channel.png", dpi=180)
    plt.close()

    # observed vs predicted
    groups = list(pred_df.groupby(["location", "anchor_index", "anchor_label"]))
    n = len(groups)
    fig, axes = plt.subplots(n, 1, figsize=(9, 3*n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, ((loc, anchor_idx, label), g) in zip(axes, groups):
        g = g.sort_values("channel")
        ax.plot(g["channel"], g["observed_dt_ms"], label="Observed for fit")
        ax.plot(g["channel"], g["predicted_dt_ms"], label="Predicted")
        ax.set_ylabel("dt to ref (ms)")
        ax.set_title(f"{loc} | anchor {anchor_idx} | {label}")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Channel")
    fig.suptitle("Observed vs predicted after fixed-shift + local-beta inversion", y=0.995)
    plt.tight_layout()
    plt.savefig(outdir / "observed_vs_predicted_by_location_anchor.png", dpi=180)
    plt.close()


def main():
    args = parse_args()
    obs, prior, outdir = load_inputs(args)
    fit_df = prepare_fit_rows(obs, args)
    if fit_df.empty:
        raise RuntimeError("No rows selected for fitting.")

    all_channels = np.arange(CHANNEL_MIN, CHANNEL_MAX + 1, dtype=float)
    prior_channels, fx, fy, fu = build_prior_interpolators(prior)

    ctrl_channels = np.linspace(CHANNEL_MIN, CHANNEL_MAX, args.n_control)
    beta0 = np.zeros(args.n_control, dtype=float)
    bounds = [(-args.beta_bound, args.beta_bound)] * args.n_control

    res = minimize(
        objective,
        beta0,
        args=(fit_df, all_channels, fx, fy, fu, args.fixed_shift, ctrl_channels, args),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 1000},
    )

    beta_ctrl = res.x
    pred_df, grp, ch_summary, curve = summarize_solution(
        fit_df, all_channels, fx, fy, fu, args.fixed_shift, ctrl_channels, beta_ctrl
    )

    curve.to_csv(outdir / "fitted_curve_fixed_shift_beta.csv", index=False)
    pred_df.to_csv(outdir / "predicted_vs_observed_rows.csv", index=False)
    grp.to_csv(outdir / "group_misfit_summary.csv", index=False)
    ch_summary.to_csv(outdir / "channel_misfit_summary.csv", index=False)
    pd.DataFrame({"control_channel": ctrl_channels, "beta_ctrl_ch": beta_ctrl}).to_csv(outdir / "beta_control_points.csv", index=False)

    metrics = {
        "success": bool(res.success),
        "message": str(res.message),
        "fun": float(res.fun),
        "fixed_shift_ch": float(args.fixed_shift),
        "n_control": int(args.n_control),
        "lambda_smooth": float(args.lambda_smooth),
        "lambda_amp": float(args.lambda_amp),
        "lambda_slope": float(args.lambda_slope),
        "beta_bound_ch": float(args.beta_bound),
        "beta_ctrl_min_ch": float(np.min(beta_ctrl)),
        "beta_ctrl_max_ch": float(np.max(beta_ctrl)),
        "beta_mean_abs_ch": float(np.mean(np.abs(beta_ctrl))),
        "weighted_rmse_ms_mean": float(ch_summary["weighted_rmse_ms"].mean()),
        "median_abs_residual_ms_mean": float(ch_summary["median_abs_residual_ms"].mean()),
    }
    with open(outdir / "fit_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_outputs(prior, curve, pred_df, ch_summary, ctrl_channels, beta_ctrl, outdir, args.fixed_shift)

    print(f"Saved outputs to: {outdir}")
    print(f"Optimization success: {res.success}")
    print(f"Message: {res.message}")
    print(f"beta range: {np.min(beta_ctrl):.3f} .. {np.max(beta_ctrl):.3f} channels")
    print(f"Mean weighted RMSE across channels: {ch_summary['weighted_rmse_ms'].mean():.3f} ms")
    print(f"Mean median |residual| across channels: {ch_summary['median_abs_residual_ms'].mean():.3f} ms")


if __name__ == "__main__":
    main()
