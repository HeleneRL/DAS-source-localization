from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize


CHANNEL_MIN = 348
CHANNEL_MAX = 2267
SOUND_SPEED_MPS = 1500.0


# ------------------------------------------------------------
# I/O
# ------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a first-pass cable inversion by optimizing a smooth lateral "
            "correction alpha(channel) relative to the prior path."
        )
    )
    parser.add_argument(
        "--obs-csv",
        type=Path,
        default=Path(r"D:\Singapore Data\Cable\inversion_observations.csv"),
        help="Merged inversion observations CSV.",
    )
    parser.add_argument(
        "--prior-csv",
        type=Path,
        default=Path(r"D:\Singapore Data\Cable\prior_geometry.csv"),
        help="Prepared prior geometry CSV.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(r"D:\Singapore Data\Cable\alpha_only_inversion_outputs"),
        help="Output directory.",
    )
    parser.add_argument(
        "--n-control",
        type=int,
        default=28,
        help="Number of spline control points for alpha(channel).",
    )
    parser.add_argument(
        "--sound-speed",
        type=float,
        default=SOUND_SPEED_MPS,
        help="Sound speed in m/s.",
    )
    parser.add_argument(
        "--lambda-smooth",
        type=float,
        default=150.0,
        help="Weight for smoothness penalty on second differences of alpha control points.",
    )
    parser.add_argument(
        "--lambda-amp",
        type=float,
        default=0.02,
        help="Weight for amplitude penalty on alpha control points.",
    )
    parser.add_argument(
        "--max-alpha-m",
        type=float,
        default=120.0,
        help="Bound alpha control points to +/- this many meters.",
    )
    parser.add_argument(
        "--weight-threshold",
        type=float,
        default=0.05,
        help="Minimum observation weight to use.",
    )
    parser.add_argument(
        "--location-filter",
        nargs="*",
        default=None,
        help="Optional subset of locations to include, e.g. loc2_tx3 loc3_tx1.",
    )
    return parser.parse_args()


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _as_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.upper().isin(["TRUE", "1", "YES", "Y"])


# ------------------------------------------------------------
# Data preparation
# ------------------------------------------------------------

def load_inputs(obs_csv: Path, prior_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not obs_csv.exists():
        raise FileNotFoundError(f"Observation CSV not found: {obs_csv}")
    if not prior_csv.exists():
        raise FileNotFoundError(f"Prior geometry CSV not found: {prior_csv}")

    obs = pd.read_csv(obs_csv)
    prior = pd.read_csv(prior_csv)
    return obs, prior



def prepare_prior(prior: pd.DataFrame) -> pd.DataFrame:
    req = [
        "channel",
        "prior_x_m",
        "prior_y_m",
        "prior_u_m",
        "normal_x",
        "normal_y",
        "tangent_x",
        "tangent_y",
        "cum_dist_3d_m",
    ]
    missing = [c for c in req if c not in prior.columns]
    if missing:
        raise ValueError(f"prior_geometry.csv missing columns: {missing}")

    g = prior.copy()
    g = g[(g["channel"] >= CHANNEL_MIN) & (g["channel"] <= CHANNEL_MAX)].copy()
    g = g.sort_values("channel").reset_index(drop=True)

    expected = np.arange(g["channel"].min(), g["channel"].max() + 1)
    if not np.array_equal(g["channel"].to_numpy(), expected):
        raise ValueError("Prior geometry channels must be contiguous over the water interval.")

    g["channel_idx"] = np.arange(len(g), dtype=int)
    return g



def prepare_observations(obs: pd.DataFrame, prior: pd.DataFrame, weight_threshold: float, location_filter: list[str] | None) -> pd.DataFrame:
    req = [
        "location",
        "anchor_index",
        "anchor_label",
        "reference_channel",
        "channel",
        "observed_dt_ref_s",
        "tx_x_m",
        "tx_y_m",
        "tx_u_m",
        "weight",
        "use_observation",
    ]
    missing = [c for c in req if c not in obs.columns]
    if missing:
        raise ValueError(f"inversion_observations.csv missing columns: {missing}")

    df = obs.copy()
    df = df[(df["channel"] >= CHANNEL_MIN) & (df["channel"] <= CHANNEL_MAX)].copy()
    df["use_observation"] = _as_bool_series(df["use_observation"])

    if "recommended_global" in df.columns:
        df["recommended_global"] = _as_bool_series(df["recommended_global"])
    if "recommended_channel" in df.columns:
        df["recommended_channel"] = _as_bool_series(df["recommended_channel"])

    if location_filter:
        df = df[df["location"].isin(location_filter)].copy()

    df = df[df["use_observation"]].copy()
    df = df[df["weight"] > weight_threshold].copy()
    df = df[df["observed_dt_ref_s"].notna()].copy()

    if df.empty:
        raise ValueError("No usable observations after filtering.")

    valid_channels = set(prior["channel"].to_numpy())
    df = df[df["channel"].isin(valid_channels)].copy()
    df = df[df["reference_channel"].isin(valid_channels)].copy()

    if df.empty:
        raise ValueError("No usable observations remain after channel matching against prior.")

    return df.reset_index(drop=True)



def build_group_records(obs_df: pd.DataFrame, channel_to_idx: dict[int, int]) -> list[dict]:
    groups: list[dict] = []

    for (location, anchor_index), g in obs_df.groupby(["location", "anchor_index"], sort=True):
        gg = g.sort_values("channel").reset_index(drop=True)
        ref_ch = int(gg["reference_channel"].iloc[0])
        if ref_ch not in channel_to_idx:
            continue

        channels = gg["channel"].astype(int).to_numpy()
        idxs = np.array([channel_to_idx[ch] for ch in channels], dtype=int)
        ref_idx = channel_to_idx[ref_ch]

        tx_xyz = np.array(
            [
                float(gg["tx_x_m"].iloc[0]),
                float(gg["tx_y_m"].iloc[0]),
                float(gg["tx_u_m"].iloc[0]),
            ],
            dtype=float,
        )

        obs_dt = gg["observed_dt_ref_s"].to_numpy(dtype=float)
        weight = gg["weight"].to_numpy(dtype=float)

        groups.append(
            {
                "location": location,
                "anchor_index": int(anchor_index),
                "anchor_label": str(gg["anchor_label"].iloc[0]),
                "reference_channel": ref_ch,
                "channels": channels,
                "channel_indices": idxs,
                "ref_idx": ref_idx,
                "tx_xyz": tx_xyz,
                "obs_dt": obs_dt,
                "weight": weight,
            }
        )

    if not groups:
        raise ValueError("No observation groups available for inversion.")

    return groups


# ------------------------------------------------------------
# Geometry / forward model
# ------------------------------------------------------------

def choose_control_channels(channels: np.ndarray, n_control: int) -> np.ndarray:
    n = len(channels)
    n_control = max(4, min(n_control, n))
    idx = np.linspace(0, n - 1, n_control)
    idx = np.unique(np.round(idx).astype(int))
    return channels[idx]



def build_candidate_curve(
    prior_xyz: np.ndarray,
    normal_xy: np.ndarray,
    channels_full: np.ndarray,
    channels_ctrl: np.ndarray,
    alpha_ctrl: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cs = CubicSpline(channels_ctrl, alpha_ctrl, bc_type="natural")
    alpha_full = cs(channels_full)

    xyz = prior_xyz.copy()
    xyz[:, 0] = prior_xyz[:, 0] + alpha_full * normal_xy[:, 0]
    xyz[:, 1] = prior_xyz[:, 1] + alpha_full * normal_xy[:, 1]
    return xyz, alpha_full



def predict_relative_times(cable_xyz: np.ndarray, tx_xyz: np.ndarray, ref_idx: int, sound_speed: float) -> np.ndarray:
    ranges = np.linalg.norm(cable_xyz - tx_xyz[None, :], axis=1)
    return (ranges - ranges[ref_idx]) / sound_speed


# ------------------------------------------------------------
# Objective
# ------------------------------------------------------------

def make_objective(
    groups: list[dict],
    prior_xyz: np.ndarray,
    normal_xy: np.ndarray,
    channels_full: np.ndarray,
    channels_ctrl: np.ndarray,
    sound_speed: float,
    lambda_smooth: float,
    lambda_amp: float,
):
    last_payload: dict[str, np.ndarray | float] = {}

    def objective(alpha_ctrl: np.ndarray) -> float:
        cable_xyz, alpha_full = build_candidate_curve(
            prior_xyz=prior_xyz,
            normal_xy=normal_xy,
            channels_full=channels_full,
            channels_ctrl=channels_ctrl,
            alpha_ctrl=alpha_ctrl,
        )

        data_value = 0.0
        residual_rows = []

        for grp in groups:
            pred_all = predict_relative_times(
                cable_xyz=cable_xyz,
                tx_xyz=grp["tx_xyz"],
                ref_idx=grp["ref_idx"],
                sound_speed=sound_speed,
            )
            pred = pred_all[grp["channel_indices"]]
            resid = pred - grp["obs_dt"]
            weighted = grp["weight"] * resid**2
            data_value += float(np.sum(weighted))

            residual_rows.append(resid)

        d2 = np.diff(alpha_ctrl, n=2)
        smooth_value = float(lambda_smooth * np.sum(d2**2))
        amp_value = float(lambda_amp * np.sum(alpha_ctrl**2))

        total = data_value + smooth_value + amp_value

        if residual_rows:
            resid_all = np.concatenate(residual_rows)
            last_payload["resid_all"] = resid_all
            last_payload["rmse_ms"] = float(np.sqrt(np.mean(resid_all**2)) * 1000.0)
            last_payload["mae_ms"] = float(np.mean(np.abs(resid_all)) * 1000.0)
        else:
            last_payload["resid_all"] = np.array([], dtype=float)
            last_payload["rmse_ms"] = np.nan
            last_payload["mae_ms"] = np.nan

        last_payload["alpha_full"] = alpha_full.copy()
        last_payload["data_value"] = data_value
        last_payload["smooth_value"] = smooth_value
        last_payload["amp_value"] = amp_value
        last_payload["total_value"] = total
        return total

    objective.last_payload = last_payload  # type: ignore[attr-defined]
    return objective


# ------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------

def make_prediction_rows(
    groups: list[dict],
    cable_xyz: np.ndarray,
    sound_speed: float,
) -> pd.DataFrame:
    rows = []
    for grp in groups:
        pred_all = predict_relative_times(cable_xyz, grp["tx_xyz"], grp["ref_idx"], sound_speed)
        pred = pred_all[grp["channel_indices"]]
        resid = pred - grp["obs_dt"]

        for ch, obs_dt, pred_dt, w, r in zip(grp["channels"], grp["obs_dt"], pred, grp["weight"], resid):
            rows.append(
                {
                    "location": grp["location"],
                    "anchor_index": grp["anchor_index"],
                    "anchor_label": grp["anchor_label"],
                    "reference_channel": grp["reference_channel"],
                    "channel": int(ch),
                    "observed_dt_ref_s": float(obs_dt),
                    "predicted_dt_ref_s": float(pred_dt),
                    "residual_s": float(r),
                    "residual_ms": float(r * 1000.0),
                    "weight": float(w),
                }
            )
    return pd.DataFrame(rows)



def summarize_groups(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (loc, anchor), g in pred_df.groupby(["location", "anchor_index"], sort=True):
        resid = g["residual_s"].to_numpy(dtype=float)
        rows.append(
            {
                "location": loc,
                "anchor_index": int(anchor),
                "anchor_label": str(g["anchor_label"].iloc[0]),
                "n_obs": int(len(g)),
                "rmse_ms": float(np.sqrt(np.mean(resid**2)) * 1000.0),
                "mae_ms": float(np.mean(np.abs(resid)) * 1000.0),
                "bias_ms": float(np.mean(resid) * 1000.0),
            }
        )
    return pd.DataFrame(rows)



def summarize_by_channel(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ch, g in pred_df.groupby("channel", sort=True):
        resid = g["residual_s"].to_numpy(dtype=float)
        rows.append(
            {
                "channel": int(ch),
                "n_obs": int(len(g)),
                "rmse_ms": float(np.sqrt(np.mean(resid**2)) * 1000.0),
                "mae_ms": float(np.mean(np.abs(resid)) * 1000.0),
                "median_abs_ms": float(np.median(np.abs(resid)) * 1000.0),
                "mean_weight": float(g["weight"].mean()),
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

def plot_path_comparison(prior: pd.DataFrame, cable_xyz: np.ndarray, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.plot(prior["prior_x_m"], prior["prior_y_m"], label="Prior path", linewidth=2.0)
    ax.plot(cable_xyz[:, 0], cable_xyz[:, 1], label="Alpha-only fit", linewidth=2.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title("Prior path vs alpha-only fitted path")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)



def plot_alpha(channels: np.ndarray, alpha_full: np.ndarray, ctrl_channels: np.ndarray, alpha_ctrl: np.ndarray, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(channels, alpha_full, linewidth=2.0, label="Interpolated alpha(channel)")
    ax.scatter(ctrl_channels, alpha_ctrl, s=35, label="Control points")
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Lateral correction alpha (m)")
    ax.set_title("Optimized lateral correction relative to prior")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)



def plot_residual_by_channel(ch_summary: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.plot(ch_summary["channel"], ch_summary["rmse_ms"], label="RMSE")
    ax.plot(ch_summary["channel"], ch_summary["median_abs_ms"], label="Median |residual|")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Residual (ms)")
    ax.set_title("Timing misfit by channel after alpha-only inversion")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)



def plot_predictions_by_location(pred_df: pd.DataFrame, outpath: Path) -> None:
    groups = list(pred_df.groupby(["location", "anchor_index"], sort=True))
    n = len(groups)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.0 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, ((loc, anchor), g) in zip(axes, groups):
        gg = g.sort_values("channel")
        ax.plot(gg["channel"], gg["observed_dt_ref_s"] * 1000.0, label="Observed", linewidth=1.8)
        ax.plot(gg["channel"], gg["predicted_dt_ref_s"] * 1000.0, label="Predicted", linewidth=1.8)
        ax.set_ylabel("dt to ref (ms)")
        ax.set_title(f"{loc} | anchor {anchor}")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Channel")
    fig.suptitle("Observed vs predicted relative arrivals", y=0.995)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    obs_raw, prior_raw = load_inputs(args.obs_csv, args.prior_csv)
    prior = prepare_prior(prior_raw)
    obs = prepare_observations(obs_raw, prior, args.weight_threshold, args.location_filter)

    channels_full = prior["channel"].to_numpy(dtype=float)
    channels_full_int = prior["channel"].to_numpy(dtype=int)
    channel_to_idx = {int(ch): i for i, ch in enumerate(channels_full_int)}

    groups = build_group_records(obs, channel_to_idx)

    prior_xyz = prior[["prior_x_m", "prior_y_m", "prior_u_m"]].to_numpy(dtype=float)
    normal_xy = prior[["normal_x", "normal_y"]].to_numpy(dtype=float)
    ctrl_channels = choose_control_channels(channels_full, args.n_control)

    objective = make_objective(
        groups=groups,
        prior_xyz=prior_xyz,
        normal_xy=normal_xy,
        channels_full=channels_full,
        channels_ctrl=ctrl_channels,
        sound_speed=args.sound_speed,
        lambda_smooth=args.lambda_smooth,
        lambda_amp=args.lambda_amp,
    )

    alpha0 = np.zeros(len(ctrl_channels), dtype=float)
    bounds = [(-args.max_alpha_m, args.max_alpha_m)] * len(ctrl_channels)

    print("Running alpha-only inversion...")
    print(f"Observation rows used : {len(obs)}")
    print(f"Location-anchor groups: {len(groups)}")
    print(f"Control points        : {len(ctrl_channels)}")

    res = minimize(
        objective,
        alpha0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 400, "disp": True},
    )

    cable_xyz_fit, alpha_full_fit = build_candidate_curve(
        prior_xyz=prior_xyz,
        normal_xy=normal_xy,
        channels_full=channels_full,
        channels_ctrl=ctrl_channels,
        alpha_ctrl=res.x,
    )

    pred_df = make_prediction_rows(groups, cable_xyz_fit, args.sound_speed)
    group_summary = summarize_groups(pred_df)
    channel_summary = summarize_by_channel(pred_df)

    fitted_curve = prior.copy()
    fitted_curve["alpha_m"] = alpha_full_fit
    fitted_curve["fit_x_m"] = cable_xyz_fit[:, 0]
    fitted_curve["fit_y_m"] = cable_xyz_fit[:, 1]
    fitted_curve["fit_u_m"] = cable_xyz_fit[:, 2]

    ctrl_df = pd.DataFrame(
        {
            "control_channel": ctrl_channels,
            "alpha_ctrl_m": res.x,
        }
    )

    metrics = {
        "success": bool(res.success),
        "message": str(res.message),
        "n_iter": int(getattr(res, "nit", -1)),
        "final_objective": float(res.fun),
        "lambda_smooth": float(args.lambda_smooth),
        "lambda_amp": float(args.lambda_amp),
        "max_alpha_m": float(args.max_alpha_m),
        "sound_speed_mps": float(args.sound_speed),
        "n_observation_rows": int(len(obs)),
        "n_groups": int(len(groups)),
        "n_control": int(len(ctrl_channels)),
        "rmse_ms": float(np.sqrt(np.mean(pred_df["residual_s"].to_numpy(dtype=float) ** 2)) * 1000.0),
        "mae_ms": float(np.mean(np.abs(pred_df["residual_s"].to_numpy(dtype=float))) * 1000.0),
        "max_abs_alpha_m": float(np.max(np.abs(alpha_full_fit))),
    }

    fitted_curve.to_csv(args.outdir / "fitted_curve_alpha_only.csv", index=False)
    ctrl_df.to_csv(args.outdir / "alpha_control_points.csv", index=False)
    pred_df.to_csv(args.outdir / "predicted_vs_observed_rows.csv", index=False)
    group_summary.to_csv(args.outdir / "group_misfit_summary.csv", index=False)
    channel_summary.to_csv(args.outdir / "channel_misfit_summary.csv", index=False)
    (args.outdir / "fit_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    plot_path_comparison(prior, cable_xyz_fit, args.outdir / "path_prior_vs_fit.png")
    plot_alpha(channels_full, alpha_full_fit, ctrl_channels, res.x, args.outdir / "alpha_vs_channel.png")
    plot_residual_by_channel(channel_summary, args.outdir / "residual_by_channel.png")
    plot_predictions_by_location(pred_df, args.outdir / "observed_vs_predicted_by_location_anchor.png")

    print(f"Saved outputs to: {args.outdir}")
    print(f"Optimizer success : {res.success}")
    print(f"Message           : {res.message}")
    print(f"Final RMSE        : {metrics['rmse_ms']:.3f} ms")
    print(f"Final MAE         : {metrics['mae_ms']:.3f} ms")
    print(f"Max |alpha|       : {metrics['max_abs_alpha_m']:.3f} m")


if __name__ == "__main__":
    main()
