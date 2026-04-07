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
        "location",
        "anchor_index",
        "anchor_label",
        "channel",
        "reference_channel",
        "tx_x_m",
        "tx_y_m",
        "tx_u_m",
        "weight",
        "use_observation",
        "recommended_channel",
        "recommended_global",
        "stable_fraction",
        "channel_trust_score",
        "mean_channel_trust_score",
        "observed_dt_ref_s",
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

    mask = (
        df["use_observation"]
        & df["fit_dt_s"].notna()
        & (pd.to_numeric(df["weight"], errors="coerce") >= min_weight)
    )
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

    # Optional per-location emphasis. Good locations can dominate cleanly.
    return df


def apply_location_weight_multipliers(df: pd.DataFrame, location_weight_map: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    mult = np.ones(len(out), dtype=float)
    for loc, factor in location_weight_map.items():
        mult *= np.where(out["location"].astype(str).eq(loc), factor, 1.0)
    out["fit_weight"] = out["fit_weight"].to_numpy(dtype=float) * mult
    out = out[out["fit_weight"] > 0].copy()
    return out


def make_prior_interpolators(prior: pd.DataFrame) -> dict[str, interp1d]:
    ch = prior["channel"].to_numpy(dtype=float)
    cols = ["prior_x_m", "prior_y_m", "prior_u_m"]
    return {
        c: interp1d(
            ch,
            prior[c].to_numpy(dtype=float),
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        for c in cols
    }


def build_control_points(ch_min: int, ch_max: int, spacing: float) -> np.ndarray:
    cps = np.arange(ch_min, ch_max + 0.5 * spacing, spacing, dtype=float)
    if cps[-1] < ch_max:
        cps = np.append(cps, float(ch_max))
    cps[0] = float(ch_min)
    cps[-1] = float(ch_max)
    return cps


def interpolate_control_curve(channels: np.ndarray, cp_channels: np.ndarray, cp_values: np.ndarray) -> np.ndarray:
    return np.interp(channels.astype(float), cp_channels.astype(float), cp_values.astype(float))


def make_effective_geometry(
    prior: pd.DataFrame,
    cp_channels: np.ndarray,
    cp_dx: np.ndarray,
    cp_dy: np.ndarray,
    fixed_global_offset_ch: float,
    interp_map: dict[str, interp1d],
) -> pd.DataFrame:
    ch = prior["channel"].to_numpy(dtype=float)

    # Channel index remap first, then XY segment motion
    mapped_prior_channel = ch + fixed_global_offset_ch

    base_x = interp_map["prior_x_m"](mapped_prior_channel)
    base_y = interp_map["prior_y_m"](mapped_prior_channel)
    base_u = interp_map["prior_u_m"](mapped_prior_channel)

    dx = interpolate_control_curve(ch, cp_channels, cp_dx)
    dy = interpolate_control_curve(ch, cp_channels, cp_dy)

    out = prior[["channel"]].copy()
    out["mapped_prior_channel_global"] = mapped_prior_channel
    out["dx_local_m"] = dx
    out["dy_local_m"] = dy
    out["x_eff_m"] = base_x + dx
    out["y_eff_m"] = base_y + dy
    out["u_eff_m"] = base_u
    return out


def predict_rows_for_geometry(
    obs: pd.DataFrame,
    eff_geom: pd.DataFrame,
    sound_speed: float,
) -> pd.DataFrame:
    ch_to_idx = {int(ch): i for i, ch in enumerate(eff_geom["channel"].to_numpy(dtype=int))}
    cable_xyz = np.column_stack(
        [
            eff_geom["x_eff_m"].to_numpy(dtype=float),
            eff_geom["y_eff_m"].to_numpy(dtype=float),
            eff_geom["u_eff_m"].to_numpy(dtype=float),
        ]
    )

    rows = []
    for (location, anchor_index, anchor_label), g in obs.groupby(["location", "anchor_index", "anchor_label"], sort=True):
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

        pred_dt_all = predict_relative_times(cable_xyz, tx_xyz, ref_idx, sound_speed)
        pred = pred_dt_all[idx]
        obs_dt = g["fit_dt_s"].to_numpy(dtype=float)
        resid = pred - obs_dt

        tmp = g[
            ["location", "anchor_index", "anchor_label", "channel", "reference_channel", "fit_dt_s", "fit_weight"]
        ].copy()
        tmp["pred_dt_s"] = pred
        tmp["residual_s"] = resid
        tmp["x_eff_m"] = eff_geom.set_index("channel").loc[g["channel"], "x_eff_m"].to_numpy(dtype=float)
        tmp["y_eff_m"] = eff_geom.set_index("channel").loc[g["channel"], "y_eff_m"].to_numpy(dtype=float)
        tmp["u_eff_m"] = eff_geom.set_index("channel").loc[g["channel"], "u_eff_m"].to_numpy(dtype=float)
        rows.append(tmp)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


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


def objective_vector(
    params: np.ndarray,
    obs: pd.DataFrame,
    prior: pd.DataFrame,
    interp_map: dict[str, interp1d],
    cp_channels: np.ndarray,
    fixed_global_offset_ch: float,
    sound_speed: float,
    lambda_xy_smooth: float,
    lambda_xy_slope: float,
    lambda_xy_anchor: float,
    lambda_length: float,
) -> np.ndarray:
    ncp = len(cp_channels)
    cp_dx = params[:ncp]
    cp_dy = params[ncp:]

    eff_geom = make_effective_geometry(
        prior=prior,
        cp_channels=cp_channels,
        cp_dx=cp_dx,
        cp_dy=cp_dy,
        fixed_global_offset_ch=fixed_global_offset_ch,
        interp_map=interp_map,
    )

    pred_rows = predict_rows_for_geometry(obs=obs, eff_geom=eff_geom, sound_speed=sound_speed)
    if pred_rows.empty:
        return np.array([1e6], dtype=float)

    data_resid = pred_rows["residual_s"].to_numpy(dtype=float) * np.sqrt(pred_rows["fit_weight"].to_numpy(dtype=float))
    out = [data_resid]

    # second-difference smoothness on dx, dy
    if ncp >= 3 and lambda_xy_smooth > 0:
        d2x = cp_dx[:-2] - 2.0 * cp_dx[1:-1] + cp_dx[2:]
        d2y = cp_dy[:-2] - 2.0 * cp_dy[1:-1] + cp_dy[2:]
        out.append(np.sqrt(lambda_xy_smooth) * d2x)
        out.append(np.sqrt(lambda_xy_smooth) * d2y)

    # first-difference control to discourage violent segment-to-segment jumps
    if ncp >= 2 and lambda_xy_slope > 0:
        d1x = np.diff(cp_dx)
        d1y = np.diff(cp_dy)
        out.append(np.sqrt(lambda_xy_slope) * d1x)
        out.append(np.sqrt(lambda_xy_slope) * d1y)

    # weak anchor to prior, only overcome when demanded by data
    if lambda_xy_anchor > 0:
        out.append(np.sqrt(lambda_xy_anchor) * cp_dx)
        out.append(np.sqrt(lambda_xy_anchor) * cp_dy)

    # mild length-preservation penalty on neighbouring segment lengths in XY
    if ncp >= 2 and lambda_length > 0:
        x_cp = np.interp(cp_channels, eff_geom["channel"], eff_geom["x_eff_m"])
        y_cp = np.interp(cp_channels, eff_geom["channel"], eff_geom["y_eff_m"])
        seg_len = np.sqrt(np.diff(x_cp) ** 2 + np.diff(y_cp) ** 2)

        base_geom = make_effective_geometry(
            prior=prior,
            cp_channels=cp_channels,
            cp_dx=np.zeros_like(cp_dx),
            cp_dy=np.zeros_like(cp_dy),
            fixed_global_offset_ch=fixed_global_offset_ch,
            interp_map=interp_map,
        )
        x0_cp = np.interp(cp_channels, base_geom["channel"], base_geom["x_eff_m"])
        y0_cp = np.interp(cp_channels, base_geom["channel"], base_geom["y_eff_m"])
        seg_len0 = np.sqrt(np.diff(x0_cp) ** 2 + np.diff(y0_cp) ** 2)

        out.append(np.sqrt(lambda_length) * (seg_len - seg_len0))

    return np.concatenate(out)


def fit_piecewise_xy(
    obs: pd.DataFrame,
    prior: pd.DataFrame,
    interp_map: dict[str, interp1d],
    cp_channels: np.ndarray,
    fixed_global_offset_ch: float,
    sound_speed: float,
    lambda_xy_smooth: float,
    lambda_xy_slope: float,
    lambda_xy_anchor: float,
    lambda_length: float,
    bound_abs_m: float,
) -> tuple[np.ndarray, np.ndarray, object, pd.DataFrame, pd.DataFrame]:
    ncp = len(cp_channels)
    x0 = np.zeros(2 * ncp, dtype=float)
    lower = -bound_abs_m * np.ones_like(x0)
    upper = +bound_abs_m * np.ones_like(x0)

    result = least_squares(
        objective_vector,
        x0=x0,
        bounds=(lower, upper),
        kwargs=dict(
            obs=obs,
            prior=prior,
            interp_map=interp_map,
            cp_channels=cp_channels,
            fixed_global_offset_ch=fixed_global_offset_ch,
            sound_speed=sound_speed,
            lambda_xy_smooth=lambda_xy_smooth,
            lambda_xy_slope=lambda_xy_slope,
            lambda_xy_anchor=lambda_xy_anchor,
            lambda_length=lambda_length,
        ),
        max_nfev=500,
        verbose=2,
    )

    cp_dx = result.x[:ncp].copy()
    cp_dy = result.x[ncp:].copy()

    eff_geom = make_effective_geometry(
        prior=prior,
        cp_channels=cp_channels,
        cp_dx=cp_dx,
        cp_dy=cp_dy,
        fixed_global_offset_ch=fixed_global_offset_ch,
        interp_map=interp_map,
    )
    pred_rows = predict_rows_for_geometry(obs=obs, eff_geom=eff_geom, sound_speed=sound_speed)
    return cp_dx, cp_dy, result, eff_geom, pred_rows


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
                }
            )
        )
        .reset_index()
    )


def group_summary(pred_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (location, anchor_index, anchor_label), g in pred_rows.groupby(["location", "anchor_index", "anchor_label"], sort=True):
        m = weighted_metrics(g["residual_s"].to_numpy(dtype=float), g["fit_weight"].to_numpy(dtype=float))
        m.update(
            {
                "location": location,
                "anchor_index": int(anchor_index),
                "anchor_label": anchor_label,
                "n_rows": int(len(g)),
            }
        )
        rows.append(m)
    return pd.DataFrame(rows)


def plot_xy_control_curves(cp_channels: np.ndarray, cp_dx: np.ndarray, cp_dy: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(cp_channels, cp_dx, marker="o", linewidth=2, label="dx(ch)")
    plt.plot(cp_channels, cp_dy, marker="o", linewidth=2, label="dy(ch)")
    plt.axhline(0.0, linestyle="--", linewidth=1.2)
    plt.xlabel("Channel")
    plt.ylabel("Local XY translation (m)")
    plt.title("Piecewise local XY segment shifts")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_effective_xy_vs_channel(eff_geom: pd.DataFrame, prior: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(eff_geom["channel"], prior["prior_x_m"], label="Prior x", linewidth=2)
    axes[0].plot(eff_geom["channel"], eff_geom["x_eff_m"], label="Fitted x", linewidth=2)
    axes[0].set_ylabel("Easting (m)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(eff_geom["channel"], prior["prior_y_m"], label="Prior y", linewidth=2)
    axes[1].plot(eff_geom["channel"], eff_geom["y_eff_m"], label="Fitted y", linewidth=2)
    axes[1].set_xlabel("Channel")
    axes[1].set_ylabel("Northing (m)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("Prior vs fitted XY coordinates by channel")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_path_compare(prior: pd.DataFrame, eff_geom: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 8))
    plt.plot(prior["prior_x_m"], prior["prior_y_m"], linewidth=2, label="Prior")
    plt.plot(eff_geom["x_eff_m"], eff_geom["y_eff_m"], linewidth=2, label="Fixed shift + piecewise XY fit")
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.title("Prior vs fixed-shift + piecewise XY fitted path")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_depth_compare(prior: pd.DataFrame, eff_geom: pd.DataFrame, out_path: Path) -> None:
    if "cum_dist_3d_m" in prior.columns:
        xaxis = prior["cum_dist_3d_m"].to_numpy(dtype=float)
        xlabel = "Cumulative 3D distance (m)"
    elif "cum_dist_horizontal_m" in prior.columns:
        xaxis = prior["cum_dist_horizontal_m"].to_numpy(dtype=float)
        xlabel = "Cumulative horizontal distance (m)"
    else:
        xaxis = prior["channel"].to_numpy(dtype=float)
        xlabel = "Channel"

    plt.figure(figsize=(12, 5))
    plt.plot(xaxis, prior["prior_u_m"], linewidth=2, label="Prior depth/u")
    plt.plot(xaxis, eff_geom["u_eff_m"], linewidth=2, label="Fit depth/u")
    plt.xlabel(xlabel)
    plt.ylabel("Up / depth-like coordinate (m)")
    plt.title("Depth profile after fixed global shift + piecewise XY fit")
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
    fig.suptitle("Observed vs predicted after fixed-shift + piecewise XY inversion", y=0.995, fontsize=14)
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
    plt.title("Timing misfit by channel after fixed-shift + piecewise XY fit")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def default_location_weights() -> dict[str, float]:
    return {
        "loc3_tx1": 2.5,
        "loc4_tx1": 2.5,
        "loc2_tx3": 1.2,
        "loc7_tx1": 1.0,
        "loc5_tx1": 0.5,
        "loc6_tx1": 0.5,
    }


def parse_location_weights(text: str | None) -> dict[str, float]:
    if text is None or str(text).strip() == "":
        return default_location_weights()
    items = {}
    for part in str(text).split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        items[k.strip()] = float(v.strip())
    return items


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit piecewise local XY shifts of cable sections after a fixed global channel offset."
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
        default=Path(r"D:\Singapore Data\Cable\piecewise_local_xy_outputs"),
    )
    parser.add_argument("--sound-speed", type=float, default=DEFAULT_SOUND_SPEED_MPS)
    parser.add_argument("--fixed-global-offset", type=float, default=DEFAULT_FIXED_GLOBAL_OFFSET_CH)

    parser.add_argument("--control-spacing-ch", type=float, default=120.0)
    parser.add_argument("--bound-abs-m", type=float, default=60.0)

    parser.add_argument("--lambda-xy-smooth", type=float, default=5.0)
    parser.add_argument("--lambda-xy-slope", type=float, default=1.0)
    parser.add_argument("--lambda-xy-anchor", type=float, default=0.005)
    parser.add_argument("--lambda-length", type=float, default=0.2)

    parser.add_argument("--min-weight", type=float, default=0.15)
    parser.add_argument("--min-stable-fraction", type=float, default=0.5)
    parser.add_argument("--all-usable", action="store_true")
    parser.add_argument("--use-raw", action="store_true")

    parser.add_argument(
        "--location-weights",
        type=str,
        default="",
        help="Comma-separated multipliers like loc3_tx1=3,loc4_tx1=3,loc5_tx1=0.5 . Empty uses defaults.",
    )

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

    loc_w = parse_location_weights(args.location_weights)
    obs = apply_location_weight_multipliers(obs, loc_w)

    interp_map = make_prior_interpolators(prior)
    cp_channels = build_control_points(CHANNEL_MIN, CHANNEL_MAX, args.control_spacing_ch)

    cp_dx, cp_dy, result, eff_geom, pred_rows = fit_piecewise_xy(
        obs=obs,
        prior=prior,
        interp_map=interp_map,
        cp_channels=cp_channels,
        fixed_global_offset_ch=args.fixed_global_offset,
        sound_speed=args.sound_speed,
        lambda_xy_smooth=args.lambda_xy_smooth,
        lambda_xy_slope=args.lambda_xy_slope,
        lambda_xy_anchor=args.lambda_xy_anchor,
        lambda_length=args.lambda_length,
        bound_abs_m=args.bound_abs_m,
    )

    if pred_rows.empty:
        raise RuntimeError("Prediction rows are empty. Fit failed to produce usable geometry.")

    ch_summary = channel_residual_summary(pred_rows)
    grp_summary = group_summary(pred_rows)
    overall = weighted_metrics(pred_rows["residual_s"].to_numpy(dtype=float), pred_rows["fit_weight"].to_numpy(dtype=float))

    cp_df = pd.DataFrame(
        {
            "control_channel": cp_channels,
            "dx_m": cp_dx,
            "dy_m": cp_dy,
            "displacement_m": np.sqrt(cp_dx ** 2 + cp_dy ** 2),
            "fixed_global_offset_ch": args.fixed_global_offset,
        }
    )

    eff_out = prior.copy()
    eff_out["fixed_global_offset_ch"] = args.fixed_global_offset
    eff_out["mapped_prior_channel_global"] = eff_geom["mapped_prior_channel_global"]
    eff_out["dx_local_m"] = eff_geom["dx_local_m"]
    eff_out["dy_local_m"] = eff_geom["dy_local_m"]
    eff_out["x_eff_m"] = eff_geom["x_eff_m"]
    eff_out["y_eff_m"] = eff_geom["y_eff_m"]
    eff_out["u_eff_m"] = eff_geom["u_eff_m"]

    cp_df.to_csv(args.output_dir / "xy_control_points.csv", index=False)
    eff_out.to_csv(args.output_dir / "prior_geometry_with_piecewise_xy_fit.csv", index=False)
    pred_rows.to_csv(args.output_dir / "predicted_vs_observed_rows.csv", index=False)
    ch_summary.to_csv(args.output_dir / "residual_by_channel_summary.csv", index=False)
    grp_summary.to_csv(args.output_dir / "group_misfit_summary.csv", index=False)

    fit_metrics = {
        "fixed_global_offset_ch": float(args.fixed_global_offset),
        "control_spacing_ch": float(args.control_spacing_ch),
        "n_control_points": int(len(cp_channels)),
        "bound_abs_m": float(args.bound_abs_m),
        "lambda_xy_smooth": float(args.lambda_xy_smooth),
        "lambda_xy_slope": float(args.lambda_xy_slope),
        "lambda_xy_anchor": float(args.lambda_xy_anchor),
        "lambda_length": float(args.lambda_length),
        "location_weights": loc_w,
        "n_fit_rows": int(len(obs)),
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "optimizer_cost": float(result.cost),
        "max_abs_dx_m": float(np.max(np.abs(cp_dx))),
        "max_abs_dy_m": float(np.max(np.abs(cp_dy))),
        "max_displacement_m": float(np.max(np.sqrt(cp_dx ** 2 + cp_dy ** 2))),
        **overall,
    }
    with open(args.output_dir / "fit_metrics.json", "w", encoding="utf-8") as f:
        json.dump(fit_metrics, f, indent=2)

    plot_xy_control_curves(cp_channels, cp_dx, cp_dy, args.output_dir / "piecewise_xy_control_curves.png")
    plot_effective_xy_vs_channel(eff_geom, prior, args.output_dir / "effective_xy_by_channel.png")
    plot_path_compare(prior, eff_geom, args.output_dir / "prior_vs_piecewise_xy_fit.png")
    plot_depth_compare(prior, eff_geom, args.output_dir / "depth_profile_piecewise_xy_fit.png")
    plot_obs_pred(pred_rows, args.output_dir / "observed_vs_predicted_by_location_anchor.png")
    plot_residual_by_channel(ch_summary, args.output_dir / "residual_by_channel.png")

    print(f"Saved outputs to: {args.output_dir}")
    print(f"Rows used in fit: {len(obs)}")
    print(f"Fixed global offset: {args.fixed_global_offset:.3f} ch")
    print(f"Control points: {len(cp_channels)}")
    print(f"Max XY displacement at control points: {np.max(np.sqrt(cp_dx ** 2 + cp_dy ** 2)):.3f} m")
    print(f"Weighted RMSE: {overall['weighted_rmse_ms']:.3f} ms")
    print(f"Weighted MAE: {overall['weighted_mae_ms']:.3f} ms")
    print(f"Median |residual|: {overall['median_abs_ms']:.3f} ms")


if __name__ == "__main__":
    main()
