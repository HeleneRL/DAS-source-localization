from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, PchipInterpolator
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
    return pd.read_csv(obs_csv), pd.read_csv(prior_csv)


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
    return df


def default_location_weights() -> dict[str, float]:
    return {
        "loc3_tx1": 6.0,
        "loc4_tx1": 6.0,
        "loc2_tx3": 2.0,
        "loc7_tx1": 1.0,
        "loc5_tx1": 0.15,
        "loc6_tx1": 0.15,
    }


def parse_location_weights(text: str | None) -> dict[str, float]:
    if text is None or str(text).strip() == "":
        return default_location_weights()
    out = {}
    for part in str(text).split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = float(v.strip())
    return out


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
        c: interp1d(ch, prior[c].to_numpy(dtype=float), kind="linear", bounds_error=False, fill_value="extrapolate")
        for c in cols
    }


def build_control_channels(ch_min: int, ch_max: int, spacing: float) -> np.ndarray:
    cps = np.arange(ch_min, ch_max + 0.5 * spacing, spacing, dtype=float)
    if cps[-1] < ch_max:
        cps = np.append(cps, float(ch_max))
    cps[0] = float(ch_min)
    cps[-1] = float(ch_max)
    return cps


def shifted_prior_base(channels: np.ndarray, fixed_global_offset_ch: float, interp_map: dict[str, interp1d]) -> dict[str, np.ndarray]:
    ch_eff = channels.astype(float) + fixed_global_offset_ch
    return {
        "mapped_prior_channel": ch_eff,
        "x": interp_map["prior_x_m"](ch_eff),
        "y": interp_map["prior_y_m"](ch_eff),
        "u": interp_map["prior_u_m"](ch_eff),
    }


def cumulative_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    s = np.zeros(len(x), dtype=float)
    s[1:] = np.cumsum(d)
    return s


def resample_curve_by_arclength(x_ctrl: np.ndarray, y_ctrl: np.ndarray, n_samples: int, dense_factor: int = 25):
    t_ctrl = np.arange(len(x_ctrl), dtype=float)
    px = PchipInterpolator(t_ctrl, x_ctrl)
    py = PchipInterpolator(t_ctrl, y_ctrl)

    n_dense = max(1000, dense_factor * n_samples)
    t_dense = np.linspace(t_ctrl[0], t_ctrl[-1], n_dense)
    x_dense = px(t_dense)
    y_dense = py(t_dense)

    s_dense = cumulative_distance(x_dense, y_dense)
    total_len = float(s_dense[-1])
    if total_len <= 1e-12:
        x_out = np.full(n_samples, x_ctrl[0], dtype=float)
        y_out = np.full(n_samples, y_ctrl[0], dtype=float)
        s_out = np.zeros(n_samples, dtype=float)
        t_out = np.zeros(n_samples, dtype=float)
        return x_out, y_out, s_out, t_out

    s_target = np.linspace(0.0, total_len, n_samples)
    x_out = np.interp(s_target, s_dense, x_dense)
    y_out = np.interp(s_target, s_dense, y_dense)
    t_out = np.interp(s_target, s_dense, t_dense)
    return x_out, y_out, s_target, t_out


def make_effective_geometry_from_control_points(
    prior: pd.DataFrame,
    cp_channels: np.ndarray,
    cp_dx: np.ndarray,
    cp_dy: np.ndarray,
    fixed_global_offset_ch: float,
    interp_map: dict[str, interp1d],
):
    ch = prior["channel"].to_numpy(dtype=float)
    base_cp = shifted_prior_base(cp_channels, fixed_global_offset_ch, interp_map)
    x_ctrl = base_cp["x"] + cp_dx
    y_ctrl = base_cp["y"] + cp_dy

    x_res, y_res, s_res, t_res = resample_curve_by_arclength(x_ctrl, y_ctrl, len(ch))
    base_full = shifted_prior_base(ch, fixed_global_offset_ch, interp_map)

    eff = prior[["channel"]].copy()
    eff["mapped_prior_channel_global"] = base_full["mapped_prior_channel"]
    eff["x_eff_m"] = x_res
    eff["y_eff_m"] = y_res
    eff["u_eff_m"] = base_full["u"]
    eff["curve_arclength_m"] = s_res
    eff["shape_parameter_t"] = t_res

    cp_df = pd.DataFrame({
        "control_channel": cp_channels,
        "mapped_prior_channel_global": cp_channels + fixed_global_offset_ch,
        "x_ctrl_base_m": base_cp["x"],
        "y_ctrl_base_m": base_cp["y"],
        "dx_m": cp_dx,
        "dy_m": cp_dy,
        "x_ctrl_fit_m": x_ctrl,
        "y_ctrl_fit_m": y_ctrl,
        "displacement_m": np.sqrt(cp_dx ** 2 + cp_dy ** 2),
    })
    return eff, cp_df


def predict_relative_times(cable_xyz: np.ndarray, tx_xyz: np.ndarray, ref_idx: int, sound_speed: float) -> np.ndarray:
    ranges = np.linalg.norm(cable_xyz - tx_xyz[None, :], axis=1)
    return (ranges - ranges[ref_idx]) / sound_speed


def predict_rows_for_geometry(obs: pd.DataFrame, eff_geom: pd.DataFrame, sound_speed: float) -> pd.DataFrame:
    ch_to_idx = {int(ch): i for i, ch in enumerate(eff_geom["channel"].to_numpy(dtype=int))}
    cable_xyz = np.column_stack([eff_geom["x_eff_m"], eff_geom["y_eff_m"], eff_geom["u_eff_m"]]).astype(float)

    x_lookup = eff_geom.set_index("channel")["x_eff_m"]
    y_lookup = eff_geom.set_index("channel")["y_eff_m"]
    u_lookup = eff_geom.set_index("channel")["u_eff_m"]

    rows = []
    for (location, anchor_index, anchor_label), g in obs.groupby(["location", "anchor_index", "anchor_label"], sort=True):
        g = g.sort_values("channel").copy()
        ref_ch = int(g["reference_channel"].iloc[0])
        if ref_ch not in ch_to_idx:
            continue
        ref_idx = ch_to_idx[ref_ch]
        idx = g["channel"].map(ch_to_idx).to_numpy(dtype=int)

        tx_xyz = np.array([float(g["tx_x_m"].iloc[0]), float(g["tx_y_m"].iloc[0]), float(g["tx_u_m"].iloc[0])], dtype=float)
        pred_dt_all = predict_relative_times(cable_xyz, tx_xyz, ref_idx, sound_speed)
        pred = pred_dt_all[idx]
        obs_dt = g["fit_dt_s"].to_numpy(dtype=float)
        resid = pred - obs_dt

        tmp = g[["location", "anchor_index", "anchor_label", "channel", "reference_channel", "fit_dt_s", "fit_weight"]].copy()
        tmp["pred_dt_s"] = pred
        tmp["residual_s"] = resid
        tmp["x_eff_m"] = x_lookup.loc[g["channel"]].to_numpy(dtype=float)
        tmp["y_eff_m"] = y_lookup.loc[g["channel"]].to_numpy(dtype=float)
        tmp["u_eff_m"] = u_lookup.loc[g["channel"]].to_numpy(dtype=float)
        rows.append(tmp)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def weighted_metrics(resid_s: np.ndarray, w: np.ndarray) -> dict[str, float]:
    resid_ms = 1000.0 * resid_s
    wsum = float(np.sum(w))
    if wsum <= 0:
        return {"weighted_rmse_ms": np.nan, "weighted_mae_ms": np.nan, "weighted_bias_ms": np.nan, "median_abs_ms": np.nan}
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
    cp_channels: np.ndarray,
    fixed_global_offset_ch: float,
    interp_map: dict[str, interp1d],
    sound_speed: float,
    lambda_anchor: float,
    lambda_smooth: float,
    lambda_heading: float,
    lambda_total_length: float,
    lambda_segment_length: float,
    lambda_end_anchor: float,
) -> np.ndarray:
    ncp = len(cp_channels)
    cp_dx = params[:ncp]
    cp_dy = params[ncp:]

    eff_geom, cp_df = make_effective_geometry_from_control_points(
        prior, cp_channels, cp_dx, cp_dy, fixed_global_offset_ch, interp_map
    )
    pred_rows = predict_rows_for_geometry(obs, eff_geom, sound_speed)
    if pred_rows.empty:
        return np.array([1e6], dtype=float)

    out = []
    out.append(pred_rows["residual_s"].to_numpy(dtype=float) * np.sqrt(pred_rows["fit_weight"].to_numpy(dtype=float)))

    x_ctrl = cp_df["x_ctrl_fit_m"].to_numpy(dtype=float)
    y_ctrl = cp_df["y_ctrl_fit_m"].to_numpy(dtype=float)
    x0 = cp_df["x_ctrl_base_m"].to_numpy(dtype=float)
    y0 = cp_df["y_ctrl_base_m"].to_numpy(dtype=float)

    if lambda_anchor > 0:
        out.append(np.sqrt(lambda_anchor) * (x_ctrl - x0))
        out.append(np.sqrt(lambda_anchor) * (y_ctrl - y0))

    if len(x_ctrl) >= 3 and lambda_smooth > 0:
        out.append(np.sqrt(lambda_smooth) * (x_ctrl[:-2] - 2.0 * x_ctrl[1:-1] + x_ctrl[2:]))
        out.append(np.sqrt(lambda_smooth) * (y_ctrl[:-2] - 2.0 * y_ctrl[1:-1] + y_ctrl[2:]))

    if len(x_ctrl) >= 3 and lambda_heading > 0:
        vx = np.diff(x_ctrl)
        vy = np.diff(y_ctrl)
        seglen = np.sqrt(vx ** 2 + vy ** 2)
        seglen = np.where(seglen <= 1e-9, 1e-9, seglen)
        tx = vx / seglen
        ty = vy / seglen
        out.append(np.sqrt(lambda_heading) * np.diff(tx))
        out.append(np.sqrt(lambda_heading) * np.diff(ty))

    if lambda_total_length > 0:
        s_fit = cumulative_distance(eff_geom["x_eff_m"].to_numpy(dtype=float), eff_geom["y_eff_m"].to_numpy(dtype=float))
        total_fit = float(s_fit[-1])
        base_full = shifted_prior_base(prior["channel"].to_numpy(dtype=float), fixed_global_offset_ch, interp_map)
        s_base = cumulative_distance(base_full["x"], base_full["y"])
        total_base = float(s_base[-1])
        out.append(np.array([np.sqrt(lambda_total_length) * (total_fit - total_base)]))

    if len(x_ctrl) >= 2 and lambda_segment_length > 0:
        seg_fit = np.sqrt(np.diff(x_ctrl) ** 2 + np.diff(y_ctrl) ** 2)
        seg_base = np.sqrt(np.diff(x0) ** 2 + np.diff(y0) ** 2)
        out.append(np.sqrt(lambda_segment_length) * (seg_fit - seg_base))

    if lambda_end_anchor > 0:
        out.append(np.sqrt(lambda_end_anchor) * np.array([x_ctrl[0] - x0[0], y_ctrl[0] - y0[0]]))
        out.append(np.sqrt(lambda_end_anchor) * np.array([x_ctrl[-1] - x0[-1], y_ctrl[-1] - y0[-1]]))

    return np.concatenate(out)


def fit_smooth_shape(
    obs: pd.DataFrame,
    prior: pd.DataFrame,
    cp_channels: np.ndarray,
    fixed_global_offset_ch: float,
    interp_map: dict[str, interp1d],
    sound_speed: float,
    lambda_anchor: float,
    lambda_smooth: float,
    lambda_heading: float,
    lambda_total_length: float,
    lambda_segment_length: float,
    lambda_end_anchor: float,
    bound_abs_m: float,
):
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
            cp_channels=cp_channels,
            fixed_global_offset_ch=fixed_global_offset_ch,
            interp_map=interp_map,
            sound_speed=sound_speed,
            lambda_anchor=lambda_anchor,
            lambda_smooth=lambda_smooth,
            lambda_heading=lambda_heading,
            lambda_total_length=lambda_total_length,
            lambda_segment_length=lambda_segment_length,
            lambda_end_anchor=lambda_end_anchor,
        ),
        max_nfev=500,
        verbose=2,
    )

    cp_dx = result.x[:ncp].copy()
    cp_dy = result.x[ncp:].copy()
    eff_geom, cp_df = make_effective_geometry_from_control_points(
        prior, cp_channels, cp_dx, cp_dy, fixed_global_offset_ch, interp_map
    )
    pred_rows = predict_rows_for_geometry(obs, eff_geom, sound_speed)
    return cp_dx, cp_dy, result, eff_geom, cp_df, pred_rows


def channel_residual_summary(pred_rows: pd.DataFrame) -> pd.DataFrame:
    g = pred_rows.copy()
    g["residual_ms"] = 1000.0 * g["residual_s"]

    def per_channel(df: pd.DataFrame) -> pd.Series:
        w = df["fit_weight"].to_numpy(dtype=float)
        r = df["residual_ms"].to_numpy(dtype=float)
        rmse = float(np.sqrt(np.sum(w * r ** 2) / max(np.sum(w), 1e-12)))
        mae = float(np.sum(w * np.abs(r)) / max(np.sum(w), 1e-12))
        return pd.Series({
            "rmse_ms": rmse,
            "mae_ms": mae,
            "median_abs_ms": float(np.median(np.abs(r))),
            "n_rows": int(len(df)),
        })

    return g.groupby("channel", sort=True).apply(per_channel).reset_index()


def group_summary(pred_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (location, anchor_index, anchor_label), g in pred_rows.groupby(["location", "anchor_index", "anchor_label"], sort=True):
        m = weighted_metrics(g["residual_s"].to_numpy(dtype=float), g["fit_weight"].to_numpy(dtype=float))
        m.update({"location": location, "anchor_index": int(anchor_index), "anchor_label": anchor_label, "n_rows": int(len(g))})
        rows.append(m)
    return pd.DataFrame(rows)


def plot_control_displacements(cp_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(cp_df["control_channel"], cp_df["dx_m"], marker="o", linewidth=2, label="dx(ch)")
    plt.plot(cp_df["control_channel"], cp_df["dy_m"], marker="o", linewidth=2, label="dy(ch)")
    plt.axhline(0.0, linestyle="--", linewidth=1.2)
    plt.xlabel("Channel")
    plt.ylabel("Control-point displacement (m)")
    plt.title("Smooth cable-shape control-point displacements")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_path_compare(prior: pd.DataFrame, eff_geom: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 8))
    plt.plot(prior["prior_x_m"], prior["prior_y_m"], linewidth=2, label="Prior")
    plt.plot(eff_geom["x_eff_m"], eff_geom["y_eff_m"], linewidth=2, label="Smooth control-point fit")
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.title("Prior vs smooth cable-shape fitted path")
    plt.axis("equal")
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
    fig.suptitle("Observed vs predicted after smooth cable-shape inversion", y=0.995, fontsize=14)
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
    plt.title("Timing misfit by channel after smooth cable-shape fit")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_xy_by_channel(prior: pd.DataFrame, eff_geom: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(prior["channel"], prior["prior_x_m"], label="Prior x", linewidth=2)
    axes[0].plot(eff_geom["channel"], eff_geom["x_eff_m"], label="Fitted x", linewidth=2)
    axes[0].set_ylabel("Easting (m)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(prior["channel"], prior["prior_y_m"], label="Prior y", linewidth=2)
    axes[1].plot(eff_geom["channel"], eff_geom["y_eff_m"], label="Fitted y", linewidth=2)
    axes[1].set_xlabel("Channel")
    axes[1].set_ylabel("Northing (m)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("Prior vs fitted XY coordinates by channel")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


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
    plt.title("Depth profile after smooth cable-shape fit")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a smooth cable centerline from XY control points after a fixed global channel offset.")
    parser.add_argument("--obs-csv", type=Path, default=Path(r"D:\Singapore Data\Cable\inversion_observations.csv"))
    parser.add_argument("--prior-csv", type=Path, default=Path(r"D:\Singapore Data\Cable\prior_geometry.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path(r"D:\Singapore Data\Cable\smooth_cable_shape_outputs"))
    parser.add_argument("--sound-speed", type=float, default=DEFAULT_SOUND_SPEED_MPS)
    parser.add_argument("--fixed-global-offset", type=float, default=DEFAULT_FIXED_GLOBAL_OFFSET_CH)

    parser.add_argument("--control-spacing-ch", type=float, default=50.0)
    parser.add_argument("--bound-abs-m", type=float, default=200.0)

    parser.add_argument("--lambda-anchor", type=float, default=0.0001)
    parser.add_argument("--lambda-smooth", type=float, default=0.05)
    parser.add_argument("--lambda-heading", type=float, default=2.0)
    parser.add_argument("--lambda-total-length", type=float, default=0.05)
    parser.add_argument("--lambda-segment-length", type=float, default=0.02)
    parser.add_argument("--lambda-end-anchor", type=float, default=0.01)

    parser.add_argument("--min-weight", type=float, default=0.15)
    parser.add_argument("--min-stable-fraction", type=float, default=0.5)
    parser.add_argument("--all-usable", action="store_true")
    parser.add_argument("--use-raw", action="store_true")
    parser.add_argument("--location-weights", type=str, default="")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    obs, prior = load_inputs(args.obs_csv, args.prior_csv)
    prior = prepare_prior(prior)
    obs = prepare_observations(
        obs, args.min_weight, args.min_stable_fraction,
        use_only_recommended=not args.all_usable,
        use_smoothed=not args.use_raw,
    )
    obs = apply_location_weight_multipliers(obs, parse_location_weights(args.location_weights))

    interp_map = make_prior_interpolators(prior)
    cp_channels = build_control_channels(CHANNEL_MIN, CHANNEL_MAX, args.control_spacing_ch)

    cp_dx, cp_dy, result, eff_geom, cp_df, pred_rows = fit_smooth_shape(
        obs=obs,
        prior=prior,
        cp_channels=cp_channels,
        fixed_global_offset_ch=args.fixed_global_offset,
        interp_map=interp_map,
        sound_speed=args.sound_speed,
        lambda_anchor=args.lambda_anchor,
        lambda_smooth=args.lambda_smooth,
        lambda_heading=args.lambda_heading,
        lambda_total_length=args.lambda_total_length,
        lambda_segment_length=args.lambda_segment_length,
        lambda_end_anchor=args.lambda_end_anchor,
        bound_abs_m=args.bound_abs_m,
    )

    if pred_rows.empty:
        raise RuntimeError("Prediction rows are empty. Fit failed to produce usable geometry.")

    ch_summary = channel_residual_summary(pred_rows)
    grp_summary = group_summary(pred_rows)
    overall = weighted_metrics(pred_rows["residual_s"].to_numpy(dtype=float), pred_rows["fit_weight"].to_numpy(dtype=float))

    eff_out = prior.copy()
    eff_out["fixed_global_offset_ch"] = args.fixed_global_offset
    eff_out["mapped_prior_channel_global"] = eff_geom["mapped_prior_channel_global"]
    eff_out["x_eff_m"] = eff_geom["x_eff_m"]
    eff_out["y_eff_m"] = eff_geom["y_eff_m"]
    eff_out["u_eff_m"] = eff_geom["u_eff_m"]
    eff_out["curve_arclength_m"] = eff_geom["curve_arclength_m"]
    eff_out["shape_parameter_t"] = eff_geom["shape_parameter_t"]

    cp_df.to_csv(args.output_dir / "shape_control_points.csv", index=False)
    eff_out.to_csv(args.output_dir / "smooth_shape_geometry.csv", index=False)
    pred_rows.to_csv(args.output_dir / "predicted_vs_observed_rows.csv", index=False)
    ch_summary.to_csv(args.output_dir / "residual_by_channel_summary.csv", index=False)
    grp_summary.to_csv(args.output_dir / "group_misfit_summary.csv", index=False)

    fit_metrics = {
        "fixed_global_offset_ch": float(args.fixed_global_offset),
        "control_spacing_ch": float(args.control_spacing_ch),
        "n_control_points": int(len(cp_channels)),
        "bound_abs_m": float(args.bound_abs_m),
        "lambda_anchor": float(args.lambda_anchor),
        "lambda_smooth": float(args.lambda_smooth),
        "lambda_heading": float(args.lambda_heading),
        "lambda_total_length": float(args.lambda_total_length),
        "lambda_segment_length": float(args.lambda_segment_length),
        "lambda_end_anchor": float(args.lambda_end_anchor),
        "location_weights": parse_location_weights(args.location_weights),
        "n_fit_rows": int(len(obs)),
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "optimizer_cost": float(result.cost),
        "max_displacement_m": float(np.max(cp_df["displacement_m"].to_numpy(dtype=float))),
        **overall,
    }
    with open(args.output_dir / "fit_metrics.json", "w", encoding="utf-8") as f:
        json.dump(fit_metrics, f, indent=2)

    plot_control_displacements(cp_df, args.output_dir / "shape_control_displacements.png")
    plot_path_compare(prior, eff_geom, args.output_dir / "prior_vs_smooth_shape_fit.png")
    plot_xy_by_channel(prior, eff_geom, args.output_dir / "effective_xy_by_channel.png")
    plot_depth_compare(prior, eff_geom, args.output_dir / "depth_profile_smooth_shape_fit.png")
    plot_obs_pred(pred_rows, args.output_dir / "observed_vs_predicted_by_location_anchor.png")
    plot_residual_by_channel(ch_summary, args.output_dir / "residual_by_channel.png")

    print(f"Saved outputs to: {args.output_dir}")
    print(f"Rows used in fit: {len(obs)}")
    print(f"Fixed global offset: {args.fixed_global_offset:.3f} ch")
    print(f"Control points: {len(cp_channels)}")
    print(f"Max control-point displacement: {np.max(cp_df['displacement_m'].to_numpy(dtype=float)):.3f} m")
    print(f"Weighted RMSE: {overall['weighted_rmse_ms']:.3f} ms")
    print(f"Weighted MAE: {overall['weighted_mae_ms']:.3f} ms")
    print(f"Median |residual|: {overall['median_abs_ms']:.3f} ms")


if __name__ == "__main__":
    main()
