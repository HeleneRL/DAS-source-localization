from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares

CHANNEL_MIN = 348
CHANNEL_MAX = 2267
DEFAULT_SOUND_SPEED_MPS = 1500.0
DEFAULT_FIXED_GLOBAL_OFFSET_CH = 61.255


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.upper().eq("TRUE")


def parse_location_weights(spec: str | None) -> Dict[str, float]:
    if not spec:
        return {}
    out: Dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Bad location weight spec '{part}', expected loc=weight")
        k, v = part.split("=", 1)
        out[k.strip()] = float(v.strip())
    return out


def load_inputs(obs_csv: Path, prior_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = pd.read_csv(obs_csv)
    prior = pd.read_csv(prior_csv)
    return obs, prior


def prepare_prior(prior: pd.DataFrame) -> pd.DataFrame:
    req = [
        "channel",
        "prior_x_m",
        "prior_y_m",
        "prior_u_m",
        "prior_z_m",
    ]
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
    location_weights: Dict[str, float],
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
        df["fit_dt_s"] = df["median_smooth_offset_ms"] / 1000.0
    else:
        df["fit_dt_s"] = df["observed_dt_ref_s"]

    mask = df["use_observation"] & df["fit_dt_s"].notna() & (df["weight"] >= min_weight)
    mask &= df["stable_fraction"].fillna(0.0) >= min_stable_fraction
    if use_only_recommended:
        mask &= df["recommended_channel"] & df["recommended_global"]

    df = df[mask].copy()
    if df.empty:
        raise ValueError("No observations left after filtering. Relax thresholds.")

    w = df["weight"].to_numpy(dtype=float)
    w *= np.clip(df["channel_trust_score"].fillna(0.0).to_numpy(dtype=float), 0.0, 1.0)
    w *= np.clip(df["mean_channel_trust_score"].fillna(0.0).to_numpy(dtype=float), 0.0, 1.0)
    w *= np.where(df["recommended_channel"].to_numpy(), 1.0, 0.5)
    w *= np.where(df["recommended_global"].to_numpy(), 1.0, 0.7)

    if location_weights:
        loc_scale = df["location"].map(location_weights).fillna(1.0).to_numpy(dtype=float)
        w *= loc_scale

    df["fit_weight"] = w
    df = df[df["fit_weight"] > 0].copy()
    if df.empty:
        raise ValueError("No observations left after weighting.")
    return df


def build_shifted_prior(prior: pd.DataFrame, fixed_global_offset_ch: float) -> pd.DataFrame:
    ch = prior["channel"].to_numpy(dtype=float)
    q = ch + fixed_global_offset_ch

    out = prior.copy()
    for col in ["prior_x_m", "prior_y_m", "prior_u_m", "prior_z_m"]:
        p = PchipInterpolator(ch, prior[col].to_numpy(dtype=float), extrapolate=True)
        out[col] = p(q)

    out = out.rename(
        columns={
            "prior_x_m": "x_shifted",
            "prior_y_m": "y_shifted",
            "prior_u_m": "u_shifted",
            "prior_z_m": "z_shifted",
        }
    )
    out["effective_prior_channel"] = q
    return out


def make_control_channels(channel_min: int, channel_max: int, spacing: int) -> np.ndarray:
    ch = list(range(channel_min, channel_max + 1, spacing))
    if ch[-1] != channel_max:
        ch.append(channel_max)
    return np.asarray(ch, dtype=float)


def eval_channel_corrections(
    control_channels: np.ndarray,
    dx_cp: np.ndarray,
    dy_cp: np.ndarray,
    channels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fx = PchipInterpolator(control_channels, dx_cp, extrapolate=True)
    fy = PchipInterpolator(control_channels, dy_cp, extrapolate=True)
    return fx(channels), fy(channels)


def compute_path_geometry(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = np.gradient(x)
    dy = np.gradient(y)
    ds = np.sqrt(dx * dx + dy * dy)
    ds = np.where(ds <= 1e-12, 1.0, ds)
    tx = dx / ds
    ty = dy / ds
    cumdist = np.zeros_like(x)
    if len(x) > 1:
        seg = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
        cumdist[1:] = np.cumsum(seg)
    return tx, ty, cumdist


def predict_relative_times(cable_xyz: np.ndarray, tx_xyz: np.ndarray, ref_idx: int, sound_speed: float) -> np.ndarray:
    ranges = np.linalg.norm(cable_xyz - tx_xyz[None, :], axis=1)
    return (ranges - ranges[ref_idx]) / sound_speed


def weighted_metrics(resid_s: np.ndarray, w: np.ndarray) -> dict[str, float]:
    resid_ms = resid_s * 1000.0
    wsum = float(np.sum(w))
    if wsum <= 0:
        return {
            "weighted_rmse_ms": np.nan,
            "weighted_mae_ms": np.nan,
            "weighted_bias_ms": np.nan,
            "median_abs_ms": np.nan,
        }
    return {
        "weighted_rmse_ms": float(np.sqrt(np.sum(w * resid_ms**2) / wsum)),
        "weighted_mae_ms": float(np.sum(w * np.abs(resid_ms)) / wsum),
        "weighted_bias_ms": float(np.sum(w * resid_ms) / wsum),
        "median_abs_ms": float(np.median(np.abs(resid_ms))),
    }


class ChannelSplineXYObjective:
    def __init__(
        self,
        obs: pd.DataFrame,
        shifted_prior: pd.DataFrame,
        control_channels: np.ndarray,
        sound_speed: float,
        lambda_anchor: float,
        lambda_smooth: float,
        lambda_slope: float,
        lambda_curvature: float,
        lambda_length: float,
        lambda_end_anchor: float,
        boundary_anchor_m: float,
    ):
        self.obs = obs.copy()
        self.shifted_prior = shifted_prior.copy()
        self.control_channels = control_channels.astype(float)
        self.sound_speed = float(sound_speed)
        self.lambda_anchor = float(lambda_anchor)
        self.lambda_smooth = float(lambda_smooth)
        self.lambda_slope = float(lambda_slope)
        self.lambda_curvature = float(lambda_curvature)
        self.lambda_length = float(lambda_length)
        self.lambda_end_anchor = float(lambda_end_anchor)
        self.boundary_anchor_m = float(boundary_anchor_m)

        self.channels = shifted_prior["channel"].to_numpy(dtype=int)
        self.channel_float = self.channels.astype(float)
        self.x0 = shifted_prior["x_shifted"].to_numpy(dtype=float)
        self.y0 = shifted_prior["y_shifted"].to_numpy(dtype=float)
        self.u0 = shifted_prior["u_shifted"].to_numpy(dtype=float)
        self.z0 = shifted_prior["z_shifted"].to_numpy(dtype=float)
        self.ch_to_idx = {int(ch): i for i, ch in enumerate(self.channels)}

        seg0 = np.sqrt(np.diff(self.x0) ** 2 + np.diff(self.y0) ** 2)
        self.total_length0 = float(np.sum(seg0))

        self.obs_groups = []
        for (location, anchor_index, anchor_label), g in self.obs.groupby(
            ["location", "anchor_index", "anchor_label"], sort=True
        ):
            g = g.sort_values("channel").copy()
            ref_ch = int(g["reference_channel"].iloc[0])
            if ref_ch not in self.ch_to_idx:
                continue
            ref_idx = self.ch_to_idx[ref_ch]
            idx = g["channel"].map(self.ch_to_idx).to_numpy(dtype=int)
            tx_xyz = np.array([
                g["tx_x_m"].iloc[0],
                g["tx_y_m"].iloc[0],
                g["tx_u_m"].iloc[0],
            ], dtype=float)
            self.obs_groups.append(
                {
                    "location": location,
                    "anchor_index": int(anchor_index),
                    "anchor_label": anchor_label,
                    "ref_idx": ref_idx,
                    "idx": idx,
                    "obs_dt": g["fit_dt_s"].to_numpy(dtype=float),
                    "w": g["fit_weight"].to_numpy(dtype=float),
                    "tx_xyz": tx_xyz,
                    "rows": g,
                }
            )
        if not self.obs_groups:
            raise ValueError("No valid observation groups after preparing objective.")

    def unpack(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(self.control_channels)
        dx_cp = p[:n]
        dy_cp = p[n:]
        return dx_cp, dy_cp

    def eval_path(self, p: np.ndarray) -> dict[str, np.ndarray]:
        dx_cp, dy_cp = self.unpack(p)
        dx, dy = eval_channel_corrections(self.control_channels, dx_cp, dy_cp, self.channel_float)
        x = self.x0 + dx
        y = self.y0 + dy
        tx, ty, cumdist = compute_path_geometry(x, y)
        return {
            "dx_cp": dx_cp,
            "dy_cp": dy_cp,
            "dx": dx,
            "dy": dy,
            "x": x,
            "y": y,
            "tx": tx,
            "ty": ty,
            "cumdist": cumdist,
        }

    def residual_vector(self, p: np.ndarray) -> np.ndarray:
        path = self.eval_path(p)
        cable_xyz = np.column_stack([path["x"], path["y"], self.u0])
        parts = []

        # data residuals
        for grp in self.obs_groups:
            pred_dt = predict_relative_times(cable_xyz, grp["tx_xyz"], grp["ref_idx"], self.sound_speed)
            pred = pred_dt[grp["idx"]]
            resid = pred - grp["obs_dt"]
            parts.append(np.sqrt(grp["w"]) * resid)

        dx_cp = path["dx_cp"]
        dy_cp = path["dy_cp"]

        # anchor displacement penalty
        if self.lambda_anchor > 0:
            parts.append(np.sqrt(self.lambda_anchor) * dx_cp)
            parts.append(np.sqrt(self.lambda_anchor) * dy_cp)

        # stronger endpoint anchoring
        if self.lambda_end_anchor > 0 and len(dx_cp) >= 2:
            endvec = np.array([dx_cp[0], dy_cp[0], dx_cp[-1], dy_cp[-1]], dtype=float)
            parts.append(np.sqrt(self.lambda_end_anchor) * endvec)

        # optional soft boundary target toward zero at ends / not hard bound, just guidance
        if self.boundary_anchor_m > 0 and len(dx_cp) >= 2:
            endmag = np.array([dx_cp[0], dy_cp[0], dx_cp[-1], dy_cp[-1]], dtype=float) / self.boundary_anchor_m
            parts.append(0.25 * endmag)

        # smoothness: second differences on control points
        if self.lambda_smooth > 0 and len(dx_cp) >= 3:
            parts.append(np.sqrt(self.lambda_smooth) * np.diff(dx_cp, n=2))
            parts.append(np.sqrt(self.lambda_smooth) * np.diff(dy_cp, n=2))

        # slope penalty: first differences
        if self.lambda_slope > 0 and len(dx_cp) >= 2:
            parts.append(np.sqrt(self.lambda_slope) * np.diff(dx_cp))
            parts.append(np.sqrt(self.lambda_slope) * np.diff(dy_cp))

        # curvature penalty on final path in channel order
        if self.lambda_curvature > 0 and len(path["x"]) >= 3:
            parts.append(np.sqrt(self.lambda_curvature) * np.diff(path["x"], n=2))
            parts.append(np.sqrt(self.lambda_curvature) * np.diff(path["y"], n=2))

        # preserve total path length roughly
        if self.lambda_length > 0:
            seg = np.sqrt(np.diff(path["x"]) ** 2 + np.diff(path["y"]) ** 2)
            total_len = np.sum(seg)
            parts.append(np.array([np.sqrt(self.lambda_length) * (total_len - self.total_length0)]))

        return np.concatenate(parts)

    def summarize_solution(self, p: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
        path = self.eval_path(p)
        cable_xyz = np.column_stack([path["x"], path["y"], self.u0])

        pred_rows = []
        resid_all = []
        w_all = []
        group_rows = []
        for grp in self.obs_groups:
            pred_dt = predict_relative_times(cable_xyz, grp["tx_xyz"], grp["ref_idx"], self.sound_speed)
            pred = pred_dt[grp["idx"]]
            resid = pred - grp["obs_dt"]
            resid_all.append(resid)
            w_all.append(grp["w"])
            gm = weighted_metrics(resid, grp["w"])
            gm.update(
                {
                    "location": grp["location"],
                    "anchor_index": grp["anchor_index"],
                    "anchor_label": grp["anchor_label"],
                    "n_rows": int(len(grp["rows"])),
                }
            )
            group_rows.append(gm)
            tmp = grp["rows"][["location", "anchor_index", "anchor_label", "channel", "reference_channel", "fit_dt_s", "fit_weight"]].copy()
            tmp["pred_dt_s"] = pred
            tmp["residual_s"] = resid
            pred_rows.append(tmp)

        pred_df = pd.concat(pred_rows, ignore_index=True)
        resid_all = np.concatenate(resid_all)
        w_all = np.concatenate(w_all)
        overall = weighted_metrics(resid_all, w_all)

        # channel summary without groupby apply to avoid warnings
        chs = []
        for ch, g in pred_df.groupby("channel", sort=True):
            r_ms = 1000.0 * g["residual_s"].to_numpy(dtype=float)
            w = g["fit_weight"].to_numpy(dtype=float)
            wsum = max(np.sum(w), 1e-12)
            chs.append(
                {
                    "channel": int(ch),
                    "weighted_rmse_ms": float(np.sqrt(np.sum(w * r_ms**2) / wsum)),
                    "weighted_mae_ms": float(np.sum(w * np.abs(r_ms)) / wsum),
                    "median_abs_ms": float(np.median(np.abs(r_ms))),
                    "n_rows": int(len(g)),
                }
            )
        ch_summary = pd.DataFrame(chs)

        fit_path = self.shifted_prior[["channel"]].copy()
        fit_path["x_shifted"] = self.x0
        fit_path["y_shifted"] = self.y0
        fit_path["u_shifted"] = self.u0
        fit_path["z_shifted"] = self.z0
        fit_path["effective_prior_channel"] = self.shifted_prior["effective_prior_channel"].to_numpy(dtype=float)
        fit_path["dx_channel_m"] = path["dx"]
        fit_path["dy_channel_m"] = path["dy"]
        fit_path["x_fit_m"] = path["x"]
        fit_path["y_fit_m"] = path["y"]
        fit_path["u_fit_m"] = self.u0
        fit_path["z_fit_m"] = self.z0
        fit_path["cumdist_fit_m"] = path["cumdist"]
        fit_path["tangent_x_fit"] = path["tx"]
        fit_path["tangent_y_fit"] = path["ty"]

        cp_df = pd.DataFrame(
            {
                "channel_control": self.control_channels,
                "dx_cp_m": path["dx_cp"],
                "dy_cp_m": path["dy_cp"],
            }
        )

        extra = {
            "overall": overall,
            "max_control_displacement_m": float(np.max(np.sqrt(path["dx_cp"] ** 2 + path["dy_cp"] ** 2))),
            "total_length_fit_m": float(path["cumdist"][-1] if len(path["cumdist"]) else 0.0),
            "total_length_shifted_prior_m": self.total_length0,
        }
        return pred_df, pd.DataFrame(group_rows), ch_summary, fit_path, cp_df, extra


def make_bounds(n_control: int, bound_abs_m: float) -> tuple[np.ndarray, np.ndarray]:
    lb = np.full(2 * n_control, -bound_abs_m, dtype=float)
    ub = np.full(2 * n_control, bound_abs_m, dtype=float)
    return lb, ub


def plot_control_displacements(cp_df: pd.DataFrame, out: Path) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(cp_df["channel_control"], cp_df["dx_cp_m"], marker="o", label="dx(ch)")
    plt.plot(cp_df["channel_control"], cp_df["dy_cp_m"], marker="o", label="dy(ch)")
    plt.axhline(0.0, linestyle="--", linewidth=1.2)
    plt.xlabel("Channel")
    plt.ylabel("Control-point displacement (m)")
    plt.title("Channel-spline XY control-point displacements")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()


def plot_path(prior: pd.DataFrame, fit_path: pd.DataFrame, out: Path) -> None:
    plt.figure(figsize=(8, 8))
    plt.plot(prior["prior_x_m"], prior["prior_y_m"], label="Prior", linewidth=2)
    plt.plot(fit_path["x_fit_m"], fit_path["y_fit_m"], label="Channel-spline XY fit", linewidth=2)
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.title("Prior vs channel-spline XY fitted path")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()


def plot_xy_by_channel(prior_shifted: pd.DataFrame, fit_path: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(fit_path["channel"], prior_shifted["x_shifted"], label="Shifted prior x")
    axes[0].plot(fit_path["channel"], fit_path["x_fit_m"], label="Fitted x")
    axes[0].set_ylabel("Easting (m)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(fit_path["channel"], prior_shifted["y_shifted"], label="Shifted prior y")
    axes[1].plot(fit_path["channel"], fit_path["y_fit_m"], label="Fitted y")
    axes[1].set_xlabel("Channel")
    axes[1].set_ylabel("Northing (m)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Shifted prior vs fitted XY coordinates by channel", y=0.98)
    plt.tight_layout()
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_depth(prior_shifted: pd.DataFrame, fit_path: pd.DataFrame, out: Path) -> None:
    if "cum_dist_3d_m" in prior_shifted.columns:
        xaxis = prior_shifted["cum_dist_3d_m"].to_numpy(dtype=float)
        xlabel = "Cumulative 3D distance (m)"
    elif "cum_dist_horizontal_m" in prior_shifted.columns:
        xaxis = prior_shifted["cum_dist_horizontal_m"].to_numpy(dtype=float)
        xlabel = "Cumulative horizontal distance (m)"
    else:
        xaxis = fit_path["cumdist_fit_m"].to_numpy(dtype=float)
        xlabel = "Cumulative fitted distance (m)"

    plt.figure(figsize=(12, 5))
    plt.plot(xaxis, prior_shifted["u_shifted"], label="Shifted prior depth/u")
    plt.plot(xaxis, fit_path["u_fit_m"], label="Fit depth/u")
    plt.xlabel(xlabel)
    plt.ylabel("Up / depth-like coordinate (m)")
    plt.title("Depth profile after channel-spline XY fit (depth unchanged)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()


def plot_obs_pred(pred_df: pd.DataFrame, out: Path) -> None:
    groups = list(pred_df.groupby(["location", "anchor_index", "anchor_label"], sort=True))
    n = len(groups)
    fig, axes = plt.subplots(n, 1, figsize=(12, max(3 * n, 6)), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, ((loc, anchor, label), g) in zip(axes, groups):
        g = g.sort_values("channel")
        ax.plot(g["channel"], 1000.0 * g["fit_dt_s"], label="Observed fit target", linewidth=1.3)
        ax.plot(g["channel"], 1000.0 * g["pred_dt_s"], label="Predicted", linewidth=1.3)
        ax.set_ylabel("dt to ref (ms)")
        ax.set_title(f"{loc} | anchor {anchor} | {label}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Channel")
    fig.suptitle("Observed vs predicted after channel-spline XY inversion", y=0.995)
    plt.tight_layout()
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_residual_by_channel(ch_summary: pd.DataFrame, out: Path) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(ch_summary["channel"], ch_summary["weighted_rmse_ms"], label="Weighted RMSE")
    plt.plot(ch_summary["channel"], ch_summary["median_abs_ms"], label="Median |residual|")
    plt.plot(ch_summary["channel"], ch_summary["weighted_mae_ms"], label="Weighted MAE")
    plt.xlabel("Channel")
    plt.ylabel("Residual (ms)")
    plt.title("Timing misfit by channel after channel-spline XY fit")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit smooth channel-indexed XY corrections to shifted prior cable geometry.")
    parser.add_argument("--obs-csv", type=Path, default=Path(r"D:\Singapore Data\Cable\inversion_observations.csv"))
    parser.add_argument("--prior-csv", type=Path, default=Path(r"D:\Singapore Data\Cable\prior_geometry.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path(r"D:\Singapore Data\Cable\channel_spline_xy_outputs"))
    parser.add_argument("--sound-speed", type=float, default=DEFAULT_SOUND_SPEED_MPS)
    parser.add_argument("--fixed-global-offset", type=float, default=DEFAULT_FIXED_GLOBAL_OFFSET_CH)
    parser.add_argument("--control-spacing-ch", type=int, default=25)
    parser.add_argument("--bound-abs-m", type=float, default=120.0)
    parser.add_argument("--min-weight", type=float, default=0.15)
    parser.add_argument("--min-stable-fraction", type=float, default=0.5)
    parser.add_argument("--all-usable", action="store_true")
    parser.add_argument("--use-raw", action="store_true")
    parser.add_argument("--location-weights", type=str, default="")
    parser.add_argument("--lambda-anchor", type=float, default=1e-4)
    parser.add_argument("--lambda-smooth", type=float, default=0.1)
    parser.add_argument("--lambda-slope", type=float, default=0.05)
    parser.add_argument("--lambda-curvature", type=float, default=0.0)
    parser.add_argument("--lambda-length", type=float, default=0.0)
    parser.add_argument("--lambda-end-anchor", type=float, default=0.001)
    parser.add_argument("--boundary-anchor-m", type=float, default=20.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    location_weights = parse_location_weights(args.location_weights)

    obs, prior = load_inputs(args.obs_csv, args.prior_csv)
    prior = prepare_prior(prior)
    obs = prepare_observations(
        obs,
        min_weight=args.min_weight,
        min_stable_fraction=args.min_stable_fraction,
        use_only_recommended=not args.all_usable,
        use_smoothed=not args.use_raw,
        location_weights=location_weights,
    )

    shifted_prior = build_shifted_prior(prior, args.fixed_global_offset)
    control_channels = make_control_channels(CHANNEL_MIN, CHANNEL_MAX, args.control_spacing_ch)

    obj = ChannelSplineXYObjective(
        obs=obs,
        shifted_prior=shifted_prior,
        control_channels=control_channels,
        sound_speed=args.sound_speed,
        lambda_anchor=args.lambda_anchor,
        lambda_smooth=args.lambda_smooth,
        lambda_slope=args.lambda_slope,
        lambda_curvature=args.lambda_curvature,
        lambda_length=args.lambda_length,
        lambda_end_anchor=args.lambda_end_anchor,
        boundary_anchor_m=args.boundary_anchor_m,
    )

    p0 = np.zeros(2 * len(control_channels), dtype=float)
    lb, ub = make_bounds(len(control_channels), args.bound_abs_m)

    res = least_squares(
        obj.residual_vector,
        p0,
        bounds=(lb, ub),
        method="trf",
        verbose=2,
        max_nfev=300,
        x_scale="jac",
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
    )

    pred_df, group_df, ch_summary, fit_path, cp_df, extra = obj.summarize_solution(res.x)

    pred_df.to_csv(args.output_dir / "predicted_vs_observed_rows.csv", index=False)
    group_df.to_csv(args.output_dir / "group_misfit_summary.csv", index=False)
    ch_summary.to_csv(args.output_dir / "channel_misfit_summary.csv", index=False)
    fit_path.to_csv(args.output_dir / "channel_spline_xy_fitted_path.csv", index=False)
    cp_df.to_csv(args.output_dir / "channel_spline_xy_control_points.csv", index=False)

    metrics = {
        "fixed_global_offset_channels": float(args.fixed_global_offset),
        "control_spacing_ch": int(args.control_spacing_ch),
        "n_control_points": int(len(control_channels)),
        "n_fit_rows": int(len(obs)),
        "sound_speed_mps": float(args.sound_speed),
        "bound_abs_m": float(args.bound_abs_m),
        "lambda_anchor": float(args.lambda_anchor),
        "lambda_smooth": float(args.lambda_smooth),
        "lambda_slope": float(args.lambda_slope),
        "lambda_curvature": float(args.lambda_curvature),
        "lambda_length": float(args.lambda_length),
        "lambda_end_anchor": float(args.lambda_end_anchor),
        "boundary_anchor_m": float(args.boundary_anchor_m),
        "use_smoothed": bool(not args.use_raw),
        "use_only_recommended": bool(not args.all_usable),
        "optimizer_success": bool(res.success),
        "optimizer_status": int(res.status),
        "optimizer_message": str(res.message),
        **extra,
    }
    with open(args.output_dir / "fit_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_control_displacements(cp_df, args.output_dir / "shape_control_displacements.png")
    plot_path(prior, fit_path, args.output_dir / "prior_vs_channel_spline_xy_fit.png")
    plot_xy_by_channel(shifted_prior, fit_path, args.output_dir / "effective_xy_by_channel.png")
    plot_depth(shifted_prior, fit_path, args.output_dir / "depth_profile_channel_spline_xy_fit.png")
    plot_obs_pred(pred_df, args.output_dir / "observed_vs_predicted_by_location_anchor.png")
    plot_residual_by_channel(ch_summary, args.output_dir / "residual_by_channel.png")

    print(f"Saved outputs to: {args.output_dir}")
    print(f"Rows used in fit: {len(obs)}")
    print(f"Fixed global offset: {args.fixed_global_offset:.3f} ch")
    print(f"Control points: {len(control_channels)}")
    print(f"Max control displacement: {extra['max_control_displacement_m']:.3f} m")
    print(f"Weighted RMSE: {extra['overall']['weighted_rmse_ms']:.3f} ms")
    print(f"Weighted MAE: {extra['overall']['weighted_mae_ms']:.3f} ms")
    print(f"Median |residual|: {extra['overall']['median_abs_ms']:.3f} ms")


if __name__ == "__main__":
    main()
