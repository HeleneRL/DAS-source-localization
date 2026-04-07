from __future__ import annotations

import os
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from das_cable_inversion import (
    build_observation_table,
    build_prior_geometry,
    linear_fill_to_full_channels,
    choose_control_channels,
    solve_inversion,
    compute_fit_diagnostics,
)


# ------------------------------------------------------------
# Fixed ENU origin used by the inversion dataset
# ------------------------------------------------------------
ENU_LAT0_DEG = 1.2160
ENU_LON0_DEG = 103.8518
ENU_H0_M = 0.0


# ------------------------------------------------------------
# Small utilities
# ------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .map({"TRUE": True, "FALSE": False})
        .fillna(False)
    )


def weighted_rmse(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return np.nan
    return float(np.sqrt(np.average(values[mask] ** 2, weights=weights[mask])))


def safe_quantile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return float(np.quantile(x, q))


def safe_col(df: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series([default] * len(df))


def latlon_to_local_xy_fixed_origin(lat_deg, lon_deg, lat0_deg, lon0_deg):
    """Simple local tangent-plane approximation reused from QC script for plan-view plotting."""
    R = 6371000.0
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    lat0 = np.radians(float(lat0_deg))
    lon0 = np.radians(float(lon0_deg))
    x = (lon - lon0) * np.cos(lat0) * R
    y = (lat - lat0) * R
    return x, y


def cumulative_arclength(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ds = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    return np.concatenate([[0.0], np.cumsum(ds)])


def project_points_onto_polyline(px, py, x_line, y_line, s_line=None):
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    x_line = np.asarray(x_line, dtype=float)
    y_line = np.asarray(y_line, dtype=float)

    if s_line is None:
        s_line = cumulative_arclength(x_line, y_line)
    else:
        s_line = np.asarray(s_line, dtype=float)

    n = len(px)
    dist = np.full(n, np.inf)
    proj_x = np.full(n, np.nan)
    proj_y = np.full(n, np.nan)
    proj_s = np.full(n, np.nan)
    seg_idx = np.full(n, -1, dtype=int)

    for i in range(len(x_line) - 1):
        x1, y1 = x_line[i], y_line[i]
        x2, y2 = x_line[i + 1], y_line[i + 1]
        dx = x2 - x1
        dy = y2 - y1
        seg_len2 = dx * dx + dy * dy

        if seg_len2 == 0:
            t = np.zeros_like(px)
            qx = np.full_like(px, x1)
            qy = np.full_like(py, y1)
            seg_len = 0.0
        else:
            t = ((px - x1) * dx + (py - y1) * dy) / seg_len2
            t = np.clip(t, 0.0, 1.0)
            qx = x1 + t * dx
            qy = y1 + t * dy
            seg_len = np.sqrt(seg_len2)

        d = np.sqrt((px - qx) ** 2 + (py - qy) ** 2)
        m = d < dist
        dist[m] = d[m]
        proj_x[m] = qx[m]
        proj_y[m] = qy[m]
        proj_s[m] = s_line[i] + t[m] * seg_len
        seg_idx[m] = i

    return dist, proj_x, proj_y, proj_s, seg_idx


def high_confidence_mask(obs: pd.DataFrame, min_weight: float) -> np.ndarray:
    mask = np.ones(len(obs), dtype=bool)
    if "use_observation" in obs.columns:
        mask &= ensure_bool(obs["use_observation"]).to_numpy(dtype=bool)
    if "passed_snr_threshold" in obs.columns:
        mask &= ensure_bool(obs["passed_snr_threshold"]).to_numpy(dtype=bool)
    if "near_window_edge" in obs.columns:
        mask &= ~ensure_bool(obs["near_window_edge"]).to_numpy(dtype=bool)
    if "weight" in obs.columns:
        w = pd.to_numeric(obs["weight"], errors="coerce").fillna(-np.inf).to_numpy()
        mask &= w >= min_weight
    return mask


# ------------------------------------------------------------
# Load inputs
# ------------------------------------------------------------

def load_observations_and_prior(input_csv: Path, channel_offset: int, min_weight: float):
    raw = pd.read_csv(input_csv)

    # Reuse the inversion origin embedded in the file when available.
    lat0 = float(raw["enu_origin_lat_deg"].dropna().iloc[0]) if "enu_origin_lat_deg" in raw.columns else ENU_LAT0_DEG
    lon0 = float(raw["enu_origin_lon_deg"].dropna().iloc[0]) if "enu_origin_lon_deg" in raw.columns else ENU_LON0_DEG
    h0 = float(raw["enu_origin_h_m"].dropna().iloc[0]) if "enu_origin_h_m" in raw.columns else ENU_H0_M

    obs = build_observation_table(raw, channel_offset)
    obs = obs[pd.to_numeric(obs["weight"], errors="coerce") >= min_weight].copy()

    prior_sparse = build_prior_geometry(raw, channel_offset)
    prior_full = linear_fill_to_full_channels(prior_sparse)

    min_ch = int(prior_full["channel"].min())
    max_ch = int(prior_full["channel"].max())
    obs = obs[(obs["channel_eff"] >= min_ch) & (obs["channel_eff"] <= max_ch)].copy()
    obs = obs[(obs["reference_channel_eff"] >= min_ch) & (obs["reference_channel_eff"] <= max_ch)].copy()
    obs = obs.reset_index(drop=True)

    tx_tbl = (
        obs.groupby("anchor_id")[["tx_x_m", "tx_y_m", "tx_u_m"]]
        .first()
        .reset_index()
    )

    return raw, obs, prior_full, tx_tbl, (lat0, lon0, h0)


def load_truth_geometry(truth_csv: Path, lat0: float, lon0: float) -> pd.DataFrame:
    truth = pd.read_csv(truth_csv)
    required = {"lat", "lon"}
    missing = required - set(truth.columns)
    if missing:
        raise ValueError(f"Truth CSV missing columns: {sorted(missing)}")

    x, y = latlon_to_local_xy_fixed_origin(truth["lat"].values, truth["lon"].values, lat0, lon0)
    truth = truth.copy()
    truth["x_m"] = x
    truth["y_m"] = y

    if "z" not in truth.columns:
        if "depth" in truth.columns:
            truth["z"] = pd.to_numeric(truth["depth"], errors="coerce")
        else:
            truth["z"] = np.nan

    # Nice if channel is present, but not required.
    for c in ["ch", "channel", "Channel", "CHAN", "chan"]:
        if c in truth.columns:
            truth["channel_like"] = pd.to_numeric(truth[c], errors="coerce")
            break
    else:
        truth["channel_like"] = np.arange(len(truth), dtype=float)

    return truth


def load_tuning_results(path: Path, source_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()
    df["source_name"] = source_name
    return df


# ------------------------------------------------------------
# Tuning summary and elbow plot
# ------------------------------------------------------------

def build_control_spacing_summary(*dfs: pd.DataFrame) -> pd.DataFrame:
    merged = pd.concat(dfs, ignore_index=True)
    merged["success_bool"] = merged["success"].astype(str).str.upper().eq("TRUE")

    # Keep all runs for context, but prefer converged ones for the main summary.
    good = merged[merged["success_bool"]].copy()
    if len(good) == 0:
        good = merged.copy()

    rows = []
    for spacing, g in good.groupby("control_spacing"):
        g = g.sort_values(["weighted_rmse_rel_opt_ms", "score", "weighted_rmse_abs_opt_ms"])
        best = g.iloc[0]
        rows.append({
            "control_spacing": float(spacing),
            "best_weighted_rmse_rel_opt_ms": float(best["weighted_rmse_rel_opt_ms"]),
            "best_weighted_rmse_abs_opt_ms": float(best["weighted_rmse_abs_opt_ms"]),
            "best_score": float(best["score"]),
            "best_n_control_points": float(best["n_control_points"]),
            "n_runs_here": int(len(g)),
            "best_source_name": str(best["source_name"]),
            "best_success": bool(best["success_bool"]),
        })
    out = pd.DataFrame(rows).sort_values("control_spacing").reset_index(drop=True)

    if len(out) > 1:
        base = float(out["best_weighted_rmse_rel_opt_ms"].min())
        out["excess_rel_rmse_ms"] = out["best_weighted_rmse_rel_opt_ms"] - base
    else:
        out["excess_rel_rmse_ms"] = 0.0
    return out


def choose_default_plot_spacings(summary: pd.DataFrame) -> list[int]:
    preferred = [5, 20, 40]
    available = set(int(x) for x in summary["control_spacing"].tolist())
    chosen = [x for x in preferred if x in available]
    if len(chosen) == 3:
        return chosen

    vals = sorted(int(x) for x in available)
    if not vals:
        return [5, 20, 40]

    # fallback: near lower / elbow-ish / moderate spacing
    targets = [5, 20, 40]
    out = []
    for t in targets:
        out.append(min(vals, key=lambda v: abs(v - t)))
    # preserve order but remove duplicates
    uniq = []
    for v in out:
        if v not in uniq:
            uniq.append(v)
    return uniq


def plot_elbow(summary: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(9, 5.5))
    x = summary["control_spacing"].values
    y = summary["best_weighted_rmse_rel_opt_ms"].values
    plt.plot(x, y, marker="o")
    plt.xlabel("Control-point spacing (channels)")
    plt.ylabel("Best weighted relative-time RMSE (ms)")
    plt.title("Control-spacing sensitivity (lower is better)")
    plt.grid(True, alpha=0.3)

    # Simple visual cue for the elbow: first point within 0.05 ms of minimum.
    ymin = np.nanmin(y)
    threshold = ymin + 0.05
    idx = np.where(y <= threshold)[0]
    if len(idx) > 0:
        i0 = int(idx[0])
        plt.axvline(x[i0], linestyle="--", alpha=0.8)
        plt.annotate(
            f"Near-flat region starts ≈ {int(x[i0])}",
            xy=(x[i0], y[i0]),
            xytext=(10, 15),
            textcoords="offset points",
        )

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


# ------------------------------------------------------------
# Inversion reruns for selected spacings
# ------------------------------------------------------------

def run_selected_inversions(
    obs: pd.DataFrame,
    prior_full: pd.DataFrame,
    selected_spacings: list[int],
    sound_speed: float,
    channel_spacing: float,
    abs_scale: float,
    rel_scale: float,
    prior_sigma_xy: float,
    prior_sigma_z: float,
    curvature_sigma_xy: float,
    curvature_sigma_z: float,
    spacing_sigma: float,
    anchor_bias_sigma: float,
    huber_delta_abs: float,
    huber_delta_rel: float,
    max_nfev: int,
):
    out = {}
    ref_channels = obs["reference_channel_eff"].unique()
    for spacing in selected_spacings:
        control_channels = choose_control_channels(
            prior_full["channel"].values,
            ref_channels,
            spacing,
        )
        solution = solve_inversion(
            obs=obs,
            prior_full=prior_full,
            control_channels=control_channels,
            sound_speed=sound_speed,
            channel_spacing=channel_spacing,
            abs_scale=abs_scale,
            rel_scale=rel_scale,
            prior_sigma_xy=prior_sigma_xy,
            prior_sigma_z=prior_sigma_z,
            curvature_sigma_xy=curvature_sigma_xy,
            curvature_sigma_z=curvature_sigma_z,
            spacing_sigma=spacing_sigma,
            anchor_bias_sigma=anchor_bias_sigma,
            huber_delta_abs=huber_delta_abs,
            huber_delta_rel=huber_delta_rel,
            max_nfev=max_nfev,
        )
        diagnostics = compute_fit_diagnostics(solution)
        out[int(spacing)] = {
            "solution": solution,
            "diagnostics": diagnostics,
        }
    return out


# ------------------------------------------------------------
# Geometry / QC helpers
# ------------------------------------------------------------

def build_segment_quality_table(layout_df: pd.DataFrame, truth_df: pd.DataFrame, fit_obs: pd.DataFrame, high_conf_mask: np.ndarray) -> pd.DataFrame:
    inv_x = pd.to_numeric(layout_df["x_m"], errors="coerce").values
    inv_y = pd.to_numeric(layout_df["y_m"], errors="coerce").values
    inv_z = pd.to_numeric(layout_df["z_m"], errors="coerce").values

    truth_x = pd.to_numeric(truth_df["x_m"], errors="coerce").values
    truth_y = pd.to_numeric(truth_df["y_m"], errors="coerce").values
    truth_z = pd.to_numeric(truth_df["z"], errors="coerce").values
    truth_s = cumulative_arclength(truth_x, truth_y)

    xy_err, truth_proj_x, truth_proj_y, truth_proj_s, _ = project_points_onto_polyline(
        inv_x, inv_y, truth_x, truth_y, s_line=truth_s
    )
    truth_z_on_inv = np.interp(truth_proj_s, truth_s, truth_z)
    abs_dz = np.abs(inv_z - truth_z_on_inv)

    obs = fit_obs.copy()
    obs["abs_res_ms"] = 1000.0 * pd.to_numeric(obs["residual_abs_opt_s"], errors="coerce")
    obs["rel_res_ms"] = 1000.0 * pd.to_numeric(obs["residual_dt_ref_opt_s"], errors="coerce")
    ch_col = "channel_eff" if "channel_eff" in obs.columns else "channel"
    obs[ch_col] = pd.to_numeric(obs[ch_col], errors="coerce")

    hc = obs.loc[high_conf_mask].copy()
    timing_hc = (
        hc.groupby(ch_col)
        .agg(
            n_obs_hc=("weight", "size"),
            median_abs_relres_ms_hc=("rel_res_ms", lambda x: np.median(np.abs(pd.to_numeric(x, errors="coerce")))),
            median_abs_absres_ms_hc=("abs_res_ms", lambda x: np.median(np.abs(pd.to_numeric(x, errors="coerce")))),
            mean_weight_hc=("weight", "mean"),
        )
        .reset_index()
        .rename(columns={ch_col: "channel"})
    )

    out = pd.DataFrame({
        "channel": layout_df["channel"].values,
        "x_m": inv_x,
        "y_m": inv_y,
        "z_m": inv_z,
        "xy_error_to_truth_m": xy_err,
        "truth_proj_x_m": truth_proj_x,
        "truth_proj_y_m": truth_proj_y,
        "abs_dz_to_truth_projected_m": abs_dz,
    })
    out = out.merge(timing_hc, on="channel", how="left")
    out["n_obs_hc"] = out["n_obs_hc"].fillna(0).astype(int)
    out["median_abs_relres_ms_hc"] = out["median_abs_relres_ms_hc"].fillna(np.nan)
    out["median_abs_absres_ms_hc"] = out["median_abs_absres_ms_hc"].fillna(np.nan)

    conditions = []
    for xe, te in zip(out["xy_error_to_truth_m"], out["median_abs_relres_ms_hc"]):
        if np.isfinite(xe) and np.isfinite(te) and (xe <= 4.0) and (te <= 20.0):
            conditions.append("good")
        elif np.isfinite(xe) and np.isfinite(te) and (xe <= 8.0) and (te <= 80.0):
            conditions.append("caution")
        else:
            conditions.append("poor")
    out["segment_quality"] = conditions
    return out


def make_layout_dataframe(prior_full: pd.DataFrame, solution: dict) -> pd.DataFrame:
    prior_xyz = solution["prior_xyz_full"]
    full_xyz = solution["full_xyz_opt"]
    cable = pd.DataFrame({
        "channel": solution["full_channels"],
        "prior_x_m": prior_xyz[:, 0],
        "prior_y_m": prior_xyz[:, 1],
        "prior_z_m": prior_xyz[:, 2],
        "x_m": full_xyz[:, 0],
        "y_m": full_xyz[:, 1],
        "z_m": full_xyz[:, 2],
    })
    cable["horizontal_shift_m"] = np.sqrt(
        (cable["x_m"] - cable["prior_x_m"]) ** 2 + (cable["y_m"] - cable["prior_y_m"]) ** 2
    )
    cable["dz_m"] = cable["z_m"] - cable["prior_z_m"]
    return cable


# ------------------------------------------------------------
# Plots for email/report
# ------------------------------------------------------------

def plot_geometry_comparison(layout_df: pd.DataFrame, truth_df: pd.DataFrame, tx_tbl: pd.DataFrame, out_png: Path, title_prefix: str) -> None:
    plt.figure(figsize=(9, 7.5))
    plt.plot(layout_df["prior_x_m"], layout_df["prior_y_m"], label="Prior", linewidth=2.0)
    plt.plot(truth_df["x_m"], truth_df["y_m"], label="Ground truth", linewidth=2.0)
    plt.plot(layout_df["x_m"], layout_df["y_m"], label="Inversion estimate", linewidth=2.4)
    plt.scatter(tx_tbl["tx_x_m"], tx_tbl["tx_y_m"], marker="x", s=65, label="Transmission sites")
    for _, row in tx_tbl.iterrows():
        plt.text(row["tx_x_m"], row["tx_y_m"], str(row["anchor_id"]), fontsize=7)
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.title(f"{title_prefix}: plan view")
    plt.axis("equal")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_z_comparison(layout_df: pd.DataFrame, truth_df: pd.DataFrame, out_png: Path, title_prefix: str) -> None:
    plt.figure(figsize=(10, 5.2))
    if "channel_like" in truth_df.columns:
        plt.plot(truth_df["channel_like"], truth_df["z"], label="Ground truth", linewidth=2.0)
    plt.plot(layout_df["channel"], layout_df["prior_z_m"], label="Prior", linewidth=2.0)
    plt.plot(layout_df["channel"], layout_df["z_m"], label="Inversion estimate", linewidth=2.4)
    plt.xlabel("Channel")
    plt.ylabel("Vertical coordinate (m)")
    plt.title(f"{title_prefix}: vertical profile")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_segment_confidence(seg_df: pd.DataFrame, out_png: Path, title: str) -> None:
    color_map = {
        "good": "tab:green",
        "caution": "goldenrod",
        "poor": "tab:red",
    }

    plt.figure(figsize=(14, 3.6))
    for quality in ["good", "caution", "poor"]:
        m = seg_df["segment_quality"].astype(str).str.lower().eq(quality)
        if np.any(m):
            plt.scatter(
                seg_df.loc[m, "channel"],
                np.zeros(np.sum(m)),
                s=34,
                label=quality,
                color=color_map[quality],
            )
    plt.xlabel("Channel")
    plt.yticks([])
    plt.ylabel("")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.grid(True, axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_observed_vs_predicted_highconf(obs_fit: pd.DataFrame, high_conf_mask: np.ndarray, out_png_rel: Path, out_png_abs: Path) -> None:
    hc = obs_fit.loc[high_conf_mask].copy()
    if len(hc) == 0:
        warnings.warn("No high-confidence observations available for predicted-vs-observed plots.")
        return

    # Relative travel times, faceted by location/anchor, using optimized geometry.
    if "anchor_id" not in hc.columns:
        if {"location", "anchor_index"}.issubset(hc.columns):
            hc["anchor_id"] = hc["location"].astype(str) + "_a" + hc["anchor_index"].astype(str)
        else:
            hc["anchor_id"] = "all"

    anchor_ids = sorted(hc["anchor_id"].dropna().astype(str).unique())
    ncol = 3
    nrow = int(np.ceil(len(anchor_ids) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 4.0 * nrow), squeeze=False)
    fig.suptitle("Observed vs predicted relative times (high-confidence subset)", y=0.995)

    for ax, aid in zip(axes.ravel(), anchor_ids):
        m = hc["anchor_id"].astype(str).eq(aid)
        x = pd.to_numeric(hc.loc[m, "observed_dt_ref_s"], errors="coerce").values
        y = pd.to_numeric(hc.loc[m, "predicted_dt_ref_s_opt"], errors="coerce").values
        w = pd.to_numeric(hc.loc[m, "weight"], errors="coerce").fillna(0.0).values
        keep = np.isfinite(x) & np.isfinite(y)
        x = x[keep]
        y = y[keep]
        w = w[keep]
        if len(x) == 0:
            ax.set_title(aid)
            ax.axis("off")
            continue
        ax.scatter(x, y, s=10 + 14 * np.clip(w, 0, 1), alpha=0.6)
        lim0 = min(np.nanmin(x), np.nanmin(y))
        lim1 = max(np.nanmax(x), np.nanmax(y))
        ax.plot([lim0, lim1], [lim0, lim1], linewidth=1.0)
        ax.set_title(aid)
        ax.set_xlabel("Observed dt_ref (s)")
        ax.set_ylabel("Predicted dt_ref (s)")
        ax.grid(True, alpha=0.2)
    for ax in axes.ravel()[len(anchor_ids):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png_rel, dpi=220)
    plt.close(fig)

    # Absolute travel times on one panel for the same high-confidence subset.
    if "predicted_t_abs_s_opt" in hc.columns:
        plt.figure(figsize=(7.2, 6.2))
        xa = pd.to_numeric(hc["observed_t_s"], errors="coerce").values
        ya = pd.to_numeric(hc["predicted_t_abs_s_opt"], errors="coerce").values
        w = pd.to_numeric(hc["weight"], errors="coerce").fillna(0.0).values
        keep = np.isfinite(xa) & np.isfinite(ya)
        xa = xa[keep]
        ya = ya[keep]
        w = w[keep]
        plt.scatter(xa, ya, s=10 + 20 * np.clip(w, 0, 1), alpha=0.45)
        lim0 = min(np.nanmin(xa), np.nanmin(ya))
        lim1 = max(np.nanmax(xa), np.nanmax(ya))
        plt.plot([lim0, lim1], [lim0, lim1], linewidth=1.2)
        plt.xlabel("Observed absolute travel time (s)")
        plt.ylabel("Predicted absolute travel time (s)")
        plt.title("Observed vs predicted absolute travel time (high-confidence observations)")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_png_abs, dpi=220)
        plt.close()


def plot_observed_vs_predicted_highconf_prior(obs_fit: pd.DataFrame, high_conf_mask: np.ndarray, out_png_rel: Path, out_png_abs: Path) -> None:
    hc = obs_fit.loc[high_conf_mask].copy()
    if len(hc) == 0:
        warnings.warn("No high-confidence observations available for prior predicted-vs-observed plots.")
        return

    if "anchor_id" not in hc.columns:
        if {"location", "anchor_index"}.issubset(hc.columns):
            hc["anchor_id"] = hc["location"].astype(str) + "_a" + hc["anchor_index"].astype(str)
        else:
            hc["anchor_id"] = "all"

    anchor_ids = sorted(hc["anchor_id"].dropna().astype(str).unique())
    ncol = 3
    nrow = int(np.ceil(len(anchor_ids) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 4.0 * nrow), squeeze=False)
    fig.suptitle("Observed vs predicted relative times from prior geometry (high-confidence subset)", y=0.995)

    for ax, aid in zip(axes.ravel(), anchor_ids):
        m = hc["anchor_id"].astype(str).eq(aid)
        x = pd.to_numeric(hc.loc[m, "observed_dt_ref_s"], errors="coerce").values
        y = pd.to_numeric(hc.loc[m, "predicted_dt_ref_s_prior"], errors="coerce").values
        w = pd.to_numeric(hc.loc[m, "weight"], errors="coerce").fillna(0.0).values
        keep = np.isfinite(x) & np.isfinite(y)
        x = x[keep]
        y = y[keep]
        w = w[keep]
        if len(x) == 0:
            ax.set_title(aid)
            ax.axis("off")
            continue
        ax.scatter(x, y, s=10 + 14 * np.clip(w, 0, 1), alpha=0.6)
        lim0 = min(np.nanmin(x), np.nanmin(y))
        lim1 = max(np.nanmax(x), np.nanmax(y))
        ax.plot([lim0, lim1], [lim0, lim1], linewidth=1.0)
        ax.set_title(aid)
        ax.set_xlabel("Observed dt_ref (s)")
        ax.set_ylabel("Predicted dt_ref from prior (s)")
        ax.grid(True, alpha=0.2)
    for ax in axes.ravel()[len(anchor_ids):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png_rel, dpi=220)
    plt.close(fig)

    if "predicted_t_abs_s_prior" in hc.columns:
        plt.figure(figsize=(7.2, 6.2))
        xa = pd.to_numeric(hc["observed_t_s"], errors="coerce").values
        ya = pd.to_numeric(hc["predicted_t_abs_s_prior"], errors="coerce").values
        w = pd.to_numeric(hc["weight"], errors="coerce").fillna(0.0).values
        keep = np.isfinite(xa) & np.isfinite(ya)
        xa = xa[keep]
        ya = ya[keep]
        w = w[keep]
        plt.scatter(xa, ya, s=10 + 20 * np.clip(w, 0, 1), alpha=0.45)
        lim0 = min(np.nanmin(xa), np.nanmin(ya))
        lim1 = max(np.nanmax(xa), np.nanmax(ya))
        plt.plot([lim0, lim1], [lim0, lim1], linewidth=1.2)
        plt.xlabel("Observed absolute travel time (s)")
        plt.ylabel("Predicted absolute travel time from prior (s)")
        plt.title("Observed vs predicted absolute travel time from prior geometry (high-confidence observations)")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_png_abs, dpi=220)
        plt.close()



def plot_prior_vs_estimate_side_by_side(obs_fit: pd.DataFrame, high_conf_mask: np.ndarray, out_png: Path):
    hc = obs_fit.loc[high_conf_mask].copy()
    if len(hc) == 0:
        warnings.warn("No high-confidence observations available.")
        return

    # Ensure anchor_id exists
    if "anchor_id" not in hc.columns:
        if {"location", "anchor_index"}.issubset(hc.columns):
            hc["anchor_id"] = hc["location"].astype(str) + "_a" + hc["anchor_index"].astype(str)
        else:
            hc["anchor_id"] = "all"

    anchor_ids = sorted(hc["anchor_id"].dropna().astype(str).unique())

    ncol = 2  # prior vs estimate
    nrow = len(anchor_ids)

    fig, axes = plt.subplots(nrow, ncol, figsize=(10, 4 * nrow), squeeze=False)
    fig.suptitle("Observed vs predicted relative times (prior vs inversion, high-confidence subset)", y=0.995)

    for i, aid in enumerate(anchor_ids):
        m = hc["anchor_id"].astype(str).eq(aid)

        x = pd.to_numeric(hc.loc[m, "observed_dt_ref_s"], errors="coerce").values
        y_prior = pd.to_numeric(hc.loc[m, "predicted_dt_ref_s_prior"], errors="coerce").values
        y_opt = pd.to_numeric(hc.loc[m, "predicted_dt_ref_s_opt"], errors="coerce").values
        w = pd.to_numeric(hc.loc[m, "weight"], errors="coerce").fillna(0.0).values

        keep = np.isfinite(x) & np.isfinite(y_prior) & np.isfinite(y_opt)
        x = x[keep]
        y_prior = y_prior[keep]
        y_opt = y_opt[keep]
        w = w[keep]

        if len(x) == 0:
            continue

        lim0 = min(np.nanmin(x), np.nanmin(y_prior), np.nanmin(y_opt))
        lim1 = max(np.nanmax(x), np.nanmax(y_prior), np.nanmax(y_opt))

        # --- PRIOR (left) ---
        ax = axes[i, 0]
        ax.scatter(x, y_prior, s=10 + 14 * np.clip(w, 0, 1), alpha=0.6)
        ax.plot([lim0, lim1], [lim0, lim1], linewidth=1.0)
        ax.set_title(f"{aid} — Prior")
        ax.set_xlabel("Observed dt_ref (s)")
        ax.set_ylabel("Predicted (prior)")
        ax.grid(True, alpha=0.2)

        # --- ESTIMATE (right) ---
        ax = axes[i, 1]
        ax.scatter(x, y_opt, s=10 + 14 * np.clip(w, 0, 1), alpha=0.6)
        ax.plot([lim0, lim1], [lim0, lim1], linewidth=1.0)
        ax.set_title(f"{aid} — Inversion")
        ax.set_xlabel("Observed dt_ref (s)")
        ax.set_ylabel("Predicted (estimate)")
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

# ------------------------------------------------------------
# Report tables and text
# ------------------------------------------------------------

def build_fit_diagnostics_table(obs: pd.DataFrame, diagnostics: dict) -> pd.DataFrame:
    fit = obs.copy()
    fit["predicted_t_abs_s_prior"] = diagnostics["pred_abs_prior"]
    fit["predicted_t_abs_s_opt"] = diagnostics["pred_abs"]
    fit["residual_abs_prior_s"] = fit["observed_t_s"] - fit["predicted_t_abs_s_prior"]
    fit["residual_abs_opt_s"] = fit["observed_t_s"] - fit["predicted_t_abs_s_opt"]
    fit["predicted_dt_ref_s_prior"] = diagnostics["pred_rel_prior"]
    fit["predicted_dt_ref_s_opt"] = diagnostics["pred_rel"]
    fit["residual_dt_ref_prior_s"] = fit["observed_dt_ref_s"] - fit["predicted_dt_ref_s_prior"]
    fit["residual_dt_ref_opt_s"] = fit["observed_dt_ref_s"] - fit["predicted_dt_ref_s_opt"]
    return fit


def write_summary_note(out_txt: Path, lat0: float, lon0: float, h0: float, control_summary: pd.DataFrame, selected_spacings: list[int], selected_results: dict, obs_fit_best: pd.DataFrame, high_conf_mask_best: np.ndarray) -> None:
    lines = []
    lines.append("Cable inversion email-report summary")
    lines.append("===================================")
    lines.append("")
    lines.append("ENU origin used for prior and inversion estimate:")
    lines.append(f"  lat0 = {lat0:.6f} deg")
    lines.append(f"  lon0 = {lon0:.6f} deg")
    lines.append(f"  h0   = {h0:.3f} m")
    lines.append("")
    lines.append("Important note:")
    lines.append("  Ground truth was rebuilt from lat/lon using the SAME ENU origin above,")
    lines.append("  so all plan-view comparisons are in one consistent local frame.")
    lines.append("")
    lines.append("Control-spacing sensitivity summary:")
    for _, row in control_summary.iterrows():
        lines.append(
            f"  spacing={int(row['control_spacing'])}: best weighted rel-RMSE={row['best_weighted_rmse_rel_opt_ms']:.3f} ms, "
            f"score={row['best_score']:.3f}, source={row['best_source_name']}"
        )
    lines.append("")
    lines.append("Selected comparison cases:")
    for spacing in selected_spacings:
        res = selected_results[spacing]
        sol = res["solution"]
        diag = res["diagnostics"]
        fit = build_fit_diagnostics_table(obs_fit_best.copy(), diag)
        hc = high_confidence_mask(fit, min_weight=0.8)
        rel_rmse_hc = 1000.0 * weighted_rmse(pd.to_numeric(fit.loc[hc, "residual_dt_ref_opt_s"], errors="coerce"), pd.to_numeric(fit.loc[hc, "weight"], errors="coerce"))
        abs_rmse_hc = 1000.0 * weighted_rmse(pd.to_numeric(fit.loc[hc, "residual_abs_opt_s"], errors="coerce"), pd.to_numeric(fit.loc[hc, "weight"], errors="coerce"))
        lines.append(
            f"  spacing={spacing}: success={bool(sol['result'].success)}, n_control_points={len(sol['control_channels'])}, "
            f"weighted rel-RMSE HC={rel_rmse_hc:.3f} ms, weighted abs-RMSE HC={abs_rmse_hc:.3f} ms"
        )
    lines.append("")
    lines.append(f"High-confidence definition: weight >= 0.8, use_observation = TRUE, passed_snr_threshold = TRUE, near_window_edge = FALSE")
    lines.append(f"High-confidence count in base dataset: {int(np.sum(high_conf_mask_best))} / {len(high_conf_mask_best)}")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build email-friendly plots and summary tables for DAS cable inversion comparison.")

    parser.add_argument("--input_csv", type=Path, default=Path(r"D:\Singapore Data\Cable\inversion_observations.csv"))
    parser.add_argument("--truth_csv", type=Path, default=Path(r"D:\Singapore Data\array-shape.csv"))
    parser.add_argument("--tuning_big_csv", type=Path, default=Path(r"D:\Singapore Data\Cable\cable_inversion_tuning_big_control_points\tuning_results_ranked.csv"))
    parser.add_argument("--tuning_control_csv", type=Path, default=Path(r"D:\Singapore Data\Cable\cable_inversion_tuning_only_control_points\tuning_results_ranked.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path(r"D:\Singapore Data\Cable\cable_inversion_email_report"))

    parser.add_argument("--selected_spacings", type=int, nargs="+", default=[5, 20, 40])
    parser.add_argument("--min_weight", type=float, default=0.8)
    parser.add_argument("--channel_offset", type=int, default=0)
    parser.add_argument("--max_nfev", type=int, default=250)

    # Fixed model used for the selected-spacing reruns.
    parser.add_argument("--sound_speed", type=float, default=1500.0)
    parser.add_argument("--channel_spacing", type=float, default=1.02)
    parser.add_argument("--abs_scale", type=float, default=0.003)
    parser.add_argument("--rel_scale", type=float, default=0.0015)
    parser.add_argument("--prior_sigma_xy", type=float, default=60.0)
    parser.add_argument("--prior_sigma_z", type=float, default=0.025)
    parser.add_argument("--curvature_sigma_xy", type=float, default=8.0)
    parser.add_argument("--curvature_sigma_z", type=float, default=0.025)
    parser.add_argument("--spacing_sigma", type=float, default=0.08)
    parser.add_argument("--anchor_bias_sigma", type=float, default=0.02)
    parser.add_argument("--huber_delta_abs", type=float, default=2.0)
    parser.add_argument("--huber_delta_rel", type=float, default=2.0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    figs_dir = args.output_dir / "figures"
    tables_dir = args.output_dir / "tables"
    ensure_dir(figs_dir)
    ensure_dir(tables_dir)

    raw, obs, prior_full, tx_tbl, (lat0, lon0, h0) = load_observations_and_prior(
        input_csv=args.input_csv,
        channel_offset=args.channel_offset,
        min_weight=args.min_weight,
    )
    truth = load_truth_geometry(args.truth_csv, lat0=lat0, lon0=lon0)

    tuning_big = load_tuning_results(args.tuning_big_csv, "big_control_points")
    tuning_control = load_tuning_results(args.tuning_control_csv, "only_control_points")
    control_summary = build_control_spacing_summary(tuning_big, tuning_control)
    control_summary.to_csv(tables_dir / "control_spacing_summary.csv", index=False)
    plot_elbow(control_summary, figs_dir / "elbow_control_spacing_vs_rel_rmse.png")

    available_spacings = set(int(x) for x in control_summary["control_spacing"].tolist())
    selected_spacings = [int(x) for x in args.selected_spacings if int(x) in available_spacings]
    if not selected_spacings:
        selected_spacings = choose_default_plot_spacings(control_summary)

    selected_results = run_selected_inversions(
        obs=obs,
        prior_full=prior_full,
        selected_spacings=selected_spacings,
        sound_speed=args.sound_speed,
        channel_spacing=args.channel_spacing,
        abs_scale=args.abs_scale,
        rel_scale=args.rel_scale,
        prior_sigma_xy=args.prior_sigma_xy,
        prior_sigma_z=args.prior_sigma_z,
        curvature_sigma_xy=args.curvature_sigma_xy,
        curvature_sigma_z=args.curvature_sigma_z,
        spacing_sigma=args.spacing_sigma,
        anchor_bias_sigma=args.anchor_bias_sigma,
        huber_delta_abs=args.huber_delta_abs,
        huber_delta_rel=args.huber_delta_rel,
        max_nfev=args.max_nfev,
    )

    # Save plots and tables for each selected spacing.
    best_spacing_for_timing = selected_spacings[0]
    best_obs_fit = None
    best_hc_mask = None

    for spacing in selected_spacings:
        solution = selected_results[spacing]["solution"]
        diagnostics = selected_results[spacing]["diagnostics"]
        layout_df = make_layout_dataframe(prior_full, solution)
        fit_df = build_fit_diagnostics_table(obs, diagnostics)
        hc_mask = high_confidence_mask(fit_df, args.min_weight)
        seg_df = build_segment_quality_table(layout_df, truth, fit_df, hc_mask)

        layout_df.to_csv(tables_dir / f"layout_spacing_{spacing}.csv", index=False)
        fit_df.to_csv(tables_dir / f"fit_spacing_{spacing}.csv", index=False)
        seg_df.to_csv(tables_dir / f"segment_qc_spacing_{spacing}.csv", index=False)

        plot_geometry_comparison(
            layout_df=layout_df,
            truth_df=truth,
            tx_tbl=tx_tbl,
            out_png=figs_dir / f"geometry_plan_spacing_{spacing}.png",
            title_prefix=f"Control spacing = {spacing} channels",
        )
        plot_z_comparison(
            layout_df=layout_df,
            truth_df=truth,
            out_png=figs_dir / f"geometry_vertical_spacing_{spacing}.png",
            title_prefix=f"Control spacing = {spacing} channels",
        )
        plot_segment_confidence(
            seg_df=seg_df,
            out_png=figs_dir / f"segment_confidence_spacing_{spacing}.png",
            title=f"Segment confidence by channel (spacing = {spacing})",
        )

        if spacing == best_spacing_for_timing:
            best_obs_fit = fit_df
            best_hc_mask = hc_mask

    if best_obs_fit is not None and best_hc_mask is not None:
        plot_observed_vs_predicted_highconf(
            obs_fit=best_obs_fit,
            high_conf_mask=best_hc_mask,
            out_png_rel=figs_dir / f"observed_vs_predicted_relative_highconf_spacing_{best_spacing_for_timing}.png",
            out_png_abs=figs_dir / f"observed_vs_predicted_absolute_highconf_spacing_{best_spacing_for_timing}.png",
        )
        plot_observed_vs_predicted_highconf_prior(
            obs_fit=best_obs_fit,
            high_conf_mask=best_hc_mask,
            out_png_rel=figs_dir / f"observed_vs_predicted_relative_prior_highconf_spacing_{best_spacing_for_timing}.png",
            out_png_abs=figs_dir / f"observed_vs_predicted_absolute_prior_highconf_spacing_{best_spacing_for_timing}.png",
        )

        plot_prior_vs_estimate_side_by_side(
            obs_fit=best_obs_fit,
            high_conf_mask=best_hc_mask,
            out_png=figs_dir / f"prior_vs_estimate_side_by_side_spacing_{best_spacing_for_timing}.png",
        )

    write_summary_note(
        out_txt=args.output_dir / "README_email_report.txt",
        lat0=lat0,
        lon0=lon0,
        h0=h0,
        control_summary=control_summary,
        selected_spacings=selected_spacings,
        selected_results=selected_results,
        obs_fit_best=obs,
        high_conf_mask_best=high_confidence_mask(obs, args.min_weight),
    )

    meta = {
        "input_csv": str(args.input_csv),
        "truth_csv": str(args.truth_csv),
        "tuning_big_csv": str(args.tuning_big_csv),
        "tuning_control_csv": str(args.tuning_control_csv),
        "output_dir": str(args.output_dir),
        "selected_spacings": selected_spacings,
        "enu_origin": {"lat": lat0, "lon": lon0, "h": h0},
        "fixed_params_for_selected_reruns": {
            "sound_speed": args.sound_speed,
            "channel_spacing": args.channel_spacing,
            "abs_scale": args.abs_scale,
            "rel_scale": args.rel_scale,
            "prior_sigma_xy": args.prior_sigma_xy,
            "prior_sigma_z": args.prior_sigma_z,
            "curvature_sigma_xy": args.curvature_sigma_xy,
            "curvature_sigma_z": args.curvature_sigma_z,
            "spacing_sigma": args.spacing_sigma,
            "anchor_bias_sigma": args.anchor_bias_sigma,
            "huber_delta_abs": args.huber_delta_abs,
            "huber_delta_rel": args.huber_delta_rel,
            "min_weight": args.min_weight,
        },
    }
    with open(args.output_dir / "report_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Done.")
    print(f"Figures written to: {figs_dir}")
    print(f"Tables written to:  {tables_dir}")
    print(f"Selected spacings used: {selected_spacings}")


if __name__ == "__main__":
    main()
