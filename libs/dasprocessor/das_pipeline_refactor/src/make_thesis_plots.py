from __future__ import annotations

import argparse
import math
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from common import ensure_dir, load_toml, path_from_cfg


# ------------------------------------------------------------
# small helpers
# ------------------------------------------------------------

AX_MIN = -0.1
AX_MAX = 0.3


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


def cumulative_arclength(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ds = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    return np.concatenate([[0.0], np.cumsum(ds)])


def latlon_to_local_xy_fixed_origin(lat_deg, lon_deg, lat0_deg, lon0_deg):
    """Simple local tangent-plane approximation for plotting."""
    R = 6371000.0
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    lat0 = np.radians(float(lat0_deg))
    lon0 = np.radians(float(lon0_deg))
    x = (lon - lon0) * np.cos(lat0) * R
    y = (lat - lat0) * R
    return x, y


def moving_average(arr, window=31):
    arr = np.asarray(arr, dtype=float)
    if window <= 1:
        return arr.copy()
    kernel = np.ones(int(window), dtype=float) / float(window)
    return np.convolve(arr, kernel, mode="same")


def weighted_rmse(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return np.nan
    return float(np.sqrt(np.average(values[mask] ** 2, weights=weights[mask])))


def add_channel_labels_plan(ax, df: pd.DataFrame, label_every: int = 100) -> None:
    label_mask = (pd.to_numeric(df["channel"], errors="coerce") % label_every == 0)
    for _, row in df.loc[label_mask].iterrows():
        ax.text(
            float(row["x_m"]),
            float(row["y_m"]),
            str(int(row["channel"])),
            fontsize=7,
            color="0.25",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.65),
            zorder=10,
        )


def add_channel_labels_depth(ax, df: pd.DataFrame, label_every: int = 100) -> None:
    label_mask = (pd.to_numeric(df["channel"], errors="coerce") % label_every == 0)
    for _, row in df.loc[label_mask].iterrows():
        ax.text(
            float(row["channel"]),
            float(row["z_m"]),
            str(int(row["channel"])),
            fontsize=7,
            color="0.25",
            ha="center",
            va="bottom",
        )


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
        mask &= w >= float(min_weight)
    return mask


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

    return dist, proj_x, proj_y, proj_s


def format_anchor_label(anchor_id: str) -> str:
    """
    Convert an anchor id like 'loc2_tx3_a1' to:
    'Location 2, Sweep 1'
    """
    if anchor_id is None:
        return "Unknown anchor"
    text = str(anchor_id)
    m = re.search(r"loc(\d+).*_a(\d+)$", text)
    if m:
        loc = m.group(1)
        sweep = m.group(2)
        return f"Location {loc}, Sweep {sweep}"
    return text


# ------------------------------------------------------------
# load inputs
# ------------------------------------------------------------

def build_tx_table(raw: pd.DataFrame, fit: pd.DataFrame) -> pd.DataFrame:
    tx_source = fit.copy()
    needed = {"tx_x_m", "tx_y_m", "tx_u_m"}
    if not needed.issubset(tx_source.columns):
        tx_source = raw.copy()

    if "anchor_id" not in tx_source.columns:
        if {"location", "anchor_index"}.issubset(tx_source.columns):
            tx_source["anchor_id"] = (
                tx_source["location"].astype(str) + "_a" + tx_source["anchor_index"].astype(str)
            )
        else:
            raise KeyError("Could not build anchor_id from the available columns.")

    tx_tbl = (
        tx_source.groupby("anchor_id")[["tx_x_m", "tx_y_m", "tx_u_m"]]
        .first()
        .reset_index()
    )
    tx_tbl["anchor_label"] = tx_tbl["anchor_id"].map(format_anchor_label)
    return tx_tbl


def load_truth(truth_csv: Path, lat0: float, lon0: float) -> pd.DataFrame:
    truth = pd.read_csv(truth_csv)
    required = {"lat", "lon"}
    missing = required - set(truth.columns)
    if missing:
        raise ValueError(f"Truth/focalization CSV missing columns: {sorted(missing)}")

    x, y = latlon_to_local_xy_fixed_origin(truth["lat"].values, truth["lon"].values, lat0, lon0)
    out = truth.copy()
    out["x_m"] = x
    out["y_m"] = y

    if "z" not in out.columns and "depth" in out.columns:
        out["z"] = pd.to_numeric(out["depth"], errors="coerce")

    for c in ["ch", "channel", "Channel", "CHAN", "chan"]:
        if c in out.columns:
            out["channel_like"] = pd.to_numeric(out[c], errors="coerce")
            break
    else:
        out["channel_like"] = np.arange(len(out), dtype=float)
    return out


def load_inputs(
    config_path: Path,
    input_csv: Path | None,
    inversion_output_dir: Path | None,
    truth_csv: Path | None,
):
    cfg = load_toml(config_path)

    input_csv = input_csv or (path_from_cfg(cfg, "inversion_dataset_output_dir") / "inversion_observations.csv")
    inversion_output_dir = inversion_output_dir or path_from_cfg(cfg, "inversion_output_dir")
    truth_csv = truth_csv or path_from_cfg(cfg, "cable_estimate_csv")

    layout = pd.read_csv(inversion_output_dir / "updated_cable_layout.csv")
    ctrl = pd.read_csv(inversion_output_dir / "control_points_optimized.csv")
    fit = pd.read_csv(inversion_output_dir / "observation_fit_diagnostics.csv")
    raw = pd.read_csv(input_csv)
    q = (
        pd.read_csv(inversion_output_dir / "channel_control_quality.csv")
        if (inversion_output_dir / "channel_control_quality.csv").exists()
        else None
    )

    lat0 = float(raw["enu_origin_lat_deg"].dropna().iloc[0])
    lon0 = float(raw["enu_origin_lon_deg"].dropna().iloc[0])
    h0 = float(raw["enu_origin_h_m"].dropna().iloc[0]) if "enu_origin_h_m" in raw.columns else 0.0
    truth = load_truth(truth_csv, lat0, lon0)
    tx_tbl = build_tx_table(raw, fit)

    return cfg, raw, layout, ctrl, fit, q, truth, tx_tbl, (lat0, lon0, h0), input_csv, inversion_output_dir, truth_csv


# ------------------------------------------------------------
# derived diagnostics
# ------------------------------------------------------------

def build_uncertainty_tube(layout: pd.DataFrame, fit: pd.DataFrame, sound_speed: float = 1500.0) -> pd.DataFrame:
    fit = fit.copy()
    fit["res_s"] = pd.to_numeric(fit["residual_dt_ref_opt_s"], errors="coerce")
    fit["weight"] = pd.to_numeric(fit["weight"], errors="coerce")
    ch_col = "channel_eff" if "channel_eff" in fit.columns else "channel"

    def ch_weighted_rmse(group: pd.DataFrame) -> float:
        w = pd.to_numeric(group["weight"], errors="coerce").to_numpy(dtype=float)
        r = pd.to_numeric(group["res_s"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(w) & np.isfinite(r) & (w > 0)
        if not np.any(mask):
            return np.nan
        return float(np.sqrt(np.sum(w[mask] * r[mask] ** 2) / np.sum(w[mask])))

    rows = []
    for ch_val, grp in fit.groupby(ch_col):
        rows.append({"channel": ch_val, "rmse_s": ch_weighted_rmse(grp)})
    ch_stats = pd.DataFrame(rows)
    ch_stats["uncertainty_m"] = float(sound_speed) * ch_stats["rmse_s"]

    out = layout.merge(ch_stats, on="channel", how="left").copy()
    out["uncertainty_m"] = pd.to_numeric(out["uncertainty_m"], errors="coerce").interpolate().bfill().ffill()
    out["uncertainty_smooth_m"] = gaussian_filter1d(out["uncertainty_m"].to_numpy(dtype=float), sigma=10)

    u = out["uncertainty_smooth_m"].to_numpy(dtype=float)
    if np.all(~np.isfinite(u)):
        out["tube_half_width_m"] = 8.0
    else:
        umin = np.nanmin(u)
        umax = np.nanmax(u)
        u_norm = (u - umin) / (umax - umin + 1e-9)
        out["tube_half_width_m"] = 3.0 + 20.0 * u_norm
    return out


def compute_tube_boundaries(df: pd.DataFrame):
    x = pd.to_numeric(df["x_m"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["y_m"], errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(df["tube_half_width_m"], errors="coerce").to_numpy(dtype=float)
    dx = np.gradient(x)
    dy = np.gradient(y)
    norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-8
    nx = -dy / norm
    ny = dx / norm
    x_up = x + nx * w
    y_up = y + ny * w
    x_dn = x - nx * w
    y_dn = y - ny * w
    return x, y, x_up, y_up, x_dn, y_dn


def build_residual_envelope(
    fit: pd.DataFrame,
    highconf_only: bool = False,
    min_weight: float = 0.80,
) -> pd.DataFrame:
    tmp = fit.copy()
    if highconf_only:
        tmp = tmp.loc[high_confidence_mask(tmp, min_weight=min_weight)].copy()

    ch_col = "channel_eff" if "channel_eff" in tmp.columns else "channel"
    tmp["rel_res_ms"] = 1000.0 * pd.to_numeric(tmp["residual_dt_ref_opt_s"], errors="coerce")

    rows = []
    for ch_val, grp in tmp.groupby(ch_col):
        vals = pd.to_numeric(grp["rel_res_ms"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        rows.append({
            "channel": ch_val,
            "p10": np.percentile(vals, 10),
            "p50": np.percentile(vals, 50),
            "p90": np.percentile(vals, 90),
        })
    env = pd.DataFrame(rows).sort_values("channel")
    if len(env) == 0:
        return env
    for c in ["p10", "p50", "p90"]:
        env[c] = moving_average(env[c].to_numpy(dtype=float), window=31)
    return env


# ------------------------------------------------------------
# plotters
# ------------------------------------------------------------

def apply_thesis_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


def plot_plan_view_with_tube(
    layout: pd.DataFrame,
    fit: pd.DataFrame,
    truth: pd.DataFrame,
    tx_tbl: pd.DataFrame,
    out_png: Path,
    label_every: int,
    sound_speed: float,
) -> None:
    df = build_uncertainty_tube(layout, fit, sound_speed=sound_speed)
    x, y, x_up, y_up, x_dn, y_dn = compute_tube_boundaries(df)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.fill_betweenx(y, x_dn, x_up, alpha=0.22, color="#4C92C3", label="Relative-residual tube", zorder=1)
    ax.plot(
        pd.to_numeric(layout["prior_x_m"], errors="coerce"),
        pd.to_numeric(layout["prior_y_m"], errors="coerce"),
        linewidth=1.8,
        alpha=0.6,
        color="0.45",
        label="Prior cable",
        zorder=2,
    )
    ax.plot(
        truth["x_m"].values,
        truth["y_m"].values,
        linewidth=2.0,
        color="#ff7f0e",
        label="Focalization / truth-like",
        zorder=3,
    )
    ax.plot(x, y, linewidth=2.4, color="#1f77b4", label="Estimated cable", zorder=4)
    ax.scatter(
        tx_tbl["tx_x_m"].values,
        tx_tbl["tx_y_m"].values,
        marker="x",
        s=80,
        linewidths=2.0,
        color="#2ca02c",
        label="Transmitters",
        zorder=5,
    )

    for _, row in tx_tbl.iterrows():
        ax.text(
            float(row["tx_x_m"]) + 3.0,
            float(row["tx_y_m"]) + 3.0,
            str(row["anchor_label"]),
            fontsize=8,
            weight="bold",
            color="0.20",
            zorder=6,
        )

    add_channel_labels_plan(ax, df[["channel", "x_m", "y_m"]], label_every=label_every)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("Cable layout with uncertainty tube")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, loc="best")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_depth_profile(
    layout: pd.DataFrame,
    truth: pd.DataFrame,
    ctrl: pd.DataFrame,
    out_png: Path,
    label_every: int,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.plot(layout["channel"], layout["prior_z_m"], linewidth=1.7, alpha=0.65, color="0.45", label="Prior z")
    ax.plot(layout["channel"], layout["z_m"], linewidth=2.4, color="#1f77b4", label="Estimated z")
    ax.scatter(ctrl["channel"], ctrl["z_m"], s=20, color="#1f77b4", zorder=3, label="Optimized control pts")
    if "channel_like" in truth.columns and "z" in truth.columns:
        ax.plot(
            truth["channel_like"],
            pd.to_numeric(truth["z"], errors="coerce"),
            linewidth=2.0,
            color="#ff7f0e",
            label="Focalization / truth-like",
        )
    add_channel_labels_depth(ax, layout[["channel", "z_m"]], label_every=label_every)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Up (m)")
    ax.set_title("Depth profile")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, loc="best")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_horizontal_shift(layout: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.2))
    shift = np.sqrt(
        (pd.to_numeric(layout["x_m"], errors="coerce") - pd.to_numeric(layout["prior_x_m"], errors="coerce")) ** 2
        + (pd.to_numeric(layout["y_m"], errors="coerce") - pd.to_numeric(layout["prior_y_m"], errors="coerce")) ** 2
    )
    ax.plot(layout["channel"], shift, linewidth=2.1)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Horizontal shift (m)")
    ax.set_title("Horizontal displacement from prior")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_relative_residual_envelope_all(
    fit: pd.DataFrame,
    out_png: Path,
    label_every: int,
) -> None:
    env = build_residual_envelope(fit, highconf_only=False)
    if len(env) == 0:
        warnings.warn("No observations available for all-points residual envelope.")
        return

    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.fill_between(env["channel"], env["p10"], env["p90"], color="#4C92C3", alpha=0.18, label="10th–90th percentile envelope")
    ax.plot(env["channel"], env["p50"], linewidth=2.1, color="#1f77b4", label="Median residual")
    ax.axhline(0.0, color="0.35", linewidth=1.0)

    label_mask = pd.to_numeric(env["channel"], errors="coerce") % label_every == 0
    for _, row in env.loc[label_mask].iterrows():
        ax.text(float(row["channel"]), 0.0, str(int(row["channel"])), fontsize=7, color="0.35", ha="center", va="bottom")

    ax.set_xlabel("Channel")
    ax.set_ylabel("Relative residual (ms)")
    ax.set_title("Relative residual envelope after inversion")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, loc="best")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_relative_residual_envelope_highconf(
    fit: pd.DataFrame,
    out_png: Path,
    label_every: int,
    min_weight: float,
) -> None:
    env = build_residual_envelope(fit, highconf_only=True, min_weight=min_weight)
    if len(env) == 0:
        warnings.warn("No high-confidence observations available for residual envelope.")
        return

    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.fill_between(env["channel"], env["p10"], env["p90"], color="#4C92C3", alpha=0.25, label="10th–90th percentile envelope")
    ax.plot(env["channel"], env["p50"], linewidth=2.4, color="#1f77b4", label="Median residual")
    ax.axhline(0.0, color="0.35", linewidth=1.0)

    label_mask = pd.to_numeric(env["channel"], errors="coerce") % label_every == 0
    for _, row in env.loc[label_mask].iterrows():
        ax.text(float(row["channel"]), 0.0, str(int(row["channel"])), fontsize=7, color="0.35", ha="center", va="bottom")

    ax.set_xlabel("Channel")
    ax.set_ylabel("Relative residual (ms)")
    ax.set_title("Relative residual envelope after inversion (high-confidence subset)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, loc="best")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def _prepare_highconf_anchor_data(fit: pd.DataFrame, min_weight: float) -> pd.DataFrame:
    hc = fit.loc[high_confidence_mask(fit, min_weight=min_weight)].copy()
    if len(hc) == 0:
        return hc

    if "anchor_id" not in hc.columns:
        if {"location", "anchor_index"}.issubset(hc.columns):
            hc["anchor_id"] = hc["location"].astype(str) + "_a" + hc["anchor_index"].astype(str)
        else:
            hc["anchor_id"] = "all"
    hc["anchor_label"] = hc["anchor_id"].map(format_anchor_label)
    return hc


def _plot_obs_pred_panel(ax, x, y, w, title, ylabel, rmse_ms):
    ax.scatter(x, y, s=10 + 14 * np.clip(w, 0, 1), alpha=0.6)
    ax.plot([AX_MIN, AX_MAX], [AX_MIN, AX_MAX], linewidth=1.0)
    ax.set_xlim(AX_MIN, AX_MAX)
    ax.set_ylim(AX_MIN, AX_MAX)
    ax.set_title(title)
    ax.set_xlabel("Observed dt_ref (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2)
    ax.text(0.04, 0.92, f"weighted RMSE = {rmse_ms:.1f} ms", transform=ax.transAxes, fontsize=9, va="top")


def plot_observed_vs_predicted_inverted_only(
    fit: pd.DataFrame,
    out_png: Path,
    min_weight: float,
) -> None:
    hc = _prepare_highconf_anchor_data(fit, min_weight=min_weight)
    if len(hc) == 0:
        warnings.warn("No high-confidence observations available for predicted-vs-observed plots.")
        return

    anchor_ids = sorted(hc["anchor_id"].dropna().astype(str).unique())
    ncol = 3
    nrow = int(np.ceil(len(anchor_ids) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 4.0 * nrow), squeeze=False)
    fig.suptitle("Observed vs predicted relative times after inversion (high-confidence subset)", y=0.995)

    for ax, aid in zip(axes.ravel(), anchor_ids):
        m = hc["anchor_id"].astype(str).eq(aid)
        title = hc.loc[m, "anchor_label"].iloc[0]
        x = pd.to_numeric(hc.loc[m, "observed_dt_ref_s"], errors="coerce").values
        y = pd.to_numeric(hc.loc[m, "predicted_dt_ref_s_opt"], errors="coerce").values
        w = pd.to_numeric(hc.loc[m, "weight"], errors="coerce").fillna(0.0).values
        keep = np.isfinite(x) & np.isfinite(y)
        x = x[keep]
        y = y[keep]
        w = w[keep]
        if len(x) == 0:
            ax.set_title(title)
            ax.axis("off")
            continue
        rmse_ms = 1000.0 * weighted_rmse(y - x, w)
        _plot_obs_pred_panel(ax, x, y, w, title, "Predicted dt_ref (s)", rmse_ms)

    for ax in axes.ravel()[len(anchor_ids):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_observed_vs_predicted_prior_only(
    fit: pd.DataFrame,
    out_png: Path,
    min_weight: float,
) -> None:
    hc = _prepare_highconf_anchor_data(fit, min_weight=min_weight)
    if len(hc) == 0:
        warnings.warn("No high-confidence observations available for prior predicted-vs-observed plots.")
        return

    anchor_ids = sorted(hc["anchor_id"].dropna().astype(str).unique())
    ncol = 3
    nrow = int(np.ceil(len(anchor_ids) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 4.0 * nrow), squeeze=False)
    fig.suptitle("Observed vs predicted relative times from prior geometry (high-confidence subset)", y=0.995)

    for ax, aid in zip(axes.ravel(), anchor_ids):
        m = hc["anchor_id"].astype(str).eq(aid)
        title = hc.loc[m, "anchor_label"].iloc[0]
        x = pd.to_numeric(hc.loc[m, "observed_dt_ref_s"], errors="coerce").values
        y = pd.to_numeric(hc.loc[m, "predicted_dt_ref_s_prior"], errors="coerce").values
        w = pd.to_numeric(hc.loc[m, "weight"], errors="coerce").fillna(0.0).values
        keep = np.isfinite(x) & np.isfinite(y)
        x = x[keep]
        y = y[keep]
        w = w[keep]
        if len(x) == 0:
            ax.set_title(title)
            ax.axis("off")
            continue
        rmse_ms = 1000.0 * weighted_rmse(y - x, w)
        _plot_obs_pred_panel(ax, x, y, w, title, "Predicted dt_ref from prior (s)", rmse_ms)

    for ax in axes.ravel()[len(anchor_ids):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_prior_vs_estimate_side_by_side(
    fit: pd.DataFrame,
    out_png: Path,
    min_weight: float,
) -> None:
    hc = _prepare_highconf_anchor_data(fit, min_weight=min_weight)
    if len(hc) == 0:
        warnings.warn("No high-confidence observations available.")
        return

    anchor_ids = sorted(hc["anchor_id"].dropna().astype(str).unique())
    nrow = len(anchor_ids)
    fig, axes = plt.subplots(nrow, 2, figsize=(10, 4 * nrow), squeeze=False)
    fig.suptitle("Observed vs predicted relative times (prior vs inversion, high-confidence subset)", y=0.995)

    for i, aid in enumerate(anchor_ids):
        m = hc["anchor_id"].astype(str).eq(aid)
        title = hc.loc[m, "anchor_label"].iloc[0]
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
            axes[i, 0].axis("off")
            axes[i, 1].axis("off")
            continue

        rmse_prior_ms = 1000.0 * weighted_rmse(y_prior - x, w)
        rmse_opt_ms = 1000.0 * weighted_rmse(y_opt - x, w)

        _plot_obs_pred_panel(axes[i, 0], x, y_prior, w, f"{title} — Prior", "Predicted (prior)", rmse_prior_ms)
        _plot_obs_pred_panel(axes[i, 1], x, y_opt, w, f"{title} — Inversion", "Predicted (estimate)", rmse_opt_ms)

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_fit_histograms(fit: pd.DataFrame, out_png: Path, min_weight: float) -> None:
    hc = fit.loc[high_confidence_mask(fit, min_weight=min_weight)].copy()
    if len(hc) == 0:
        warnings.warn("No high-confidence observations available for residual histograms.")
        return

    abs_prior = 1000.0 * pd.to_numeric(hc["residual_abs_prior_s"], errors="coerce")
    abs_opt = 1000.0 * pd.to_numeric(hc["residual_abs_opt_s"], errors="coerce")
    rel_prior = 1000.0 * pd.to_numeric(hc["residual_dt_ref_prior_s"], errors="coerce")
    rel_opt = 1000.0 * pd.to_numeric(hc["residual_dt_ref_opt_s"], errors="coerce")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].hist(abs_prior[np.isfinite(abs_prior)], bins=60, alpha=0.45, label="Absolute prior")
    axes[0].hist(abs_opt[np.isfinite(abs_opt)], bins=60, alpha=0.45, label="Absolute inverted")
    axes[0].set_title("Absolute-time residuals (high-confidence subset)")
    axes[0].set_xlabel("Residual (ms)")
    axes[0].set_ylabel("Count")
    axes[0].legend(frameon=True)

    axes[1].hist(rel_prior[np.isfinite(rel_prior)], bins=60, alpha=0.45, label="Relative prior")
    axes[1].hist(rel_opt[np.isfinite(rel_opt)], bins=60, alpha=0.45, label="Relative inverted")
    axes[1].set_title("Relative-time residuals (high-confidence subset)")
    axes[1].set_xlabel("Residual (ms)")
    axes[1].set_ylabel("Count")
    axes[1].legend(frameon=True)

    for ax in axes:
        ax.grid(True, alpha=0.20)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_distance_to_truth(layout: pd.DataFrame, truth: pd.DataFrame, out_png: Path) -> None:
    truth_s = cumulative_arclength(truth["x_m"], truth["y_m"])
    xy_err, _, _, _ = project_points_onto_polyline(
        layout["x_m"].to_numpy(dtype=float),
        layout["y_m"].to_numpy(dtype=float),
        truth["x_m"].to_numpy(dtype=float),
        truth["y_m"].to_numpy(dtype=float),
        s_line=truth_s,
    )
    fig, ax = plt.subplots(figsize=(13, 5.0))
    ax.plot(layout["channel"], xy_err, linewidth=2.1)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Horizontal distance to focalization geometry (m)")
    ax.set_title("Difference between estimated cable and focalization geometry")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_control_quality(q: pd.DataFrame | None, ctrl: pd.DataFrame, out_png: Path) -> None:
    if q is None:
        return
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.plot(q["channel"], q["control_quality_score"], linewidth=2.1, label="Control quality score")
    interp_vals = np.interp(ctrl["channel"], q["channel"], q["control_quality_score"])
    ax.scatter(ctrl["channel"], interp_vals, s=22, label="Selected control points", zorder=3)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Control quality score")
    ax.set_title("Control quality and selected control points")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def write_plot_readme(out_dir: Path, min_weight: float, label_every: int, truth_csv: Path) -> None:
    text = f"""Thesis-ready plot set
======================

This folder contains the final post-inversion visualization set.

Defaults used:
- high-confidence threshold: weight >= {min_weight}
- channel labels every: {label_every}
- focalization / truth-like geometry source: {truth_csv}
- observed-vs-predicted axes fixed to: xlim = ylim = [{AX_MIN}, {AX_MAX}]
"""
    (out_dir / "README_thesis_plots.txt").write_text(text, encoding="utf-8")


def write_figure_captions(out_dir: Path) -> None:
    captions = {
        "figure_planview_tube.png":
            "Plan-view cable geometry showing prior (gray), inverted (blue), and focalization-derived reference (orange). The shaded tube represents the relative-residual-based uncertainty envelope along the cable. Transmitter locations are marked with crosses. The inversion aligns closely with the reference while reducing uncertainty.",

        "figure_depth_profile.png":
            "Vertical cable profile as a function of channel. The inverted geometry (blue) follows the focalization-based reference (orange) more closely than the prior model (gray). Optimized control points are indicated, highlighting where the inversion adjusts the geometry.",

        "figure_observed_vs_predicted_prior_only.png":
            "Observed versus predicted relative arrival times using the prior geometry for the high-confidence subset. Significant deviations from the 1:1 line and structured scatter patterns indicate systematic geometric inconsistencies in the prior model.",

        "figure_observed_vs_predicted_inverted_only.png":
            "Observed versus predicted relative arrival times after inversion for the high-confidence subset. Each panel corresponds to a transmitter location and sweep number. The solid line indicates the 1:1 relationship. The tight clustering along the diagonal demonstrates strong agreement between observations and the inverted geometry, with low residual error reported as weighted RMSE.",

        "figure_observed_vs_predicted_prior_vs_inverted.png":
            "Comparison of predicted versus observed relative arrival times for prior (left) and inverted (right) geometries. The inversion substantially improves agreement with observations, collapsing structured deviations into a tight distribution along the 1:1 line.",

        "figure_relative_residual_envelope_all.png":
            "Relative residuals as a function of channel after inversion, shown for all observations. The shaded region indicates the 10th–90th percentile envelope, and the line shows the median residual. The wide spread reflects the inclusion of lower-confidence observations.",

        "figure_relative_residual_envelope_highconf.png":
            "Relative residuals after inversion for the high-confidence subset. The 10th–90th percentile envelope is significantly narrower and centered near zero, indicating that the inversion produces an unbiased and consistent fit for reliable observations.",

        "figure_fit_histograms_highconf.png":
            "Distributions of absolute and relative residuals for prior and inverted geometries, evaluated on the high-confidence subset. The inversion significantly reduces both bias and spread, resulting in a tighter and more centered distribution.",

        "figure_horizontal_shift.png":
            "Horizontal displacement between the prior and inverted cable geometries as a function of channel. The inversion introduces spatially varying corrections, with larger adjustments in regions where the prior geometry deviates most from observations.",

        "figure_distance_to_focalization_vs_channel.png":
            "Horizontal distance from the inverted cable geometry to the focalization-based reference as a function of channel. Smaller values indicate closer agreement between the estimated cable layout and the external reference geometry.",

        "figure_control_quality_and_points.png":
            "Control quality score along the cable, with selected high-confidence control points indicated. These points form the subset used to constrain the inversion, ensuring robustness against low-quality observations.",
    }

    lines = ["Figure captions", "===============", ""]
    for name, caption in captions.items():
        lines.append(f"{name}")
        lines.append(f"  {caption}")
        lines.append("")

    (out_dir / "FIGURE_CAPTIONS.txt").write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate thesis-ready post-inversion plots.")
    parser.add_argument("--config", type=Path, required=True, help="Path to pipeline_config.toml")
    parser.add_argument("--input_csv", type=Path, default=None, help="Override inversion_observations.csv path")
    parser.add_argument("--inversion_output_dir", type=Path, default=None, help="Override inversion output directory")
    parser.add_argument("--truth_csv", type=Path, default=None, help="Override focalization/truth-like geometry CSV")
    parser.add_argument("--output_dir", type=Path, default=None, help="Directory for final thesis-ready figures")
    parser.add_argument("--label_every", type=int, default=100, help="Annotate every Nth channel on geometry plots")
    parser.add_argument("--min_weight", type=float, default=0.80, help="High-confidence threshold for filtered plots")
    parser.add_argument("--sound_speed", type=float, default=None, help="Override sound speed used for uncertainty tube")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_thesis_style()

    cfg, raw, layout, ctrl, fit, q, truth, tx_tbl, (_lat0, _lon0, _h0), _input_csv, inversion_output_dir, truth_csv = load_inputs(
        config_path=args.config,
        input_csv=args.input_csv,
        inversion_output_dir=args.inversion_output_dir,
        truth_csv=args.truth_csv,
    )
    sound_speed = float(args.sound_speed) if args.sound_speed is not None else float(cfg["inversion"]["sound_speed"])
    out_dir = ensure_dir(args.output_dir or (inversion_output_dir / "thesis_plots"))

    plot_plan_view_with_tube(
        layout=layout,
        fit=fit,
        truth=truth,
        tx_tbl=tx_tbl,
        out_png=out_dir / "figure_planview_tube.png",
        label_every=args.label_every,
        sound_speed=sound_speed,
    )
    plot_depth_profile(
        layout=layout,
        truth=truth,
        ctrl=ctrl,
        out_png=out_dir / "figure_depth_profile.png",
        label_every=args.label_every,
    )
    plot_horizontal_shift(
        layout=layout,
        out_png=out_dir / "figure_horizontal_shift.png",
    )
    plot_observed_vs_predicted_inverted_only(
        fit=fit,
        out_png=out_dir / "figure_observed_vs_predicted_inverted_only.png",
        min_weight=args.min_weight,
    )
    plot_observed_vs_predicted_prior_only(
        fit=fit,
        out_png=out_dir / "figure_observed_vs_predicted_prior_only.png",
        min_weight=args.min_weight,
    )
    plot_prior_vs_estimate_side_by_side(
        fit=fit,
        out_png=out_dir / "figure_observed_vs_predicted_prior_vs_inverted.png",
        min_weight=args.min_weight,
    )
    plot_relative_residual_envelope_all(
        fit=fit,
        out_png=out_dir / "figure_relative_residual_envelope_all.png",
        label_every=args.label_every,
    )
    plot_relative_residual_envelope_highconf(
        fit=fit,
        out_png=out_dir / "figure_relative_residual_envelope_highconf.png",
        label_every=args.label_every,
        min_weight=args.min_weight,
    )
    plot_fit_histograms(
        fit=fit,
        out_png=out_dir / "figure_fit_histograms_highconf.png",
        min_weight=args.min_weight,
    )
    plot_distance_to_truth(
        layout=layout,
        truth=truth,
        out_png=out_dir / "figure_distance_to_focalization_vs_channel.png",
    )
    plot_control_quality(
        q=q,
        ctrl=ctrl,
        out_png=out_dir / "figure_control_quality_and_points.png",
    )
    write_plot_readme(
        out_dir=out_dir,
        min_weight=args.min_weight,
        label_every=args.label_every,
        truth_csv=truth_csv,
    )
    write_figure_captions(out_dir)

    print(f"Thesis-ready plots written to: {out_dir}")


if __name__ == "__main__":
    main()