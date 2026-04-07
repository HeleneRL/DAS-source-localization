import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS / FILE PATHS
# ============================================================

UPDATED_LAYOUT_CSV = r"D:\Singapore Data\Cable\cable_inversion_output_40_no_offset_no_z_move_lin_z\updated_cable_layout.csv"
ANCHOR_BIASES_CSV = r"D:\Singapore Data\Cable\cable_inversion_output_40_no_offset_no_z_move_lin_z\anchor_biases.csv"
CONTROL_POINTS_CSV = r"D:\Singapore Data\Cable\cable_inversion_output_40_no_offset_no_z_move_lin_z\control_points_optimized.csv"
INVERSION_SUMMARY_CSV = r"D:\Singapore Data\Cable\cable_inversion_output_40_no_offset_no_z_move_lin_z\inversion_summary.csv"
FIT_DIAGNOSTICS_CSV = r"D:\Singapore Data\Cable\cable_inversion_output_40_no_offset_no_z_move_lin_z\observation_fit_diagnostics.csv"
OBSERVATIONS_CSV = r"D:\Singapore Data\Cable\inversion_observations.csv"
TRUTH_CSV = r"D:\Singapore Data\array-shape.csv"
CHANNEL_PROGRESSION_METRICS_CSV = r"D:\Singapore Data\Cable\channel_progression_check_40_no_offset_no_z_move_z_lin\channel_progression_metrics.csv"
CHANNEL_PROGRESSION_DIAGNOSTICS_CSV = r"D:\Singapore Data\Cable\channel_progression_check_40_no_offset_no_z_move_z_lin\channel_progression_diagnostics.csv"

OUTPUT_DIR = r"D:\Singapore Data\Cable\qc_40_no_offset_no_z_move_lin_z_v2"

HIGH_WEIGHT_THRESHOLD = 0.7
EXCLUDE_ENDPOINTS_N = 25

# If you want to tighten the high-confidence subset:
# HIGH_WEIGHT_THRESHOLD = 0.8


# ============================================================
# Utilities
# ============================================================

def ensure_bool(series):
    if series.dtype == bool:
        return series
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .map({"TRUE": True, "FALSE": False})
        .fillna(False)
    )


def latlon_to_local_xy(lat_deg, lon_deg, lat0_deg, lon0_deg):
    """
    Equirectangular local tangent-plane approximation.
    Good enough for small areas.
    """
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
    """
    Project each point (px, py) to nearest point on polyline (x_line, y_line).

    Returns:
        dist, proj_x, proj_y, proj_s, seg_idx, seg_t
    """
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
    seg_t = np.full(n, np.nan)

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
        seg_t[m] = t[m]

    return dist, proj_x, proj_y, proj_s, seg_idx, seg_t


def weighted_rmse(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return np.nan
    return np.sqrt(np.average(values[mask] ** 2, weights=weights[mask]))


def safe_quantile(x, q):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    return np.quantile(x, q)


def get_truth_channel_col(df):
    for c in ["ch", "channel", "Channel", "CHAN", "chan"]:
        if c in df.columns:
            return c
    raise ValueError("Could not find truth channel column in truth CSV.")


def score_segment(xy_error_m, med_abs_rel_ms):
    if (xy_error_m <= 4.0) and (med_abs_rel_ms <= 20.0):
        return "good"
    if (xy_error_m <= 8.0) and (med_abs_rel_ms <= 80.0):
        return "caution"
    return "poor"


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --------------------------------------------------------
    # Load files
    # --------------------------------------------------------
    layout = pd.read_csv(UPDATED_LAYOUT_CSV)
    raw_obs = pd.read_csv(OBSERVATIONS_CSV)
    fit = pd.read_csv(FIT_DIAGNOSTICS_CSV)
    truth = pd.read_csv(TRUTH_CSV)

    summary_df = pd.read_csv(INVERSION_SUMMARY_CSV) if os.path.exists(INVERSION_SUMMARY_CSV) else None
    anchor_bias_df = pd.read_csv(ANCHOR_BIASES_CSV) if os.path.exists(ANCHOR_BIASES_CSV) else None
    control_df = pd.read_csv(CONTROL_POINTS_CSV) if os.path.exists(CONTROL_POINTS_CSV) else None
    prog_diag_df = pd.read_csv(CHANNEL_PROGRESSION_DIAGNOSTICS_CSV) if os.path.exists(CHANNEL_PROGRESSION_DIAGNOSTICS_CSV) else None
    prog_metrics_df = pd.read_csv(CHANNEL_PROGRESSION_METRICS_CSV) if os.path.exists(CHANNEL_PROGRESSION_METRICS_CSV) else None

    # --------------------------------------------------------
    # Clean booleans / numerics in raw observations
    # --------------------------------------------------------
    for c in ["use_observation", "passed_snr_threshold", "near_window_edge", "recommended_channel"]:
        if c in raw_obs.columns:
            raw_obs[c] = ensure_bool(raw_obs[c])

    if "weight" not in raw_obs.columns:
        raw_obs["weight"] = 1.0
    raw_obs["weight"] = pd.to_numeric(raw_obs["weight"], errors="coerce")

    # --------------------------------------------------------
    # Merge fit diagnostics with raw observation metadata
    # fit is authoritative filtered subset
    # --------------------------------------------------------
    merge_keys = [
        "location",
        "anchor_index",
        "channel",
        "reference_channel",
        "observed_t_s",
        "observed_dt_ref_s",
    ]

    missing_obs_keys = [c for c in merge_keys if c not in raw_obs.columns]
    missing_fit_keys = [c for c in merge_keys if c not in fit.columns]
    if missing_obs_keys:
        raise ValueError(f"observations_csv missing merge keys: {missing_obs_keys}")
    if missing_fit_keys:
        raise ValueError(f"fit_diagnostics_csv missing merge keys: {missing_fit_keys}")

    numeric_merge_cols = ["anchor_index", "channel", "reference_channel", "observed_t_s", "observed_dt_ref_s"]
    for c in numeric_merge_cols:
        raw_obs[c] = pd.to_numeric(raw_obs[c], errors="coerce")
        fit[c] = pd.to_numeric(fit[c], errors="coerce")

    obs = pd.merge(
        fit,
        raw_obs,
        on=merge_keys,
        how="left",
        suffixes=("", "_raw")
    )

    if len(obs) != len(fit):
        warnings.warn(
            f"Merged observation table length ({len(obs)}) differs from fit table length ({len(fit)})."
        )

    # If anchor_id not present in fit, rebuild it
    if "anchor_id" not in obs.columns:
        obs["anchor_id"] = obs["location"].astype(str) + "_a" + obs["anchor_index"].astype(str)

    # Make sure these exist after merge
    for c in ["use_observation", "passed_snr_threshold", "near_window_edge", "weight"]:
        if c not in obs.columns:
            if c == "weight":
                obs[c] = 1.0
            else:
                obs[c] = False

    for c in ["use_observation", "passed_snr_threshold", "near_window_edge"]:
        obs[c] = ensure_bool(obs[c])

    obs["weight"] = pd.to_numeric(obs["weight"], errors="coerce")

    # --------------------------------------------------------
    # Truth XY from lat/lon ONLY
    # --------------------------------------------------------
    if ("lat_deg" not in layout.columns) or ("lon_deg" not in layout.columns):
        raise ValueError("updated_layout_csv must contain lat_deg and lon_deg")

    truth_ch_col = get_truth_channel_col(truth)
    for c in ["lat", "lon", "z"]:
        if c not in truth.columns:
            raise ValueError(f"truth_csv missing required column: {c}")

    lat0 = np.mean(np.r_[layout["lat_deg"].values, truth["lat"].values])
    lon0 = np.mean(np.r_[layout["lon_deg"].values, truth["lon"].values])

    inv_x_ll, inv_y_ll = latlon_to_local_xy(layout["lat_deg"].values, layout["lon_deg"].values, lat0, lon0)
    truth_x_ll, truth_y_ll = latlon_to_local_xy(truth["lat"].values, truth["lon"].values, lat0, lon0)

    inv_z = pd.to_numeric(layout["z_m"], errors="coerce").values if "z_m" in layout.columns else np.full(len(layout), np.nan)
    truth_z = pd.to_numeric(truth["z"], errors="coerce").values

    # --------------------------------------------------------
    # Geometry projection
    # --------------------------------------------------------
    inv_s = cumulative_arclength(inv_x_ll, inv_y_ll)
    truth_s = cumulative_arclength(truth_x_ll, truth_y_ll)

    xy_err, truth_proj_x, truth_proj_y, truth_proj_s, _, _ = project_points_onto_polyline(
        inv_x_ll, inv_y_ll, truth_x_ll, truth_y_ll, s_line=truth_s
    )

    truth_z_on_inv = np.interp(truth_proj_s, truth_s, truth_z)
    dz = inv_z - truth_z_on_inv
    abs_dz = np.abs(dz)

    n = len(layout)
    core_mask = np.ones(n, dtype=bool)
    k = int(EXCLUDE_ENDPOINTS_N)
    if 2 * k < n:
        core_mask[:k] = False
        core_mask[-k:] = False

    # --------------------------------------------------------
    # Timing QC
    # --------------------------------------------------------
    needed_fit_cols = [
        "residual_abs_opt_s",
        "residual_dt_ref_opt_s",
        "predicted_t_abs_s_opt",
        "predicted_dt_ref_s_opt",
    ]
    missing_fit = [c for c in needed_fit_cols if c not in obs.columns]
    if missing_fit:
        raise ValueError(f"fit diagnostics missing columns after merge: {missing_fit}")

    high_conf_mask = (
        obs["use_observation"].fillna(False).values.astype(bool)
        & obs["passed_snr_threshold"].fillna(False).values.astype(bool)
        & (~obs["near_window_edge"].fillna(False).values.astype(bool))
        & np.isfinite(obs["weight"].values)
        & (obs["weight"].values >= HIGH_WEIGHT_THRESHOLD)
    )

    abs_rmse_all_ms = 1000.0 * weighted_rmse(
        pd.to_numeric(obs["residual_abs_opt_s"], errors="coerce").values,
        obs["weight"].values
    )
    rel_rmse_all_ms = 1000.0 * weighted_rmse(
        pd.to_numeric(obs["residual_dt_ref_opt_s"], errors="coerce").values,
        obs["weight"].values
    )

    abs_rmse_hc_ms = 1000.0 * weighted_rmse(
        pd.to_numeric(obs.loc[high_conf_mask, "residual_abs_opt_s"], errors="coerce").values,
        obs.loc[high_conf_mask, "weight"].values
    )
    rel_rmse_hc_ms = 1000.0 * weighted_rmse(
        pd.to_numeric(obs.loc[high_conf_mask, "residual_dt_ref_opt_s"], errors="coerce").values,
        obs.loc[high_conf_mask, "weight"].values
    )

    # --------------------------------------------------------
    # Per-channel QC
    # --------------------------------------------------------
    ch_col = "channel_eff" if "channel_eff" in obs.columns else "channel"
    if ch_col not in obs.columns:
        raise ValueError("Need channel_eff or channel in merged observations")

    obs[ch_col] = pd.to_numeric(obs[ch_col], errors="coerce")
    obs["abs_res_ms"] = 1000.0 * pd.to_numeric(obs["residual_abs_opt_s"], errors="coerce")
    obs["rel_res_ms"] = 1000.0 * pd.to_numeric(obs["residual_dt_ref_opt_s"], errors="coerce")

    channel_timing = (
        obs.groupby(ch_col)
        .agg(
            n_obs=("weight", "size"),
            median_abs_absres_ms=("abs_res_ms", lambda x: np.median(np.abs(pd.to_numeric(x, errors="coerce")))),
            median_abs_relres_ms=("rel_res_ms", lambda x: np.median(np.abs(pd.to_numeric(x, errors="coerce")))),
            mean_weight=("weight", "mean"),
        )
        .reset_index()
        .rename(columns={ch_col: "channel"})
    )

    channel_geom = pd.DataFrame({
        "channel": layout["channel"].values,
        "x_m_from_latlon": inv_x_ll,
        "y_m_from_latlon": inv_y_ll,
        "z_m": inv_z,
        "xy_error_to_truth_m": xy_err,
        "truth_proj_x_m": truth_proj_x,
        "truth_proj_y_m": truth_proj_y,
        "truth_proj_s_m": truth_proj_s,
        "truth_z_projected_by_arclength_m": truth_z_on_inv,
        "dz_to_truth_projected_m": dz,
        "abs_dz_to_truth_projected_m": abs_dz,
        "core_mask": core_mask,
    })

    if {"prior_x_m", "prior_y_m"}.issubset(layout.columns) and {"x_m", "y_m"}.issubset(layout.columns):
        channel_geom["horizontal_shift_from_prior_m"] = np.sqrt(
            (pd.to_numeric(layout["x_m"], errors="coerce").values - pd.to_numeric(layout["prior_x_m"], errors="coerce").values) ** 2 +
            (pd.to_numeric(layout["y_m"], errors="coerce").values - pd.to_numeric(layout["prior_y_m"], errors="coerce").values) ** 2
        )
    elif "horizontal_shift_m" in layout.columns:
        channel_geom["horizontal_shift_from_prior_m"] = pd.to_numeric(layout["horizontal_shift_m"], errors="coerce").values
    else:
        channel_geom["horizontal_shift_from_prior_m"] = np.nan

    channel_qc = channel_geom.merge(channel_timing, on="channel", how="left")
    channel_qc["median_abs_absres_ms"] = channel_qc["median_abs_absres_ms"].fillna(0.0)
    channel_qc["median_abs_relres_ms"] = channel_qc["median_abs_relres_ms"].fillna(0.0)
    channel_qc["n_obs"] = channel_qc["n_obs"].fillna(0).astype(int)
    channel_qc["mean_weight"] = channel_qc["mean_weight"].fillna(0.0)

    channel_qc["segment_quality"] = [
        score_segment(xe, te)
        for xe, te in zip(channel_qc["xy_error_to_truth_m"], channel_qc["median_abs_relres_ms"])
    ]

    channel_qc.to_csv(os.path.join(OUTPUT_DIR, "channel_qc_table.csv"), index=False)

    # --------------------------------------------------------
    # Anchor QC
    # --------------------------------------------------------
    anchor_qc = (
        obs.groupby("anchor_id")
        .agg(
            n_obs=("weight", "size"),
            mean_weight=("weight", "mean"),
            median_abs_absres_ms=("abs_res_ms", lambda x: np.median(np.abs(pd.to_numeric(x, errors="coerce")))),
            median_abs_relres_ms=("rel_res_ms", lambda x: np.median(np.abs(pd.to_numeric(x, errors="coerce")))),
            rmse_abs_ms=("abs_res_ms", lambda x: np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2))),
            rmse_rel_ms=("rel_res_ms", lambda x: np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2))),
        )
        .reset_index()
    )

    if anchor_bias_df is not None and {"anchor_id", "anchor_bias_s"}.issubset(anchor_bias_df.columns):
        anchor_qc = anchor_qc.merge(anchor_bias_df[["anchor_id", "anchor_bias_s"]], on="anchor_id", how="left")

    anchor_qc.to_csv(os.path.join(OUTPUT_DIR, "anchor_qc_table.csv"), index=False)

    # --------------------------------------------------------
    # Reference-channel timing sanity check
    # --------------------------------------------------------
    ref_col = "reference_channel_eff" if "reference_channel_eff" in obs.columns else "reference_channel"
    if ref_col not in obs.columns:
        raise ValueError("Need reference_channel_eff or reference_channel")

    ref_tbl = obs.copy()
    ref_tbl[ref_col] = pd.to_numeric(ref_tbl[ref_col], errors="coerce")
    ref_tbl["observed_t_s"] = pd.to_numeric(ref_tbl["observed_t_s"], errors="coerce")
    ref_tbl["predicted_t_abs_s_opt"] = pd.to_numeric(ref_tbl["predicted_t_abs_s_opt"], errors="coerce")

    ref_summary = (
        ref_tbl.groupby("anchor_id")
        .agg(
            reference_channel=(ref_col, lambda x: pd.Series(x).dropna().iloc[0] if len(pd.Series(x).dropna()) else np.nan),
            observed_median_t_ref_s=("observed_t_s", "median"),
            predicted_median_t_ref_s=("predicted_t_abs_s_opt", "median"),
            n_obs=("observed_t_s", "size"),
        )
        .reset_index()
    )
    ref_summary["delta_pred_minus_obs_s"] = (
        ref_summary["predicted_median_t_ref_s"] - ref_summary["observed_median_t_ref_s"]
    )
    ref_summary.to_csv(os.path.join(OUTPUT_DIR, "reference_channel_timing_summary.csv"), index=False)

    # --------------------------------------------------------
    # Channel progression summary
    # --------------------------------------------------------
    prog_core_median = np.nan
    prog_core_p95 = np.nan
    if prog_diag_df is not None and "channel_shift_truth_minus_est" in prog_diag_df.columns and 2 * k < len(layout):
        core_ch_min = layout["channel"].iloc[k]
        core_ch_max = layout["channel"].iloc[-k - 1]
        prog_core = prog_diag_df.loc[
            prog_diag_df["est_channel"].between(core_ch_min, core_ch_max),
            "channel_shift_truth_minus_est"
        ].dropna()
        if len(prog_core):
            prog_core_median = np.median(prog_core)
            prog_core_p95 = np.quantile(np.abs(prog_core), 0.95)

    # --------------------------------------------------------
    # Scorecard
    # --------------------------------------------------------
    scorecard = pd.DataFrame({
        "metric": [
            "xy_rmse_all_m",
            "xy_median_all_m",
            "xy_p95_all_m",
            "xy_rmse_core_m",
            "xy_median_core_m",
            "xy_p95_core_m",
            "median_abs_dz_all_m",
            "median_abs_dz_core_m",
            "weighted_rmse_abs_all_ms",
            "weighted_rmse_rel_all_ms",
            "weighted_rmse_abs_highconf_ms",
            "weighted_rmse_rel_highconf_ms",
            "median_channel_shift_core",
            "p95_abs_channel_shift_core",
        ],
        "value": [
            np.sqrt(np.mean(xy_err ** 2)),
            np.median(xy_err),
            safe_quantile(xy_err, 0.95),
            np.sqrt(np.mean(xy_err[core_mask] ** 2)),
            np.median(xy_err[core_mask]),
            safe_quantile(xy_err[core_mask], 0.95),
            np.median(abs_dz),
            np.median(abs_dz[core_mask]),
            abs_rmse_all_ms,
            rel_rmse_all_ms,
            abs_rmse_hc_ms,
            rel_rmse_hc_ms,
            prog_core_median,
            prog_core_p95,
        ]
    })
    scorecard.to_csv(os.path.join(OUTPUT_DIR, "qc_scorecard.csv"), index=False)

    # --------------------------------------------------------
    # Text report
    # --------------------------------------------------------
    lines = []
    lines.append("DAS cable inversion quality check")
    lines.append("================================")
    lines.append("")
    lines.append("Geometry to truth (XY, all channels):")
    lines.append(f"  RMSE:   {np.sqrt(np.mean(xy_err ** 2)):.3f} m")
    lines.append(f"  Median: {np.median(xy_err):.3f} m")
    lines.append(f"  P95:    {safe_quantile(xy_err, 0.95):.3f} m")
    lines.append("")
    lines.append("Geometry to truth (XY, core excluding endpoints):")
    lines.append(f"  RMSE:   {np.sqrt(np.mean(xy_err[core_mask] ** 2)):.3f} m")
    lines.append(f"  Median: {np.median(xy_err[core_mask]):.3f} m")
    lines.append(f"  P95:    {safe_quantile(xy_err[core_mask], 0.95):.3f} m")
    lines.append("")
    lines.append("Depth mismatch to truth (using truth projected by arclength):")
    lines.append(f"  Median |dz| all:  {np.median(abs_dz):.3f} m")
    lines.append(f"  Median |dz| core: {np.median(abs_dz[core_mask]):.3f} m")
    lines.append("")
    lines.append("Timing residuals (weighted RMSE):")
    lines.append(f"  Absolute, all:   {abs_rmse_all_ms:.3f} ms")
    lines.append(f"  Relative, all:   {rel_rmse_all_ms:.3f} ms")
    lines.append(f"  Absolute, high-confidence: {abs_rmse_hc_ms:.3f} ms")
    lines.append(f"  Relative, high-confidence: {rel_rmse_hc_ms:.3f} ms")
    lines.append("")
    lines.append("Channel progression consistency:")
    lines.append(f"  Median channel shift (core): {prog_core_median:.3f}")
    lines.append(f"  P95 |channel shift| (core): {prog_core_p95:.3f}")
    lines.append("")

    suspect = channel_qc.sort_values("xy_error_to_truth_m", ascending=False).head(15)
    lines.append("Most suspect channels by geometry:")
    for _, row in suspect.iterrows():
        lines.append(
            f"  ch {int(row['channel'])}: xy_error={row['xy_error_to_truth_m']:.2f} m, "
            f"median|relres|={row['median_abs_relres_ms']:.2f} ms, quality={row['segment_quality']}"
        )

    with open(os.path.join(OUTPUT_DIR, "qc_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    # Plan view
    plt.figure(figsize=(10, 8))
    if {"prior_x_m", "prior_y_m"}.issubset(layout.columns):
        plt.plot(layout["prior_x_m"], layout["prior_y_m"], label="Prior geometry")
    plt.plot(inv_x_ll, inv_y_ll, label="Inverted geometry")
    plt.plot(truth_x_ll, truth_y_ll, label="Ground truth")
    if control_df is not None and {"x_m", "y_m"}.issubset(control_df.columns):
        plt.scatter(control_df["x_m"], control_df["y_m"], s=25, label="Optimized control pts")
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.title("Plan-view geometry comparison")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "qc_planview_prior_inverted_truth.png"), dpi=200)
    plt.close()

    # Inverted colored by XY error
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot(truth_x_ll, truth_y_ll, color="k", lw=2, label="Truth")

    vmin = np.nanmin(xy_err)
    vmax = max(np.nanmax(xy_err), 1e-6)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for i in range(len(inv_x_ll) - 1):
        c = plt.cm.viridis(norm(xy_err[i]))
        ax.plot(inv_x_ll[i:i+2], inv_y_ll[i:i+2], color=c, lw=4)

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Nearest XY error to truth (m)")

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("Inverted geometry colored by XY error")
    ax.axis("equal")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "qc_inverted_colored_by_xy_error.png"), dpi=200)
    plt.close(fig)
    # Depth comparison
    plt.figure(figsize=(12, 5))
    if "prior_z_m" in layout.columns:
        plt.plot(layout["channel"], pd.to_numeric(layout["prior_z_m"], errors="coerce"), label="Prior z")
    plt.plot(layout["channel"], inv_z, label="Inverted z")
    plt.plot(layout["channel"], truth_z_on_inv, label="Truth z projected by arclength")
    plt.xlabel("Channel")
    plt.ylabel("Up / depth coordinate (m)")
    plt.title("Depth comparison along the inverted cable")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "qc_depth_comparison.png"), dpi=200)
    plt.close()

    # Horizontal movement from prior
    plt.figure(figsize=(12, 5))
    plt.plot(channel_qc["channel"], channel_qc["horizontal_shift_from_prior_m"])
    plt.xlabel("Channel")
    plt.ylabel("Horizontal shift from prior (m)")
    plt.title("Horizontal movement from prior")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "qc_horizontal_shift_from_prior.png"), dpi=200)
    plt.close()

    # Absolute residual histogram
    plt.figure(figsize=(10, 6))
    if "residual_abs_prior_s" in obs.columns:
        plt.hist(1000 * pd.to_numeric(obs["residual_abs_prior_s"], errors="coerce"), bins=80, alpha=0.5, label="Absolute prior")
    plt.hist(1000 * pd.to_numeric(obs["residual_abs_opt_s"], errors="coerce"), bins=80, alpha=0.5, label="Absolute inverted")
    plt.xlabel("Residual (ms)")
    plt.ylabel("Count")
    plt.title("Absolute timing residuals")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "qc_absolute_residual_hist.png"), dpi=200)
    plt.close()

    # Relative residual histogram
    plt.figure(figsize=(10, 6))
    if "residual_dt_ref_prior_s" in obs.columns:
        plt.hist(1000 * pd.to_numeric(obs["residual_dt_ref_prior_s"], errors="coerce"), bins=80, alpha=0.5, label="Relative prior")
    plt.hist(1000 * pd.to_numeric(obs["residual_dt_ref_opt_s"], errors="coerce"), bins=80, alpha=0.5, label="Relative inverted")
    plt.xlabel("Residual (ms)")
    plt.ylabel("Count")
    plt.title("Relative timing residuals")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "qc_relative_residual_hist.png"), dpi=200)
    plt.close()

    # Relative residuals by anchor
    plt.figure(figsize=(14, 7))
    anchor_order = anchor_qc["anchor_id"].tolist()
    anchor_to_x = {a: i for i, a in enumerate(anchor_order)}
    xs = np.array([anchor_to_x[a] for a in obs["anchor_id"]])
    plt.scatter(xs, np.abs(obs["rel_res_ms"]), s=12, alpha=0.25, c=xs, cmap="tab20")
    plt.xticks(range(len(anchor_order)), anchor_order, rotation=45, ha="right")
    plt.ylabel("|relative residual| (ms)")
    plt.title("Relative timing residuals by anchor")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "qc_relative_residuals_by_anchor.png"), dpi=200)
    plt.close()

    # Anchor biases
    if anchor_bias_df is not None and {"anchor_id", "anchor_bias_s"}.issubset(anchor_bias_df.columns):
        plt.figure(figsize=(12, 5))
        plt.bar(anchor_bias_df["anchor_id"], anchor_bias_df["anchor_bias_s"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Anchor bias (s)")
        plt.title("Estimated per-anchor time biases")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "qc_anchor_biases.png"), dpi=200)
        plt.close()

    # Observed vs predicted dt_ref, high-confidence
    hc = obs.loc[high_conf_mask].copy()
    if len(hc) > 0:
        anchors = sorted(hc["anchor_id"].unique())
        ncol = 3
        nrow = int(np.ceil(len(anchors) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(14, 4 * nrow), squeeze=False)
        for ax, aid in zip(axes.ravel(), anchors):
            m = hc["anchor_id"] == aid
            x = pd.to_numeric(hc.loc[m, "observed_dt_ref_s"], errors="coerce").values
            y = pd.to_numeric(hc.loc[m, "predicted_dt_ref_s_opt"], errors="coerce").values
            ax.scatter(x, y, s=8, alpha=0.5)
            lim0 = min(np.nanmin(x), np.nanmin(y))
            lim1 = max(np.nanmax(x), np.nanmax(y))
            ax.plot([lim0, lim1], [lim0, lim1], color="k", lw=1)
            ax.set_title(aid)
            ax.set_xlabel("Observed dt_ref (s)")
            ax.set_ylabel("Predicted dt_ref (s)")
        for ax in axes.ravel()[len(anchors):]:
            ax.axis("off")
        fig.suptitle("Observed vs predicted relative times (high-confidence subset)", y=0.995)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "qc_observed_vs_predicted_dtref_highconf.png"), dpi=200)
        plt.close(fig)

    # Reference-channel timing sanity
    plt.figure(figsize=(12, 5))
    x = np.arange(len(ref_summary))
    plt.scatter(x, ref_summary["observed_median_t_ref_s"], s=100, label="Observed median t at reference ch")
    plt.scatter(x, ref_summary["predicted_median_t_ref_s"], s=100, label="Predicted median t at reference ch")
    plt.xticks(x, ref_summary["anchor_id"], rotation=45, ha="right")
    plt.ylabel("Time (s)")
    plt.title("Reference-channel absolute-time sanity check by anchor")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "qc_reference_channel_timing.png"), dpi=200)
    plt.close()

    # Geometry + timing by channel
    plt.figure(figsize=(13, 6))
    ax1 = plt.gca()
    ax1.plot(channel_qc["channel"], channel_qc["xy_error_to_truth_m"], label="XY error to truth (m)")
    ax1.set_xlabel("Channel")
    ax1.set_ylabel("XY error to truth (m)")
    ax2 = ax1.twinx()
    ax2.plot(channel_qc["channel"], channel_qc["median_abs_relres_ms"], color="tab:orange",
             label="Median |relative residual| (ms)")
    ax2.set_ylabel("Median |relative residual| (ms)")
    ln1, lb1 = ax1.get_legend_handles_labels()
    ln2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(ln1 + ln2, lb1 + lb2, loc="upper right")
    plt.title("Geometry error and timing error by channel")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "qc_channel_geometry_timing.png"), dpi=200)
    plt.close()

    # Segment quality ribbon
    color_map = {"good": "tab:green", "caution": "goldenrod", "poor": "tab:red"}
    plt.figure(figsize=(14, 2.8))
    for label in ["good", "caution", "poor"]:
        m = channel_qc["segment_quality"] == label
        plt.scatter(channel_qc.loc[m, "channel"], np.zeros(np.sum(m)), s=40, color=color_map[label], label=label)
    plt.yticks([])
    plt.xlabel("Channel")
    plt.title("Segment quality classification")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "qc_segment_quality_ribbon.png"), dpi=200)
    plt.close()

    print("Done.")
    print(f"QC outputs written to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()