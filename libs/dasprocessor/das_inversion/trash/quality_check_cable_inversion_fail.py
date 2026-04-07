import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def latlon_to_local_xy(lat, lon, lat0, lon0):
    R = 6371000.0
    lat = np.radians(np.asarray(lat, dtype=float))
    lon = np.radians(np.asarray(lon, dtype=float))
    lat0 = np.radians(float(lat0))
    lon0 = np.radians(float(lon0))
    x = (lon - lon0) * np.cos(lat0) * R
    y = (lat - lat0) * R
    return x, y


def cumulative_arclength(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ds = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    return np.concatenate([[0.0], np.cumsum(ds)])


def project_points_onto_polyline(px, py, x_line, y_line, s_line):
    n_pts = len(px)
    dist = np.full(n_pts, np.inf)
    proj_x = np.full(n_pts, np.nan)
    proj_y = np.full(n_pts, np.nan)
    proj_s = np.full(n_pts, np.nan)
    seg_idx = np.full(n_pts, -1, dtype=int)

    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)

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
        else:
            t = ((px - x1) * dx + (py - y1) * dy) / seg_len2
            t = np.clip(t, 0.0, 1.0)
            qx = x1 + t * dx
            qy = y1 + t * dy
        d = np.sqrt((px - qx) ** 2 + (py - qy) ** 2)
        mask = d < dist
        dist[mask] = d[mask]
        proj_x[mask] = qx[mask]
        proj_y[mask] = qy[mask]
        proj_s[mask] = s_line[i] + t[mask] * np.sqrt(seg_len2)
        seg_idx[mask] = i
    return dist, proj_x, proj_y, proj_s, seg_idx


def get_metric(summary_df, name, default=np.nan):
    m = summary_df.loc[summary_df['metric'] == name, 'value']
    if len(m) == 0:
        return default
    try:
        return float(m.iloc[0])
    except Exception:
        return m.iloc[0]


def robust_stats(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(mean=np.nan, median=np.nan, rmse=np.nan, p95=np.nan, max=np.nan)
    return {
        'mean': float(np.mean(x)),
        'median': float(np.median(x)),
        'rmse': float(np.sqrt(np.mean(x ** 2))),
        'p95': float(np.percentile(x, 95)),
        'max': float(np.max(x)),
    }


def weighted_rmse(residual_s, weights):
    residual_s = np.asarray(residual_s, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(residual_s) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return np.nan
    return float(np.sqrt(np.average(residual_s[mask] ** 2, weights=weights[mask])))


def classify_segments(channel, error_xy, residual_rel_ms=None, geom_thr=(5, 10), rel_thr=(5, 20)):
    channel = np.asarray(channel)
    error_xy = np.asarray(error_xy)
    if residual_rel_ms is None:
        residual_rel_ms = np.zeros_like(error_xy)
    else:
        residual_rel_ms = np.asarray(residual_rel_ms)

    labels = np.empty(len(channel), dtype=object)
    for i in range(len(channel)):
        g = error_xy[i]
        r = abs(residual_rel_ms[i]) if np.isfinite(residual_rel_ms[i]) else 0.0
        if g <= geom_thr[0] and r <= rel_thr[0]:
            labels[i] = 'good'
        elif g <= geom_thr[1] and r <= rel_thr[1]:
            labels[i] = 'caution'
        else:
            labels[i] = 'poor'
    return labels


def main():
    parser = argparse.ArgumentParser(description='Quality-check report for DAS cable inversion.')
    parser.add_argument('--updated_layout_csv', required=True)
    parser.add_argument('--observations_csv', required=True)
    parser.add_argument('--fit_diagnostics_csv', required=True)
    parser.add_argument('--truth_csv', required=True)
    parser.add_argument('--summary_csv', required=True)
    parser.add_argument('--anchor_biases_csv', required=True)
    parser.add_argument('--control_points_csv', required=True)
    parser.add_argument('--channel_progression_diagnostics_csv', required=False, default=None)
    parser.add_argument('--channel_progression_metrics_csv', required=False, default=None)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--high_weight_threshold', type=float, default=0.7)
    parser.add_argument('--exclude_endpoints_n', type=int, default=25)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cable = pd.read_csv(args.updated_layout_csv)
    obs = pd.read_csv(args.observations_csv)
    fit = pd.read_csv(args.fit_diagnostics_csv)
    truth = pd.read_csv(args.truth_csv)
    summary = pd.read_csv(args.summary_csv)
    anchor_biases = pd.read_csv(args.anchor_biases_csv)
    ctrl = pd.read_csv(args.control_points_csv)
    prog = pd.read_csv(args.channel_progression_diagnostics_csv) if args.channel_progression_diagnostics_csv else None
    prog_metrics = pd.read_csv(args.channel_progression_metrics_csv) if args.channel_progression_metrics_csv else None

    # common local xy from lat lon for geometry comparison
    lat0 = np.mean(np.concatenate([cable['lat_deg'].values, truth['lat'].values]))
    lon0 = np.mean(np.concatenate([cable['lon_deg'].values, truth['lon'].values]))
    cable_x_ll, cable_y_ll = latlon_to_local_xy(cable['lat_deg'].values, cable['lon_deg'].values, lat0, lon0)
    truth_x_ll, truth_y_ll = latlon_to_local_xy(truth['lat'].values, truth['lon'].values, lat0, lon0)
    truth_s = cumulative_arclength(truth_x_ll, truth_y_ll)
    cable_s = cumulative_arclength(cable_x_ll, cable_y_ll)
    dist_xy, proj_x, proj_y, proj_s, _ = project_points_onto_polyline(cable_x_ll, cable_y_ll, truth_x_ll, truth_y_ll, truth_s)

    # matched truth z and channel by projected arc length
    truth_z_by_s = np.interp(proj_s, truth_s, truth['z'].values)
    truth_ch_by_s = np.interp(proj_s, truth_s, truth['ch'].values)
    z_diff = cable['z_m'].values - truth_z_by_s

    # ignore endpoint projection pathologies for some summaries
    endn = int(args.exclude_endpoints_n)
    mask_core = np.ones(len(cable), dtype=bool)
    if len(cable) > 2 * endn:
        mask_core[:endn] = False
        mask_core[-endn:] = False

    geom_all = robust_stats(dist_xy)
    geom_core = robust_stats(dist_xy[mask_core])
    z_all = robust_stats(np.abs(z_diff))
    z_core = robust_stats(np.abs(z_diff[mask_core]))

    # join fit diagnostics with resulting layout for geometry/timing combined QC
    fit = fit.copy()
    fit['channel_eff'] = pd.to_numeric(fit.get('channel_eff', fit['channel']), errors='coerce')
    fit = fit.merge(cable[['channel', 'horizontal_shift_m', 'x_m', 'y_m', 'z_m']], how='left', left_on='channel_eff', right_on='channel')
    fit['residual_abs_opt_ms'] = 1000.0 * fit['residual_abs_opt_s']
    fit['residual_dt_ref_opt_ms'] = 1000.0 * fit['residual_dt_ref_opt_s']

    channel_geom = pd.DataFrame({
        'channel': cable['channel'],
        'xy_error_to_truth_m': dist_xy,
        'truth_channel_matched': truth_ch_by_s,
        'z_error_to_truth_m': z_diff,
        'horizontal_shift_from_prior_m': cable['horizontal_shift_m'],
    })

    # channelwise timing summaries
    ch_timing = fit.groupby('channel_eff').agg(
        n_obs=('residual_dt_ref_opt_ms', 'size'),
        mean_abs_rel_res_ms=('residual_dt_ref_opt_ms', lambda s: np.mean(np.abs(s))),
        median_abs_rel_res_ms=('residual_dt_ref_opt_ms', lambda s: np.median(np.abs(s))),
        p95_abs_rel_res_ms=('residual_dt_ref_opt_ms', lambda s: np.percentile(np.abs(s), 95)),
        weighted_mean_abs_rel_res_ms=('residual_dt_ref_opt_ms', lambda s: np.nan),
    ).reset_index().rename(columns={'channel_eff': 'channel'})
    # weighted metric manually
    rows = []
    for ch, g in fit.groupby('channel_eff'):
        rows.append((ch, weighted_rmse(g['residual_dt_ref_opt_s'].values, g['weight'].values) * 1000.0 if 'weight' in g else np.nan))
    ch_w = pd.DataFrame(rows, columns=['channel', 'weighted_rmse_rel_ms'])
    ch_timing = ch_timing.merge(ch_w, on='channel', how='left')

    channel_qc = channel_geom.merge(ch_timing, on='channel', how='left')
    channel_qc['segment_quality'] = classify_segments(
        channel_qc['channel'].values,
        channel_qc['xy_error_to_truth_m'].values,
        channel_qc['median_abs_rel_res_ms'].fillna(0).values,
    )
    channel_qc.to_csv(os.path.join(args.output_dir, 'channel_qc_table.csv'), index=False)

    # observation subsets
    def subset_mask(df):
        m = np.ones(len(df), dtype=bool)
        if 'use_observation' in df:
            m &= df['use_observation'].fillna(False).astype(bool).values
        if 'passed_snr_threshold' in df:
            m &= df['passed_snr_threshold'].fillna(False).astype(bool).values
        if 'near_window_edge' in df:
            m &= ~df['near_window_edge'].fillna(True).astype(bool).values
        return m

    high_conf = fit[subset_mask(fit)].copy()
    if 'weight' in high_conf:
        high_conf = high_conf[high_conf['weight'] >= args.high_weight_threshold].copy()

    # scorecard
    scorecard = {
        'n_cable_channels': len(cable),
        'n_observations': len(fit),
        'n_high_conf_observations': len(high_conf),
        'geometry_rmse_xy_m_all': geom_all['rmse'],
        'geometry_median_xy_m_all': geom_all['median'],
        'geometry_p95_xy_m_all': geom_all['p95'],
        'geometry_rmse_xy_m_core': geom_core['rmse'],
        'geometry_median_xy_m_core': geom_core['median'],
        'geometry_p95_xy_m_core': geom_core['p95'],
        'z_median_abs_m_all': z_all['median'],
        'z_p95_abs_m_all': z_all['p95'],
        'z_median_abs_m_core': z_core['median'],
        'z_p95_abs_m_core': z_core['p95'],
        'weighted_rmse_abs_ms_all': weighted_rmse(fit['residual_abs_opt_s'].values, fit['weight'].values) * 1000.0,
        'weighted_rmse_rel_ms_all': weighted_rmse(fit['residual_dt_ref_opt_s'].values, fit['weight'].values) * 1000.0,
        'weighted_rmse_abs_ms_high_conf': weighted_rmse(high_conf['residual_abs_opt_s'].values, high_conf['weight'].values) * 1000.0 if len(high_conf) else np.nan,
        'weighted_rmse_rel_ms_high_conf': weighted_rmse(high_conf['residual_dt_ref_opt_s'].values, high_conf['weight'].values) * 1000.0 if len(high_conf) else np.nan,
        'inversion_rmse_abs_opt_ms_reported': get_metric(summary, 'rmse_abs_opt_ms'),
        'inversion_rmse_rel_opt_ms_reported': get_metric(summary, 'rmse_rel_opt_ms'),
        'median_horizontal_shift_from_prior_m': float(np.median(cable['horizontal_shift_m'])),
        'p95_horizontal_shift_from_prior_m': float(np.percentile(cable['horizontal_shift_m'], 95)),
        'channel_shift_median_core': float(np.median(prog.loc[mask_core, 'channel_shift_truth_minus_est'])) if prog is not None else np.nan,
        'channel_shift_p95_abs_core': float(np.percentile(np.abs(prog.loc[mask_core, 'channel_shift_truth_minus_est']), 95)) if prog is not None else np.nan,
    }
    pd.DataFrame([scorecard]).to_csv(os.path.join(args.output_dir, 'qc_scorecard.csv'), index=False)

    # anchor scorecard
    anchor_tbl = fit.groupby('anchor_id').agg(
        n_obs=('anchor_id', 'size'),
        wrmse_abs_ms=('residual_abs_opt_s', lambda s: np.nan),
        wrmse_rel_ms=('residual_dt_ref_opt_s', lambda s: np.nan),
        median_abs_rel_ms=('residual_dt_ref_opt_s', lambda s: 1000.0 * np.median(np.abs(s))),
        median_abs_abs_ms=('residual_abs_opt_s', lambda s: 1000.0 * np.median(np.abs(s))),
        reference_channel=('reference_channel_eff', 'first'),
    ).reset_index()
    rows = []
    for aid, g in fit.groupby('anchor_id'):
        rows.append((aid,
                     weighted_rmse(g['residual_abs_opt_s'].values, g['weight'].values) * 1000.0,
                     weighted_rmse(g['residual_dt_ref_opt_s'].values, g['weight'].values) * 1000.0))
    anchor_wr = pd.DataFrame(rows, columns=['anchor_id', 'wrmse_abs_ms', 'wrmse_rel_ms'])
    anchor_tbl = anchor_tbl.drop(columns=['wrmse_abs_ms', 'wrmse_rel_ms']).merge(anchor_wr, on='anchor_id', how='left')
    anchor_tbl = anchor_tbl.merge(anchor_biases, on='anchor_id', how='left')
    anchor_tbl.to_csv(os.path.join(args.output_dir, 'anchor_qc_table.csv'), index=False)

    # figures
    # 1 overlay prior/inverted/truth
    plt.figure(figsize=(10, 9))
    plt.plot(cable['prior_x_m'], cable['prior_y_m'], label='Prior geometry', lw=1.8, alpha=0.9)
    plt.plot(cable['x_m'], cable['y_m'], label='Inverted geometry', lw=2.2)
    plt.plot(truth['x'], truth['y'], label='Ground truth', lw=2.0)
    plt.scatter(ctrl['x_m'], ctrl['y_m'], s=18, label='Optimized control pts')
    plt.xlabel('East (m)')
    plt.ylabel('North (m)')
    plt.title('Plan-view geometry comparison')
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'qc_planview_prior_inverted_truth.png'), dpi=220)
    plt.close()

    # 2 inverted colored by xy error
    plt.figure(figsize=(9, 9))
    plt.plot(truth_x_ll, truth_y_ll, color='k', lw=1.5, label='Truth')
    sc = plt.scatter(cable_x_ll, cable_y_ll, c=dist_xy, s=10, cmap='viridis')
    plt.colorbar(sc, label='Nearest XY error to truth (m)')
    plt.xlabel('East (m)')
    plt.ylabel('North (m)')
    plt.title('Inverted geometry colored by XY error')
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'qc_inverted_colored_by_xy_error.png'), dpi=220)
    plt.close()

    # 3 channelwise geometry and timing combined
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(channel_qc['channel'], channel_qc['xy_error_to_truth_m'], label='XY error to truth (m)')
    ax1.set_xlabel('Channel')
    ax1.set_ylabel('XY error to truth (m)')
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(channel_qc['channel'], channel_qc['median_abs_rel_res_ms'], color='tab:orange', alpha=0.8, label='Median |relative residual| (ms)')
    ax2.set_ylabel('Median |relative residual| (ms)')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    plt.title('Geometry error and timing error by channel')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'qc_channel_geometry_timing.png'), dpi=220)
    plt.close(fig)

    # 4 segment quality ribbon
    qual_color = {'good': 'tab:green', 'caution': 'goldenrod', 'poor': 'tab:red'}
    plt.figure(figsize=(12, 2.4))
    for q in ['good', 'caution', 'poor']:
        m = channel_qc['segment_quality'] == q
        plt.scatter(channel_qc.loc[m, 'channel'], np.zeros(m.sum()), s=12, color=qual_color[q], label=q)
    plt.yticks([])
    plt.xlabel('Channel')
    plt.title('Segment quality classification')
    plt.legend(ncol=3, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'qc_segment_quality_ribbon.png'), dpi=220)
    plt.close()

    # 5 relative residual by anchor boxplot-ish scatter
    anchors = sorted(anchor_tbl['anchor_id'].tolist())
    pos = {a: i for i, a in enumerate(anchors)}
    plt.figure(figsize=(12, 5))
    for aid, g in fit.groupby('anchor_id'):
        x = np.full(len(g), pos[aid]) + np.random.uniform(-0.15, 0.15, len(g))
        plt.scatter(x, np.abs(g['residual_dt_ref_opt_ms']), s=6, alpha=0.2)
    plt.xticks(range(len(anchors)), anchors, rotation=45, ha='right')
    plt.ylabel('|relative residual| (ms)')
    plt.title('Relative timing residuals by anchor')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'qc_relative_residuals_by_anchor.png'), dpi=220)
    plt.close()

    # 6 observed vs predicted dt_ref for high confidence only
    fig, axes = plt.subplots(4, 3, figsize=(13, 14), squeeze=False)
    hc_plot = high_conf if len(high_conf) else fit
    aids = sorted(hc_plot['anchor_id'].unique())
    for ax, aid in zip(axes.ravel(), aids):
        g = hc_plot[hc_plot['anchor_id'] == aid]
        ax.scatter(g['observed_dt_ref_s'], g['predicted_dt_ref_s_opt'], s=8, alpha=0.35)
        lim0 = np.nanmin(np.concatenate([g['observed_dt_ref_s'].values, g['predicted_dt_ref_s_opt'].values]))
        lim1 = np.nanmax(np.concatenate([g['observed_dt_ref_s'].values, g['predicted_dt_ref_s_opt'].values]))
        ax.plot([lim0, lim1], [lim0, lim1], color='k', lw=1)
        ax.set_title(aid)
        ax.set_xlabel('Observed dt_ref (s)')
        ax.set_ylabel('Predicted dt_ref (s)')
    for ax in axes.ravel()[len(aids):]:
        ax.axis('off')
    fig.suptitle('Observed vs predicted relative times (high-confidence subset)', y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'qc_observed_vs_predicted_dtref_highconf.png'), dpi=220)
    plt.close(fig)

    # 7 reference channel timing sanity: predicted and observed absolute time at reference channels by anchor
    ref_rows = []
    for aid, g in fit.groupby('anchor_id'):
        ref_ch = g['reference_channel_eff'].iloc[0]
        gref = g[g['channel_eff'] == ref_ch]
        if len(gref):
            ref_rows.append({
                'anchor_id': aid,
                'reference_channel': ref_ch,
                'obs_ref_t_s_median': gref['observed_t_s'].median(),
                'pred_ref_t_s_median': gref['predicted_t_abs_s_opt'].median(),
                'n_reference_obs': len(gref),
            })
    ref_tbl = pd.DataFrame(ref_rows)
    ref_tbl.to_csv(os.path.join(args.output_dir, 'reference_channel_timing_summary.csv'), index=False)
    if len(ref_tbl):
        plt.figure(figsize=(10, 4.5))
        x = np.arange(len(ref_tbl))
        plt.scatter(x, ref_tbl['obs_ref_t_s_median'], label='Observed median t at reference ch')
        plt.scatter(x, ref_tbl['pred_ref_t_s_median'], label='Predicted median t at reference ch')
        plt.xticks(x, ref_tbl['anchor_id'], rotation=45, ha='right')
        plt.ylabel('Time (s)')
        plt.title('Reference-channel absolute-time sanity check by anchor')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'qc_reference_channel_timing.png'), dpi=220)
        plt.close()

    # 8 anchor bias bar plot
    plt.figure(figsize=(10, 4.5))
    plt.bar(anchor_biases['anchor_id'], anchor_biases['anchor_bias_s'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Anchor bias (s)')
    plt.title('Estimated per-anchor time biases')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'qc_anchor_biases.png'), dpi=220)
    plt.close()

    # 9 Z profile vs prior and truth
    truth_z_on_cable_s = np.interp(cable_s, truth_s, truth['z'].values)
    plt.figure(figsize=(12, 5))
    plt.plot(cable['channel'], cable['prior_z_m'], label='Prior z')
    plt.plot(cable['channel'], cable['z_m'], label='Inverted z')
    plt.plot(cable['channel'], truth_z_on_cable_s, label='Truth z projected by arclength')
    plt.xlabel('Channel')
    plt.ylabel('Up / depth coordinate (m)')
    plt.title('Depth comparison along the inverted cable')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'qc_depth_comparison.png'), dpi=220)
    plt.close()

    # 10 horizontal shift from prior and movement of control points
    plt.figure(figsize=(12, 4.5))
    plt.plot(cable['channel'], cable['horizontal_shift_m'])
    plt.xlabel('Channel')
    plt.ylabel('Horizontal shift from prior (m)')
    plt.title('Horizontal movement from prior')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'qc_horizontal_shift_from_prior.png'), dpi=220)
    plt.close()

    # 11 prior vs inverted timing histograms from fit csv
    plt.figure(figsize=(10, 5))
    plt.hist(1000 * fit['residual_abs_prior_s'], bins=120, alpha=0.5, label='Absolute prior')
    plt.hist(1000 * fit['residual_abs_opt_s'], bins=120, alpha=0.5, label='Absolute inverted')
    plt.xlabel('Residual (ms)')
    plt.ylabel('Count')
    plt.title('Absolute timing residuals')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'qc_absolute_residual_hist.png'), dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.hist(1000 * fit['residual_dt_ref_prior_s'], bins=120, alpha=0.5, label='Relative prior')
    plt.hist(1000 * fit['residual_dt_ref_opt_s'], bins=120, alpha=0.5, label='Relative inverted')
    plt.xlabel('Residual (ms)')
    plt.ylabel('Count')
    plt.title('Relative timing residuals')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'qc_relative_residual_hist.png'), dpi=220)
    plt.close()

    # 12 simple text report
    with open(os.path.join(args.output_dir, 'qc_report.txt'), 'w', encoding='utf-8') as f:
        f.write('DAS cable inversion quality check\n')
        f.write('================================\n\n')
        f.write('Geometry to truth (XY, all channels):\n')
        f.write(f"  RMSE:   {geom_all['rmse']:.3f} m\n")
        f.write(f"  Median: {geom_all['median']:.3f} m\n")
        f.write(f"  P95:    {geom_all['p95']:.3f} m\n\n")
        f.write('Geometry to truth (XY, core excluding endpoints):\n')
        f.write(f"  RMSE:   {geom_core['rmse']:.3f} m\n")
        f.write(f"  Median: {geom_core['median']:.3f} m\n")
        f.write(f"  P95:    {geom_core['p95']:.3f} m\n\n")
        f.write('Depth mismatch to truth (using truth projected by arclength):\n')
        f.write(f"  Median |dz| all:  {z_all['median']:.3f} m\n")
        f.write(f"  Median |dz| core: {z_core['median']:.3f} m\n\n")
        f.write('Timing residuals (weighted RMSE):\n')
        f.write(f"  Absolute, all:   {scorecard['weighted_rmse_abs_ms_all']:.3f} ms\n")
        f.write(f"  Relative, all:   {scorecard['weighted_rmse_rel_ms_all']:.3f} ms\n")
        f.write(f"  Absolute, high-confidence: {scorecard['weighted_rmse_abs_ms_high_conf']:.3f} ms\n")
        f.write(f"  Relative, high-confidence: {scorecard['weighted_rmse_rel_ms_high_conf']:.3f} ms\n\n")
        if prog is not None:
            f.write('Channel progression consistency:\n')
            f.write(f"  Median channel shift (core): {scorecard['channel_shift_median_core']:.3f}\n")
            f.write(f"  P95 |channel shift| (core): {scorecard['channel_shift_p95_abs_core']:.3f}\n\n")
        f.write('Most suspect channels by geometry:\n')
        bad = channel_qc.sort_values('xy_error_to_truth_m', ascending=False).head(15)
        for _, row in bad.iterrows():
            f.write(f"  ch {int(row['channel'])}: xy_error={row['xy_error_to_truth_m']:.2f} m, median|relres|={row['median_abs_rel_res_ms']:.2f} ms, quality={row['segment_quality']}\n")

    print('QC outputs written to:', os.path.abspath(args.output_dir))


if __name__ == '__main__':
    main()
