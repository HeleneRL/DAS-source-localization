import os
import ast
import argparse
import importlib.util
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


DEFAULT_INV_SCRIPT = os.path.join(os.path.dirname(__file__), 'das_cable_inversion.py')


def load_inversion_module(script_path):
    spec = importlib.util.spec_from_file_location('das_cable_inversion_module', script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_grid(values, cast=float):
    if values is None:
        return None
    if isinstance(values, list):
        return [cast(v) for v in values]
    text = str(values).strip()
    if text.startswith('['):
        parsed = ast.literal_eval(text)
        return [cast(v) for v in parsed]
    return [cast(v) for v in text.split(',') if str(v).strip()]


def read_estimated_layout(path):
    df = pd.read_csv(path)
    rename = {}
    if 'ch' in df.columns and 'channel' not in df.columns:
        rename['ch'] = 'channel'
    if 'x' in df.columns and 'x_m' not in df.columns:
        rename['x'] = 'x_m'
    if 'y' in df.columns and 'y_m' not in df.columns:
        rename['y'] = 'y_m'
    if 'z' in df.columns and 'z_m' not in df.columns:
        rename['z'] = 'z_m'
    df = df.rename(columns=rename)
    required = ['channel', 'x_m', 'y_m']
    for c in required:
        if c not in df.columns:
            raise ValueError(f'{path} is missing required column: {c}')
    if 'z_m' not in df.columns:
        df['z_m'] = np.nan
    out = df[['channel', 'x_m', 'y_m', 'z_m']].copy()
    out['channel'] = pd.to_numeric(out['channel'], errors='coerce').astype('Int64')
    out = out.dropna(subset=['channel', 'x_m', 'y_m']).copy()
    out['channel'] = out['channel'].astype(int)
    return out.sort_values('channel').reset_index(drop=True)


def read_true_layout(path, channel_offset=0):
    df = pd.read_csv(path)
    rename = {}
    if 'ch' in df.columns and 'channel' not in df.columns:
        rename['ch'] = 'channel'
    if 'x' in df.columns and 'x_m' not in df.columns:
        rename['x'] = 'x_m'
    if 'y' in df.columns and 'y_m' not in df.columns:
        rename['y'] = 'y_m'
    if 'z' in df.columns and 'z_m' not in df.columns:
        rename['z'] = 'z_m'
    df = df.rename(columns=rename)
    required = ['channel', 'x_m', 'y_m']
    for c in required:
        if c not in df.columns:
            raise ValueError(f'{path} is missing required column: {c}')
    if 'z_m' not in df.columns:
        df['z_m'] = np.nan
    df = df[['channel', 'x_m', 'y_m', 'z_m']].copy()
    df['channel'] = pd.to_numeric(df['channel'], errors='coerce') + channel_offset
    df = df.dropna(subset=['channel', 'x_m', 'y_m']).copy()
    df['channel'] = df['channel'].astype(int)
    return df.sort_values('channel').reset_index(drop=True)


def match_layouts(est_df, true_df):
    merged = est_df.merge(true_df, on='channel', how='inner', suffixes=('_est', '_true'))
    if merged.empty:
        raise ValueError('No overlapping channels between estimated layout and true layout.')
    merged['dx_m'] = merged['x_m_est'] - merged['x_m_true']
    merged['dy_m'] = merged['y_m_est'] - merged['y_m_true']
    merged['dz_m'] = merged['z_m_est'] - merged['z_m_true']
    merged['xy_error_m'] = np.hypot(merged['dx_m'], merged['dy_m'])
    merged['xyz_error_m'] = np.sqrt(merged['dx_m']**2 + merged['dy_m']**2 + np.nan_to_num(merged['dz_m'])**2)
    return merged.sort_values('channel').reset_index(drop=True)


def rigid_align_2d(xy_est, xy_true):
    ce = xy_est.mean(axis=0)
    ct = xy_true.mean(axis=0)
    Xe = xy_est - ce
    Xt = xy_true - ct
    H = Xe.T @ Xt
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    aligned = Xe @ R.T + ct
    t = ct - ce @ R.T
    return aligned, R, t


def compute_metrics(matched):
    metrics = {
        'n_overlap_channels': len(matched),
        'channel_min': int(matched['channel'].min()),
        'channel_max': int(matched['channel'].max()),
        'rmse_x_m': float(np.sqrt(np.mean(matched['dx_m']**2))),
        'rmse_y_m': float(np.sqrt(np.mean(matched['dy_m']**2))),
        'rmse_xy_m': float(np.sqrt(np.mean(matched['xy_error_m']**2))),
        'mae_xy_m': float(np.mean(np.abs(matched['xy_error_m']))),
        'median_xy_m': float(np.median(matched['xy_error_m'])),
        'p95_xy_m': float(np.quantile(matched['xy_error_m'], 0.95)),
        'max_xy_m': float(np.max(matched['xy_error_m'])),
        'mean_dx_m': float(np.mean(matched['dx_m'])),
        'mean_dy_m': float(np.mean(matched['dy_m'])),
        'median_dx_m': float(np.median(matched['dx_m'])),
        'median_dy_m': float(np.median(matched['dy_m'])),
    }
    if matched['z_m_est'].notna().any() and matched['z_m_true'].notna().any():
        valid_z = matched[['z_m_est', 'z_m_true']].dropna()
        if len(valid_z) > 0:
            dz = valid_z['z_m_est'] - valid_z['z_m_true']
            metrics['rmse_z_m'] = float(np.sqrt(np.mean(dz**2)))
            metrics['median_abs_z_m'] = float(np.median(np.abs(dz)))
    xy_est = matched[['x_m_est', 'y_m_est']].to_numpy()
    xy_true = matched[['x_m_true', 'y_m_true']].to_numpy()
    aligned, R, t = rigid_align_2d(xy_est, xy_true)
    err = np.linalg.norm(aligned - xy_true, axis=1)
    metrics['rmse_xy_rigid_aligned_m'] = float(np.sqrt(np.mean(err**2)))
    metrics['median_xy_rigid_aligned_m'] = float(np.median(err))
    metrics['p95_xy_rigid_aligned_m'] = float(np.quantile(err, 0.95))
    metrics['rigid_rotation_deg'] = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    metrics['rigid_translation_x_m'] = float(t[0])
    metrics['rigid_translation_y_m'] = float(t[1])
    return metrics, aligned, err


def save_metrics(metrics, output_dir, filename='comparison_metrics.csv'):
    pd.DataFrame({'metric': list(metrics.keys()), 'value': list(metrics.values())}).to_csv(
        os.path.join(output_dir, filename), index=False
    )


def make_comparison_plots(matched, aligned_xy, aligned_err, output_dir, prefix='comparison'):
    os.makedirs(output_dir, exist_ok=True)

    # Plan view overlay
    plt.figure(figsize=(10, 8))
    plt.plot(matched['x_m_true'], matched['y_m_true'], label='True geometry')
    plt.plot(matched['x_m_est'], matched['y_m_est'], label='Estimated geometry')
    plt.xlabel('East (m)')
    plt.ylabel('North (m)')
    plt.title('XY plan view: estimated vs true')
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}_plan_overlay.png'), dpi=220)
    plt.close()

    # Plan view with rigid alignment
    plt.figure(figsize=(10, 8))
    plt.plot(matched['x_m_true'], matched['y_m_true'], label='True geometry')
    plt.plot(aligned_xy[:, 0], aligned_xy[:, 1], label='Estimated after rigid alignment')
    plt.xlabel('East (m)')
    plt.ylabel('North (m)')
    plt.title('XY plan view after removing global translation/rotation')
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}_plan_overlay_rigid_aligned.png'), dpi=220)
    plt.close()

    # Quiver residuals
    step = max(1, len(matched) // 250)
    sel = matched.iloc[::step]
    plt.figure(figsize=(11, 8))
    plt.plot(matched['x_m_true'], matched['y_m_true'], label='True geometry')
    plt.plot(matched['x_m_est'], matched['y_m_est'], label='Estimated geometry', alpha=0.7)
    plt.quiver(
        sel['x_m_true'], sel['y_m_true'], sel['dx_m'], sel['dy_m'],
        angles='xy', scale_units='xy', scale=1, width=0.002
    )
    plt.xlabel('East (m)')
    plt.ylabel('North (m)')
    plt.title('Vector error from true to estimated geometry')
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}_xy_quiver_error.png'), dpi=220)
    plt.close()

    # Error vs channel
    plt.figure(figsize=(12, 6))
    plt.plot(matched['channel'], matched['xy_error_m'], label='Direct XY error')
    plt.plot(matched['channel'], aligned_err, label='Rigid-aligned XY error')
    plt.xlabel('Channel')
    plt.ylabel('Error (m)')
    plt.title('XY geometry error vs channel')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}_xy_error_vs_channel.png'), dpi=220)
    plt.close()

    # dx/dy vs channel
    plt.figure(figsize=(12, 6))
    plt.plot(matched['channel'], matched['dx_m'], label='dx = x_est - x_true')
    plt.plot(matched['channel'], matched['dy_m'], label='dy = y_est - y_true')
    plt.axhline(0, linewidth=1)
    plt.xlabel('Channel')
    plt.ylabel('Difference (m)')
    plt.title('Coordinate differences vs channel')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}_dx_dy_vs_channel.png'), dpi=220)
    plt.close()

    # Histogram/CDF
    plt.figure(figsize=(10, 6))
    plt.hist(matched['xy_error_m'], bins=80, alpha=0.7)
    plt.xlabel('XY error (m)')
    plt.ylabel('Count')
    plt.title('Distribution of direct XY errors')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}_xy_error_hist.png'), dpi=220)
    plt.close()

    plt.figure(figsize=(10, 6))
    xs = np.sort(matched['xy_error_m'].to_numpy())
    ys = np.arange(1, len(xs) + 1) / len(xs)
    plt.plot(xs, ys)
    plt.xlabel('XY error (m)')
    plt.ylabel('CDF')
    plt.title('CDF of direct XY errors')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}_xy_error_cdf.png'), dpi=220)
    plt.close()

    # Z comparison if available
    if matched['z_m_est'].notna().any() and matched['z_m_true'].notna().any():
        valid = matched[['channel', 'z_m_est', 'z_m_true']].dropna()
        if len(valid) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(valid['channel'], valid['z_m_true'], label='True z')
            plt.plot(valid['channel'], valid['z_m_est'], label='Estimated z')
            plt.xlabel('Channel')
            plt.ylabel('z / up (m)')
            plt.title('Z comparison')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{prefix}_z_comparison.png'), dpi=220)
            plt.close()


def interpolate_truth_z_to_estimated_channels(est_df, true_df):
    truth_z = true_df[['channel', 'z_m']].dropna().sort_values('channel')
    if len(truth_z) < 2:
        out = est_df.copy()
        out['z_m_from_true'] = np.nan
        return out
    f = interp1d(truth_z['channel'], truth_z['z_m'], kind='linear', bounds_error=False, fill_value='extrapolate')
    out = est_df.copy()
    out['z_m_from_true'] = f(out['channel'])
    return out


def run_single_inversion(module, obs_csv, out_dir, params):
    df = pd.read_csv(obs_csv)
    origin_lat = float(df['enu_origin_lat_deg'].dropna().iloc[0])
    origin_lon = float(df['enu_origin_lon_deg'].dropna().iloc[0])
    origin_h = float(df['enu_origin_h_m'].dropna().iloc[0])

    obs = module.build_observation_table(df, params['channel_offset'])
    prior_sparse = module.build_prior_geometry(df, params['channel_offset'])
    prior_full = module.linear_fill_to_full_channels(prior_sparse)

    min_ch, max_ch = prior_full['channel'].min(), prior_full['channel'].max()
    obs = obs[(obs['channel_eff'] >= min_ch) & (obs['channel_eff'] <= max_ch)].copy()
    obs = obs[(obs['reference_channel_eff'] >= min_ch) & (obs['reference_channel_eff'] <= max_ch)].copy()

    control_channels = module.choose_control_channels(
        prior_full['channel'].values,
        obs['reference_channel_eff'].unique(),
        params['control_spacing'],
    )

    solution = module.solve_inversion(
        obs=obs,
        prior_full=prior_full,
        control_channels=control_channels,
        sound_speed=params['sound_speed'],
        channel_spacing=params['channel_spacing'],
        abs_scale=params['abs_scale'],
        rel_scale=params['rel_scale'],
        prior_sigma_xy=params['prior_sigma_xy'],
        prior_sigma_z=params['prior_sigma_z'],
        curvature_sigma_xy=params['curvature_sigma_xy'],
        curvature_sigma_z=params['curvature_sigma_z'],
        spacing_sigma=params['spacing_sigma'],
        anchor_bias_sigma=params['anchor_bias_sigma'],
        max_nfev=params['max_nfev'],
    )
    diagnostics = module.compute_fit_diagnostics(solution)
    module.save_outputs(obs, prior_full, solution, diagnostics, out_dir, origin_lat, origin_lon, origin_h)
    module.make_plots(obs, solution, diagnostics, out_dir)

    return os.path.join(out_dir, 'updated_cable_layout.csv')


def compare_files(estimated_layout_csv, true_layout_csv, output_dir, true_channel_offset=0, export_truth_z_fill=False):
    os.makedirs(output_dir, exist_ok=True)
    est_df = read_estimated_layout(estimated_layout_csv)
    true_df = read_true_layout(true_layout_csv, channel_offset=true_channel_offset)
    matched = match_layouts(est_df, true_df)
    metrics, aligned_xy, aligned_err = compute_metrics(matched)
    matched.to_csv(os.path.join(output_dir, 'channelwise_geometry_comparison.csv'), index=False)
    save_metrics(metrics, output_dir)
    make_comparison_plots(matched, aligned_xy, aligned_err, output_dir)

    if export_truth_z_fill:
        filled = interpolate_truth_z_to_estimated_channels(est_df, true_df)
        filled.to_csv(os.path.join(output_dir, 'estimated_layout_with_truth_z.csv'), index=False)

    return metrics


def tuning_grid(args):
    return list(product(
        parse_grid(args.grid_control_spacing, int),
        parse_grid(args.grid_prior_sigma_xy, float),
        parse_grid(args.grid_prior_sigma_z, float),
        parse_grid(args.grid_curvature_sigma_xy, float),
        parse_grid(args.grid_curvature_sigma_z, float),
        parse_grid(args.grid_spacing_sigma, float),
    ))


def main():
    parser = argparse.ArgumentParser(description='Compare estimated DAS cable geometry with a reference geometry and optionally tune inversion parameters.')
    parser.add_argument('--mode', choices=['compare', 'tune'], required=True)
    parser.add_argument('--estimated_layout_csv', type=str, help='Path to updated_cable_layout.csv from the inversion script. Required for compare mode.')
    parser.add_argument('--true_layout_csv', type=str, required=True, help='Path to array-shape.csv (or similar true/reference cable file).')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--true_channel_offset', type=int, default=0, help='Optional channel offset to add to the reference geometry channels before comparison.')
    parser.add_argument('--export_truth_z_fill', action='store_true', help='Also write estimated_layout_with_truth_z.csv using z interpolated from the reference geometry.')

    parser.add_argument('--inversion_observations_csv', type=str, help='Required for tune mode: original inversion_observations.csv')
    parser.add_argument('--inversion_script', type=str, default=DEFAULT_INV_SCRIPT)
    parser.add_argument('--channel_offset', type=int, default=61)
    parser.add_argument('--sound_speed', type=float, default=1500.0)
    parser.add_argument('--channel_spacing', type=float, default=1.02)
    parser.add_argument('--abs_scale', type=float, default=0.003)
    parser.add_argument('--rel_scale', type=float, default=0.0015)
    parser.add_argument('--anchor_bias_sigma', type=float, default=0.02)
    parser.add_argument('--max_nfev', type=int, default=250)

    parser.add_argument('--grid_control_spacing', type=str, default='60,75,90')
    parser.add_argument('--grid_prior_sigma_xy', type=str, default='20,30,40')
    parser.add_argument('--grid_prior_sigma_z', type=str, default='4,6,8')
    parser.add_argument('--grid_curvature_sigma_xy', type=str, default='2,3,5')
    parser.add_argument('--grid_curvature_sigma_z', type=str, default='1.0,1.5,2.5')
    parser.add_argument('--grid_spacing_sigma', type=str, default='0.05,0.08,0.12')
    parser.add_argument('--top_k_plots', type=int, default=5)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == 'compare':
        if not args.estimated_layout_csv:
            raise ValueError('--estimated_layout_csv is required in compare mode.')
        metrics = compare_files(
            estimated_layout_csv=args.estimated_layout_csv,
            true_layout_csv=args.true_layout_csv,
            output_dir=args.output_dir,
            true_channel_offset=args.true_channel_offset,
            export_truth_z_fill=args.export_truth_z_fill,
        )
        print('Comparison complete.')
        for k, v in metrics.items():
            print(f'{k}: {v}')
        return

    # tune mode
    if not args.inversion_observations_csv:
        raise ValueError('--inversion_observations_csv is required in tune mode.')

    module = load_inversion_module(args.inversion_script)
    grid = tuning_grid(args)
    records = []
    top_root = os.path.join(args.output_dir, 'runs')
    os.makedirs(top_root, exist_ok=True)

    for i, (control_spacing, prior_sigma_xy, prior_sigma_z, curvature_sigma_xy, curvature_sigma_z, spacing_sigma) in enumerate(grid, start=1):
        run_name = (
            f'run_{i:03d}_cs{control_spacing}'
            f'_psxy{prior_sigma_xy:g}_psz{prior_sigma_z:g}'
            f'_csxy{curvature_sigma_xy:g}_csz{curvature_sigma_z:g}'
            f'_ss{spacing_sigma:g}'
        )
        run_dir = os.path.join(top_root, run_name)
        os.makedirs(run_dir, exist_ok=True)
        params = {
            'control_spacing': control_spacing,
            'prior_sigma_xy': prior_sigma_xy,
            'prior_sigma_z': prior_sigma_z,
            'curvature_sigma_xy': curvature_sigma_xy,
            'curvature_sigma_z': curvature_sigma_z,
            'spacing_sigma': spacing_sigma,
            'channel_offset': args.channel_offset,
            'sound_speed': args.sound_speed,
            'channel_spacing': args.channel_spacing,
            'abs_scale': args.abs_scale,
            'rel_scale': args.rel_scale,
            'anchor_bias_sigma': args.anchor_bias_sigma,
            'max_nfev': args.max_nfev,
        }
        try:
            est_csv = run_single_inversion(module, args.inversion_observations_csv, run_dir, params)
            metrics = compare_files(
                estimated_layout_csv=est_csv,
                true_layout_csv=args.true_layout_csv,
                output_dir=os.path.join(run_dir, 'comparison'),
                true_channel_offset=args.true_channel_offset,
                export_truth_z_fill=False,
            )
            row = {**params, **metrics, 'run_name': run_name, 'run_dir': run_dir, 'status': 'ok'}
            print(f"[{i}/{len(grid)}] {run_name}: rmse_xy={metrics['rmse_xy_m']:.3f} m, median={metrics['median_xy_m']:.3f} m")
        except Exception as exc:
            row = {**params, 'run_name': run_name, 'run_dir': run_dir, 'status': f'failed: {exc}'}
            print(f'[{i}/{len(grid)}] {run_name} failed: {exc}')
        records.append(row)

    results = pd.DataFrame(records)
    sort_cols = [c for c in ['rmse_xy_m', 'median_xy_m', 'p95_xy_m'] if c in results.columns]
    if sort_cols:
        results = results.sort_values(sort_cols).reset_index(drop=True)
    results.to_csv(os.path.join(args.output_dir, 'tuning_results.csv'), index=False)

    # Copy top-k summary plots into one folder reference by CSV only
    topk_dir = os.path.join(args.output_dir, 'top_runs_summary')
    os.makedirs(topk_dir, exist_ok=True)
    ok = results[results['status'] == 'ok'].copy()
    if len(ok) > 0:
        ok.head(args.top_k_plots).to_csv(os.path.join(topk_dir, 'top_runs.csv'), index=False)

    print('Tuning complete.')
    print(f'Results saved to: {os.path.join(args.output_dir, "tuning_results.csv")}')


if __name__ == '__main__':
    main()
