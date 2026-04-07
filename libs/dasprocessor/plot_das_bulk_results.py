from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CHANNEL_MIN = 348
CHANNEL_MAX = 2267



REQUIRED_COLUMNS = {
    'location',
    'anchor_index',
    'anchor_label',
    'reference_channel',
    'channel',
    'peak_time_s_from_sequence_start',
    'anchor_time_s_from_sequence_start',
    'peak_global_sample',
    'peak_raw_envelope',
    'peak_prominence_raw',
    'baseline_median',
    'baseline_mad',
    'snr_like',
    'passed_snr_threshold',
    'near_window_edge',
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



def load_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[(df["channel"] >= CHANNEL_MIN) & (df["channel"] <= CHANNEL_MAX)].copy()
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns in {csv_path}: {sorted(missing)}')

    df['passed_snr_threshold'] = df['passed_snr_threshold'].astype(str).str.lower().isin(['true', '1', 'yes'])
    df['near_window_edge'] = df['near_window_edge'].astype(str).str.lower().isin(['true', '1', 'yes'])

    df['relative_offset_s'] = (
        df['peak_time_s_from_sequence_start'] - df['anchor_time_s_from_sequence_start']
    )

    ref_by_anchor = (
        df.loc[df['channel'] == df['reference_channel'], ['anchor_index', 'peak_time_s_from_sequence_start']]
        .drop_duplicates(subset=['anchor_index'])
        .rename(columns={'peak_time_s_from_sequence_start': 'reference_peak_time_s'})
    )
    if ref_by_anchor.empty:
        raise ValueError('Could not find reference channel rows in CSV.')

    df = df.merge(ref_by_anchor, on='anchor_index', how='left')
    df['relative_to_reference_s'] = (
        df['peak_time_s_from_sequence_start'] - df['reference_peak_time_s']
    )
    df['relative_to_reference_ms'] = 1000.0 * df['relative_to_reference_s']

    # robust local stability metric from first difference of arrival curve
    df = df.sort_values(['anchor_index', 'channel']).reset_index(drop=True)
    stability_list = []
    for anchor_index, grp in df.groupby('anchor_index', sort=True):
        g = grp.copy().sort_values('channel')
        arrival = g['relative_to_reference_s'].to_numpy()
        grad = np.full_like(arrival, np.nan, dtype=float)
        if len(arrival) >= 3:
            grad[1:-1] = 0.5 * (arrival[2:] - arrival[:-2])
            grad[0] = arrival[1] - arrival[0]
            grad[-1] = arrival[-1] - arrival[-2]
        elif len(arrival) == 2:
            grad[:] = arrival[1] - arrival[0]
        else:
            grad[:] = np.nan
        g['arrival_gradient_s_per_channel'] = grad
        stability_list.append(g)

    df = pd.concat(stability_list, ignore_index=True)
    return df



def smooth_valid_curve(
    channels: np.ndarray,
    values: np.ndarray,
    valid_mask: np.ndarray,
    max_gap_channels: int = 25,
    median_window: int = 21,
) -> np.ndarray:
    smoothed = np.full_like(values, np.nan, dtype=float)
    if valid_mask.sum() < 3:
        return smoothed

    x_valid = channels[valid_mask]
    y_valid = values[valid_mask]
    y_interp = np.interp(channels, x_valid, y_valid)

    half = median_window // 2
    y_med = np.full_like(y_interp, np.nan, dtype=float)
    for i in range(len(y_interp)):
        i0 = max(0, i - half)
        i1 = min(len(y_interp), i + half + 1)
        y_med[i] = np.median(y_interp[i0:i1])

    nearest_dist = np.min(np.abs(channels[:, None] - x_valid[None, :]), axis=1)
    smoothed[nearest_dist <= max_gap_channels] = y_med[nearest_dist <= max_gap_channels]
    return smoothed



def plot_arrival_vs_channel(df: pd.DataFrame, outdir: Path, location_name: str) -> None:
    for anchor_index, grp in df.groupby('anchor_index', sort=True):
        anchor_label = grp['anchor_label'].iloc[0]
        ref_channel = int(grp['reference_channel'].iloc[0])

        valid = (
            grp['passed_snr_threshold']
            & (~grp['near_window_edge'])
            & np.isfinite(grp['snr_like'])
        )

        x = grp['channel'].to_numpy()
        y = grp['relative_to_reference_ms'].to_numpy()
        y_smooth = smooth_valid_curve(x, grp['relative_to_reference_s'].to_numpy(), valid.to_numpy())

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(x[~valid], y[~valid], s=8, alpha=0.35, label='Lower-confidence / flagged')
        ax.scatter(x[valid], y[valid], s=10, label='Valid detections')
        ax.axvline(ref_channel, linestyle='--', linewidth=1.0, label=f'Reference ch {ref_channel}')
        if np.isfinite(y_smooth).any():
            ax.plot(x, 1000.0 * y_smooth, linewidth=2.0, label='Smoothed valid curve')
        ax.axhline(0.0, linestyle=':', linewidth=1.0)
        ax.set_title(f'{location_name} | {anchor_label} | arrival offset relative to reference')
        ax.set_xlabel('Channel')
        ax.set_ylabel('Arrival offset relative to reference (ms)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        fig.savefig(outdir / f'{location_name}_anchor{anchor_index}_{anchor_label}_arrival_vs_channel.png', dpi=220)
        plt.close(fig)



def plot_confidence_vs_channel(df: pd.DataFrame, outdir: Path, location_name: str) -> None:
    for anchor_index, grp in df.groupby('anchor_index', sort=True):
        anchor_label = grp['anchor_label'].iloc[0]
        valid = grp['passed_snr_threshold'] & (~grp['near_window_edge'])

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(grp['channel'], grp['snr_like'], s=10, alpha=0.9, label='snr_like')
        bad = ~valid
        if bad.any():
            ax.scatter(grp.loc[bad, 'channel'], grp.loc[bad, 'snr_like'], s=16, marker='x', label='Flagged')
        ax.set_title(f'{location_name} | {anchor_label} | confidence metric by channel')
        ax.set_xlabel('Channel')
        ax.set_ylabel('snr_like = (peak - median(env)) / MAD(env)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        fig.savefig(outdir / f'{location_name}_anchor{anchor_index}_{anchor_label}_snr_like_vs_channel.png', dpi=220)
        plt.close(fig)



def plot_prominence_vs_channel(df: pd.DataFrame, outdir: Path, location_name: str) -> None:
    for anchor_index, grp in df.groupby('anchor_index', sort=True):
        anchor_label = grp['anchor_label'].iloc[0]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(grp['channel'], grp['peak_prominence_raw'], s=10, label='Peak prominence')
        ax.set_title(f'{location_name} | {anchor_label} | raw peak prominence by channel')
        ax.set_xlabel('Channel')
        ax.set_ylabel('Peak prominence of Hilbert envelope')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        fig.savefig(outdir / f'{location_name}_anchor{anchor_index}_{anchor_label}_prominence_vs_channel.png', dpi=220)
        plt.close(fig)



def plot_heatmaps(df: pd.DataFrame, outdir: Path, location_name: str) -> None:
    anchors = sorted(df['anchor_index'].unique())
    channels = np.sort(df['channel'].unique())

    arrival_mat = np.full((len(anchors), len(channels)), np.nan, dtype=float)
    snr_mat = np.full((len(anchors), len(channels)), np.nan, dtype=float)
    stable_mat = np.full((len(anchors), len(channels)), np.nan, dtype=float)

    a_to_i = {a: i for i, a in enumerate(anchors)}
    c_to_i = {c: i for i, c in enumerate(channels)}

    for _, row in df.iterrows():
        i = a_to_i[row['anchor_index']]
        j = c_to_i[row['channel']]
        arrival_mat[i, j] = row['relative_to_reference_ms']
        snr_mat[i, j] = row['snr_like']
        stable_mat[i, j] = abs(row['arrival_gradient_s_per_channel']) * 1000.0 if pd.notna(row['arrival_gradient_s_per_channel']) else np.nan

    extent = [channels.min(), channels.max(), anchors[-1] + 0.5, anchors[0] - 0.5]

    fig, ax = plt.subplots(figsize=(13, 4.8))
    im = ax.imshow(arrival_mat, aspect='auto', interpolation='nearest', extent=extent)
    ax.set_title(f'{location_name} | arrival offset heatmap (ms)')
    ax.set_xlabel('Channel')
    ax.set_ylabel('Anchor index')
    fig.colorbar(im, ax=ax, label='Arrival offset relative to reference (ms)')
    plt.tight_layout()
    fig.savefig(outdir / f'{location_name}_arrival_offset_heatmap.png', dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 4.8))
    im = ax.imshow(snr_mat, aspect='auto', interpolation='nearest', extent=extent)
    ax.set_title(f'{location_name} | confidence heatmap')
    ax.set_xlabel('Channel')
    ax.set_ylabel('Anchor index')
    fig.colorbar(im, ax=ax, label='snr_like')
    plt.tight_layout()
    fig.savefig(outdir / f'{location_name}_snr_like_heatmap.png', dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 4.8))
    im = ax.imshow(stable_mat, aspect='auto', interpolation='nearest', extent=extent)
    ax.set_title(f'{location_name} | local roughness heatmap')
    ax.set_xlabel('Channel')
    ax.set_ylabel('Anchor index')
    fig.colorbar(im, ax=ax, label='|d(arrival)/d(channel)| (ms/channel)')
    plt.tight_layout()
    fig.savefig(outdir / f'{location_name}_roughness_heatmap.png', dpi=220)
    plt.close(fig)



def plot_stability_mask(df: pd.DataFrame, outdir: Path, location_name: str, roughness_threshold_ms_per_channel: float) -> None:
    for anchor_index, grp in df.groupby('anchor_index', sort=True):
        anchor_label = grp['anchor_label'].iloc[0]
        roughness = np.abs(grp['arrival_gradient_s_per_channel']) * 1000.0
        valid = grp['passed_snr_threshold'] & (~grp['near_window_edge'])
        stable = valid & (roughness <= roughness_threshold_ms_per_channel)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(grp['channel'], grp['relative_to_reference_ms'], s=8, alpha=0.25, label='All channels')
        ax.scatter(grp.loc[valid, 'channel'], grp.loc[valid, 'relative_to_reference_ms'], s=10, label='Valid')
        ax.scatter(grp.loc[stable, 'channel'], grp.loc[stable, 'relative_to_reference_ms'], s=12, label='Stable subset')
        ax.set_title(f'{location_name} | {anchor_label} | stable-region view')
        ax.set_xlabel('Channel')
        ax.set_ylabel('Arrival offset relative to reference (ms)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        fig.savefig(outdir / f'{location_name}_anchor{anchor_index}_{anchor_label}_stable_regions.png', dpi=220)
        plt.close(fig)



def save_summary(df: pd.DataFrame, outdir: Path, location_name: str, roughness_threshold_ms_per_channel: float) -> None:
    rows = []
    for anchor_index, grp in df.groupby('anchor_index', sort=True):
        valid = grp['passed_snr_threshold'] & (~grp['near_window_edge'])
        roughness = np.abs(grp['arrival_gradient_s_per_channel']) * 1000.0
        stable = valid & (roughness <= roughness_threshold_ms_per_channel)
        rows.append({
            'location': location_name,
            'anchor_index': int(anchor_index),
            'anchor_label': grp['anchor_label'].iloc[0],
            'n_channels': int(len(grp)),
            'n_valid': int(valid.sum()),
            'valid_fraction': float(valid.mean()),
            'n_stable': int(stable.sum()),
            'stable_fraction': float(stable.mean()),
            'median_snr_like': float(np.nanmedian(grp['snr_like'])),
            'median_abs_roughness_ms_per_channel': float(np.nanmedian(roughness)),
        })
    pd.DataFrame(rows).to_csv(outdir / f'{location_name}_summary.csv', index=False)



def main() -> None:
    parser = argparse.ArgumentParser(description='Plot diagnostics for bulk DAS matched-filter CSV results.')
    parser.add_argument('--csv', type=Path, required=True, help='Per-location CSV result file.')
    parser.add_argument('--outdir', type=Path, default=None, help='Output directory for plots.')
    parser.add_argument('--roughness-threshold-ms-per-channel', type=float, default=0.20)
    args = parser.parse_args()

    csv_path = args.csv
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = load_results(csv_path)
    location_name = str(df['location'].iloc[0])

    outdir = args.outdir if args.outdir is not None else csv_path.with_suffix('')
    ensure_dir(outdir)

    plot_arrival_vs_channel(df, outdir, location_name)
    plot_confidence_vs_channel(df, outdir, location_name)
    plot_prominence_vs_channel(df, outdir, location_name)
    plot_heatmaps(df, outdir, location_name)
    plot_stability_mask(df, outdir, location_name, args.roughness_threshold_ms_per_channel)
    save_summary(df, outdir, location_name, args.roughness_threshold_ms_per_channel)

    print(f'Saved plots to: {outdir}')


if __name__ == '__main__':
    main()
