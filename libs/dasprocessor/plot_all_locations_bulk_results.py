from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def rolling_nanmedian(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    if window % 2 == 0:
        window += 1
    half = window // 2
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(len(x)):
        i0 = max(0, i - half)
        i1 = min(len(x), i + half + 1)
        out[i] = np.nanmedian(x[i0:i1])
    return out


def rolling_nanmean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    if window % 2 == 0:
        window += 1
    half = window // 2
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(len(x)):
        i0 = max(0, i - half)
        i1 = min(len(x), i + half + 1)
        out[i] = np.nanmean(x[i0:i1])
    return out


def fill_small_gaps(channels: np.ndarray, values: np.ndarray, valid_mask: np.ndarray, max_gap_channels: int) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=float)
    if valid_mask.sum() < 2:
        return out

    x_valid = channels[valid_mask]
    y_valid = values[valid_mask]
    y_interp = np.interp(channels, x_valid, y_valid)

    nearest_dist = np.min(np.abs(channels[:, None] - x_valid[None, :]), axis=1)
    out[nearest_dist <= max_gap_channels] = y_interp[nearest_dist <= max_gap_channels]
    return out


def aggressive_smooth_curve(
    channels: np.ndarray,
    values_s: np.ndarray,
    valid_mask: np.ndarray,
    max_gap_channels: int = 40,
    median_window: int = 61,
    mean_window: int = 101,
) -> np.ndarray:
    interp = fill_small_gaps(channels, values_s, valid_mask, max_gap_channels=max_gap_channels)
    if not np.isfinite(interp).any():
        return interp

    med = rolling_nanmedian(interp, median_window)
    smooth = rolling_nanmean(med, mean_window)

    x_valid = channels[np.isfinite(interp)]
    if len(x_valid) == 0:
        return np.full_like(values_s, np.nan, dtype=float)
    nearest_dist = np.min(np.abs(channels[:, None] - x_valid[None, :]), axis=1)
    smooth[nearest_dist > max_gap_channels] = np.nan
    return smooth


def contiguous_ranges(mask: np.ndarray, channels: np.ndarray) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = None
    prev_ch = None
    for keep, ch in zip(mask, channels):
        if keep and start is None:
            start = int(ch)
        if not keep and start is not None:
            ranges.append((start, int(prev_ch)))
            start = None
        prev_ch = ch
    if start is not None and prev_ch is not None:
        ranges.append((start, int(prev_ch)))
    return ranges


def load_results(csv_path: Path, channel_min: int, channel_max: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns in {csv_path}: {sorted(missing)}')

    df['passed_snr_threshold'] = df['passed_snr_threshold'].astype(str).str.lower().isin(['true', '1', 'yes'])
    df['near_window_edge'] = df['near_window_edge'].astype(str).str.lower().isin(['true', '1', 'yes'])

    df = df[(df['channel'] >= channel_min) & (df['channel'] <= channel_max)].copy()
    if df.empty:
        raise ValueError(f'No rows left after filtering to channels {channel_min}..{channel_max}')

    df['relative_offset_s'] = (
        df['peak_time_s_from_sequence_start'] - df['anchor_time_s_from_sequence_start']
    )

    ref_by_trace = (
        df.loc[df['channel'] == df['reference_channel'], ['location', 'anchor_index', 'peak_time_s_from_sequence_start']]
        .drop_duplicates(subset=['location', 'anchor_index'])
        .rename(columns={'peak_time_s_from_sequence_start': 'reference_peak_time_s'})
    )
    if ref_by_trace.empty:
        raise ValueError('Could not find reference-channel rows in CSV.')

    df = df.merge(ref_by_trace, on=['location', 'anchor_index'], how='left')
    df['relative_to_reference_s'] = df['peak_time_s_from_sequence_start'] - df['reference_peak_time_s']
    df['relative_to_reference_ms'] = 1000.0 * df['relative_to_reference_s']

    df = df.sort_values(['location', 'anchor_index', 'channel']).reset_index(drop=True)
    rows = []
    for (location, anchor_index), grp in df.groupby(['location', 'anchor_index'], sort=True):
        g = grp.copy().sort_values('channel')
        arrival = g['relative_to_reference_s'].to_numpy()
        grad = np.full(len(g), np.nan, dtype=float)
        if len(g) >= 3:
            grad[1:-1] = 0.5 * (arrival[2:] - arrival[:-2])
            grad[0] = arrival[1] - arrival[0]
            grad[-1] = arrival[-1] - arrival[-2]
        elif len(g) == 2:
            grad[:] = arrival[1] - arrival[0]
        g['arrival_gradient_s_per_channel'] = grad

        valid = (
            g['passed_snr_threshold'].to_numpy()
            & (~g['near_window_edge'].to_numpy())
            & np.isfinite(g['snr_like'].to_numpy())
        )
        smooth = aggressive_smooth_curve(
            channels=g['channel'].to_numpy(),
            values_s=g['relative_to_reference_s'].to_numpy(),
            valid_mask=valid,
        )
        g['smoothed_relative_to_reference_s'] = smooth
        g['smoothed_relative_to_reference_ms'] = 1000.0 * smooth
        residual_ms = g['relative_to_reference_ms'].to_numpy() - g['smoothed_relative_to_reference_ms'].to_numpy()
        g['residual_to_smooth_ms'] = residual_ms
        rows.append(g)

    return pd.concat(rows, ignore_index=True)


def make_location_channel_summary(df: pd.DataFrame, residual_threshold_ms: float) -> pd.DataFrame:
    rows = []
    for (location, channel), grp in df.groupby(['location', 'channel'], sort=True):
        base_valid = grp['passed_snr_threshold'] & (~grp['near_window_edge']) & np.isfinite(grp['snr_like'])
        residual_ok = np.abs(grp['residual_to_smooth_ms']) <= residual_threshold_ms
        stable = base_valid & residual_ok.fillna(False)
        rows.append({
            'location': location,
            'channel': int(channel),
            'n_traces': int(len(grp)),
            'valid_fraction': float(base_valid.mean()),
            'stable_fraction': float(stable.mean()),
            'median_snr_like': float(np.nanmedian(grp['snr_like'])),
            'median_prominence': float(np.nanmedian(grp['peak_prominence_raw'])),
            'median_arrival_ms': float(np.nanmedian(grp['relative_to_reference_ms'])),
            'median_smoothed_arrival_ms': float(np.nanmedian(grp['smoothed_relative_to_reference_ms'])),
            'median_abs_residual_ms': float(np.nanmedian(np.abs(grp['residual_to_smooth_ms']))),
        })
    return pd.DataFrame(rows)


def make_overall_channel_summary(locch: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for channel, grp in locch.groupby('channel', sort=True):
        rows.append({
            'channel': int(channel),
            'n_locations': int(len(grp)),
            'mean_valid_fraction': float(np.nanmean(grp['valid_fraction'])),
            'mean_stable_fraction': float(np.nanmean(grp['stable_fraction'])),
            'median_of_location_snr': float(np.nanmedian(grp['median_snr_like'])),
            'median_of_location_abs_residual_ms': float(np.nanmedian(grp['median_abs_residual_ms'])),
            'n_locations_poor': int((grp['stable_fraction'] < 0.5).sum()),
            'n_locations_good': int((grp['stable_fraction'] >= 0.5).sum()),
        })
    return pd.DataFrame(rows)


def heatmap_from_summary(summary: pd.DataFrame, value_col: str, locations: list[str], channels: np.ndarray) -> np.ndarray:
    loc_to_i = {loc: i for i, loc in enumerate(locations)}
    ch_to_j = {int(ch): j for j, ch in enumerate(channels)}
    mat = np.full((len(locations), len(channels)), np.nan, dtype=float)
    for _, row in summary.iterrows():
        mat[loc_to_i[row['location']], ch_to_j[int(row['channel'])]] = row[value_col]
    return mat


def plot_heatmap(mat: np.ndarray, locations: list[str], channels: np.ndarray, title: str, cbar_label: str, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 5.8))
    extent = [channels.min(), channels.max(), len(locations) - 0.5, -0.5]
    im = ax.imshow(mat, aspect='auto', interpolation='nearest', extent=extent)
    ax.set_yticks(range(len(locations)))
    ax.set_yticklabels(locations)
    ax.set_xlabel('Channel')
    ax.set_ylabel('Location')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label)
    plt.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_channel_quality_overall(overall: pd.DataFrame, outpath: Path) -> None:
    ch = overall['channel'].to_numpy()

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    axes[0].plot(ch, overall['mean_stable_fraction'])
    axes[0].set_ylabel('Mean stable fraction')
    axes[0].set_title('Overall channel quality across all locations')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ch, overall['median_of_location_snr'])
    axes[1].set_ylabel('Median location SNR-like')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(ch, overall['n_locations_poor'])
    axes[2].set_ylabel('# poor locations')
    axes[2].set_xlabel('Channel')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_stacked_smoothed_curves(df: pd.DataFrame, locations: list[str], outpath: Path) -> None:
    n = len(locations)
    fig, axes = plt.subplots(n, 1, figsize=(15, max(3.0 * n, 8.0)), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, location in zip(axes, locations):
        loc = df[df['location'] == location].copy()
        for anchor_index, grp in loc.groupby('anchor_index', sort=True):
            ax.plot(
                grp['channel'],
                grp['relative_to_reference_ms'],
                linewidth=0.8,
                alpha=0.18,
            )
            ax.plot(
                grp['channel'],
                grp['smoothed_relative_to_reference_ms'],
                linewidth=2.0,
                label=f"anchor {int(anchor_index)} smooth",
            )

        loc_med = (
            loc.groupby('channel', sort=True)['smoothed_relative_to_reference_ms']
            .median()
            .reset_index()
        )
        ax.plot(loc_med['channel'], loc_med['smoothed_relative_to_reference_ms'], linewidth=3.0, linestyle='--', label='location median smooth')
        ax.axhline(0.0, linestyle=':', linewidth=1.0)
        ax.set_ylabel('Offset (ms)')
        ax.set_title(location)
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=3, fontsize=8)

    axes[-1].set_xlabel('Channel')
    fig.suptitle('Aggressively smoothed arrival curves by location', y=0.995)
    plt.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_overlay_location_medians(df: pd.DataFrame, locations: list[str], outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 6.5))
    for location in locations:
        loc = df[df['location'] == location].copy()
        loc_med = (
            loc.groupby('channel', sort=True)['smoothed_relative_to_reference_ms']
            .median()
            .reset_index()
        )
        ax.plot(loc_med['channel'], loc_med['smoothed_relative_to_reference_ms'], linewidth=2.0, label=location)
    ax.axhline(0.0, linestyle=':', linewidth=1.0)
    ax.set_title('Median smoothed arrival curve by location')
    ax.set_xlabel('Channel')
    ax.set_ylabel('Arrival offset relative to reference (ms)')
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3)
    plt.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_residual_heatmap(df: pd.DataFrame, locations: list[str], channels: np.ndarray, outpath: Path) -> None:
    rows = []
    for (location, channel), grp in df.groupby(['location', 'channel'], sort=True):
        rows.append({
            'location': location,
            'channel': int(channel),
            'median_abs_residual_ms': float(np.nanmedian(np.abs(grp['residual_to_smooth_ms']))),
        })
    tmp = pd.DataFrame(rows)
    mat = heatmap_from_summary(tmp, 'median_abs_residual_ms', locations, channels)
    plot_heatmap(
        mat,
        locations,
        channels,
        title='Median absolute residual to aggressive smooth curve',
        cbar_label='Median |residual| (ms)',
        outpath=outpath,
    )


def save_summaries(locch: pd.DataFrame, overall: pd.DataFrame, outdir: Path, poor_channel_threshold: float) -> None:
    locch.to_csv(outdir / 'all_locations_location_channel_summary.csv', index=False)
    overall.to_csv(outdir / 'all_locations_overall_channel_summary.csv', index=False)

    poor_mask = overall['mean_stable_fraction'].to_numpy() < poor_channel_threshold
    channels = overall['channel'].to_numpy()
    ranges = contiguous_ranges(poor_mask, channels)
    pd.DataFrame(ranges, columns=['channel_start', 'channel_end']).to_csv(
        outdir / 'all_locations_poor_channel_ranges.csv', index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare bulk DAS matched-filter results across all locations.')
    parser.add_argument('--csv', type=Path, required=True, help='Combined all-locations CSV file.')
    parser.add_argument('--outdir', type=Path, default=None, help='Output directory for plots.')
    parser.add_argument('--channel-min', type=int, default=348)
    parser.add_argument('--channel-max', type=int, default=2267)
    parser.add_argument('--residual-threshold-ms', type=float, default=80.0, help='Residual-to-smooth threshold for stable_fraction summary.')
    parser.add_argument('--poor-channel-threshold', type=float, default=0.50, help='Channels below this mean stable fraction are marked as poor overall.')
    args = parser.parse_args()

    csv_path = args.csv
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    outdir = args.outdir if args.outdir is not None else csv_path.with_suffix('')
    ensure_dir(outdir)

    df = load_results(csv_path, channel_min=args.channel_min, channel_max=args.channel_max)
    locations = sorted(df['location'].unique())
    channels = np.sort(df['channel'].unique())

    locch = make_location_channel_summary(df, residual_threshold_ms=args.residual_threshold_ms)
    overall = make_overall_channel_summary(locch)

    valid_mat = heatmap_from_summary(locch, 'valid_fraction', locations, channels)
    stable_mat = heatmap_from_summary(locch, 'stable_fraction', locations, channels)
    snr_mat = heatmap_from_summary(locch, 'median_snr_like', locations, channels)
    arr_mat = heatmap_from_summary(locch, 'median_smoothed_arrival_ms', locations, channels)

    plot_heatmap(
        valid_mat,
        locations,
        channels,
        title='Valid-detection fraction by location and channel',
        cbar_label='Fraction of anchors passing base validity',
        outpath=outdir / 'all_locations_valid_fraction_heatmap.png',
    )
    plot_heatmap(
        stable_mat,
        locations,
        channels,
        title='Stable-detection fraction by location and channel',
        cbar_label='Fraction of anchors valid and close to smooth curve',
        outpath=outdir / 'all_locations_stable_fraction_heatmap.png',
    )
    plot_heatmap(
        snr_mat,
        locations,
        channels,
        title='Median SNR-like confidence by location and channel',
        cbar_label='Median snr_like',
        outpath=outdir / 'all_locations_median_snr_heatmap.png',
    )
    plot_heatmap(
        arr_mat,
        locations,
        channels,
        title='Median aggressively smoothed arrival offset by location and channel',
        cbar_label='Arrival offset relative to reference (ms)',
        outpath=outdir / 'all_locations_smoothed_arrival_heatmap.png',
    )

    plot_channel_quality_overall(overall, outdir / 'all_locations_overall_channel_quality.png')
    plot_stacked_smoothed_curves(df, locations, outdir / 'all_locations_stacked_smoothed_curves.png')
    plot_overlay_location_medians(df, locations, outdir / 'all_locations_overlay_location_medians.png')
    plot_residual_heatmap(df, locations, channels, outdir / 'all_locations_residual_heatmap.png')
    save_summaries(locch, overall, outdir, poor_channel_threshold=args.poor_channel_threshold)

    print(f'Saved all-location comparison plots to: {outdir}')


if __name__ == '__main__':
    main()
