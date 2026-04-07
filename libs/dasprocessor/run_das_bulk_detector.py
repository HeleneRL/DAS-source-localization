from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.signal import chirp, correlate, hilbert, peak_prominences

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


# ============================================================
# Config helpers
# ============================================================


def load_toml(path: Path) -> dict[str, Any]:
    with path.open('rb') as f:
        return tomllib.load(f)


# ============================================================
# HDF5 sequence reader
# ============================================================


@dataclass
class FileSpan:
    path: Path
    start_sample: int
    stop_sample: int
    n_samples: int
    start_utc: dt.datetime


class HDF5Sequence:
    def __init__(self, folder: Path):
        self.folder = folder
        self.filepaths = sorted(folder.glob('*.hdf5'))
        if not self.filepaths:
            raise FileNotFoundError(f'No .hdf5 files found in {folder}')

        self._handles: list[h5py.File] = []
        self._datasets = []
        self.file_spans: list[FileSpan] = []

        self.fs: float | None = None
        self.dx: float | None = None
        self.n_channels: int | None = None
        self.global_start_utc: dt.datetime | None = None

        running = 0
        for fp in self.filepaths:
            f = h5py.File(fp, 'r')
            self._handles.append(f)
            data = f['data']
            self._datasets.append(data)
            n_samples, n_channels = data.shape

            header = f['header']
            dt_s = float(header['dt'][()])
            fs = 1.0 / dt_s
            dx = float(header['dx'][()]) if 'dx' in header else float('nan')
            start_unix = float(header['time'][()])
            start_utc = dt.datetime.utcfromtimestamp(start_unix)

            if self.fs is None:
                self.fs = fs
                self.dx = dx
                self.n_channels = n_channels
                self.global_start_utc = start_utc
            else:
                if not np.isclose(fs, self.fs):
                    raise ValueError(f'Sampling rate mismatch in {fp}: {fs} vs {self.fs}')
                if n_channels != self.n_channels:
                    raise ValueError(f'n_channels mismatch in {fp}: {n_channels} vs {self.n_channels}')
                if not np.isclose(dx, self.dx, equal_nan=True):
                    raise ValueError(f'dx mismatch in {fp}: {dx} vs {self.dx}')

            span = FileSpan(
                path=fp,
                start_sample=running,
                stop_sample=running + n_samples,
                n_samples=n_samples,
                start_utc=start_utc,
            )
            self.file_spans.append(span)
            running += n_samples

        assert self.fs is not None
        assert self.n_channels is not None
        assert self.global_start_utc is not None
        self.total_samples = running
        self.duration_s = self.total_samples / self.fs

    def close(self) -> None:
        for f in self._handles:
            try:
                f.close()
            except Exception:
                pass
        self._handles.clear()
        self._datasets.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def read_block(
        self,
        channel_start: int,
        channel_stop: int,
        global_start_sample: int,
        global_stop_sample: int,
    ) -> np.ndarray:
        if not (0 <= channel_start < channel_stop <= self.n_channels):
            raise IndexError(
                f'Channel block [{channel_start}, {channel_stop}) outside 0..{self.n_channels}'
            )
        if global_start_sample < 0 or global_stop_sample > self.total_samples:
            raise ValueError(
                f'Sample window [{global_start_sample}, {global_stop_sample}) '
                f'outside 0..{self.total_samples}'
            )
        if global_stop_sample <= global_start_sample:
            raise ValueError('global_stop_sample must be > global_start_sample')

        out_len = global_stop_sample - global_start_sample
        out = np.empty((out_len, channel_stop - channel_start), dtype=np.float32)
        cursor = 0

        for ds, span in zip(self._datasets, self.file_spans):
            overlap_start = max(global_start_sample, span.start_sample)
            overlap_stop = min(global_stop_sample, span.stop_sample)
            if overlap_stop <= overlap_start:
                continue

            local_start = overlap_start - span.start_sample
            local_stop = overlap_stop - span.start_sample
            chunk = ds[local_start:local_stop, channel_start:channel_stop]
            chunk = np.asarray(chunk, dtype=np.float32)
            n = chunk.shape[0]
            out[cursor:cursor + n, :] = chunk
            cursor += n

        if cursor != out_len:
            raise RuntimeError(f'Internal read error: expected {out_len} samples, got {cursor}')

        return out


# ============================================================
# Signal helpers
# ============================================================


def make_lfm_reference(fs: float, f0: float, f1: float, duration_s: float, window: str) -> np.ndarray:
    n = int(round(duration_s * fs))
    t = np.arange(n, dtype=np.float64) / fs
    ref = chirp(t, f0=f0, f1=f1, t1=duration_s, method='linear')
    if window.lower() == 'hann':
        ref *= np.hanning(n)
    elif window.lower() not in ('none', 'rect', 'rectangular'):
        raise ValueError(f'Unsupported reference_window: {window}')
    ref -= np.mean(ref)
    ref /= np.linalg.norm(ref) + 1e-12
    return ref.astype(np.float32)


@dataclass
class PeakResult:
    peak_local_index: int
    peak_global_sample: int
    peak_time_s: float
    peak_time_utc: str
    peak_raw: float
    prominence_raw: float
    baseline_median: float
    baseline_mad: float
    snr_like: float
    passed_snr_threshold: bool
    near_window_edge: bool



def detect_best_peak(
    x: np.ndarray,
    ref: np.ndarray,
    fs: float,
    raw_start_sample: int,
    warn_if_peak_within_s: float,
    peak_threshold_snr: float,
    correlation_method: str,
) -> PeakResult:
    x = np.asarray(x, dtype=np.float32)
    x = x - np.mean(x)

    xc = correlate(x, ref, mode='valid', method=correlation_method)
    env = np.abs(hilbert(xc))

    peak_idx = int(np.argmax(env))
    peak_raw = float(env[peak_idx])

    prominences = peak_prominences(env, np.array([peak_idx], dtype=int))[0]
    prominence_raw = float(prominences[0]) if len(prominences) else float('nan')

    baseline_median = float(np.median(env))
    mad = float(np.median(np.abs(env - baseline_median)))
    baseline_mad = mad
    snr_like = float((peak_raw - baseline_median) / (mad + 1e-12))

    peak_global_sample = raw_start_sample + peak_idx
    peak_time_s = peak_global_sample / fs
    peak_time_utc = peak_time_s  # temporary placeholder; overwritten by caller

    edge_guard = int(round(warn_if_peak_within_s * fs))
    near_window_edge = bool(peak_idx < edge_guard or peak_idx >= len(env) - edge_guard)

    return PeakResult(
        peak_local_index=peak_idx,
        peak_global_sample=peak_global_sample,
        peak_time_s=peak_time_s,
        peak_time_utc=str(peak_time_utc),
        peak_raw=peak_raw,
        prominence_raw=prominence_raw,
        baseline_median=baseline_median,
        baseline_mad=baseline_mad,
        snr_like=snr_like,
        passed_snr_threshold=bool(snr_like >= peak_threshold_snr),
        near_window_edge=near_window_edge,
    )


# ============================================================
# Main processing
# ============================================================


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



def process_location(
    cfg: dict[str, Any],
    location_name: str,
    combined_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    general = cfg['general']
    acquisition = cfg['acquisition']
    signal_cfg = cfg['signal']
    search_cfg = cfg['search']
    loc_cfg = cfg['locations'][location_name]

    data_root = Path(general['data_root'])
    folder = data_root / loc_cfg['folder']
    if not folder.exists():
        raise FileNotFoundError(f'Folder does not exist: {folder}')

    output_root = Path(general['output_root'])
    ensure_dir(output_root)

    channel_start = int(acquisition.get('channel_start', 0))
    channel_stop = int(acquisition['max_channel_exclusive'])
    block_size = int(acquisition['channel_block_size'])

    search_before_s = float(search_cfg['search_before_s'])
    search_after_s = float(search_cfg['search_after_s'])
    peak_threshold_snr = float(search_cfg['peak_threshold_snr'])
    warn_if_peak_within_s = float(search_cfg['warn_if_peak_within_s'])
    correlation_method = str(search_cfg.get('correlation_method', 'fft'))

    ref = None

    rows: list[dict[str, Any]] = []

    with HDF5Sequence(folder) as seq:
        fs = float(seq.fs)
        dx = float(seq.dx)
        n_channels = int(seq.n_channels)

        if ref is None:
            ref = make_lfm_reference(
                fs=fs,
                f0=float(signal_cfg['lfm_f0_hz']),
                f1=float(signal_cfg['lfm_f1_hz']),
                duration_s=float(signal_cfg['lfm_duration_s']),
                window=str(signal_cfg['reference_window']),
            )

        if not np.isclose(fs, float(signal_cfg['fs_nominal_hz'])):
            print(
                f'[warn] {location_name}: fs={fs:.6f} differs from nominal '
                f"{float(signal_cfg['fs_nominal_hz']):.6f}"
            )

        if channel_stop > n_channels:
            raise ValueError(
                f'{location_name}: requested max_channel_exclusive={channel_stop}, '
                f'but file has only {n_channels} channels'
            )

        print(f'\n=== Processing {location_name} ===')
        print(f'Folder                : {folder}')
        print(f'Files                 : {len(seq.filepaths)}')
        print(f'fs                    : {fs:.3f} Hz')
        print(f'dx                    : {dx:.6f} m')
        print(f'Channels processed    : [{channel_start}, {channel_stop})')
        print(f'Duration              : {seq.duration_s:.3f} s')
        print(f'Global start UTC      : {seq.global_start_utc}')

        ref_len = len(ref)
        search_before_n = int(round(search_before_s * fs))
        search_after_n = int(round(search_after_s * fs))

        for anchor_idx, anchor_time_s in enumerate(loc_cfg['anchor_times_s']):
            anchor_label = loc_cfg['anchor_labels'][anchor_idx]
            anchor_sample = int(round(float(anchor_time_s) * fs))

            raw_start = anchor_sample - search_before_n
            raw_stop = anchor_sample + search_after_n + ref_len

            if raw_start < 0 or raw_stop > seq.total_samples:
                raise ValueError(
                    f'{location_name}/{anchor_label}: raw window '
                    f'[{raw_start}, {raw_stop}) outside 0..{seq.total_samples}. '
                    f'Increase recording span or reduce search window.'
                )

            print(
                f'  Anchor {anchor_idx+1}: {anchor_label}, '
                f'anchor_time_s={anchor_time_s:.6f}, raw_window_s='
                f'[{raw_start/fs:.3f}, {raw_stop/fs:.3f})'
            )

            for ch0 in range(channel_start, channel_stop, block_size):
                ch1 = min(ch0 + block_size, channel_stop)
                block = seq.read_block(ch0, ch1, raw_start, raw_stop)

                for j, ch in enumerate(range(ch0, ch1)):
                    peak = detect_best_peak(
                        x=block[:, j],
                        ref=ref,
                        fs=fs,
                        raw_start_sample=raw_start,
                        warn_if_peak_within_s=warn_if_peak_within_s,
                        peak_threshold_snr=peak_threshold_snr,
                        correlation_method=correlation_method,
                    )

                    peak_time_utc = seq.global_start_utc + dt.timedelta(seconds=peak.peak_time_s)
                    peak_time_local = peak_time_utc + dt.timedelta(hours=8)

                    row = {
                        'location': location_name,
                        'folder': str(folder),
                        'reference_channel': int(loc_cfg['reference_channel']),
                        'anchor_index': anchor_idx + 1,
                        'anchor_label': anchor_label,
                        'anchor_time_s_from_sequence_start': float(anchor_time_s),
                        'search_before_s': search_before_s,
                        'search_after_s': search_after_s,
                        'channel': ch,
                        'distance_m_if_dx_valid': float(ch * dx) if np.isfinite(dx) else math.nan,
                        'peak_global_sample': peak.peak_global_sample,
                        'peak_time_s_from_sequence_start': peak.peak_time_s,
                        'peak_time_utc': peak_time_utc.isoformat(sep=' '),
                        'peak_time_local_sg': peak_time_local.isoformat(sep=' '),
                        'peak_local_index_within_search': peak.peak_local_index,
                        'peak_raw_envelope': peak.peak_raw,
                        'peak_prominence_raw': peak.prominence_raw,
                        'baseline_median': peak.baseline_median,
                        'baseline_mad': peak.baseline_mad,
                        'snr_like': peak.snr_like,
                        'passed_snr_threshold': peak.passed_snr_threshold,
                        'near_window_edge': peak.near_window_edge,
                    }
                    rows.append(row)
                    combined_rows.append(row)

                print(f'    processed channels [{ch0}, {ch1})')

    return rows


# ============================================================
# Output helpers
# ============================================================


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        raise ValueError(f'No rows to write for {path}')

    fieldnames = list(rows[0].keys())
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# CLI
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(description='Bulk matched-filter detector for Singapore DAS LFM sweeps.')
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('singapore_das_config.toml'),
        help='Path to TOML config file.',
    )
    args = parser.parse_args()

    cfg = load_toml(args.config)

    output_root = Path(cfg['general']['output_root'])
    ensure_dir(output_root)

    combined_rows: list[dict[str, Any]] = []
    per_location_counts: dict[str, int] = {}

    for location_name in cfg['general']['locations_to_process']:
        rows = process_location(cfg, location_name, combined_rows)
        per_location_counts[location_name] = len(rows)

        if cfg['outputs'].get('save_per_location_csv', True):
            out_csv = output_root / f'{location_name}_bulk_lfm35_45_results.csv'
            write_csv(out_csv, rows)
            print(f'[saved] {out_csv}')

    if cfg['outputs'].get('save_combined_csv', True):
        combined_csv = output_root / 'all_locations_bulk_lfm35_45_results.csv'
        write_csv(combined_csv, combined_rows)
        print(f'[saved] {combined_csv}')

    if cfg['outputs'].get('save_run_metadata_json', True):
        metadata = {
            'created_utc': dt.datetime.utcnow().isoformat(sep=' ') + 'Z',
            'config_path': str(args.config.resolve()),
            'per_location_counts': per_location_counts,
            'total_rows': len(combined_rows),
        }
        metadata_path = output_root / 'run_metadata.json'
        with metadata_path.open('w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        print(f'[saved] {metadata_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
