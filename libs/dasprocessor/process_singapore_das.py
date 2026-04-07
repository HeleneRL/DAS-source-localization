"""
Process DAS HDF5 files folder-by-folder:
- reads OptoDAS HDF5 files
- extracts metadata (fs, dx, gaugeLength, channels, etc.)
- bandpass filters between 3.5 and 4.5 kHz
- saves one compressed backup per source file
- writes a folder-level metadata summary JSON

Designed to use your existing simpleDASreader8.py codebase.

Example usage:
    python process_singapore_das.py --input "D:\\Singapore Data\\loc1_tx2"
    python process_singapore_das.py --input "D:\\Singapore Data\\loc3_tx1" --output "D:\\Singapore Data\\processed"
    python process_singapore_das.py --root "D:\\Singapore Data"

Notes:
- This processes ONE HDF5 file at a time for memory safety.
- It loads ALL channels in each file.
- Output is saved as compressed .npz for easier storage.
- Later, this can be extended with LFM arrival detection.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt

# Import from your existing package
# Adjust these imports if your package/module layout is different.
from .simpleDASreader8 import load_DAS_files, get_filemeta


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DEFAULT_FMIN = 3500.0
DEFAULT_FMAX = 4500.0
DEFAULT_FILTER_ORDER = 6


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_hdf5_files(folder: Path) -> list[Path]:
    return sorted([p for p in folder.glob("*.hdf5") if p.is_file()])


def make_bandpass_sos(fs: float, fmin: float, fmax: float, order: int = 6):
    nyq = 0.5 * fs
    if not (0 < fmin < fmax < nyq):
        raise ValueError(
            f"Invalid band [{fmin}, {fmax}] for sampling rate fs={fs} Hz "
            f"(Nyquist = {nyq} Hz)"
        )
    return butter(order, [fmin, fmax], btype="bandpass", fs=fs, output="sos")


def sanitize_meta(obj: Any) -> Any:
    """
    Make metadata JSON-safe.
    """
    if isinstance(obj, dict):
        return {str(k): sanitize_meta(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_meta(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, bytes):
        return obj.decode(errors="replace")
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


# ---------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------

def inspect_file_metadata(hdf5_path: Path) -> dict[str, Any]:
    """
    Read metadata using your existing reader.
    """
    meta = get_filemeta(str(hdf5_path), metaDetail=1)

    header = meta.get("header", {})
    timing = meta.get("timing", {})
    cable_spec = meta.get("cableSpec", {})

    dx = header.get("dx", None)
    gauge_length = header.get("gaugeLength", None)
    dt = header.get("dt", None)
    fs = None if dt is None else 1.0 / float(dt)

    channels = header.get("channels", None)
    n_channels = None if channels is None else len(channels)

    out = {
        "file": str(hdf5_path),
        "fs_hz": fs,
        "dt_s": dt,
        "dx_m": dx,
        "gauge_length_m": gauge_length,
        "n_channels": n_channels,
        "first_channel": None if channels is None or len(channels) == 0 else int(channels[0]),
        "last_channel": None if channels is None or len(channels) == 0 else int(channels[-1]),
        "header_name": header.get("name", None),
        "header_unit": header.get("unit", None),
        "experiment": header.get("experiment", None),
        "timing": sanitize_meta(timing),
        "cableSpec": sanitize_meta(cable_spec),
    }
    return out


def summarize_folder_metadata(hdf5_files: list[Path]) -> dict[str, Any]:
    """
    Check whether key metadata are consistent across files in a folder.
    """
    per_file = [inspect_file_metadata(p) for p in hdf5_files]

    def unique_non_none(key: str):
        vals = [x[key] for x in per_file if x.get(key) is not None]
        return sorted(set(vals))

    summary = {
        "n_files": len(per_file),
        "fs_hz_unique": unique_non_none("fs_hz"),
        "dx_m_unique": unique_non_none("dx_m"),
        "gauge_length_m_unique": unique_non_none("gauge_length_m"),
        "n_channels_unique": unique_non_none("n_channels"),
        "files": per_file,
    }

    summary["consistent_fs"] = len(summary["fs_hz_unique"]) <= 1
    summary["consistent_dx"] = len(summary["dx_m_unique"]) <= 1
    summary["consistent_gauge_length"] = len(summary["gauge_length_m_unique"]) <= 1
    summary["consistent_n_channels"] = len(summary["n_channels_unique"]) <= 1

    return summary


# ---------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------

def process_one_file(
    hdf5_path: Path,
    output_folder: Path,
    fmin: float = DEFAULT_FMIN,
    fmax: float = DEFAULT_FMAX,
    filter_order: int = DEFAULT_FILTER_ORDER,
    integrate: bool = True,
    save_dtype: str = "float32",
) -> Path:
    """
    Load one HDF5 file, bandpass filter all channels, save compressed backup.
    """
    print(f"\nReading: {hdf5_path}")

    # Load all channels from one file.
    # Your reader returns a DASDataFrame with .meta attached.
    signal = load_DAS_files(
        str(hdf5_path),
        chIndex=slice(None),
        integrate=integrate,
        unwr=False,
        showProgress=False,
    )

    y = signal.to_numpy()
    meta = signal.meta

    fs = 1.0 / float(meta["dt"])
    dx = float(meta.get("dx", np.nan))
    gauge_length = float(meta.get("gaugeLength", np.nan))
    channels = np.asarray(signal.columns)

    print(f"  shape            : {y.shape}")
    print(f"  fs               : {fs:.3f} Hz")
    print(f"  dx               : {dx} m")
    print(f"  gauge length     : {gauge_length} m")
    print(f"  filter band      : [{fmin}, {fmax}] Hz")

    sos = make_bandpass_sos(fs, fmin, fmax, order=filter_order)

    # Zero-phase filtering is better if later you want timing/detection.
    y_filt = sosfiltfilt(sos, y, axis=0)

    if save_dtype == "float32":
        y_filt = y_filt.astype(np.float32, copy=False)

    ensure_dir(output_folder)

    out_name = hdf5_path.stem + f"_bp_{int(fmin)}_{int(fmax)}.npz"
    out_path = output_folder / out_name

    np.savez_compressed(
        out_path,
        y=y_filt,
        fs=np.array(fs, dtype=np.float64),
        dt=np.array(meta["dt"], dtype=np.float64),
        dx=np.array(dx, dtype=np.float64),
        gauge_length=np.array(gauge_length, dtype=np.float64),
        channels=channels,
        source_file=np.array(str(hdf5_path)),
        filter_band_hz=np.array([fmin, fmax], dtype=np.float64),
        meta_json=np.array(json.dumps(sanitize_meta(meta))),
    )

    print(f"  saved            : {out_path}")
    return out_path


def process_folder(
    input_folder: Path,
    output_root: Path | None = None,
    fmin: float = DEFAULT_FMIN,
    fmax: float = DEFAULT_FMAX,
    filter_order: int = DEFAULT_FILTER_ORDER,
    integrate: bool = True,
) -> None:
    """
    Process all HDF5 files in one folder.
    """
    input_folder = input_folder.resolve()
    if output_root is None:
        output_root = input_folder / "processed_bp_3500_4500"
    else:
        output_root = output_root.resolve() / input_folder.name

    ensure_dir(output_root)

    hdf5_files = find_hdf5_files(input_folder)
    if not hdf5_files:
        print(f"No .hdf5 files found in {input_folder}")
        return

    print(f"\n=== Processing folder: {input_folder} ===")
    print(f"Found {len(hdf5_files)} HDF5 files")

    # First inspect metadata across the folder
    summary = summarize_folder_metadata(hdf5_files)

    print("\nFolder metadata summary:")
    print(f"  fs unique               : {summary['fs_hz_unique']}")
    print(f"  dx unique               : {summary['dx_m_unique']}")
    print(f"  gauge length unique     : {summary['gauge_length_m_unique']}")
    print(f"  n_channels unique       : {summary['n_channels_unique']}")

    if not summary["consistent_dx"]:
        print("  WARNING: dx differs across files")
    if not summary["consistent_gauge_length"]:
        print("  WARNING: gauge length differs across files")
    if not summary["consistent_fs"]:
        print("  WARNING: fs differs across files")

    # Save folder summary
    summary_path = output_root / "folder_metadata_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    print(f"\nSaved metadata summary to: {summary_path}")

    # Process each file separately
    saved_files = []
    for hdf5_path in hdf5_files:
        try:
            out_path = process_one_file(
                hdf5_path=hdf5_path,
                output_folder=output_root,
                fmin=fmin,
                fmax=fmax,
                filter_order=filter_order,
                integrate=integrate,
            )
            saved_files.append(str(out_path))
        except Exception as e:
            print(f"FAILED on {hdf5_path.name}: {e}")

    # Save manifest
    manifest = {
        "input_folder": str(input_folder),
        "output_folder": str(output_root),
        "n_input_files": len(hdf5_files),
        "n_saved_files": len(saved_files),
        "saved_files": saved_files,
        "filter_band_hz": [fmin, fmax],
        "filter_order": filter_order,
        "integrate": integrate,
    }
    manifest_path = output_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to: {manifest_path}")


def process_root(
    root_folder: Path,
    output_root: Path | None = None,
    fmin: float = DEFAULT_FMIN,
    fmax: float = DEFAULT_FMAX,
    filter_order: int = DEFAULT_FILTER_ORDER,
    integrate: bool = True,
) -> None:
    """
    Process all subfolders under the given root that contain HDF5 files.
    """
    root_folder = root_folder.resolve()
    subfolders = sorted([p for p in root_folder.iterdir() if p.is_dir()])

    candidate_folders = [p for p in subfolders if len(find_hdf5_files(p)) > 0]

    if not candidate_folders:
        print(f"No subfolders with HDF5 files found under {root_folder}")
        return

    print(f"Found {len(candidate_folders)} data folders under {root_folder}")
    for folder in candidate_folders:
        process_folder(
            input_folder=folder,
            output_root=output_root,
            fmin=fmin,
            fmax=fmax,
            filter_order=filter_order,
            integrate=integrate,
        )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Bandpass-filter DAS HDF5 data and save compressed backups.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=str, help="Single folder containing HDF5 files")
    group.add_argument("--root", type=str, help="Root folder containing multiple experiment folders")

    parser.add_argument("--output", type=str, default=None, help="Output root folder")
    parser.add_argument("--fmin", type=float, default=DEFAULT_FMIN, help="Bandpass lower cutoff in Hz")
    parser.add_argument("--fmax", type=float, default=DEFAULT_FMAX, help="Bandpass upper cutoff in Hz")
    parser.add_argument("--order", type=int, default=DEFAULT_FILTER_ORDER, help="Butterworth filter order")
    parser.add_argument(
        "--no-integrate",
        action="store_true",
        help="Disable time integration inside load_DAS_files (default is integrate=True to match old workflow)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output = None if args.output is None else Path(args.output)
    integrate = not args.no_integrate

    if args.input:
        process_folder(
            input_folder=Path(args.input),
            output_root=output,
            fmin=args.fmin,
            fmax=args.fmax,
            filter_order=args.order,
            integrate=integrate,
        )
    else:
        process_root(
            root_folder=Path(args.root),
            output_root=output,
            fmin=args.fmin,
            fmax=args.fmax,
            filter_order=args.order,
            integrate=integrate,
        )


if __name__ == "__main__":
    main()