from __future__ import annotations

from pathlib import Path
import csv
import datetime as dt

import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import chirp, correlate, hilbert, butter, sosfiltfilt, find_peaks


# ============================================================
# USER SETTINGS
# ============================================================

FILES = [
    r"D:\Singapore Data\loc1_tx2\034644.hdf5",
    r"D:\Singapore Data\loc1_tx2\034654.hdf5",
    r"D:\Singapore Data\loc1_tx2\034704.hdf5",
    r"D:\Singapore Data\loc1_tx2\034714.hdf5",
    r"D:\Singapore Data\loc1_tx2\034724.hdf5",
]

CHANNEL_START = 1600
CHANNEL_STOP = 2001     # exclusive
CHANNEL_STEP = 10

# Sweep definition for loc1
SWEEP_F0 = 3500.0
SWEEP_F1 = 8500.0
SWEEP_DURATION = 5.0

# Detection preprocessing
USE_BANDPASS = True
BANDPASS_LOW = 3300.0
BANDPASS_HIGH = 7000.0   # clip the weak top end a bit at first
BANDPASS_ORDER = 6

# Search / peak settings
PEAK_MIN_DISTANCE_SEC = 1.0
PEAK_MIN_PROMINENCE = 0.05   # tune later
PEAK_MIN_HEIGHT = None       # optional

SAVE_CSV = r"D:\Singapore Data\loc1_tx2\lfm_arrivals_loc1.csv"

DEBUG_CHANNEL = 1800
DEBUG_SWEEP_INDEX = 0


# ============================================================
# RAW HDF5 LOADING
# ============================================================

def read_hdf5_channel_block(filepath: str, ch_start: int, ch_stop: int):
    with h5py.File(filepath, "r") as f:
        data = f["data"]
        n_samples, n_channels = data.shape

        if not (0 <= ch_start < ch_stop <= n_channels):
            raise ValueError(f"Invalid channel slice {ch_start}:{ch_stop} for file with {n_channels} channels")

        y = data[:, ch_start:ch_stop].astype(np.float64)

        header = f["header"]
        dt_s = float(header["dt"][()])
        fs = 1.0 / dt_s
        dx = float(header["dx"][()]) if "dx" in header else np.nan
        start_unix = float(header["time"][()])
        start_utc = dt.datetime.utcfromtimestamp(start_unix)

    return y, fs, dx, start_utc, n_channels


# ============================================================
# REFERENCE CHIRP
# ============================================================

def make_lfm_reference(fs: float, f0: float, f1: float, duration: float, apply_taper: bool = True):
    n = int(round(duration * fs))
    t = np.arange(n) / fs

    ref = chirp(t, f0=f0, f1=f1, t1=duration, method="linear")

    if apply_taper:
        win = np.hanning(n)
        ref = ref * win

    # normalize
    ref = ref - np.mean(ref)
    ref = ref / (np.linalg.norm(ref) + 1e-12)

    return ref


# ============================================================
# PREPROCESSING
# ============================================================

def bandpass_filter(x: np.ndarray, fs: float, f_low: float, f_high: float, order: int = 6):
    sos = butter(order, [f_low, f_high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def preprocess_for_detection(y: np.ndarray, fs: float):
    y = y - np.mean(y, axis=0, keepdims=True)

    if USE_BANDPASS:
        y = bandpass_filter(y, fs, BANDPASS_LOW, BANDPASS_HIGH, BANDPASS_ORDER)

    # channel-wise normalization for fairer comparison
    std = np.std(y, axis=0, keepdims=True)
    y = y / (std + 1e-12)

    return y


# ============================================================
# MATCHED FILTER DETECTOR
# ============================================================

def matched_filter_one_channel(x: np.ndarray, ref: np.ndarray):
    """
    x: 1D signal
    ref: 1D reference chirp

    Returns:
        env: matched-filter envelope
        peak_idx: index in env
        peak_val: value at peak
        peak_info: dict from scipy.find_peaks
    """
    xc = correlate(x, ref, mode="valid")
    env = np.abs(hilbert(xc))
    env = env / (np.max(env) + 1e-12)

    distance_samples = max(1, int(round(PEAK_MIN_DISTANCE_SEC * 1)))  # will set properly outside if needed

    peaks, props = find_peaks(
        env,
        prominence=PEAK_MIN_PROMINENCE,
        height=PEAK_MIN_HEIGHT,
    )

    if len(peaks) == 0:
        return env, None, None, None

    best = np.argmax(env[peaks])
    peak_idx = int(peaks[best])
    peak_val = float(env[peak_idx])

    peak_info = {
        "prominence": float(props["prominences"][best]) if "prominences" in props else np.nan,
        "height": float(props["peak_heights"][best]) if "peak_heights" in props else peak_val,
    }

    return env, peak_idx, peak_val, peak_info


# ============================================================
# DETECTION ACROSS CHANNELS
# ============================================================

def detect_arrivals_in_file(
    filepath: str,
    ch_start: int,
    ch_stop: int,
    ref: np.ndarray,
    sweep_id: int,
):
    y, fs, dx, start_utc, n_channels_total = read_hdf5_channel_block(filepath, ch_start, ch_stop)
    y = preprocess_for_detection(y, fs)

    results = []

    for local_ch in range(y.shape[1]):
        ch = ch_start + local_ch
        x = y[:, local_ch]

        env, peak_idx, peak_val, peak_info = matched_filter_one_channel(x, ref)

        if peak_idx is None:
            continue

        # correlate(mode="valid") means env index corresponds to start sample of reference match
        arrival_sample = peak_idx
        arrival_time_s = arrival_sample / fs
        distance_m = ch * dx

        results.append({
            "sweep_id": sweep_id,
            "file": filepath,
            "channel": ch,
            "distance_m": distance_m,
            "arrival_sample": arrival_sample,
            "arrival_time_s": arrival_time_s,
            "peak_value": peak_val,
            "prominence": peak_info["prominence"],
            "height": peak_info["height"],
        })

    return results, fs, dx, start_utc


# ============================================================
# DEBUG PLOT
# ============================================================

def plot_debug_for_channel(filepath: str, channel: int, ref: np.ndarray):
    y, fs, dx, start_utc, _ = read_hdf5_channel_block(filepath, channel, channel + 1)
    y = preprocess_for_detection(y, fs)
    x = y[:, 0]

    env, peak_idx, peak_val, peak_info = matched_filter_one_channel(x, ref)

    xc = correlate(x, ref, mode="valid")
    xc_env = np.abs(hilbert(xc))
    xc_env = xc_env / (np.max(xc_env) + 1e-12)

    t_x = np.arange(len(x)) / fs
    t_env = np.arange(len(xc_env)) / fs

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)

    axes[0].plot(t_x, x)
    axes[0].set_title(f"Preprocessed signal, ch {channel}, file {Path(filepath).name}")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(t_env, xc_env, label="Matched filter envelope")
    if peak_idx is not None:
        axes[1].axvline(peak_idx / fs, color="r", linestyle="--", label="Chosen peak")
    axes[1].set_title("Matched filter output")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Normalized envelope")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# SAVE
# ============================================================

def save_results_csv(results: list[dict], csv_path: str):
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        print("No detections to save.")
        return

    fieldnames = list(results[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved: {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    # build reference once
    # probe fs from first file
    _, fs, dx, _, _ = read_hdf5_channel_block(FILES[0], CHANNEL_START, CHANNEL_START + 1)
    ref = make_lfm_reference(fs, SWEEP_F0, SWEEP_F1, SWEEP_DURATION, apply_taper=True)

    all_results = []

    for i, fp in enumerate(FILES):
        print(f"Processing sweep/file {i+1}: {Path(fp).name}")
        results, fs, dx, start_utc = detect_arrivals_in_file(
            filepath=fp,
            ch_start=CHANNEL_START,
            ch_stop=CHANNEL_STOP,
            ref=ref,
            sweep_id=i + 1,
        )
        all_results.extend(results)

    save_results_csv(all_results, SAVE_CSV)

    # Debug one example
    plot_debug_for_channel(FILES[DEBUG_SWEEP_INDEX], DEBUG_CHANNEL, ref)


if __name__ == "__main__":
    main()