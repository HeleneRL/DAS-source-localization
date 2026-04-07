from __future__ import annotations

"""
Standalone script for loc1 LFM arrival detection.

What it does
------------
1. Concatenates a set of raw HDF5 files into one continuous timeline.
2. Uses a small reference-channel cluster to detect the 5 global LFM sweep times.
3. Defines anchor_time_s as the median detected sweep time across the reference channels.
4. For every requested channel, estimates one arrival time near each detected sweep.
5. Saves detections to CSV.
6. Produces useful debug plots.

Assumptions for loc1
--------------------
- The useful signal is the 5 x LFM sweep from 3.5 kHz to 8.5 kHz over 5 seconds.
- The sweeps are all contained across the chosen files.
- The source is stationary during these sweeps.

Notes
-----
- This reads raw HDF5 directly with h5py.
- It does NOT use simpleDASreader8.load_DAS_files().
- It does NOT save backups.
- It uses matched filtering on the raw channel time series.

Recommended first use
---------------------
- Start with a limited channel range around where you already see sweeps well.
- Example:
    CHANNEL_START = 1600
    CHANNEL_STOP  = 2001

If the global reference-channel detection works but some channel picks are noisy,
tighten LOCAL_SEARCH_BEFORE_SEC / LOCAL_SEARCH_AFTER_SEC and/or raise the thresholds.
"""

from pathlib import Path
import csv
import datetime as dt

import h5py
import numpy as np
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

# Reference-channel cluster used to define the median anchor times
REFERENCE_CHANNEL_START = 1800
REFERENCE_CHANNEL_STOP = 1811   # exclusive -> 1800..1810

# Channel scan range for arrival picking
CHANNEL_START = 1600
CHANNEL_STOP = 2001   # exclusive
CHANNEL_STEP = 1

# LFM reference
SWEEP_F0 = 3500.0
SWEEP_F1 = 8500.0
SWEEP_DURATION_SEC = 5.0
N_EXPECTED_SWEEPS = 5

# Preprocessing bandpass
USE_BANDPASS = True
BANDPASS_LOW = 3300.0
BANDPASS_HIGH = 7000.0   # intentionally clipped high end for better SNR first pass
BANDPASS_ORDER = 6

# Global reference-channel sweep detection
GLOBAL_MIN_PEAK_SPACING_SEC = 4.0
GLOBAL_MIN_PROMINENCE = 0.08
GLOBAL_MIN_HEIGHT = None

# Local channel arrival search around each detected anchor time
LOCAL_SEARCH_BEFORE_SEC = 0.75
LOCAL_SEARCH_AFTER_SEC = 0.75
LOCAL_MIN_PROMINENCE = 0.04
LOCAL_MIN_HEIGHT = None

# Save outputs
SAVE_CSV = r"D:\Singapore Data\loc1_tx2\loc1_lfm_arrivals_global_anchor.csv"

# Debug plots
PLOT_GLOBAL_REF = True
PLOT_LOCAL_DEBUG = True
DEBUG_LOCAL_CHANNEL = 1800
DEBUG_LOCAL_SWEEP_ID = 3   # 1..N_EXPECTED_SWEEPS


# ============================================================
# RAW HDF5 LOADING
# ============================================================

def read_hdf5_channel_block(filepath: str, ch_start: int, ch_stop: int):
    with h5py.File(filepath, "r") as f:
        data = f["data"]
        n_samples, n_channels = data.shape

        if not (0 <= ch_start < ch_stop <= n_channels):
            raise ValueError(
                f"Invalid channel slice {ch_start}:{ch_stop} for file with {n_channels} channels"
            )

        y = data[:, ch_start:ch_stop].astype(np.float64)

        header = f["header"]
        dt_s = float(header["dt"][()])
        fs = 1.0 / dt_s
        dx = float(header["dx"][()]) if "dx" in header else np.nan
        start_unix = float(header["time"][()])
        start_utc = dt.datetime.utcfromtimestamp(start_unix)

    return y, fs, dx, start_utc, n_channels


def load_concat_channel_block(filepaths: list[str], ch_start: int, ch_stop: int):
    blocks = []
    fs_list = []
    dx_list = []
    start_times = []
    n_channels_list = []
    file_sample_counts = []

    for fp in filepaths:
        y, fs, dx, start_utc, n_channels = read_hdf5_channel_block(fp, ch_start, ch_stop)
        blocks.append(y)
        fs_list.append(fs)
        dx_list.append(dx)
        start_times.append(start_utc)
        n_channels_list.append(n_channels)
        file_sample_counts.append(y.shape[0])
        print(f"Loaded {Path(fp).name}: shape={y.shape}, fs={fs:.3f} Hz")

    fs0 = fs_list[0]
    if not np.allclose(fs_list, fs0):
        raise ValueError(f"Sampling rate differs across files: {fs_list}")

    dx0 = dx_list[0]
    if not np.allclose(dx_list, dx0, equal_nan=True):
        raise ValueError(f"dx differs across files: {dx_list}")

    nch0 = n_channels_list[0]
    if any(n != nch0 for n in n_channels_list):
        raise ValueError(f"n_channels differs across files: {n_channels_list}")

    # sanity check time continuity
    dt_s = 1.0 / fs0
    for i in range(1, len(start_times)):
        expected = start_times[i - 1] + dt.timedelta(seconds=file_sample_counts[i - 1] * dt_s)
        gap = (start_times[i] - expected).total_seconds()
        if abs(gap) > 0.1:
            print(
                f"Warning: file timing gap/overlap between {Path(filepaths[i-1]).name} and "
                f"{Path(filepaths[i]).name}: {gap:.6f} s"
            )

    y_concat = np.vstack(blocks)
    return y_concat, fs0, dx0, start_times[0], file_sample_counts


# ============================================================
# REFERENCE CHIRP
# ============================================================

def make_lfm_reference(fs: float, f0: float, f1: float, duration: float, taper: bool = True):
    n = int(round(duration * fs))
    t = np.arange(n) / fs

    ref = chirp(t, f0=f0, f1=f1, t1=duration, method="linear")

    if taper:
        ref = ref * np.hanning(n)

    ref = ref - np.mean(ref)
    ref = ref / (np.linalg.norm(ref) + 1e-12)
    return ref


# ============================================================
# PREPROCESSING
# ============================================================

def bandpass_filter(y: np.ndarray, fs: float, f_low: float, f_high: float, order: int = 6):
    sos = butter(order, [f_low, f_high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, y, axis=0)


def preprocess_for_detection(y: np.ndarray, fs: float):
    y = y - np.mean(y, axis=0, keepdims=True)

    if USE_BANDPASS:
        y = bandpass_filter(y, fs, BANDPASS_LOW, BANDPASS_HIGH, BANDPASS_ORDER)

    std = np.std(y, axis=0, keepdims=True)
    y = y / (std + 1e-12)
    return y


# ============================================================
# MATCHED FILTER UTILITIES
# ============================================================

def matched_filter_envelope(x: np.ndarray, ref: np.ndarray):
    """
    Returns normalized envelope of valid-mode correlation.
    Envelope sample index corresponds to the START of a candidate reference match.
    """
    xc = correlate(x, ref, mode="valid")
    env = np.abs(hilbert(xc))
    env = env / (np.max(env) + 1e-12)
    return env


def refine_peak_parabolic(y: np.ndarray, idx: int):
    """
    Simple 3-point parabolic refinement around a peak index.
    Returns fractional index. Falls back to idx near boundaries.
    """
    if idx <= 0 or idx >= len(y) - 1:
        return float(idx)

    y1, y2, y3 = y[idx - 1], y[idx], y[idx + 1]
    denom = (y1 - 2 * y2 + y3)
    if abs(denom) < 1e-12:
        return float(idx)

    delta = 0.5 * (y1 - y3) / denom
    return float(idx + delta)


# ============================================================
# GLOBAL SWEEP DETECTION ON REFERENCE CHANNELS
# ============================================================

def detect_global_sweep_times_one_channel(
    x_ref: np.ndarray,
    ref: np.ndarray,
    fs: float,
    n_expected: int,
):
    env = matched_filter_envelope(x_ref, ref)

    min_distance = max(1, int(round(GLOBAL_MIN_PEAK_SPACING_SEC * fs)))

    peaks, props = find_peaks(
        env,
        distance=min_distance,
        prominence=GLOBAL_MIN_PROMINENCE,
        height=GLOBAL_MIN_HEIGHT,
    )

    if len(peaks) == 0:
        raise RuntimeError("No peaks found on reference channel.")

    peak_vals = env[peaks]

    if len(peaks) >= n_expected:
        strongest_idx = np.argsort(peak_vals)[-n_expected:]
        peaks_sel = np.sort(peaks[strongest_idx])
    else:
        print(
            f"Warning: only found {len(peaks)} peaks, fewer than expected {n_expected}."
        )
        peaks_sel = np.sort(peaks)

    peak_times = [float(refine_peak_parabolic(env, int(p)) / fs) for p in peaks_sel]
    return env, peak_times


def detect_global_sweeps_median_anchor(
    y_ref_block: np.ndarray,
    ref: np.ndarray,
    fs: float,
    ref_channel_start: int,
    n_expected: int,
):
    """
    Detect sweep times separately on each reference channel, then use the median
    across channels for each sweep index.
    """
    per_channel_times = []
    envs = []

    n_ref_channels = y_ref_block.shape[1]
    for i in range(n_ref_channels):
        ch = ref_channel_start + i
        x_ref = y_ref_block[:, i]

        try:
            env, peak_times = detect_global_sweep_times_one_channel(
                x_ref=x_ref,
                ref=ref,
                fs=fs,
                n_expected=n_expected,
            )
            if len(peak_times) == n_expected:
                per_channel_times.append(peak_times)
                envs.append((ch, env))
            else:
                print(f"Reference ch {ch}: found {len(peak_times)} peaks, skipping from median anchor.")
        except RuntimeError:
            print(f"Reference ch {ch}: no peaks found, skipping from median anchor.")

    if len(per_channel_times) == 0:
        raise RuntimeError("No valid reference channels found for median anchor.")

    per_channel_times = np.array(per_channel_times)   # shape: (n_good_ref_channels, n_expected)

    median_anchor_times = np.median(per_channel_times, axis=0)

    global_sweeps = []
    for k, t_med in enumerate(median_anchor_times):
        global_sweeps.append({
            "sweep_id": k + 1,
            "anchor_time_s": float(t_med),
        })

    return envs, global_sweeps, per_channel_times


# ============================================================
# LOCAL ARRIVAL PICKING PER CHANNEL PER SWEEP
# ============================================================

def pick_arrival_near_anchor(
    x: np.ndarray,
    ref: np.ndarray,
    fs: float,
    anchor_time_s: float,
):
    """
    Pick one arrival near a known anchor time in one channel.
    """
    n_ref = len(ref)
    n_x = len(x)

    # Search in raw signal time
    t0 = max(0.0, anchor_time_s - LOCAL_SEARCH_BEFORE_SEC)
    t1 = min(n_x / fs, anchor_time_s + LOCAL_SEARCH_AFTER_SEC + SWEEP_DURATION_SEC)

    s0 = max(0, int(np.floor(t0 * fs)))
    s1 = min(n_x, int(np.ceil(t1 * fs)))

    x_win = x[s0:s1]
    if len(x_win) < n_ref + 3:
        return None

    env = matched_filter_envelope(x_win, ref)

    peaks, props = find_peaks(
        env,
        prominence=LOCAL_MIN_PROMINENCE,
        height=LOCAL_MIN_HEIGHT,
    )

    if len(peaks) == 0:
        return None

    # Choose the strongest local peak
    best = np.argmax(env[peaks])
    p = int(peaks[best])

    p_refined = refine_peak_parabolic(env, p)
    arrival_sample_global = s0 + p_refined
    arrival_time_global_s = arrival_sample_global / fs

    return {
        "arrival_sample_global": float(arrival_sample_global),
        "arrival_time_global_s": float(arrival_time_global_s),
        "peak_value": float(env[p]),
        "prominence": float(props["prominences"][best]) if "prominences" in props else np.nan,
        "search_start_s": s0 / fs,
        "search_stop_s": s1 / fs,
        "local_env": env,
        "local_env_time0_s": s0 / fs,
    }


def detect_arrivals_all_channels(
    y: np.ndarray,
    fs: float,
    dx: float,
    ch_start: int,
    ch_step: int,
    ref: np.ndarray,
    global_sweeps: list[dict],
):
    results = []

    n_channels = y.shape[1]
    for local_ch in range(n_channels):
        ch = ch_start + local_ch * ch_step
        x = y[:, local_ch]

        for sweep in global_sweeps:
            picked = pick_arrival_near_anchor(
                x=x,
                ref=ref,
                fs=fs,
                anchor_time_s=sweep["anchor_time_s"],
            )

            if picked is None:
                continue

            results.append({
                "sweep_id": sweep["sweep_id"],
                "anchor_time_s": sweep["anchor_time_s"],
                "channel": ch,
                "distance_m": float(ch * dx),
                "arrival_time_global_s": picked["arrival_time_global_s"],
                "arrival_minus_anchor_s": picked["arrival_time_global_s"] - sweep["anchor_time_s"],
                "arrival_sample_global": picked["arrival_sample_global"],
                "peak_value": picked["peak_value"],
                "prominence": picked["prominence"],
                "search_start_s": picked["search_start_s"],
                "search_stop_s": picked["search_stop_s"],
            })

    return results


# ============================================================
# DEBUG PLOTS
# ============================================================

def plot_global_reference_debug(
    env: np.ndarray,
    global_sweeps: list[dict],
    fs: float,
    x_ref: np.ndarray,
    title: str,
):
    t_sig = np.arange(len(x_ref)) / fs
    t_env = np.arange(len(env)) / fs

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)

    axes[0].plot(t_sig, x_ref)
    axes[0].set_title(f"Reference channel preprocessed signal: {title}")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(t_env, env, label="Matched-filter envelope")
    for sw in global_sweeps:
        axes[1].axvline(sw["anchor_time_s"], color="r", linestyle="--", alpha=0.8)
        axes[1].text(
            sw["anchor_time_s"],
            0.95,
            f"S{sw['sweep_id']}",
            rotation=90,
            va="top",
            ha="left",
        )
    axes[1].set_title("Global sweep detection on reference channel")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Normalized envelope")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_local_debug(
    y: np.ndarray,
    fs: float,
    ch_start: int,
    debug_channel: int,
    ref: np.ndarray,
    global_sweeps: list[dict],
    debug_sweep_id: int,
):
    local_idx = debug_channel - ch_start
    if not (0 <= local_idx < y.shape[1]):
        print(f"Debug channel {debug_channel} not in loaded block.")
        return

    x = y[:, local_idx]
    sweep = None
    for sw in global_sweeps:
        if sw["sweep_id"] == debug_sweep_id:
            sweep = sw
            break

    if sweep is None:
        print(f"Debug sweep_id {debug_sweep_id} not found.")
        return

    picked = pick_arrival_near_anchor(
        x=x,
        ref=ref,
        fs=fs,
        anchor_time_s=sweep["anchor_time_s"],
    )

    if picked is None:
        print("No local pick found for debug plot.")
        return

    s0 = int(round(picked["search_start_s"] * fs))
    s1 = int(round(picked["search_stop_s"] * fs))
    x_win = x[s0:s1]

    t_x = np.arange(len(x_win)) / fs + picked["search_start_s"]
    t_env = np.arange(len(picked["local_env"])) / fs + picked["local_env_time0_s"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)

    axes[0].plot(t_x, x_win)
    axes[0].axvline(sweep["anchor_time_s"], color="k", linestyle="--", label="Anchor time")
    axes[0].axvline(picked["arrival_time_global_s"], color="r", linestyle="--", label="Picked arrival")
    axes[0].set_title(
        f"Local debug: ch {debug_channel}, sweep {debug_sweep_id}"
    )
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()

    axes[1].plot(t_env, picked["local_env"], label="Matched-filter envelope")
    axes[1].axvline(sweep["anchor_time_s"], color="k", linestyle="--", label="Anchor time")
    axes[1].axvline(picked["arrival_time_global_s"], color="r", linestyle="--", label="Picked arrival")
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

    print(f"Saved CSV: {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    # Load reference-channel block over concatenated time
    y_ref_block, fs, dx, start_utc, file_sample_counts = load_concat_channel_block(
        FILES,
        REFERENCE_CHANNEL_START,
        REFERENCE_CHANNEL_STOP,
    )
    y_ref_block = preprocess_for_detection(y_ref_block, fs)

    # Build chirp reference
    ref = make_lfm_reference(
        fs=fs,
        f0=SWEEP_F0,
        f1=SWEEP_F1,
        duration=SWEEP_DURATION_SEC,
        taper=True,
    )

    # Detect all sweep anchors globally using median across reference channels
    envs_ref, global_sweeps, per_channel_times = detect_global_sweeps_median_anchor(
        y_ref_block=y_ref_block,
        ref=ref,
        fs=fs,
        ref_channel_start=REFERENCE_CHANNEL_START,
        n_expected=N_EXPECTED_SWEEPS,
    )

    print("\nDetected global sweeps from median anchor across reference channels:")
    for sw in global_sweeps:
        print(f"  Sweep {sw['sweep_id']}: anchor_time = {sw['anchor_time_s']:.6f} s")

    print("\nPer-reference-channel detected sweep times:")
    for i, times in enumerate(per_channel_times):
        ch = REFERENCE_CHANNEL_START + i
        times_str = ", ".join(f"{t:.6f}" for t in times)
        print(f"  ch {ch}: {times_str}")

    if PLOT_GLOBAL_REF:
        # Plot the first reference channel as representative,
        # but with median anchor times overlaid.
        rep_ch, rep_env = envs_ref[0]
        rep_x = y_ref_block[:, 0]

        plot_global_reference_debug(
            env=rep_env,
            global_sweeps=global_sweeps,
            fs=fs,
            x_ref=rep_x,
            title=f"representative ref ch {rep_ch}, anchor = median({REFERENCE_CHANNEL_START}:{REFERENCE_CHANNEL_STOP-1})",
        )

    # Load full channel block over concatenated time
    y_block, fs2, dx2, _, _ = load_concat_channel_block(
        FILES,
        CHANNEL_START,
        CHANNEL_STOP,
    )
    if not np.isclose(fs, fs2):
        raise RuntimeError("Sampling rate mismatch between reference load and block load.")
    y_block = preprocess_for_detection(y_block, fs2)

    # Detect arrivals for all channels near each sweep anchor
    results = detect_arrivals_all_channels(
        y=y_block[:, ::CHANNEL_STEP],
        fs=fs2,
        dx=dx2,
        ch_start=CHANNEL_START,
        ch_step=CHANNEL_STEP,
        ref=ref,
        global_sweeps=global_sweeps,
    )

    print(f"\nTotal channel-sweep detections saved: {len(results)}")
    save_results_csv(results, SAVE_CSV)

    # Optional local debug plot
    if PLOT_LOCAL_DEBUG:
        plot_local_debug(
            y=y_block,
            fs=fs2,
            ch_start=CHANNEL_START,
            debug_channel=DEBUG_LOCAL_CHANNEL,
            ref=ref,
            global_sweeps=global_sweeps,
            debug_sweep_id=DEBUG_LOCAL_SWEEP_ID,
        )

    # Optional quick summary by channel
    if results:
        counts = {}
        for r in results:
            counts[r["channel"]] = counts.get(r["channel"], 0) + 1

        print("\nDetection counts for first few channels:")
        for ch in sorted(counts)[:10]:
            print(f"  ch {ch}: {counts[ch]} sweeps")


if __name__ == "__main__":
    main()