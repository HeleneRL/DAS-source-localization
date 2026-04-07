from __future__ import annotations

"""
Standalone script for loc2+ LFM arrival detection using ONE clear reference channel.

What it does
------------
1. Reads all HDF5 files in one folder and concatenates them in time.
2. Uses ONE reference channel only.
3. Computes full-record matched-filter envelopes for:
   - 3.5–4.5 kHz chirp
   - 3.5–8.5 kHz chirp
4. Takes the two strongest peaks from each envelope:
   - earlier 3.5–4.5 peak = sweep 1
   - later   3.5–4.5 peak = sweep 2
   - earlier 3.5–8.5 peak = sweep 3
   - later   3.5–8.5 peak = sweep 4
5. Uses those anchor times to pick arrivals in all requested channels.
6. Saves detections to CSV.
7. Produces debug plots.

Sequence assumed from loc2 onwards
----------------------------------
2 x 4.5–5.5 kHz sweep, then delay
2 x 3.5–4.5 kHz sweep, then delay   <- target sweeps 1, 2
2 x 3.5–8.5 kHz sweep, then delay   <- target sweeps 3, 4
then tonals and packets (ignored)

Notes
-----
- This reads raw HDF5 directly with h5py.
- It does NOT use simpleDASreader8.load_DAS_files().
- It does NOT save backups.
- It uses matched filtering on raw channel time series.
- This version is intentionally simple and follows the data visually.
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

FOLDER = Path(r"D:\Singapore Data\loc2_tx3")

# Output naming style
OUTPUT_PREFIX = "loc2"

# SINGLE reference channel
REFERENCE_CHANNEL = 1960

# Channel scan range for arrival picking
CHANNEL_START = 1900
CHANNEL_STOP = 2101   # exclusive
CHANNEL_STEP = 1

# Sweep definitions
SWEEP_DURATION_SEC = 5.0

REFS = {
    1: {"f0": 3500.0, "f1": 4500.0, "label": "3500_4500"},
    2: {"f0": 3500.0, "f1": 4500.0, "label": "3500_4500"},
    3: {"f0": 3500.0, "f1": 8500.0, "label": "3500_8500"},
    4: {"f0": 3500.0, "f1": 8500.0, "label": "3500_8500"},
}

# Preprocessing bandpass
USE_BANDPASS = True
BANDPASS_LOW = 3300.0
BANDPASS_HIGH = 7000.0
BANDPASS_ORDER = 6

# Peak finding
GLOBAL_MIN_PROMINENCE = 0.08
GLOBAL_MIN_HEIGHT = None

LOCAL_MIN_PROMINENCE = 0.04
LOCAL_MIN_HEIGHT = None

# Optional: require some minimum spacing between the two chosen peaks
# for each chirp family, to avoid duplicates on the same sweep.
GLOBAL_MIN_PEAK_SPACING_SEC = 3.0

# Local arrival search around each anchor
LOCAL_SEARCH_BEFORE_SEC = 0.75
LOCAL_SEARCH_AFTER_SEC = 0.75

# Save outputs
SAVE_CSV = FOLDER / f"{OUTPUT_PREFIX}_lfm_arrivals_global_anchor.csv"

# Debug plots
PLOT_GLOBAL_REF = True
PLOT_LOCAL_DEBUG = True
DEBUG_LOCAL_CHANNEL = 1960
DEBUG_LOCAL_SWEEP_ID = 3   # 1..4


# ============================================================
# RAW HDF5 LOADING
# ============================================================

def list_hdf5_files(folder: Path) -> list[str]:
    files = sorted(str(p) for p in folder.glob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No .hdf5 files found in {folder}")
    return files


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
# REFERENCE CHIRPS
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
# MATCHED FILTER
# ============================================================

def matched_filter_envelope(x: np.ndarray, ref: np.ndarray):
    xc = correlate(x, ref, mode="valid")
    env = np.abs(hilbert(xc))
    env = env / (np.max(env) + 1e-12)
    return env


def refine_peak_parabolic(y: np.ndarray, idx: int):
    if idx <= 0 or idx >= len(y) - 1:
        return float(idx)

    y1, y2, y3 = y[idx - 1], y[idx], y[idx + 1]
    denom = (y1 - 2 * y2 + y3)
    if abs(denom) < 1e-12:
        return float(idx)

    delta = 0.5 * (y1 - y3) / denom
    return float(idx + delta)


# ============================================================
# GLOBAL DETECTION ON ONE REFERENCE CHANNEL
# ============================================================

def detect_two_strongest_peaks(
    env: np.ndarray,
    fs: float,
    prominence: float,
    height=None,
    min_spacing_sec: float = 0.0,
):
    min_distance = max(1, int(round(min_spacing_sec * fs)))

    peaks, props = find_peaks(
        env,
        prominence=prominence,
        height=height,
        distance=min_distance if min_spacing_sec > 0 else None,
    )

    if len(peaks) < 2:
        raise RuntimeError(f"Expected at least 2 peaks, found {len(peaks)}")

    peak_vals = env[peaks]
    strongest_idx = np.argsort(peak_vals)[-2:]
    peaks_sel = np.sort(peaks[strongest_idx])

    out = []
    for p in peaks_sel:
        p_refined = refine_peak_parabolic(env, int(p))
        idx_in_all = np.where(peaks == p)[0][0]
        prom = float(props["prominences"][idx_in_all]) if "prominences" in props else np.nan

        out.append({
            "peak_idx": int(p),
            "peak_time_s": float(p_refined / fs),
            "peak_value": float(env[p]),
            "prominence": prom,
        })

    return out


def detect_global_sweeps_single_reference(
    x_ref: np.ndarray,
    fs: float,
    ref_45: np.ndarray,
    ref_85: np.ndarray,
):
    env_45 = matched_filter_envelope(x_ref, ref_45)
    env_85 = matched_filter_envelope(x_ref, ref_85)

    peaks_45 = detect_two_strongest_peaks(
        env_45,
        fs=fs,
        prominence=GLOBAL_MIN_PROMINENCE,
        height=GLOBAL_MIN_HEIGHT,
        min_spacing_sec=GLOBAL_MIN_PEAK_SPACING_SEC,
    )
    peaks_85 = detect_two_strongest_peaks(
        env_85,
        fs=fs,
        prominence=GLOBAL_MIN_PROMINENCE,
        height=GLOBAL_MIN_HEIGHT,
        min_spacing_sec=GLOBAL_MIN_PEAK_SPACING_SEC,
    )

    peaks_45 = sorted(peaks_45, key=lambda d: d["peak_time_s"])
    peaks_85 = sorted(peaks_85, key=lambda d: d["peak_time_s"])

    global_sweeps = [
        {
            "sweep_id": 1,
            "anchor_time_s": peaks_45[0]["peak_time_s"],
            "sweep_kind": "3500_4500",
            "f0_hz": 3500.0,
            "f1_hz": 4500.0,
            "peak_value": peaks_45[0]["peak_value"],
            "prominence": peaks_45[0]["prominence"],
        },
        {
            "sweep_id": 2,
            "anchor_time_s": peaks_45[1]["peak_time_s"],
            "sweep_kind": "3500_4500",
            "f0_hz": 3500.0,
            "f1_hz": 4500.0,
            "peak_value": peaks_45[1]["peak_value"],
            "prominence": peaks_45[1]["prominence"],
        },
        {
            "sweep_id": 3,
            "anchor_time_s": peaks_85[0]["peak_time_s"],
            "sweep_kind": "3500_8500",
            "f0_hz": 3500.0,
            "f1_hz": 8500.0,
            "peak_value": peaks_85[0]["peak_value"],
            "prominence": peaks_85[0]["prominence"],
        },
        {
            "sweep_id": 4,
            "anchor_time_s": peaks_85[1]["peak_time_s"],
            "sweep_kind": "3500_8500",
            "f0_hz": 3500.0,
            "f1_hz": 8500.0,
            "peak_value": peaks_85[1]["peak_value"],
            "prominence": peaks_85[1]["prominence"],
        },
    ]

    return env_45, env_85, global_sweeps


# ============================================================
# LOCAL ARRIVAL PICKING
# ============================================================

def pick_arrival_near_anchor(
    x: np.ndarray,
    ref: np.ndarray,
    fs: float,
    anchor_time_s: float,
):
    n_ref = len(ref)
    n_x = len(x)

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
    refs_by_sweep: dict[int, np.ndarray],
    global_sweeps: list[dict],
):
    results = []

    n_channels = y.shape[1]
    for local_ch in range(n_channels):
        ch = ch_start + local_ch * ch_step
        x = y[:, local_ch]

        for sweep in global_sweeps:
            sid = sweep["sweep_id"]
            ref = refs_by_sweep[sid]

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
                "sweep_kind": sweep["sweep_kind"],
                "f0_hz": sweep["f0_hz"],
                "f1_hz": sweep["f1_hz"],
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

def plot_global_reference_debug_simple(
    x_ref: np.ndarray,
    fs: float,
    env_45: np.ndarray,
    env_85: np.ndarray,
    global_sweeps: list[dict],
    ref_channel: int,
):
    t_sig = np.arange(len(x_ref)) / fs
    t45 = np.arange(len(env_45)) / fs
    t85 = np.arange(len(env_85)) / fs

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=False)

    axes[0].plot(t_sig, x_ref)
    for sw in global_sweeps:
        color = "r" if sw["sweep_id"] in [1, 2] else "m"
        axes[0].axvline(sw["anchor_time_s"], color=color, linestyle="--", alpha=0.8)
    axes[0].set_title(f"Reference channel preprocessed signal: ch {ref_channel}")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(t45, env_45, label="Matched filter: 3.5-4.5 kHz")
    for sw in global_sweeps:
        if sw["sweep_id"] in [1, 2]:
            axes[1].axvline(sw["anchor_time_s"], color="r", linestyle="--", alpha=0.8)
            axes[1].text(
                sw["anchor_time_s"], 0.95, f"S{sw['sweep_id']}",
                rotation=90, va="top", ha="left"
            )
    axes[1].set_title("Full-record matched-filter envelope: 3.5-4.5 kHz")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Normalized envelope")
    axes[1].legend()

    axes[2].plot(t85, env_85, color="tab:green", label="Matched filter: 3.5-8.5 kHz")
    for sw in global_sweeps:
        if sw["sweep_id"] in [3, 4]:
            axes[2].axvline(sw["anchor_time_s"], color="m", linestyle="--", alpha=0.8)
            axes[2].text(
                sw["anchor_time_s"], 0.95, f"S{sw['sweep_id']}",
                rotation=90, va="top", ha="left"
            )
    axes[2].set_title("Full-record matched-filter envelope: 3.5-8.5 kHz")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Normalized envelope")
    axes[2].legend()

    plt.tight_layout()
    plt.show()


def plot_local_debug(
    y: np.ndarray,
    fs: float,
    ch_start: int,
    debug_channel: int,
    refs_by_sweep: dict[int, np.ndarray],
    global_sweeps: list[dict],
    debug_sweep_id: int,
):
    local_idx = debug_channel - ch_start
    if not (0 <= local_idx < y.shape[1]):
        print(f"Debug channel {debug_channel} not in loaded block.")
        return

    x = y[:, local_idx]
    sweep = next((sw for sw in global_sweeps if sw["sweep_id"] == debug_sweep_id), None)
    if sweep is None:
        print(f"Debug sweep_id {debug_sweep_id} not found.")
        return

    ref = refs_by_sweep[debug_sweep_id]

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
    axes[0].set_title(f"Local debug: ch {debug_channel}, sweep {debug_sweep_id} ({sweep['sweep_kind']})")
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

def save_results_csv(results: list[dict], csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        print("No detections to save.")
        return

    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved CSV: {csv_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    files = list_hdf5_files(FOLDER)

    # Load ONLY the chosen reference channel
    y_ref_block, fs, dx, start_utc, file_sample_counts = load_concat_channel_block(
        files,
        REFERENCE_CHANNEL,
        REFERENCE_CHANNEL + 1,
    )
    y_ref_block = preprocess_for_detection(y_ref_block, fs)
    x_ref = y_ref_block[:, 0]

    # Build chirp references
    ref_45 = make_lfm_reference(fs, 3500.0, 4500.0, SWEEP_DURATION_SEC, taper=True)
    ref_85 = make_lfm_reference(fs, 3500.0, 8500.0, SWEEP_DURATION_SEC, taper=True)

    refs_by_sweep = {
        1: ref_45,
        2: ref_45,
        3: ref_85,
        4: ref_85,
    }

    # Detect anchors from the single reference channel
    env_45, env_85, global_sweeps = detect_global_sweeps_single_reference(
        x_ref=x_ref,
        fs=fs,
        ref_45=ref_45,
        ref_85=ref_85,
    )

    print("\nDetected global sweeps from single reference channel:")
    for sw in global_sweeps:
        print(
            f"  Sweep {sw['sweep_id']}: {sw['sweep_kind']} "
            f"anchor_time = {sw['anchor_time_s']:.6f} s, "
            f"peak = {sw['peak_value']:.3f}, prominence = {sw['prominence']:.3f}"
        )

    if PLOT_GLOBAL_REF:
        plot_global_reference_debug_simple(
            x_ref=x_ref,
            fs=fs,
            env_45=env_45,
            env_85=env_85,
            global_sweeps=global_sweeps,
            ref_channel=REFERENCE_CHANNEL,
        )

    # Load full channel block
    y_block, fs2, dx2, _, _ = load_concat_channel_block(
        files,
        CHANNEL_START,
        CHANNEL_STOP,
    )
    if not np.isclose(fs, fs2):
        raise RuntimeError("Sampling rate mismatch between reference load and block load.")
    y_block = preprocess_for_detection(y_block, fs2)

    # Detect arrivals in all channels
    results = detect_arrivals_all_channels(
        y=y_block[:, ::CHANNEL_STEP],
        fs=fs2,
        dx=dx2,
        ch_start=CHANNEL_START,
        ch_step=CHANNEL_STEP,
        refs_by_sweep=refs_by_sweep,
        global_sweeps=global_sweeps,
    )

    print(f"\nTotal channel-sweep detections saved: {len(results)}")
    save_results_csv(results, SAVE_CSV)

    if PLOT_LOCAL_DEBUG:
        plot_local_debug(
            y=y_block,
            fs=fs2,
            ch_start=CHANNEL_START,
            debug_channel=DEBUG_LOCAL_CHANNEL,
            refs_by_sweep=refs_by_sweep,
            global_sweeps=global_sweeps,
            debug_sweep_id=DEBUG_LOCAL_SWEEP_ID,
        )

    if results:
        counts = {}
        for r in results:
            counts[r["channel"]] = counts.get(r["channel"], 0) + 1

        print("\nDetection counts for first few channels:")
        for ch in sorted(counts)[:10]:
            print(f"  ch {ch}: {counts[ch]} sweeps")


if __name__ == "__main__":
    main()