from __future__ import annotations

from pathlib import Path
import datetime as dt
import csv

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import h5py


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

# Channel scan range
CH_START = 1200
CH_STOP = 2401      # exclusive
CH_STEP = 100

# Spectrogram settings
WINDOW = "hann"
NPERSEG = 4096
NOVERLAP = 3584

# Expected LFM parameters
LFM_F0 = 3500.0
LFM_F1 = 8500.0
LFM_DURATION = 5.0
N_SWEEPS = 5

# Files are 10 s each and sweep starts at each file start for loc1 in your list
# If needed, adjust this later.
SWEEP_START_TIMES_SEC = [0.0, 10.0, 20.0, 30.0, 40.0]

# Track scoring parameters
TRACK_HALF_BW_HZ = 120.0         # energy counted "on track" within ± this
OFFTRACK_OFFSET_HZ = 350.0       # off-track comparison bands centered here away
OFFTRACK_HALF_BW_HZ = 120.0

# Optional frequency display / saved output
SAVE_CSV = r"D:\Singapore Data\loc1_tx2\lfm_scan_scores.csv"
SAVE_FIG = r"D:\Singapore Data\loc1_tx2\lfm_scan_scores.png"


# ============================================================
# RAW HDF5 HELPERS
# ============================================================

def read_one_file_one_channel(filepath: str, channel: int):
    with h5py.File(filepath, "r") as f:
        data = f["data"]
        n_samples, n_channels = data.shape

        if not (0 <= channel < n_channels):
            raise IndexError(
                f"Requested channel {channel}, but file has channels 0..{n_channels-1}"
            )

        y = data[:, channel].astype(np.float64)

        header = f["header"]
        dt_s = float(header["dt"][()])
        fs = 1.0 / dt_s
        dx = float(header["dx"][()]) if "dx" in header else np.nan
        start_unix = float(header["time"][()])
        start_utc = dt.datetime.utcfromtimestamp(start_unix)

        return y, fs, dx, start_utc, n_channels


def load_sequence_one_channel(filepaths: list[str], channel: int):
    signals = []
    starts = []
    fs_list = []
    dx_list = []
    n_channels_list = []

    for fp in filepaths:
        y, fs, dx, start_utc, n_channels = read_one_file_one_channel(fp, channel)
        signals.append(y)
        starts.append(start_utc)
        fs_list.append(fs)
        dx_list.append(dx)
        n_channels_list.append(n_channels)

    fs0 = fs_list[0]
    if not np.allclose(fs_list, fs0):
        raise ValueError(f"Sampling rate differs across files: {fs_list}")

    dx0 = dx_list[0]
    if not np.allclose(dx_list, dx0, equal_nan=True):
        raise ValueError(f"dx differs across files: {dx_list}")

    nch0 = n_channels_list[0]
    if any(n != nch0 for n in n_channels_list):
        raise ValueError(f"n_channels differs across files: {n_channels_list}")

    x = np.concatenate(signals)
    return x, fs0, dx0, starts[0], nch0


# ============================================================
# LFM SCORE
# ============================================================

def expected_lfm_freq(t_rel: np.ndarray, start_t: float, f0: float, f1: float, dur: float):
    """
    Expected instantaneous frequency at times t_rel for one sweep starting at start_t.
    Returns NaN outside the sweep interval.
    """
    tau = t_rel - start_t
    out = np.full_like(t_rel, np.nan, dtype=float)
    mask = (tau >= 0.0) & (tau <= dur)
    out[mask] = f0 + (f1 - f0) * (tau[mask] / dur)
    return out


def channel_lfm_score(
    x: np.ndarray,
    fs: float,
    sweep_starts: list[float],
    f0: float,
    f1: float,
    dur: float,
):
    """
    Compute an LFM visibility score for one channel.
    """
    f, t_sec, Sxx = spectrogram(
        x,
        fs=fs,
        window=WINDOW,
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
        detrend=False,
        scaling="density",
        mode="magnitude",
    )

    P = Sxx**2  # power-like quantity

    on_vals = []
    off_vals = []

    for start_t in sweep_starts:
        f_exp = expected_lfm_freq(t_sec, start_t, f0, f1, dur)
        valid_cols = np.where(np.isfinite(f_exp))[0]

        for j in valid_cols:
            fe = f_exp[j]

            on_mask = (f >= fe - TRACK_HALF_BW_HZ) & (f <= fe + TRACK_HALF_BW_HZ)

            off_mask_1 = (f >= fe + OFFTRACK_OFFSET_HZ - OFFTRACK_HALF_BW_HZ) & (
                f <= fe + OFFTRACK_OFFSET_HZ + OFFTRACK_HALF_BW_HZ
            )
            off_mask_2 = (f >= fe - OFFTRACK_OFFSET_HZ - OFFTRACK_HALF_BW_HZ) & (
                f <= fe - OFFTRACK_OFFSET_HZ + OFFTRACK_HALF_BW_HZ
            )

            if np.any(on_mask):
                on_vals.append(np.mean(P[on_mask, j]))

            off_parts = []
            if np.any(off_mask_1):
                off_parts.append(np.mean(P[off_mask_1, j]))
            if np.any(off_mask_2):
                off_parts.append(np.mean(P[off_mask_2, j]))

            if off_parts:
                off_vals.append(np.mean(off_parts))

    on_energy = float(np.mean(on_vals)) if on_vals else 0.0
    off_energy = float(np.mean(off_vals)) if off_vals else 1e-20

    ratio = on_energy / max(off_energy, 1e-20)
    score_db = 10.0 * np.log10(ratio)

    return {
        "score_db": score_db,
        "on_energy": on_energy,
        "off_energy": off_energy,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    results = []

    # Probe dx once
    _, fs_probe, dx, start_utc, n_channels = load_sequence_one_channel(FILES, CH_START)

    print(f"fs = {fs_probe:.3f} Hz")
    print(f"dx = {dx:.6f} m")
    print(f"n_channels = {n_channels}")
    print(f"Scanning channels {CH_START}:{CH_STOP}:{CH_STEP}")

    for ch in range(CH_START, CH_STOP, CH_STEP):
        print(f"Processing channel {ch}...")
        x, fs, dx, _, _ = load_sequence_one_channel(FILES, ch)

        metrics = channel_lfm_score(
            x=x,
            fs=fs,
            sweep_starts=SWEEP_START_TIMES_SEC,
            f0=LFM_F0,
            f1=LFM_F1,
            dur=LFM_DURATION,
        )

        dist_km = (ch * dx) / 1000.0

        results.append({
            "channel": ch,
            "distance_km": dist_km,
            **metrics,
        })

    # Sort strongest first for printing
    results_sorted = sorted(results, key=lambda r: r["score_db"], reverse=True)

    print("\nTop 10 channels:")
    for r in results_sorted[:10]:
        print(
            f"ch={r['channel']:4d}, dist={r['distance_km']:.3f} km, "
            f"score={r['score_db']:.2f} dB"
        )

    # Save CSV
    if SAVE_CSV is not None:
        out_csv = Path(SAVE_CSV)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["channel", "distance_km", "score_db", "on_energy", "off_energy"]
            )
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved CSV: {out_csv}")

    # Plot
    channels = np.array([r["channel"] for r in results])
    dists = np.array([r["distance_km"] for r in results])
    scores = np.array([r["score_db"] for r in results])

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)

    axes[0].plot(channels, scores)
    axes[0].set_xlabel("Channel")
    axes[0].set_ylabel("LFM score [dB]")
    axes[0].set_title("LFM visibility score vs channel")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(dists, scores)
    axes[1].set_xlabel("Distance [km]")
    axes[1].set_ylabel("LFM score [dB]")
    axes[1].set_title("LFM visibility score vs distance")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if SAVE_FIG is not None:
        out_fig = Path(SAVE_FIG)
        out_fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_fig, dpi=220, bbox_inches="tight")
        print(f"Saved figure: {out_fig}")

    plt.show()


if __name__ == "__main__":
    main()