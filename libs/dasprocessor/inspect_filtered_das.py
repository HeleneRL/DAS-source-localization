from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram


# ---------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------

DATA_FOLDER = Path(r"D:\Singapore Data\processed\loc3_tx1")

FILE_1 = DATA_FOLDER / "040644_bp_3500_4500.npz"
FILE_2 = DATA_FOLDER / "040654_bp_3500_4500.npz"
META_FILE = DATA_FOLDER / "folder_metadata_summary.json"

# Choose which file to use for the 2D time-channel image and spectrogram
FILE_FOR_IMAGE = FILE_1
FILE_FOR_SPEC = FILE_1


# Time window (seconds) for image/spectrogram inspection
T_START = 0.0
T_STOP = 10.0


# Channels to show in channel-time image
CH_START = 0
CH_STOP = 4000   # exclusive
CH_STEP = 1

# Spectrogram channel
SPEC_CHANNEL = 1200

# Spectrogram settings
N_PER_SEG = 1024
N_OVERLAP = 768

# Percentile clipping for image display
IMG_CLIP_LOW = 1
IMG_CLIP_HIGH = 99

# Optionally decimate channels in the image for speed
IMAGE_CHANNEL_DECIMATION = 4

# Time downsampling for image (simple stride)
IMAGE_TIME_DECIMATION = 20


# ---------------------------------------------------------------------
# LOADERS
# ---------------------------------------------------------------------

def load_npz(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    y = d["y"]
    fs = float(np.asarray(d["fs"]).squeeze())
    channels = d["channels"] if "channels" in d else np.arange(y.shape[1])
    dx = float(np.asarray(d["dx"]).squeeze()) if "dx" in d else None
    gauge_length = float(np.asarray(d["gauge_length"]).squeeze()) if "gauge_length" in d else None
    source_file = str(np.asarray(d["source_file"]).squeeze()) if "source_file" in d else None
    return {
        "y": y,
        "fs": fs,
        "channels": np.asarray(channels),
        "dx": dx,
        "gauge_length": gauge_length,
        "source_file": source_file,
    }


def load_folder_metadata(meta_path: Path):
    if not meta_path.exists():
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def time_to_sample(t, fs):
    return int(round(t * fs))


def clip_time_window(y, fs, t0, t1):
    n = y.shape[0]
    i0 = max(0, time_to_sample(t0, fs))
    i1 = min(n, time_to_sample(t1, fs))
    return y[i0:i1], i0, i1


def robust_limits(x, p_low=1, p_high=99):
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    if lo == hi:
        lo = np.min(x)
        hi = np.max(x)
    return lo, hi


# ---------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------

def plot_channel_metrics(npz_paths):
    """
    Plot per-channel statistics to help identify 'real' / active channel ranges.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for npz_path in npz_paths:
        data = load_npz(npz_path)
        y = data["y"]
        channels = data["channels"]

        rms = np.sqrt(np.mean(y**2, axis=0))
        std = np.std(y, axis=0)
        peak = np.max(np.abs(y), axis=0)

        label = npz_path.stem

        axes[0].plot(channels, rms, label=label)
        axes[1].plot(channels, std, label=label)
        axes[2].plot(channels, peak, label=label)

    axes[0].set_title("Per-channel RMS")
    axes[1].set_title("Per-channel standard deviation")
    axes[2].set_title("Per-channel peak |amplitude|")
    axes[2].set_xlabel("Channel")
    axes[0].set_ylabel("RMS")
    axes[1].set_ylabel("STD")
    axes[2].set_ylabel("Peak abs")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()


def plot_channel_time_image(npz_path, t0, t1, ch_start, ch_stop, ch_step=1,
                            t_decim=1, ch_decim=1):
    """
    Plot data as a time-vs-channel image.
    Very useful for spotting the spatial region that looks like real DAS cable.
    """
    data = load_npz(npz_path)
    y = data["y"]
    fs = data["fs"]
    channels = data["channels"]

    y_win, i0, i1 = clip_time_window(y, fs, t0, t1)

    # Channel slicing
    ch_idx = np.arange(ch_start, min(ch_stop, y.shape[1]), ch_step)
    ch_idx = ch_idx[::ch_decim]

    # Time decimation
    y_img = y_win[::t_decim][:, ch_idx]

    times = (np.arange(y_win.shape[0])[::t_decim] + i0) / fs
    ch_vals = channels[ch_idx]

    # Display transpose so y-axis is channel
    img = y_img.T

    vmin, vmax = robust_limits(img, IMG_CLIP_LOW, IMG_CLIP_HIGH)

    plt.figure(figsize=(15, 7))
    plt.imshow(
        img,
        aspect="auto",
        origin="lower",
        extent=[times[0], times[-1], ch_vals[0], ch_vals[-1]],
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(label="Amplitude")
    plt.xlabel("Time [s]")
    plt.ylabel("Channel")
    plt.title(f"Channel-time image: {npz_path.name}")
    plt.tight_layout()


def plot_channel_summary_image(npz_path, metric="rms", block_size=2500):
    """
    Optional overview image: summarize long files by computing one metric
    over consecutive time blocks, then plotting block-vs-channel.
    This can reveal consistently active spatial regions across the file.
    """
    data = load_npz(npz_path)
    y = data["y"]
    fs = data["fs"]
    channels = data["channels"]

    n_samples, n_channels = y.shape
    n_blocks = n_samples // block_size
    if n_blocks < 1:
        print("File too short for summary image with this block_size.")
        return

    y = y[:n_blocks * block_size]
    yb = y.reshape(n_blocks, block_size, n_channels)

    if metric == "rms":
        summary = np.sqrt(np.mean(yb**2, axis=1))
    elif metric == "std":
        summary = np.std(yb, axis=1)
    elif metric == "peak":
        summary = np.max(np.abs(yb), axis=1)
    else:
        raise ValueError("metric must be one of: rms, std, peak")

    # summary shape = (n_blocks, n_channels), transpose for imshow
    img = summary.T
    block_times = np.arange(n_blocks) * block_size / fs

    vmin, vmax = robust_limits(img, 1, 99)

    plt.figure(figsize=(15, 7))
    plt.imshow(
        img,
        aspect="auto",
        origin="lower",
        extent=[block_times[0], block_times[-1], channels[0], channels[-1]],
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(label=metric.upper())
    plt.xlabel("Time block start [s]")
    plt.ylabel("Channel")
    plt.title(f"{metric.upper()} summary image: {npz_path.name}")
    plt.tight_layout()


def plot_spectrogram(npz_path, channel, t0, t1, nperseg=1024, noverlap=768):
    """
    Plot spectrogram for one channel to look for an LFM sweep.
    """
    data = load_npz(npz_path)
    y = data["y"]
    fs = data["fs"]

    if not (0 <= channel < y.shape[1]):
        raise ValueError(f"Channel {channel} is out of range 0..{y.shape[1]-1}")

    y_win, i0, i1 = clip_time_window(y, fs, t0, t1)
    x = y_win[:, channel]

    f, t, Sxx = spectrogram(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
        mode="magnitude",
    )

    # Shift local spectrogram time axis to file time axis
    t = t + i0 / fs

    # Convert to dB safely
    Sxx_db = 20 * np.log10(Sxx + 1e-12)

    plt.figure(figsize=(14, 6))
    plt.pcolormesh(t, f, Sxx_db, shading="gouraud")
    plt.colorbar(label="Magnitude [dB]")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title(f"Spectrogram: {npz_path.name}, channel {channel}")
    plt.ylim([3000, 5000])  # around your band of interest
    plt.tight_layout()


def print_basic_info(npz_path):
    data = load_npz(npz_path)
    y = data["y"]

    print(f"\nFile: {npz_path}")
    print(f"  shape           : {y.shape}")
    print(f"  fs              : {data['fs']:.6f} Hz")
    print(f"  duration        : {y.shape[0] / data['fs']:.3f} s")
    print(f"  channels        : {data['channels'][0]} .. {data['channels'][-1]}")
    print(f"  n_channels      : {len(data['channels'])}")
    print(f"  dx              : {data['dx']}")
    print(f"  gauge_length    : {data['gauge_length']}")
    print(f"  source_file     : {data['source_file']}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    print_basic_info(FILE_1)
    print_basic_info(FILE_2)

    meta = load_folder_metadata(META_FILE)
    if meta is not None:
        print("\nFolder metadata summary:")
        print(f"  fs_hz_unique           : {meta.get('fs_hz_unique')}")
        print(f"  dx_m_unique            : {meta.get('dx_m_unique')}")
        print(f"  gauge_length_m_unique  : {meta.get('gauge_length_m_unique')}")
        print(f"  n_channels_unique      : {meta.get('n_channels_unique')}")

    # 1) Per-channel metrics for comparing files
    plot_channel_metrics([FILE_1, FILE_2])

    # 2) Time-vs-channel image for one chosen file
    plot_channel_time_image(
        FILE_FOR_IMAGE,
        t0=T_START,
        t1=T_STOP,
        ch_start=CH_START,
        ch_stop=CH_STOP,
        ch_step=CH_STEP,
        t_decim=IMAGE_TIME_DECIMATION,
        ch_decim=IMAGE_CHANNEL_DECIMATION,
    )

    # 3) Optional long-view summary image
    plot_channel_summary_image(FILE_FOR_IMAGE, metric="rms", block_size=2500)

    # 4) Spectrogram for one channel
    plot_spectrogram(
        FILE_FOR_SPEC,
        channel=SPEC_CHANNEL,
        t0=T_START,
        t1=T_STOP,
        nperseg=N_PER_SEG,
        noverlap=N_OVERLAP,
    )

    plt.show()


if __name__ == "__main__":
    main()