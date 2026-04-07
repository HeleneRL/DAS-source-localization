from __future__ import annotations

from pathlib import Path
import datetime as dt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import spectrogram

import h5py


# ============================================================
# USER SETTINGS
# ============================================================

'''
FILES = [
    r"D:\Singapore Data\loc3_tx1\040714.hdf5",
    r"D:\Singapore Data\loc3_tx1\040724.hdf5",
    r"D:\Singapore Data\loc3_tx1\040734.hdf5",
    r"D:\Singapore Data\loc3_tx1\040744.hdf5",
    r"D:\Singapore Data\loc3_tx1\040644.hdf5",
    r"D:\Singapore Data\loc3_tx1\040654.hdf5",
    r"D:\Singapore Data\loc3_tx1\040704.hdf5"
]


FILES = [
    r"D:\Singapore Data\loc4_tx1\041454.hdf5",
    r"D:\Singapore Data\loc4_tx1\041504.hdf5",
    r"D:\Singapore Data\loc4_tx1\041514.hdf5",
    r"D:\Singapore Data\loc4_tx1\041524.hdf5",
    r"D:\Singapore Data\loc4_tx1\041534.hdf5",
    r"D:\Singapore Data\loc4_tx1\041424.hdf5",
    r"D:\Singapore Data\loc4_tx1\041434.hdf5",
    r"D:\Singapore Data\loc4_tx1\041444.hdf5"
]

FILES = [
    r"D:\Singapore Data\loc7_tx1\043654.hdf5",
    r"D:\Singapore Data\loc7_tx1\043704.hdf5",
    r"D:\Singapore Data\loc7_tx1\043714.hdf5",
    r"D:\Singapore Data\loc7_tx1\043624.hdf5",
    r"D:\Singapore Data\loc7_tx1\043634.hdf5",
    r"D:\Singapore Data\loc7_tx1\043644.hdf5"
]
'''


FILES = [
    r"D:\Singapore Data\loc3_tx1\040644.hdf5",
    r"D:\Singapore Data\loc3_tx1\040654.hdf5",
    r"D:\Singapore Data\loc3_tx1\040704.hdf5",
    r"D:\Singapore Data\loc3_tx1\040714.hdf5",
    r"D:\Singapore Data\loc3_tx1\040724.hdf5",
    r"D:\Singapore Data\loc3_tx1\040734.hdf5",
    r"D:\Singapore Data\loc3_tx1\040744.hdf5"
]

# Try channel 122 first to mimic the example image
CHANNEL = 2100
LOCATION = "loc3_tx1"

# Singapore local time = UTC+8
UTC_OFFSET_HOURS = 8

# Spectrogram settings
WINDOW = "hann"
NPERSEG = 4096
NOVERLAP = 3584

# Frequency range to display
FREQ_MIN = 3400
FREQ_MAX = 7000

# Robust color scaling
DB_FLOOR_PERCENTILE = 5
DB_CEIL_PERCENTILE = 99.8

# Optional save path
SAVE_PATH = None
# Example:
# SAVE_PATH = r"D:\Singapore Data\loc1_tx2\channel_122_spectrogram.png"


# ============================================================
# HDF5 HELPERS
# ============================================================

def _decode_if_bytes(x):
    if isinstance(x, bytes):
        return x.decode(errors="replace")
    return x


def read_one_file_one_channel(filepath: str, channel: int):
    """
    Read one channel directly from one raw OptoDAS HDF5 file.

    Returns:
        y          : 1D np.ndarray, shape (n_samples,)
        fs         : float
        dx         : float
        start_utc  : datetime.datetime
        n_channels : int
    """
    with h5py.File(filepath, "r") as f:
        data = f["data"]
        n_samples, n_channels = data.shape

        if not (0 <= channel < n_channels):
            raise IndexError(
                f"Requested channel {channel}, but file has channels 0..{n_channels-1}"
            )

        # raw single-channel read
        y = data[:, channel].astype(np.float64)

        header = f["header"]

        dt_s = float(header["dt"][()])
        fs = 1.0 / dt_s

        # dx may be missing in some files, but in your Singapore files it exists
        dx = float(header["dx"][()]) if "dx" in header else np.nan

        # UNIX timestamp in UTC
        start_unix = float(header["time"][()])
        start_utc = dt.datetime.utcfromtimestamp(start_unix)

        return y, fs, dx, start_utc, n_channels


def load_sequence_one_channel(filepaths: list[str], channel: int):
    """
    Load a single channel from several sequential HDF5 files and concatenate.
    """
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
        print(f"Loaded {Path(fp).name}: {len(y)} samples, fs={fs:.3f} Hz")

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
# MAIN PLOTTING
# ============================================================

def main():
    x, fs, dx, start_utc, n_channels = load_sequence_one_channel(FILES, CHANNEL)

    # Distance estimate from channel index and dx
    dist_km = (CHANNEL * dx) / 1000.0 if np.isfinite(dx) else np.nan

    # Spectrogram
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

    Sxx_db = 20 * np.log10(Sxx + 1e-20)

    # Frequency window
    fmask = (f >= FREQ_MIN) & (f <= FREQ_MAX)
    f_plot = f[fmask]
    S_plot = Sxx_db[fmask, :]

    # Convert spectrogram time centers to local time
    local_start = start_utc + dt.timedelta(hours=UTC_OFFSET_HOURS)
    t_local = [local_start + dt.timedelta(seconds=float(s)) for s in t_sec]
    t_local_num = mdates.date2num(t_local)

    # Color scaling
    vmin = np.percentile(S_plot, DB_FLOOR_PERCENTILE)
    vmax = np.percentile(S_plot, DB_CEIL_PERCENTILE)

    # Plot with frequency on x-axis and time on y-axis
    fig, ax = plt.subplots(figsize=(7, 12))

    pcm = ax.pcolormesh(
        f_plot,
        t_local_num,
        S_plot.T,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )

    title = (
        f"{LOCATION}, channel {CHANNEL}, at dist {dist_km:.6f} km"
        if np.isfinite(dist_km)
        else f"SJI Jetty, channel {CHANNEL}"
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Time (local)")

    ax.set_xlim(FREQ_MIN, FREQ_MAX)
    ax.yaxis_date()
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("Magnitude (dB)")

    plt.tight_layout()

    print()
    print(f"Channel           : {CHANNEL}")
    print(f"n_channels/file   : {n_channels}")
    print(f"dx                : {dx:.6f} m")
    print(f"distance          : {dist_km:.6f} km")
    print(f"fs                : {fs:.3f} Hz")
    print(f"UTC start         : {start_utc}")
    print(f"Local start       : {local_start}")
    print(f"Duration          : {len(x)/fs:.3f} s")

    if SAVE_PATH is not None:
        out = Path(SAVE_PATH)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=220, bbox_inches="tight")
        print(f"Saved figure to: {out}")

    plt.show()


if __name__ == "__main__":
    main()