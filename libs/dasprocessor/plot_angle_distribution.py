import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt

# Folder with your DOA-info JSON files
INFO_DIR = r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\new_cable_subarray_info"

# --- Define angle bins (off-axis) ---
# Special bin for exactly 0°, then [1–5), [5–10), ..., [85–90]
BIN_EDGES = [
    -0.5,   # so that 0° ends up in bin 0
    0.5,    # 0 bin: [-0.5, 0.5)
    5, 10, 15, 20, 25, 30,
    35, 40, 45, 50, 55, 60,
    65, 70, 75, 80, 85, 90
]
BIN_LABELS = [
    "0",
    "1–5", "5–10", "10–15", "15–20", "20–25",
    "25–30", "30–35", "35–40", "40–45", "45–50",
    "50–55", "55–60", "60–65", "65–70", "70–75",
    "75–80", "80–85", "85–90",
]

N_BINS = len(BIN_LABELS)
assert len(BIN_EDGES) == N_BINS + 1, "BIN_EDGES must be one longer than BIN_LABELS"


def parse_start_channel_from_filename(fname: str) -> str:
    """
    Extract 'start_ch_XXX' from file name like:
    doa_info_start_ch_100_arrlen_30.json
    Used for legend labels.
    """
    m = re.search(r"start_ch_(\d+)_arrlen_(\d+)", fname)
    if m:
        return f"start ch {m.group(1)}"
    return fname


def bin_angles_for_file(json_path: str) -> np.ndarray:
    """
    Load one DOA-info JSON and count how many valid packets
    fall into each angle-off-axis bucket.

    Returns: counts, shape = (N_BINS,)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    counts = np.zeros(N_BINS, dtype=int)

    for pkt_str, entry in info.items():
        if not entry.get("valid", False):
            continue

        angle_off = entry.get("angle_off_axis_deg", None)
        if angle_off is None:
            continue

        angle = float(angle_off)

        # Ignore clearly invalid values
        if angle < 0:
            continue

        # Digitize into bins:
        # right=True gives:
        #   bin 0 : (-inf, 0.5]
        #   bin 1 : (0.5, 5]
        #   bin 2 : (5, 10]
        # etc.
        idx = np.digitize(angle, BIN_EDGES, right=True) - 1

        # Clamp anything that falls right at the top edge (e.g. 90°)
        if idx < 0 or idx >= N_BINS:
            # You can print a warning if you like:
            # print(f"Angle {angle:.2f} out of bin range, skipping.")
            continue

        counts[idx] += 1

    return counts


def main():
    # Collect all doa_info JSONs
    files = [
        os.path.join(INFO_DIR, f)
        for f in os.listdir(INFO_DIR)
        if f.startswith("doa_info_") and f.endswith(".json")
    ]
    files = sorted(files)

    if not files:
        print("No doa_info_*.json files found.")
        return

    all_counts = []
    labels = []

    for path in files:
        counts = bin_angles_for_file(path)
        all_counts.append(counts)
        labels.append(parse_start_channel_from_filename(os.path.basename(path)))

    all_counts = np.array(all_counts)  # shape: (n_files, N_BINS)
    n_files = all_counts.shape[0]

    # --- Plotting ---
    x = np.arange(N_BINS)

    fig, ax = plt.subplots(figsize=(14, 6))

    # total width for all bars per bin
    total_width = 0.8
    bar_width = total_width / n_files

    # Center the group of bars around each bin position
    offsets = (np.arange(n_files) - (n_files - 1) / 2) * bar_width

    for i in range(n_files):
        ax.bar(
            x + offsets[i],
            all_counts[i],
            width=bar_width,
            label=labels[i],
            alpha=0.8
        )

    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS, rotation=45, ha="right")

    ax.set_xlabel("Angle off axis (degrees)")
    ax.set_ylabel("Number of packets")
    ax.set_title("Distribution of DOA angle-off-axis by subarray")

    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Subarray", fontsize=9)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
