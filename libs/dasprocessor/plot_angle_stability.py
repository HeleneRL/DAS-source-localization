import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt

INFO_DIR   = r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\new_cable_subarray_info"
MAX_PACKET = 127  # packets 0..127

all_subarrays_angles = {}   # key = subarray name, value = np.array of angles (len = MAX_PACKET+1)


def parse_start_channel_from_filename(fname: str) -> int | None:
    """
    Parse start channel from file name like:
    doa_info_start_ch_70_arrlen_30.json
    """
    m = re.search(r"start_ch_(\d+)_arrlen_(\d+)", fname)
    if m:
        return int(m.group(1))
    return None


def plot_angles_for_file(json_path: str) -> None:
    # ----- Load JSON -----
    with open(json_path, "r", encoding="utf-8") as f:
        info = json.load(f)   # keys: packet_idx as strings

    # Prepare arrays for all packets 0..MAX_PACKET
    packets = np.arange(0, MAX_PACKET + 1, dtype=int)
    angles  = np.full_like(packets, fill_value=np.nan, dtype=float)

    for p in packets:
        key = str(p)
        if key not in info:
            continue

        entry = info[key]
        if not entry.get("valid", False):
            continue

        angle_off = entry.get("angle_off_axis_deg", None)
        if angle_off is None:
            continue

        angles[p] = float(angle_off)

    # If everything is NaN, skip individual plot (but also don't store)
    if np.all(np.isnan(angles)):
        print(f"No valid angles in {os.path.basename(json_path)}, skipping.")
        return

    # ---- FILTER OUT ZERO-ANGLE PACKETS for the per-subarray plot ----
    mask_plot = ~np.isnan(angles) #& (angles != 0.0)
    packets_plot = packets[mask_plot]
    angles_plot  = angles[mask_plot]

    # ----- Individual plot -----
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        packets_plot,
        angles_plot,
        marker="o",
        linestyle="-",
        markersize=3,
        label="off-axis angle"
    )

    ax.set_xlabel("Packet index")
    ax.set_ylabel("Off-axis angle (deg)")

    start_ch = parse_start_channel_from_filename(os.path.basename(json_path))
    if start_ch is not None:
        title = f"Off-axis angle vs packet index\nsubarray start_ch = {start_ch}"
        sub_name = f"start_ch_{start_ch}"
    else:
        title = f"Off-axis angle vs packet index\n{os.path.basename(json_path)}"
        sub_name = os.path.basename(json_path)

    ax.set_title(title)

    ax.set_xlim(0, MAX_PACKET)
    ax.set_ylim(0, 90)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.show()

    # Store the FULL angles array (0..MAX_PACKET) for combined plot
    all_subarrays_angles[sub_name] = angles


def main():
    # List all doa_info JSONs in the folder
    files = [
        f for f in os.listdir(INFO_DIR)
        if f.endswith(".json") and f.startswith("doa_info_start_ch_")
    ]
    files.sort()

    # 1) Individual plots + fill all_subarrays_angles
    for fname in files:
        json_path = os.path.join(INFO_DIR, fname)
        print(f"Processing {json_path} ...")
        plot_angles_for_file(json_path)

    # 2) Combined plot with valid ranges
    if not all_subarrays_angles:
        print("No subarrays with valid angles, skipping combined plot.")
        return

    # ---- Valid packet ranges per start channel (inclusive) ----
    # Intervals are (start_packet, end_packet), both inclusive
    VALID_RANGES = {
        100: [(43, 89)],
        120: [(0, 6), (48, 92)],
        160: [(5, 29), (75, 95)],
        180: [(25, 53), (77, 93)],
        245: [(0, 16), (40, 60)],
        265: [(12, 40), (48, 67)],
        300: [(0, 11), (37, 51), (58, 71)],  # interpreted 558–71 as 58–71
    }

    plt.figure(figsize=(14, 6))
    packets = np.arange(0, MAX_PACKET + 1)

    for sub_name, angles_full in all_subarrays_angles.items():
        # Extract start_ch from sub_name, e.g. "start_ch_100"
        m = re.search(r"start_ch_(\d+)", sub_name)
        start_ch = int(m.group(1)) if m else None

        # Base mask: valid, non-NaN, non-zero angles
        mask = ~np.isnan(angles_full) #& (angles_full != 0.0)

        # If we have valid ranges for this start_ch, apply them
        # if start_ch in VALID_RANGES:
        #     range_mask = np.zeros_like(mask, dtype=bool)
        #     for lo, hi in VALID_RANGES[start_ch]:
        #         range_mask |= (packets >= lo) & (packets <= hi)
        #     mask &= range_mask

        # Nothing left to plot?
        if not np.any(mask):
            continue

        pkt_plot = packets[mask].astype(float)
        ang_plot = angles_full[mask].astype(float)

        # --- BREAK LINES WHERE THERE ARE PACKET GAPS ---
        # Find indices where the packet index jumps by more than 1
        jumps = np.where(np.diff(pkt_plot) > 1)[0] + 1

        # Insert NaNs at those positions to break the line
        pkt_plot = np.insert(pkt_plot, jumps, np.nan)
        ang_plot = np.insert(ang_plot, jumps, np.nan)

        plt.plot(
            pkt_plot,
            ang_plot,
            marker="o",
            linestyle="-",
            markersize=3,
            label=sub_name
        )

    plt.xlabel("Packet index")
    plt.ylabel("Off-axis angle (degrees)")
    plt.title("Off-axis angle vs packet index")
    plt.xlim(0, MAX_PACKET)
    plt.ylim(0, 90)
    plt.grid(alpha=0.3)
    plt.legend(title="Subarray", loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

