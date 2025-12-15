import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

# Folder with your JSON files
INFO_DIR = r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\new_cable_subarray_info"

def parse_start_channel_from_filename(fname: str) -> int | None:
    """
    Try to parse start channel from file name like:
    doa_info_start_ch_70_arrlen_30.json
    """
    m = re.search(r"start_ch_(\d+)_arrlen_(\d+)", fname)
    if m:
        return int(m.group(1))
    return None

def plot_doa_channels_for_file(json_path: str) -> None:
    # ----- Load JSON -----
    with open(json_path, "r", encoding="utf-8") as f:
        info = json.load(f)   # keys: packet_idx as strings

    # Sort packets by integer key
    packet_idxs = sorted(int(k) for k in info.keys())

    # Arrays to hold values; we will use NaN where invalid
    packets = []
    ch_available = []
    inliers1 = []
    inliers2 = []

    inliers2_valid = []

    for p in packet_idxs:
        entry = info[str(p)]
        packets.append(p)

        # Always use channels_available if present, otherwise 0
        ch_available.append(entry.get("channels_available", 0))

        if entry.get("valid", False):
            # Valid packet → use real inliers
            n1 = entry.get("n_inliers_stage1", 0)
            n2 = entry.get("n_inliers_stage2", 0)
            inliers1.append(n1)
            inliers2.append(n2)
            inliers2_valid.append(n2)
        else:
            # Invalid packet → only "available" bar should show anything
            # Inliers are forced to zero
            inliers1.append(0)
            inliers2.append(0)

    packets = np.array(packets, dtype=int)
    ch_available = np.array(ch_available, dtype=float)
    inliers1 = np.array(inliers1, dtype=float)
    inliers2 = np.array(inliers2, dtype=float)

    if len(packets) == 0:
        print(f"No valid packets in {os.path.basename(json_path)}, skipping.")
        return

    # ----- Plot -----
    fig, ax = plt.subplots(figsize=(12, 5))

    width = 0.25
    x = packets.astype(float)

    ax.bar(x - width, ch_available, width=width, label="channels with ToA", alpha=0.7)
    ax.bar(x,         inliers1,     width=width, label="inliers stage 1",  alpha=0.7)
    ax.bar(x + width, inliers2,     width=width, label="inliers stage 2 (final fit)",  alpha=0.7)


    # Horizontal line at requested = 30 channels
    ax.axhline(30, color="k", linestyle="--", linewidth=1.2, label="requested channels (30)")
    ax.axhline(3, color="k", linestyle="--", linewidth=1.2, label="minimum channels for valid fit (3)")

    # ---- Add average stage-2 inlier horizontal line ----
    # Only include packets that are valid AND have stage-2 inliers recorder


    ax.axhline(np.mean(inliers2_valid), color="purple", linestyle="--", linewidth=1,
                   label=f"Avg channels in final fit = {np.mean(inliers2_valid):.1f}")

    # Labels & title
    ax.set_xlabel("Packet index")
    ax.set_ylabel("Number of channels")
    start_ch = parse_start_channel_from_filename(os.path.basename(json_path))
    if start_ch is not None:
        title = f"DOA channel usage for subarray with start_ch = {start_ch}"
    else:
        title = f"DOA channel usage vs packet\n{os.path.basename(json_path)}"
    ax.set_title(title)

    # Make x-axis ticks readable (maybe every 2 or 4 packets)
    if len(packets) > 30:
        step = 4
    else:
        step = 1
    xticks = packets[::step]
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(p) for p in xticks], rotation=0)

    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()
    # Optionally save:
    # out_png = json_path.replace(".json", "_channels_vs_packet.png")
    # fig.savefig(out_png, dpi=300)
    # print(f"Saved figure to: {out_png}")


def main():
    # List all doa_info JSONs in the folder
    for fname in os.listdir(INFO_DIR):
        if not fname.endswith(".json"):
            continue
        if not fname.startswith("doa_info_"):
            continue

        json_path = os.path.join(INFO_DIR, fname)
        print(f"Processing {json_path} ...")
        plot_doa_channels_for_file(json_path)


if __name__ == "__main__":
    main()
