import json
import numpy as np
import matplotlib.pyplot as plt

# Path to your JSON file
json_path = r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\B_4\peaks-merged-all_hilbert_channels.json"

# Parameters
max_packet = 127  # packets 0..127

# Load JSON
with open(json_path, "r") as f:
    data = json.load(f)  # data[channel_str][packet_str] = arrival_time

# Determine channel range from keys in JSON
channel_keys = [int(ch) for ch in data.keys()]
min_channel = min(channel_keys)
max_channel = max(channel_keys)

channels = list(range(min_channel, max_channel + 1))
n_channels = len(channels)
n_packets = max_packet + 1  # packets 0..127

# Build presence matrix: shape (n_packets, n_channels)
# presence[p, j] = 1 if there is a timestamp for packet p at channel channels[j]
presence = np.zeros((n_packets, n_channels), dtype=int)

for j, ch in enumerate(channels):
    ch_str = str(ch)
    if ch_str not in data:
        continue
    packet_dict = data[ch_str]  # dict of packet_str -> timestamp
    for p_str in packet_dict.keys():
        p = int(p_str)
        if 0 <= p <= max_packet:
            presence[p, j] = 1

# ----------------------------------------------------------------------
# 1) Heatmap: packet vs channel coverage
# ----------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(12, 6))

im = ax1.imshow(
    presence,
    origin="lower",       # packet 0 at bottom
    aspect="auto",
    cmap="Blues",
    interpolation="none",
)

ax1.set_xlabel("Channel index")
ax1.set_ylabel("Packet index")

# x-ticks (sparse for readability)
xtick_step = 20
xticks = np.arange(0, n_channels, xtick_step)
ax1.set_xticks(xticks)
ax1.set_xticklabels([str(channels[i]) for i in xticks])

# y-ticks (e.g. every 8 packets)
ytick_step = 8
yticks = np.arange(0, n_packets, ytick_step)
ax1.set_yticks(yticks)
ax1.set_yticklabels(yticks.astype(str))

ax1.set_title("ToA coverage (blue = ToA detected)")

plt.tight_layout()
plt.show()
# fig1.savefig("packet_channel_coverage.png", dpi=300)


# ============================================================
# PLOT 2 — Percentage of channels detecting each packet
# ============================================================

# For each packet, compute % of channels with a detected ToA
percent_detected = 100 * presence.sum(axis=1) / n_channels    # shape (128,)

fig2, ax2 = plt.subplots(figsize=(14, 5))

ax2.bar(np.arange(n_packets), percent_detected, width=0.8)

ax2.set_xlabel("Packet index")
ax2.set_ylabel("Percentage of channels detecting packet (%)")
ax2.set_title("Channel coverage per packet (channels 16–351)")

# Add grid for readability
ax2.grid(True, linestyle="--", alpha=0.6)

# Optional nicer x ticks
ax2.set_xticks(np.arange(0, n_packets, 8))

ax2.hlines(
    y=70,
    xmin=0,
    xmax=48,
    colors="red",
    linewidth=2
)

# 2) Slanted line packet 48 -> 127 going from 70 -> 15
ax2.plot(
    [48, 127],
    [70, 15],
    color="red",
    linewidth=2
)

plt.tight_layout()
plt.show()

# fig2.savefig("channel_coverage_percentage.png", dpi=300)
