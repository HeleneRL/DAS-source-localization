from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = Path(r"D:\Singapore Data\loc1_tx2\loc1_lfm_arrivals_global_anchor.csv")

SAVE_SUMMARY_CSV = Path(r"D:\Singapore Data\loc1_tx2\loc1_lfm_arrivals_stability_summary.csv")
SAVE_CANDIDATE_CHANNELS_CSV = Path(r"D:\Singapore Data\loc1_tx2\loc1_candidate_good_channels.csv")
SAVE_CANDIDATE_REGIONS_CSV = Path(r"D:\Singapore Data\loc1_tx2\loc1_candidate_good_regions.csv")
SAVE_FINAL_CHANNELS_CSV = Path(r"D:\Singapore Data\loc1_tx2\loc1_final_good_channels.csv")
SAVE_FINAL_REGIONS_CSV = Path(r"D:\Singapore Data\loc1_tx2\loc1_final_good_regions.csv")

# ------------------------------------------------------------
# Selection thresholds
# ------------------------------------------------------------
REQUIRED_N_SWEEPS = 5
MAX_STD_DT = 0.01      # seconds
MAX_RANGE_DT = 0.03    # seconds

# Minimum contiguous region size for final keep
MIN_REGION_CHANNELS = 5


def find_contiguous_regions(channels: list[int]) -> list[tuple[int, int]]:
    """
    Convert sorted channel list into contiguous channel regions.
    Example: [10,11,12,15,16] -> [(10,12), (15,16)]
    """
    if not channels:
        return []

    channels = sorted(channels)

    regions = []
    start = channels[0]
    prev = channels[0]

    for ch in channels[1:]:
        if ch == prev + 1:
            prev = ch
        else:
            regions.append((start, prev))
            start = ch
            prev = ch

    regions.append((start, prev))
    return regions


def build_regions_df(channel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build contiguous region summary from a dataframe that contains at least:
    channel, distance_m, std_dt, range_dt, mean_dt
    """
    if channel_df.empty:
        return pd.DataFrame(columns=[
            "start_channel", "stop_channel", "n_channels",
            "start_distance_m", "stop_distance_m",
            "mean_std_dt", "mean_range_dt", "mean_dt_mean"
        ])

    channel_list = channel_df["channel"].astype(int).tolist()
    regions = find_contiguous_regions(channel_list)

    rows = []
    for start_ch, stop_ch in regions:
        block = channel_df[
            (channel_df["channel"] >= start_ch) &
            (channel_df["channel"] <= stop_ch)
        ].copy()

        rows.append({
            "start_channel": int(start_ch),
            "stop_channel": int(stop_ch),
            "n_channels": int(stop_ch - start_ch + 1),
            "start_distance_m": float(block["distance_m"].min()),
            "stop_distance_m": float(block["distance_m"].max()),
            "mean_std_dt": float(block["std_dt"].mean()),
            "mean_range_dt": float(block["range_dt"].mean()),
            "mean_dt_mean": float(block["mean_dt"].mean()),
        })

    return pd.DataFrame(rows)


def keep_only_channels_in_regions(
    channel_df: pd.DataFrame,
    regions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep only channels belonging to regions listed in regions_df.
    """
    if channel_df.empty or regions_df.empty:
        return channel_df.iloc[0:0].copy()

    keep_mask = pd.Series(False, index=channel_df.index)

    for _, row in regions_df.iterrows():
        keep_mask |= (
            (channel_df["channel"] >= row["start_channel"]) &
            (channel_df["channel"] <= row["stop_channel"])
        )

    return channel_df[keep_mask].copy().sort_values("channel").reset_index(drop=True)


def main():
    df = pd.read_csv(CSV_PATH)

    # --------------------------------------------------------
    # Per-channel stability summary
    # --------------------------------------------------------
    g = df.groupby("channel")

    summary = g.agg(
        distance_m=("distance_m", "first"),
        n_sweeps=("sweep_id", "nunique"),
        mean_dt=("arrival_minus_anchor_s", "mean"),
        std_dt=("arrival_minus_anchor_s", "std"),
        min_dt=("arrival_minus_anchor_s", "min"),
        max_dt=("arrival_minus_anchor_s", "max"),
    ).reset_index()

    summary["range_dt"] = summary["max_dt"] - summary["min_dt"]

    # Save full summary
    SAVE_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SAVE_SUMMARY_CSV, index=False)
    print(f"\nSaved summary to: {SAVE_SUMMARY_CSV}")

    summary_full = summary[summary["n_sweeps"] == REQUIRED_N_SWEEPS].copy()
    summary_full = summary_full.sort_values(["std_dt", "range_dt"])

    print("\nBest 20 channels by stability:")
    print(summary_full.head(20).to_string(index=False))

    # --------------------------------------------------------
    # Candidate good channels from pointwise thresholds
    # --------------------------------------------------------
    candidate = summary[
        (summary["n_sweeps"] == REQUIRED_N_SWEEPS) &
        (summary["std_dt"] <= MAX_STD_DT) &
        (summary["range_dt"] <= MAX_RANGE_DT)
    ].copy()

    candidate = candidate.sort_values("channel").reset_index(drop=True)

    candidate_regions = build_regions_df(candidate)

    candidate.to_csv(SAVE_CANDIDATE_CHANNELS_CSV, index=False)
    candidate_regions.to_csv(SAVE_CANDIDATE_REGIONS_CSV, index=False)

    print(f"\nSaved candidate good channels to: {SAVE_CANDIDATE_CHANNELS_CSV}")
    print(f"Saved candidate good regions to: {SAVE_CANDIDATE_REGIONS_CSV}")

    print(f"\nCandidate good channels: {len(candidate)}")
    if not candidate.empty:
        print(candidate.head(20).to_string(index=False))

    print("\nCandidate contiguous regions:")
    if not candidate_regions.empty:
        print(candidate_regions.to_string(index=False))
    else:
        print("None")

    # --------------------------------------------------------
    # Final regions after minimum contiguous region length
    # --------------------------------------------------------
    final_regions = candidate_regions[
        candidate_regions["n_channels"] >= MIN_REGION_CHANNELS
    ].copy().reset_index(drop=True)

    final_channels = keep_only_channels_in_regions(candidate, final_regions)

    final_channels.to_csv(SAVE_FINAL_CHANNELS_CSV, index=False)
    final_regions.to_csv(SAVE_FINAL_REGIONS_CSV, index=False)

    print(f"\nSaved final good channels to: {SAVE_FINAL_CHANNELS_CSV}")
    print(f"Saved final good regions to: {SAVE_FINAL_REGIONS_CSV}")

    print(f"\nFinal kept channels: {len(final_channels)}")
    print(f"Final kept regions (min {MIN_REGION_CHANNELS} channels):")
    if not final_regions.empty:
        print(final_regions.to_string(index=False))
    else:
        print("None")

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------
    plt.figure(figsize=(12, 5))
    plt.plot(summary["channel"], summary["std_dt"], label="All channels")
    if not candidate.empty:
        plt.scatter(candidate["channel"], candidate["std_dt"], s=10, label="Candidate kept")
    if not final_channels.empty:
        plt.scatter(final_channels["channel"], final_channels["std_dt"], s=14, label="Final kept")
    plt.xlabel("Channel")
    plt.ylabel("Std of arrival_minus_anchor_s [s]")
    plt.title("Sweep-to-sweep timing stability by channel")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(12, 5))
    plt.plot(summary["channel"], summary["mean_dt"], label="All channels")
    if not candidate.empty:
        plt.scatter(candidate["channel"], candidate["mean_dt"], s=10, label="Candidate kept")
    if not final_channels.empty:
        plt.scatter(final_channels["channel"], final_channels["mean_dt"], s=14, label="Final kept")
    plt.xlabel("Channel")
    plt.ylabel("Mean arrival_minus_anchor_s [s]")
    plt.title("Mean relative arrival time by channel")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(12, 5))
    plt.plot(summary["channel"], summary["range_dt"], label="All channels")
    if not candidate.empty:
        plt.scatter(candidate["channel"], candidate["range_dt"], s=10, label="Candidate kept")
    if not final_channels.empty:
        plt.scatter(final_channels["channel"], final_channels["range_dt"], s=14, label="Final kept")
    plt.xlabel("Channel")
    plt.ylabel("Range of arrival_minus_anchor_s [s]")
    plt.title("Sweep-to-sweep timing range by channel")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()