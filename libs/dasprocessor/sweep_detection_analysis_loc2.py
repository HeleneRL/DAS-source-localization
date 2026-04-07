from __future__ import annotations

from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# USER DEFAULTS
# ============================================================

DEFAULT_FOLDER = Path(r"D:\Singapore Data\loc2_tx3")

ARRIVALS_CSV_NAME = "*_lfm_arrivals_global_anchor.csv"

# ------------------------------------------------------------
# Selection thresholds
# ------------------------------------------------------------
REQUIRED_N_SWEEPS = 4
MAX_STD_DT = 0.01      # seconds
MAX_RANGE_DT = 0.03    # seconds

# Minimum contiguous region size for final keep
MIN_REGION_CHANNELS = 5

# Whether to also save sweep-kind specific summaries
SAVE_PER_KIND_SUMMARIES = True


# ============================================================
# HELPERS
# ============================================================

def find_arrivals_csv(folder: Path) -> Path:
    matches = sorted(folder.glob(ARRIVALS_CSV_NAME))
    if not matches:
        raise FileNotFoundError(
            f"No arrivals CSV matching {ARRIVALS_CSV_NAME} found in {folder}"
        )
    if len(matches) > 1:
        print("Warning: multiple arrivals CSVs found, using the first one:")
        for m in matches:
            print(f"  {m.name}")
    return matches[0]


def infer_prefix_from_csv(csv_path: Path) -> str:
    name = csv_path.stem
    suffix = "_lfm_arrivals_global_anchor"
    if name.endswith(suffix):
        return name[:-len(suffix)]
    return name


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


def summarize_channels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-channel stability summary.
    Expected columns in df:
      channel, distance_m, sweep_id, arrival_minus_anchor_s
    """
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
    return summary


def select_good_channels(
    summary: pd.DataFrame,
    required_n_sweeps: int,
    max_std_dt: float,
    max_range_dt: float,
    min_region_channels: int,
):
    summary_full = summary[summary["n_sweeps"] == required_n_sweeps].copy()
    summary_full = summary_full.sort_values(["std_dt", "range_dt"])

    candidate = summary[
        (summary["n_sweeps"] == required_n_sweeps) &
        (summary["std_dt"] <= max_std_dt) &
        (summary["range_dt"] <= max_range_dt)
    ].copy()

    candidate = candidate.sort_values("channel").reset_index(drop=True)
    candidate_regions = build_regions_df(candidate)

    final_regions = candidate_regions[
        candidate_regions["n_channels"] >= min_region_channels
    ].copy().reset_index(drop=True)

    final_channels = keep_only_channels_in_regions(candidate, final_regions)

    return {
        "summary_full": summary_full,
        "candidate": candidate,
        "candidate_regions": candidate_regions,
        "final_channels": final_channels,
        "final_regions": final_regions,
    }


def save_selection_outputs(
    out_dir: Path,
    prefix: str,
    summary: pd.DataFrame,
    candidate: pd.DataFrame,
    candidate_regions: pd.DataFrame,
    final_channels: pd.DataFrame,
    final_regions: pd.DataFrame,
    suffix: str = "",
):
    suffix_part = f"_{suffix}" if suffix else ""

    save_summary_csv = out_dir / f"{prefix}_lfm_arrivals_stability_summary{suffix_part}.csv"
    save_candidate_channels_csv = out_dir / f"{prefix}_candidate_good_channels{suffix_part}.csv"
    save_candidate_regions_csv = out_dir / f"{prefix}_candidate_good_regions{suffix_part}.csv"
    save_final_channels_csv = out_dir / f"{prefix}_final_good_channels{suffix_part}.csv"
    save_final_regions_csv = out_dir / f"{prefix}_final_good_regions{suffix_part}.csv"

    summary.to_csv(save_summary_csv, index=False)
    candidate.to_csv(save_candidate_channels_csv, index=False)
    candidate_regions.to_csv(save_candidate_regions_csv, index=False)
    final_channels.to_csv(save_final_channels_csv, index=False)
    final_regions.to_csv(save_final_regions_csv, index=False)

    print(f"\nSaved summary to: {save_summary_csv}")
    print(f"Saved candidate good channels to: {save_candidate_channels_csv}")
    print(f"Saved candidate good regions to: {save_candidate_regions_csv}")
    print(f"Saved final good channels to: {save_final_channels_csv}")
    print(f"Saved final good regions to: {save_final_regions_csv}")


def print_selection_report(title: str, results: dict, min_region_channels: int):
    summary_full = results["summary_full"]
    candidate = results["candidate"]
    candidate_regions = results["candidate_regions"]
    final_channels = results["final_channels"]
    final_regions = results["final_regions"]

    print(f"\n=== {title} ===")

    print("\nBest 20 channels by stability:")
    if summary_full.empty:
        print("None")
    else:
        print(summary_full.head(20).to_string(index=False))

    print(f"\nCandidate good channels: {len(candidate)}")
    if not candidate.empty:
        print(candidate.head(20).to_string(index=False))

    print("\nCandidate contiguous regions:")
    if not candidate_regions.empty:
        print(candidate_regions.to_string(index=False))
    else:
        print("None")

    print(f"\nFinal kept channels: {len(final_channels)}")
    print(f"Final kept regions (min {min_region_channels} channels):")
    if not final_regions.empty:
        print(final_regions.to_string(index=False))
    else:
        print("None")


def plot_selection(
    summary: pd.DataFrame,
    candidate: pd.DataFrame,
    final_channels: pd.DataFrame,
    title_prefix: str = "",
):
    title_prefix = f"{title_prefix} - " if title_prefix else ""

    plt.figure(figsize=(12, 5))
    plt.plot(summary["channel"], summary["std_dt"], label="All channels")
    if not candidate.empty:
        plt.scatter(candidate["channel"], candidate["std_dt"], s=10, label="Candidate kept")
    if not final_channels.empty:
        plt.scatter(final_channels["channel"], final_channels["std_dt"], s=14, label="Final kept")
    plt.xlabel("Channel")
    plt.ylabel("Std of arrival_minus_anchor_s [s]")
    plt.title(f"{title_prefix}Sweep-to-sweep timing stability by channel")
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
    plt.title(f"{title_prefix}Mean relative arrival time by channel")
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
    plt.title(f"{title_prefix}Sweep-to-sweep timing range by channel")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze loc2+ LFM arrival detections and find stable channel regions."
    )
    parser.add_argument(
        "--folder",
        type=Path,
        default=DEFAULT_FOLDER,
        help="Folder containing the *_lfm_arrivals_global_anchor.csv file.",
    )
    parser.add_argument(
        "--required-n-sweeps",
        type=int,
        default=REQUIRED_N_SWEEPS,
        help="Required number of detected sweeps per channel.",
    )
    parser.add_argument(
        "--max-std-dt",
        type=float,
        default=MAX_STD_DT,
        help="Maximum std of arrival_minus_anchor_s to keep a channel.",
    )
    parser.add_argument(
        "--max-range-dt",
        type=float,
        default=MAX_RANGE_DT,
        help="Maximum range of arrival_minus_anchor_s to keep a channel.",
    )
    parser.add_argument(
        "--min-region-channels",
        type=int,
        default=MIN_REGION_CHANNELS,
        help="Minimum contiguous region size to keep.",
    )

    args = parser.parse_args()

    folder = args.folder
    csv_path = find_arrivals_csv(folder)
    prefix = infer_prefix_from_csv(csv_path)

    print(f"Using arrivals CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = {
        "channel", "distance_m", "sweep_id", "arrival_minus_anchor_s"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    # --------------------------------------------------------
    # Overall summary using all 4 sweeps
    # --------------------------------------------------------
    summary = summarize_channels(df)

    overall_results = select_good_channels(
        summary=summary,
        required_n_sweeps=args.required_n_sweeps,
        max_std_dt=args.max_std_dt,
        max_range_dt=args.max_range_dt,
        min_region_channels=args.min_region_channels,
    )

    save_selection_outputs(
        out_dir=folder,
        prefix=prefix,
        summary=summary,
        candidate=overall_results["candidate"],
        candidate_regions=overall_results["candidate_regions"],
        final_channels=overall_results["final_channels"],
        final_regions=overall_results["final_regions"],
        suffix="",
    )

    print_selection_report(
        title=f"{prefix} - ALL 4 SWEEPS",
        results=overall_results,
        min_region_channels=args.min_region_channels,
    )

    plot_selection(
        summary=summary,
        candidate=overall_results["candidate"],
        final_channels=overall_results["final_channels"],
        title_prefix=f"{prefix} all sweeps",
    )

    # --------------------------------------------------------
    # Optional per-sweep-kind summaries
    # --------------------------------------------------------
    if SAVE_PER_KIND_SUMMARIES and "sweep_kind" in df.columns:
        for kind in sorted(df["sweep_kind"].dropna().unique()):
            df_kind = df[df["sweep_kind"] == kind].copy()
            summary_kind = summarize_channels(df_kind)

            n_sweeps_kind = int(df_kind.groupby("channel")["sweep_id"].nunique().max())
            # Usually 2 for loc2+ per chirp family
            results_kind = select_good_channels(
                summary=summary_kind,
                required_n_sweeps=n_sweeps_kind,
                max_std_dt=args.max_std_dt,
                max_range_dt=args.max_range_dt,
                min_region_channels=args.min_region_channels,
            )

            kind_suffix = kind
            save_selection_outputs(
                out_dir=folder,
                prefix=prefix,
                summary=summary_kind,
                candidate=results_kind["candidate"],
                candidate_regions=results_kind["candidate_regions"],
                final_channels=results_kind["final_channels"],
                final_regions=results_kind["final_regions"],
                suffix=kind_suffix,
            )

            print_selection_report(
                title=f"{prefix} - {kind}",
                results=results_kind,
                min_region_channels=args.min_region_channels,
            )

            plot_selection(
                summary=summary_kind,
                candidate=results_kind["candidate"],
                final_channels=results_kind["final_channels"],
                title_prefix=f"{prefix} {kind}",
            )

    plt.show()


if __name__ == "__main__":
    main()