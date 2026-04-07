from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CHANNEL_MIN_DEFAULT = 348
CHANNEL_MAX_DEFAULT = 2267
ROLLING_MEDIAN_WIN = 41
ROLLING_MEAN_WIN = 61
MIN_VALID_POINTS_PER_GROUP = 25
ANCHOR_AGREEMENT_GOOD_MS = 80.0
ANCHOR_AGREEMENT_BAD_MS = 250.0
RESIDUAL_GOOD_MS = 60.0
RESIDUAL_BAD_MS = 200.0
SNR_GOOD = 8.0
SNR_BAD = 3.0
STABLE_FRAC_GOOD = 0.60
STABLE_FRAC_BAD = 0.25
VALID_FRAC_GOOD = 0.75
VALID_FRAC_BAD = 0.40
GLOBAL_RECOMMENDED_FRAC = 0.60
GLOBAL_TRUST_GOOD = 0.60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build trust plots and recommended channel mask from all-locations DAS bulk matched-filter results."
    )
    parser.add_argument("--csv", required=True, help="Path to all_locations_bulk_lfm35_45_results.csv")
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory. Defaults to sibling folder of CSV named trust_map_outputs.",
    )
    parser.add_argument("--channel-min", type=int, default=CHANNEL_MIN_DEFAULT)
    parser.add_argument("--channel-max", type=int, default=CHANNEL_MAX_DEFAULT)
    parser.add_argument("--rolling-median", type=int, default=ROLLING_MEDIAN_WIN)
    parser.add_argument("--rolling-mean", type=int, default=ROLLING_MEAN_WIN)
    parser.add_argument("--min-valid-points", type=int, default=MIN_VALID_POINTS_PER_GROUP)
    return parser.parse_args()


def ensure_odd(n: int) -> int:
    n = max(3, int(n))
    return n if n % 2 == 1 else n + 1


def normalize_anchor_column(df: pd.DataFrame) -> pd.DataFrame:
    candidates = ["anchor_label", "anchor_name", "anchor", "anchor_id", "replicate"]
    for c in candidates:
        if c in df.columns:
            out = df.copy()
            out["anchor"] = out[c].astype(str)
            return out
    raise ValueError(f"Could not find an anchor column. Tried: {candidates}")


def require_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def smooth_series(channels: np.ndarray, values_ms: np.ndarray, med_win: int, mean_win: int) -> np.ndarray:
    s = pd.Series(values_ms, index=channels, dtype=float)
    s = s.interpolate(method="index", limit=12, limit_direction="both")
    s = s.rolling(med_win, center=True, min_periods=max(3, med_win // 5)).median()
    s = s.rolling(mean_win, center=True, min_periods=max(3, mean_win // 5)).mean()
    return s.to_numpy(dtype=float)


def score_high_good(x: np.ndarray, bad: float, good: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = (x - bad) / (good - bad + 1e-12)
    return np.clip(y, 0.0, 1.0)


def score_low_good(x: np.ndarray, good: float, bad: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = (bad - x) / (bad - good + 1e-12)
    return np.clip(y, 0.0, 1.0)


def contiguous_true_ranges(channels: np.ndarray, mask: np.ndarray) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    start = None
    prev = None
    for ch, ok in zip(channels, mask):
        if ok and start is None:
            start = ch
            prev = ch
        elif ok:
            prev = ch
        elif start is not None:
            out.append((int(start), int(prev), int(prev - start + 1)))
            start = None
            prev = None
    if start is not None:
        out.append((int(start), int(prev), int(prev - start + 1)))
    return out


def standardize_input(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        df,
        [
            "location",
            "reference_channel",
            "anchor_index",
            "channel",
            "peak_time_s_from_sequence_start",
            "snr_like",
            "passed_snr_threshold",
            "near_window_edge",
        ],
    )

    out = df.copy()
    out = normalize_anchor_column(out)
    out = out.rename(
        columns={
            "peak_time_s_from_sequence_start": "t_peak_s",
            "snr_like": "snr",
            "passed_snr_threshold": "passed_snr_threshold",
        }
    )

    out["channel"] = out["channel"].astype(int)
    out["reference_channel"] = out["reference_channel"].astype(int)
    out["anchor_index"] = out["anchor_index"].astype(int)
    out["t_peak_s"] = pd.to_numeric(out["t_peak_s"], errors="coerce")
    out["snr"] = pd.to_numeric(out["snr"], errors="coerce")
    out["passed_snr_threshold"] = out["passed_snr_threshold"].fillna(False).astype(bool)
    out["near_window_edge"] = out["near_window_edge"].fillna(False).astype(bool)

    out["relative_to_reference_s"] = np.nan
    for (loc, anchor_idx), g in out.groupby(["location", "anchor_index"], sort=False):
        ref_ch = int(g["reference_channel"].iloc[0])
        ref_rows = g[g["channel"] == ref_ch]
        if ref_rows.empty:
            continue
        t_ref = pd.to_numeric(ref_rows["t_peak_s"], errors="coerce").iloc[0]
        if not np.isfinite(t_ref):
            continue
        idx = (out["location"] == loc) & (out["anchor_index"] == anchor_idx)
        out.loc[idx, "relative_to_reference_s"] = out.loc[idx, "t_peak_s"] - t_ref

    out["is_valid"] = (
        (out["snr"] > 0)
        & out["passed_snr_threshold"]
        & (~out["near_window_edge"])
        & np.isfinite(out["t_peak_s"])
        & np.isfinite(out["relative_to_reference_s"])
    )
    return out


def main() -> None:
    args = parse_args()
    med_win = ensure_odd(args.rolling_median)
    mean_win = ensure_odd(args.rolling_mean)

    csv_path = Path(args.csv)
    outdir = Path(args.outdir) if args.outdir else csv_path.parent / "trust_map_outputs"
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = standardize_input(df)
    df = df[(df["channel"] >= args.channel_min) & (df["channel"] <= args.channel_max)].copy()
    if df.empty:
        raise ValueError("No rows left after channel filtering.")

    df["offset_ms"] = 1000.0 * df["relative_to_reference_s"].astype(float)
    df["base_valid"] = (
        df["is_valid"]
        & (~df["near_window_edge"])
        & np.isfinite(df["offset_ms"])
        & np.isfinite(df["snr"])
    )

    channels = np.arange(args.channel_min, args.channel_max + 1)
    locations = sorted(df["location"].astype(str).unique())
    anchors = sorted(df["anchor"].astype(str).unique())

    smooth_rows: list[pd.DataFrame] = []

    for loc in locations:
        loc_df = df[df["location"] == loc].copy()
        for anchor in anchors:
            grp = loc_df[loc_df["anchor"] == anchor].copy().sort_values("channel")
            merged = pd.DataFrame({"channel": channels})
            grp = merged.merge(grp, on="channel", how="left")
            grp["location"] = loc
            grp["anchor"] = anchor
            grp["base_valid"] = grp["base_valid"].fillna(False).astype(bool)
            grp["near_window_edge"] = grp["near_window_edge"].fillna(False).astype(bool)

            vals = grp["offset_ms"].to_numpy(dtype=float)
            valid_mask = grp["base_valid"].to_numpy(dtype=bool)
            smooth_input = np.where(valid_mask, vals, np.nan)

            if np.sum(np.isfinite(smooth_input)) >= args.min_valid_points:
                smooth_ms = smooth_series(channels, smooth_input, med_win, mean_win)
            else:
                smooth_ms = np.full_like(channels, np.nan, dtype=float)

            residual_ms = vals - smooth_ms
            stable = valid_mask & np.isfinite(residual_ms) & (np.abs(residual_ms) <= RESIDUAL_BAD_MS)

            tmp = pd.DataFrame(
                {
                    "location": loc,
                    "anchor": anchor,
                    "channel": channels,
                    "offset_ms": vals,
                    "smooth_offset_ms": smooth_ms,
                    "residual_ms": residual_ms,
                    "snr_like": grp["snr"].to_numpy(dtype=float),
                    "base_valid": valid_mask,
                    "stable": stable,
                }
            )
            smooth_rows.append(tmp)

    smooth_df = pd.concat(smooth_rows, ignore_index=True)

    wide_smooth = smooth_df.pivot_table(
        index=["location", "channel"],
        columns="anchor",
        values="smooth_offset_ms",
        aggfunc="first",
    )
    if len(anchors) >= 2:
        first_anchor, second_anchor = anchors[0], anchors[1]
        disagreement = (
            (wide_smooth[first_anchor] - wide_smooth[second_anchor])
            .abs()
            .rename("anchor_disagreement_ms")
            .reset_index()
        )
    else:
        disagreement = pd.DataFrame(columns=["location", "channel", "anchor_disagreement_ms"])

    agg = smooth_df.groupby(["location", "channel"], as_index=False).agg(
        valid_fraction=("base_valid", "mean"),
        stable_fraction=("stable", "mean"),
        median_snr_like=("snr_like", lambda s: np.nanmedian(s.to_numpy(dtype=float))),
        median_smooth_offset_ms=("smooth_offset_ms", lambda s: np.nanmedian(s.to_numpy(dtype=float))),
        median_abs_residual_ms=("residual_ms", lambda s: np.nanmedian(np.abs(s.to_numpy(dtype=float)))),
        n_anchor_rows=("anchor", "count"),
    )
    agg = agg.merge(disagreement, on=["location", "channel"], how="left")

    agg["score_valid_frac"] = score_high_good(agg["valid_fraction"], VALID_FRAC_BAD, VALID_FRAC_GOOD)
    agg["score_stable_frac"] = score_high_good(agg["stable_fraction"], STABLE_FRAC_BAD, STABLE_FRAC_GOOD)
    agg["score_snr"] = score_high_good(agg["median_snr_like"], SNR_BAD, SNR_GOOD)
    agg["score_agreement"] = score_low_good(
        agg["anchor_disagreement_ms"], ANCHOR_AGREEMENT_GOOD_MS, ANCHOR_AGREEMENT_BAD_MS
    )
    agg["score_residual"] = score_low_good(
        agg["median_abs_residual_ms"], RESIDUAL_GOOD_MS, RESIDUAL_BAD_MS
    )

    agg["channel_trust_score"] = (
        0.20 * agg["score_valid_frac"]
        + 0.25 * agg["score_stable_frac"]
        + 0.20 * agg["score_snr"]
        + 0.20 * agg["score_agreement"]
        + 0.15 * agg["score_residual"]
    )

    agg["recommended_channel"] = (
        (agg["valid_fraction"] >= 0.50)
        & (agg["stable_fraction"] >= 0.50)
        & (agg["median_snr_like"] >= 4.0)
        & (agg["anchor_disagreement_ms"].fillna(0.0) <= 150.0)
        & (agg["median_abs_residual_ms"] <= 120.0)
    )

    overall = agg.groupby("channel", as_index=False).agg(
        mean_valid_fraction=("valid_fraction", "mean"),
        mean_stable_fraction=("stable_fraction", "mean"),
        mean_median_snr_like=("median_snr_like", "mean"),
        mean_anchor_disagreement_ms=("anchor_disagreement_ms", "mean"),
        mean_abs_residual_ms=("median_abs_residual_ms", "mean"),
        mean_channel_trust_score=("channel_trust_score", "mean"),
        n_locations_recommended=("recommended_channel", "sum"),
        recommended_fraction=("recommended_channel", "mean"),
    )
    overall["recommended_global"] = (
        (overall["recommended_fraction"] >= GLOBAL_RECOMMENDED_FRAC)
        & (overall["mean_channel_trust_score"] >= GLOBAL_TRUST_GOOD)
    )

    smooth_df.to_csv(outdir / "channel_trust_smoothed_rows.csv", index=False)
    agg.to_csv(outdir / "location_channel_trust_summary.csv", index=False)
    overall.to_csv(outdir / "overall_channel_trust_summary.csv", index=False)

    ranges = contiguous_true_ranges(
        overall["channel"].to_numpy(), overall["recommended_global"].to_numpy(dtype=bool)
    )
    if ranges:
        ranges_df = pd.DataFrame(ranges, columns=["channel_start", "channel_end", "n_channels"])
    else:
        ranges_df = pd.DataFrame(columns=["channel_start", "channel_end", "n_channels"])
    ranges_df.to_csv(outdir / "recommended_global_channel_ranges.csv", index=False)

    plt.rcParams.update({"figure.dpi": 140})

    if not disagreement.empty:
        pivot = disagreement.pivot(index="location", columns="channel", values="anchor_disagreement_ms").reindex(index=locations)
        fig, ax = plt.subplots(figsize=(16, 5))
        im = ax.imshow(pivot.to_numpy(), aspect="auto", interpolation="nearest", origin="upper")
        ax.set_title("Anchor disagreement by location and channel")
        ax.set_xlabel("Channel")
        ax.set_ylabel("Location")
        xticks = np.linspace(0, len(pivot.columns) - 1, 10, dtype=int)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(int(pivot.columns[i])) for i in xticks])
        ax.set_yticks(np.arange(len(locations)))
        ax.set_yticklabels(locations)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("|anchor1 - anchor2| smooth offset (ms)")
        fig.tight_layout()
        fig.savefig(outdir / "anchor_disagreement_heatmap.png", bbox_inches="tight")
        plt.close(fig)

    pivot = agg.pivot(index="location", columns="channel", values="channel_trust_score").reindex(index=locations)
    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", interpolation="nearest", origin="upper", vmin=0, vmax=1)
    ax.set_title("Channel trust score by location and channel")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Location")
    xticks = np.linspace(0, len(pivot.columns) - 1, 10, dtype=int)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(int(pivot.columns[i])) for i in xticks])
    ax.set_yticks(np.arange(len(locations)))
    ax.set_yticklabels(locations)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Trust score (0-1)")
    fig.tight_layout()
    fig.savefig(outdir / "channel_trust_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    pivot = agg.pivot(index="location", columns="channel", values="recommended_channel").reindex(index=locations).astype(float)
    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", interpolation="nearest", origin="upper", vmin=0, vmax=1)
    ax.set_title("Recommended channel mask by location")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Location")
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(int(pivot.columns[i])) for i in xticks])
    ax.set_yticks(np.arange(len(locations)))
    ax.set_yticklabels(locations)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Recommended = 1")
    fig.tight_layout()
    fig.savefig(outdir / "recommended_mask_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    axes[0].plot(overall["channel"], overall["mean_channel_trust_score"], label="mean trust score")
    axes[0].plot(overall["channel"], overall["recommended_fraction"], label="recommended fraction")
    axes[0].legend()
    axes[0].set_ylabel("Score / fraction")
    axes[0].set_title("Overall channel trust summary across all locations")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(overall["channel"], overall["mean_median_snr_like"], label="mean median snr_like")
    axes[1].axhline(4.0, linestyle="--", linewidth=1)
    axes[1].set_ylabel("snr_like")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(overall["channel"], overall["mean_anchor_disagreement_ms"], label="mean anchor disagreement")
    axes[2].axhline(150.0, linestyle="--", linewidth=1)
    axes[2].set_ylabel("Disagreement (ms)")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(overall["channel"], overall["mean_abs_residual_ms"], label="mean abs residual")
    axes[3].axhline(120.0, linestyle="--", linewidth=1)
    axes[3].set_ylabel("Residual (ms)")
    axes[3].set_xlabel("Channel")
    axes[3].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "overall_channel_trust_summary.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(overall["channel"], overall["mean_channel_trust_score"], label="mean trust score")
    ax.plot(overall["channel"], overall["recommended_fraction"], label="recommended fraction")
    for start, end, _ in ranges:
        ax.axvspan(start, end, alpha=0.18)
    ax.set_title("Global recommended channel regions")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Score / fraction")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "global_recommended_channel_regions.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(len(locations), 1, figsize=(16, 2.8 * len(locations)), sharex=True)
    if len(locations) == 1:
        axes = [axes]
    for ax, loc in zip(axes, locations):
        sub = agg[agg["location"] == loc].sort_values("channel")
        ax.plot(sub["channel"], sub["channel_trust_score"], label="trust score")
        ax.plot(sub["channel"], sub["stable_fraction"], label="stable fraction")
        ax.plot(sub["channel"], sub["valid_fraction"], label="valid fraction", alpha=0.85)
        rec = sub["recommended_channel"].to_numpy(dtype=bool)
        for start, end, _ in contiguous_true_ranges(sub["channel"].to_numpy(), rec):
            ax.axvspan(start, end, alpha=0.12)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel(loc)
        ax.grid(True, alpha=0.3)
    axes[0].legend(ncol=3, loc="upper right")
    axes[-1].set_xlabel("Channel")
    fig.suptitle("Trust diagnostics by location")
    fig.tight_layout()
    fig.savefig(outdir / "trust_diagnostics_by_location.png", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved outputs to: {outdir}")
    print("Files written:")
    for p in sorted(outdir.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()