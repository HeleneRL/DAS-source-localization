from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import load_toml, ensure_dir, path_from_cfg


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
            "pick_quality_score",
        ],
    )

    out = df.copy()
    out = normalize_anchor_column(out)
    out = out.rename(columns={"peak_time_s_from_sequence_start": "t_peak_s", "snr_like": "snr"})

    out["channel"] = out["channel"].astype(int)
    out["reference_channel"] = out["reference_channel"].astype(int)
    out["anchor_index"] = out["anchor_index"].astype(int)
    out["t_peak_s"] = pd.to_numeric(out["t_peak_s"], errors="coerce")
    out["snr"] = pd.to_numeric(out["snr"], errors="coerce")
    out["pick_quality_score"] = pd.to_numeric(out["pick_quality_score"], errors="coerce")
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
    parser = argparse.ArgumentParser(description="Build channel trust summaries from detector output.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_toml(args.config)
    tcfg = cfg["trust"]

    med_win = ensure_odd(int(tcfg["rolling_median_win"]))
    mean_win = ensure_odd(int(tcfg["rolling_mean_win"]))
    in_csv = path_from_cfg(cfg, "raw_detection_output_dir") / "all_locations_detections.csv"
    outdir = ensure_dir(path_from_cfg(cfg, "trust_output_dir"))

    df = pd.read_csv(in_csv)
    df = standardize_input(df)
    df = df[(df["channel"] >= int(tcfg["channel_min"])) & (df["channel"] <= int(tcfg["channel_max"]))].copy()
    if df.empty:
        raise ValueError("No rows left after channel filtering.")

    df["offset_ms"] = 1000.0 * df["relative_to_reference_s"].astype(float)
    df["base_valid"] = (
        df["is_valid"]
        & (~df["near_window_edge"])
        & np.isfinite(df["offset_ms"])
        & np.isfinite(df["snr"])
    )

    channels = np.arange(int(tcfg["channel_min"]), int(tcfg["channel_max"]) + 1)
    locations = sorted(df["location"].astype(str).unique())
    anchors = sorted(df["anchor"].astype(str).unique())

    smooth_rows: list[pd.DataFrame] = []
    for loc in locations:
        loc_df = df[df["location"] == loc].copy()
        for anchor in anchors:
            grp = loc_df[loc_df["anchor"] == anchor].copy().sort_values("channel")
            merged = pd.DataFrame({"channel": channels}).merge(grp, on="channel", how="left")
            merged["location"] = loc
            merged["anchor"] = anchor
            merged["base_valid"] = merged["base_valid"].fillna(False).astype(bool)
            merged["near_window_edge"] = merged["near_window_edge"].fillna(False).astype(bool)

            vals = merged["offset_ms"].to_numpy(dtype=float)
            valid_mask = merged["base_valid"].to_numpy(dtype=bool)
            smooth_input = np.where(valid_mask, vals, np.nan)

            if np.sum(np.isfinite(smooth_input)) >= int(tcfg["min_valid_points_per_group"]):
                smooth_ms = smooth_series(channels, smooth_input, med_win, mean_win)
            else:
                smooth_ms = np.full_like(channels, np.nan, dtype=float)

            residual_ms = vals - smooth_ms
            stable = valid_mask & np.isfinite(residual_ms) & (np.abs(residual_ms) <= float(tcfg["residual_bad_ms"]))

            smooth_rows.append(
                pd.DataFrame(
                    {
                        "location": loc,
                        "anchor": anchor,
                        "channel": channels,
                        "offset_ms": vals,
                        "smooth_offset_ms": smooth_ms,
                        "residual_ms": residual_ms,
                        "snr_like": merged["snr"].to_numpy(dtype=float),
                        "pick_quality_score": merged["pick_quality_score"].to_numpy(dtype=float),
                        "base_valid": valid_mask,
                        "stable": stable,
                    }
                )
            )

    smooth_df = pd.concat(smooth_rows, ignore_index=True)

    wide_smooth = smooth_df.pivot_table(index=["location", "channel"], columns="anchor", values="smooth_offset_ms", aggfunc="first")
    if len(anchors) >= 2:
        first_anchor, second_anchor = anchors[0], anchors[1]
        disagreement = (
            (wide_smooth[first_anchor] - wide_smooth[second_anchor]).abs().rename("anchor_disagreement_ms").reset_index()
        )
    else:
        disagreement = pd.DataFrame(columns=["location", "channel", "anchor_disagreement_ms"])

    agg = smooth_df.groupby(["location", "channel"], as_index=False).agg(
        valid_fraction=("base_valid", "mean"),
        stable_fraction=("stable", "mean"),
        median_snr_like=("snr_like", lambda s: np.nanmedian(s.to_numpy(dtype=float))),
        median_pick_quality_score=("pick_quality_score", lambda s: np.nanmedian(s.to_numpy(dtype=float))),
        median_smooth_offset_ms=("smooth_offset_ms", lambda s: np.nanmedian(s.to_numpy(dtype=float))),
        median_abs_residual_ms=("residual_ms", lambda s: np.nanmedian(np.abs(s.to_numpy(dtype=float)))),
        n_anchor_rows=("anchor", "count"),
    )
    agg = agg.merge(disagreement, on=["location", "channel"], how="left")

    agg["score_valid_frac"] = score_high_good(agg["valid_fraction"], float(tcfg["valid_frac_bad"]), float(tcfg["valid_frac_good"]))
    agg["score_stable_frac"] = score_high_good(agg["stable_fraction"], float(tcfg["stable_frac_bad"]), float(tcfg["stable_frac_good"]))
    agg["score_snr"] = score_high_good(agg["median_snr_like"], float(tcfg["snr_bad"]), float(tcfg["snr_good"]))
    agg["score_agreement"] = score_low_good(
        agg["anchor_disagreement_ms"], float(tcfg["anchor_agreement_good_ms"]), float(tcfg["anchor_agreement_bad_ms"])
    )
    agg["score_residual"] = score_low_good(
        agg["median_abs_residual_ms"], float(tcfg["residual_good_ms"]), float(tcfg["residual_bad_ms"])
    )

    agg["channel_trust_score"] = (
        0.18 * agg["score_valid_frac"]
        + 0.24 * agg["score_stable_frac"]
        + 0.18 * agg["score_snr"]
        + 0.18 * agg["score_agreement"]
        + 0.12 * agg["score_residual"]
        + 0.10 * agg["median_pick_quality_score"].clip(0.0, 1.0)
    )

    agg["recommended_channel"] = (
        (agg["valid_fraction"] >= 0.50)
        & (agg["stable_fraction"] >= 0.50)
        & (agg["median_snr_like"] >= 4.0)
        & (agg["median_pick_quality_score"] >= 0.40)
        & (agg["anchor_disagreement_ms"].fillna(0.0) <= 150.0)
        & (agg["median_abs_residual_ms"] <= 120.0)
    )

    overall = agg.groupby("channel", as_index=False).agg(
        mean_valid_fraction=("valid_fraction", "mean"),
        mean_stable_fraction=("stable_fraction", "mean"),
        mean_median_snr_like=("median_snr_like", "mean"),
        mean_median_pick_quality_score=("median_pick_quality_score", "mean"),
        mean_anchor_disagreement_ms=("anchor_disagreement_ms", "mean"),
        mean_abs_residual_ms=("median_abs_residual_ms", "mean"),
        mean_channel_trust_score=("channel_trust_score", "mean"),
        n_locations_recommended=("recommended_channel", "sum"),
        recommended_fraction=("recommended_channel", "mean"),
    )
    overall["recommended_global"] = (
        (overall["recommended_fraction"] >= float(tcfg["global_recommended_frac"]))
        & (overall["mean_channel_trust_score"] >= float(tcfg["global_trust_good"]))
    )

    smooth_df.to_csv(outdir / "channel_trust_smoothed_rows.csv", index=False)
    agg.to_csv(outdir / "location_channel_trust_summary.csv", index=False)
    overall.to_csv(outdir / "overall_channel_trust_summary.csv", index=False)

    ranges = contiguous_true_ranges(overall["channel"].to_numpy(), overall["recommended_global"].to_numpy(dtype=bool))
    if ranges:
        pd.DataFrame(ranges, columns=["channel_start", "channel_end", "n_channels"]).to_csv(
            outdir / "recommended_global_channel_ranges.csv", index=False
        )
    else:
        pd.DataFrame(columns=["channel_start", "channel_end", "n_channels"]).to_csv(
            outdir / "recommended_global_channel_ranges.csv", index=False
        )

    plt.rcParams.update({"figure.dpi": 140})

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

    fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True)
    axes[0].plot(overall["channel"], overall["mean_channel_trust_score"], label="mean trust score")
    axes[0].plot(overall["channel"], overall["recommended_fraction"], label="recommended fraction")
    axes[0].legend()
    axes[0].set_ylabel("Score / fraction")
    axes[0].set_title("Overall channel trust summary")

    axes[1].plot(overall["channel"], overall["mean_median_snr_like"])
    axes[1].set_ylabel("snr_like")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(overall["channel"], overall["mean_median_pick_quality_score"])
    axes[2].set_ylabel("pick quality")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(overall["channel"], overall["mean_anchor_disagreement_ms"])
    axes[3].set_ylabel("anchor disagreement (ms)")
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(overall["channel"], overall["mean_abs_residual_ms"])
    axes[4].set_ylabel("residual (ms)")
    axes[4].set_xlabel("Channel")
    axes[4].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "overall_channel_trust_summary.png", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved outputs to: {outdir}")


if __name__ == "__main__":
    main()
