from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


CHANNEL_MIN = 348
CHANNEL_MAX = 2267
DEFAULT_SOUND_SPEED_MPS = 1500.0


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.upper().eq("TRUE")


def load_inputs(obs_csv: Path, prior_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = pd.read_csv(obs_csv)
    prior = pd.read_csv(prior_csv)
    return obs, prior


def prepare_prior(prior: pd.DataFrame) -> pd.DataFrame:
    req = [
        "channel",
        "prior_x_m",
        "prior_y_m",
        "prior_u_m",
        "tangent_x",
        "tangent_y",
        "normal_x",
        "normal_y",
    ]
    missing = [c for c in req if c not in prior.columns]
    if missing:
        raise ValueError(f"prior_geometry.csv missing columns: {missing}")

    out = prior.copy()
    out = out[(out["channel"] >= CHANNEL_MIN) & (out["channel"] <= CHANNEL_MAX)].copy()
    out = out.sort_values("channel").reset_index(drop=True)
    return out


def prepare_observations(
    obs: pd.DataFrame,
    min_weight: float,
    min_stable_fraction: float,
    use_only_recommended: bool,
    use_smoothed: bool,
) -> pd.DataFrame:
    df = obs.copy()
    df = df[(df["channel"] >= CHANNEL_MIN) & (df["channel"] <= CHANNEL_MAX)].copy()

    required = [
        "location",
        "anchor_index",
        "anchor_label",
        "channel",
        "reference_channel",
        "tx_x_m",
        "tx_y_m",
        "tx_u_m",
        "weight",
        "use_observation",
        "recommended_channel",
        "recommended_global",
        "stable_fraction",
        "channel_trust_score",
        "mean_channel_trust_score",
        "observed_dt_ref_s",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"inversion_observations.csv missing columns: {missing}")

    df["use_observation"] = as_bool(df["use_observation"])
    df["recommended_channel"] = as_bool(df["recommended_channel"])
    df["recommended_global"] = as_bool(df["recommended_global"])

    if use_smoothed and "median_smooth_offset_ms" in df.columns:
        df["fit_dt_s"] = pd.to_numeric(df["median_smooth_offset_ms"], errors="coerce") / 1000.0
    else:
        df["fit_dt_s"] = pd.to_numeric(df["observed_dt_ref_s"], errors="coerce")

    mask = df["use_observation"] & df["fit_dt_s"].notna() & (pd.to_numeric(df["weight"], errors="coerce") >= min_weight)
    mask &= pd.to_numeric(df["stable_fraction"], errors="coerce").fillna(0.0) >= min_stable_fraction

    if use_only_recommended:
        mask &= df["recommended_channel"] & df["recommended_global"]

    df = df[mask].copy()
    if df.empty:
        raise ValueError("No observations left after filtering. Relax thresholds.")

    w = pd.to_numeric(df["weight"], errors="coerce").to_numpy(dtype=float)
    w *= np.clip(pd.to_numeric(df["channel_trust_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float), 0.0, 1.0)
    w *= np.clip(pd.to_numeric(df["mean_channel_trust_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float), 0.0, 1.0)
    w *= np.where(df["recommended_channel"].to_numpy(), 1.0, 0.5)
    w *= np.where(df["recommended_global"].to_numpy(), 1.0, 0.7)

    df["fit_weight"] = w
    df = df[df["fit_weight"] > 0].copy()
    return df


def make_prior_interpolators(prior: pd.DataFrame) -> dict[str, interp1d]:
    ch = prior["channel"].to_numpy(dtype=float)
    cols = [
        "prior_x_m",
        "prior_y_m",
        "prior_u_m",
        "tangent_x",
        "tangent_y",
        "normal_x",
        "normal_y",
    ]
    return {
        c: interp1d(
            ch,
            prior[c].to_numpy(dtype=float),
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        for c in cols
    }


def evaluate_shifted_prior(
    channels: np.ndarray,
    shift_ch: float,
    interp_map: dict[str, interp1d],
    fixed_prior_offset_ch: float = 0.0,
) -> dict[str, np.ndarray]:
    ch_eff = channels.astype(float) + fixed_prior_offset_ch + shift_ch
    out = {name: fn(ch_eff) for name, fn in interp_map.items()}

    t = np.column_stack([out["tangent_x"], out["tangent_y"]])
    tnorm = np.linalg.norm(t, axis=1, keepdims=True)
    tnorm = np.where(tnorm <= 1e-12, 1.0, tnorm)
    t = t / tnorm
    out["tangent_x"] = t[:, 0]
    out["tangent_y"] = t[:, 1]

    n = np.column_stack([out["normal_x"], out["normal_y"]])
    nnorm = np.linalg.norm(n, axis=1, keepdims=True)
    nnorm = np.where(nnorm <= 1e-12, 1.0, nnorm)
    n = n / nnorm
    out["normal_x"] = n[:, 0]
    out["normal_y"] = n[:, 1]

    out["channel_effective"] = ch_eff
    return out


def predict_relative_times(cable_xyz: np.ndarray, tx_xyz: np.ndarray, ref_idx: int, sound_speed: float) -> np.ndarray:
    ranges = np.linalg.norm(cable_xyz - tx_xyz[None, :], axis=1)
    return (ranges - ranges[ref_idx]) / sound_speed


def weighted_metrics(resid_s: np.ndarray, w: np.ndarray) -> dict[str, float]:
    resid_ms = resid_s * 1000.0
    wsum = float(np.sum(w))
    if wsum <= 0:
        return {
            "weighted_rmse_ms": np.nan,
            "weighted_mae_ms": np.nan,
            "weighted_bias_ms": np.nan,
            "median_abs_ms": np.nan,
        }

    wrmse = float(np.sqrt(np.sum(w * resid_ms**2) / wsum))
    wmae = float(np.sum(w * np.abs(resid_ms)) / wsum)
    wbias = float(np.sum(w * resid_ms) / wsum)
    medabs = float(np.median(np.abs(resid_ms)))

    return {
        "weighted_rmse_ms": wrmse,
        "weighted_mae_ms": wmae,
        "weighted_bias_ms": wbias,
        "median_abs_ms": medabs,
    }


def misfit_for_shift(
    shift_ch: float,
    obs: pd.DataFrame,
    prior_channels: np.ndarray,
    interp_map: dict[str, interp1d],
    sound_speed: float,
    fixed_prior_offset_ch: float = 0.0,
) -> tuple[float, dict, pd.DataFrame]:
    shifted = evaluate_shifted_prior(
        prior_channels,
        shift_ch,
        interp_map,
        fixed_prior_offset_ch=fixed_prior_offset_ch,
    )

    cable_xyz = np.column_stack([
        shifted["prior_x_m"],
        shifted["prior_y_m"],
        shifted["prior_u_m"],
    ])

    ch_to_idx = {int(ch): i for i, ch in enumerate(prior_channels)}

    rows = []
    resid_all = []
    w_all = []
    group_summary = []

    for (location, anchor_index), g in obs.groupby(["location", "anchor_index"], sort=True):
        g = g.sort_values("channel").copy()
        ref_ch = int(g["reference_channel"].iloc[0])
        if ref_ch not in ch_to_idx:
            continue

        ref_idx = ch_to_idx[ref_ch]

        tx_xyz = np.array([
            float(g["tx_x_m"].iloc[0]),
            float(g["tx_y_m"].iloc[0]),
            float(g["tx_u_m"].iloc[0]),
        ], dtype=float)

        pred_dt = predict_relative_times(cable_xyz, tx_xyz, ref_idx, sound_speed)

        idx = g["channel"].map(ch_to_idx).to_numpy(dtype=int)
        pred = pred_dt[idx]
        obs_dt = g["fit_dt_s"].to_numpy(dtype=float)
        w = g["fit_weight"].to_numpy(dtype=float)
        resid = pred - obs_dt

        resid_all.append(resid)
        w_all.append(w)

        gm = weighted_metrics(resid, w)
        gm.update({
            "location": location,
            "anchor_index": int(anchor_index),
            "anchor_label": g["anchor_label"].iloc[0],
            "n_rows": int(len(g)),
            "residual_shift_channels": float(shift_ch),
            "fixed_prior_offset_ch": float(fixed_prior_offset_ch),
            "total_effective_shift_ch": float(fixed_prior_offset_ch + shift_ch),
        })
        group_summary.append(gm)

        temp = g[
            [
                "location",
                "anchor_index",
                "anchor_label",
                "channel",
                "reference_channel",
                "fit_dt_s",
                "fit_weight",
            ]
        ].copy()
        temp["pred_dt_s"] = pred
        temp["residual_s"] = resid
        temp["residual_shift_channels"] = shift_ch
        temp["fixed_prior_offset_ch"] = fixed_prior_offset_ch
        temp["total_effective_shift_ch"] = fixed_prior_offset_ch + shift_ch
        rows.append(temp)

    if not resid_all:
        return np.inf, {}, pd.DataFrame()

    resid_all = np.concatenate(resid_all)
    w_all = np.concatenate(w_all)

    score = float(np.sum(w_all * resid_all**2) / max(np.sum(w_all), 1e-12))
    overall = weighted_metrics(resid_all, w_all)
    overall["objective_s2"] = score

    pred_rows = pd.concat(rows, ignore_index=True)
    group_df = pd.DataFrame(group_summary)

    return score, {"overall": overall, "group": group_df}, pred_rows


def scan_shifts(
    obs: pd.DataFrame,
    prior_channels: np.ndarray,
    interp_map: dict[str, interp1d],
    sound_speed: float,
    shift_min: float,
    shift_max: float,
    shift_step: float,
    fixed_prior_offset_ch: float = 0.0,
) -> tuple[pd.DataFrame, float, dict, pd.DataFrame]:
    shifts = np.arange(shift_min, shift_max + 0.5 * shift_step, shift_step, dtype=float)

    records = []
    best_score = np.inf
    best_shift = None
    best_metrics = None
    best_pred = None

    for s in shifts:
        score, metrics, pred_rows = misfit_for_shift(
            s,
            obs,
            prior_channels,
            interp_map,
            sound_speed,
            fixed_prior_offset_ch=fixed_prior_offset_ch,
        )

        rec = {
            "residual_shift_channels": s,
            "fixed_prior_offset_ch": fixed_prior_offset_ch,
            "total_effective_shift_ch": fixed_prior_offset_ch + s,
        }

        if metrics:
            rec.update(metrics["overall"])
        else:
            rec.update({
                "objective_s2": np.inf,
                "weighted_rmse_ms": np.nan,
                "weighted_mae_ms": np.nan,
                "weighted_bias_ms": np.nan,
                "median_abs_ms": np.nan,
            })

        records.append(rec)

        if score < best_score:
            best_score = score
            best_shift = s
            best_metrics = metrics
            best_pred = pred_rows

    return pd.DataFrame(records), float(best_shift), best_metrics, best_pred


def refine_shift_parabolic(scan_df: pd.DataFrame) -> float:
    df = scan_df.sort_values("residual_shift_channels").reset_index(drop=True)
    idx = int(df["objective_s2"].idxmin())
    if idx <= 0 or idx >= len(df) - 1:
        return float(df.loc[idx, "residual_shift_channels"])

    x1, x2, x3 = df.loc[idx - 1: idx + 1, "residual_shift_channels"].to_numpy(dtype=float)
    y1, y2, y3 = df.loc[idx - 1: idx + 1, "objective_s2"].to_numpy(dtype=float)

    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    if abs(denom) < 1e-12:
        return float(x2)

    a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    b = (x3**2 * (y1 - y2) + x2**2 * (y3 - y1) + x1**2 * (y2 - y3)) / denom

    if abs(a) < 1e-12:
        return float(x2)

    xv = -b / (2 * a)
    return float(xv)


def make_path_plot(
    prior: pd.DataFrame,
    interp_map: dict[str, interp1d],
    best_shift: float,
    fixed_prior_offset: float,
    out_path: Path,
) -> None:
    ch = prior["channel"].to_numpy(dtype=float)
    shifted = evaluate_shifted_prior(
        ch,
        best_shift,
        interp_map,
        fixed_prior_offset_ch=fixed_prior_offset,
    )
    total_shift = fixed_prior_offset + best_shift

    plt.figure(figsize=(8, 8))
    plt.plot(prior["prior_x_m"], prior["prior_y_m"], label="Prior path", linewidth=2)
    plt.plot(
        shifted["prior_x_m"],
        shifted["prior_y_m"],
        label=(
            f"Shifted prior "
            f"(base={fixed_prior_offset:.2f}, residual={best_shift:.2f}, total={total_shift:.2f} ch)"
        ),
        linewidth=2,
    )
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.title("Prior path vs shifted prior")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def make_shift_scan_plot(
    scan_df: pd.DataFrame,
    best_shift: float,
    fixed_prior_offset: float,
    out_path: Path,
) -> None:
    total_shift = fixed_prior_offset + best_shift

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(scan_df["residual_shift_channels"], scan_df["weighted_rmse_ms"], label="Weighted RMSE (ms)")
    ax1.plot(scan_df["residual_shift_channels"], scan_df["median_abs_ms"], label="Median |residual| (ms)")
    ax1.axvline(
        best_shift,
        linestyle="--",
        linewidth=1.2,
        label=f"Best residual shift = {best_shift:.2f} ch\nTotal = {total_shift:.2f} ch",
    )
    ax1.set_xlabel("Residual tangential shift (channels)")
    ax1.set_ylabel("Misfit (ms)")
    ax1.set_title("Global tangential shift scan")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_obs_pred_plot(pred_rows: pd.DataFrame, out_path: Path) -> None:
    groups = list(pred_rows.groupby(["location", "anchor_index"], sort=True))
    n = len(groups)

    fig, axes = plt.subplots(n, 1, figsize=(12, max(3 * n, 6)), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, ((loc, anchor), g) in zip(axes, groups):
        g = g.sort_values("channel")
        ax.plot(g["channel"], 1000.0 * g["fit_dt_s"], label="Observed (fit target)", linewidth=1.4)
        ax.plot(g["channel"], 1000.0 * g["pred_dt_s"], label="Predicted", linewidth=1.4)
        ax.set_ylabel("dt to ref (ms)")
        ax.set_title(f"{loc} | anchor {anchor}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Channel")
    fig.suptitle("Observed vs predicted relative arrivals after global tangential shift", y=0.995, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_residual_channel_plot(pred_rows: pd.DataFrame, out_path: Path) -> None:
    g = pred_rows.copy()
    g["residual_ms"] = 1000.0 * g["residual_s"]

    def w_rmse(df: pd.DataFrame) -> float:
        w = df["fit_weight"].to_numpy(dtype=float)
        r = df["residual_ms"].to_numpy(dtype=float)
        return float(np.sqrt(np.sum(w * r**2) / max(np.sum(w), 1e-12)))

    def w_mae(df: pd.DataFrame) -> float:
        w = df["fit_weight"].to_numpy(dtype=float)
        r = np.abs(df["residual_ms"].to_numpy(dtype=float))
        return float(np.sum(w * r) / max(np.sum(w), 1e-12))

    summary = (
        g.groupby("channel", sort=True)
        .apply(
            lambda df: pd.Series({
                "rmse_ms": w_rmse(df),
                "mae_ms": w_mae(df),
                "median_abs_ms": float(np.median(np.abs(df["residual_ms"]))),
                "n": int(len(df)),
            })
        )
        .reset_index()
    )

    plt.figure(figsize=(12, 5))
    plt.plot(summary["channel"], summary["rmse_ms"], label="Weighted RMSE")
    plt.plot(summary["channel"], summary["median_abs_ms"], label="Median |residual|")
    plt.plot(summary["channel"], summary["mae_ms"], label="Weighted MAE")
    plt.xlabel("Channel")
    plt.ylabel("Residual (ms)")
    plt.title("Timing misfit by channel after global tangential shift")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def make_shifted_depth_plot(
    prior: pd.DataFrame,
    interp_map: dict[str, interp1d],
    best_shift: float,
    fixed_prior_offset: float,
    out_path: Path,
) -> None:
    ch = prior["channel"].to_numpy(dtype=float)
    shifted = evaluate_shifted_prior(
        ch,
        best_shift,
        interp_map,
        fixed_prior_offset_ch=fixed_prior_offset,
    )
    total_shift = fixed_prior_offset + best_shift

    if "cum_dist_3d_m" in prior.columns:
        xaxis = prior["cum_dist_3d_m"].to_numpy(dtype=float)
        xlabel = "Cumulative 3D distance (m)"
    elif "cum_dist_horizontal_m" in prior.columns:
        xaxis = prior["cum_dist_horizontal_m"].to_numpy(dtype=float)
        xlabel = "Cumulative horizontal distance (m)"
    else:
        xaxis = ch
        xlabel = "Channel"

    plt.figure(figsize=(12, 5))
    plt.plot(xaxis, prior["prior_u_m"], label="Prior depth/u")
    plt.plot(
        xaxis,
        shifted["prior_u_m"],
        label=f"Shifted prior depth/u (total shift={total_shift:.2f} ch)",
    )
    plt.xlabel(xlabel)
    plt.ylabel("Up / depth-like coordinate (m)")
    plt.title("Depth profile before and after global tangential shift")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a global tangential shift of the prior cable path.")
    parser.add_argument(
        "--obs-csv",
        type=Path,
        default=Path(r"D:\Singapore Data\Cable\inversion_observations.csv"),
    )
    parser.add_argument(
        "--prior-csv",
        type=Path,
        default=Path(r"D:\Singapore Data\Cable\prior_geometry.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"D:\Singapore Data\Cable\global_tangential_shift_outputs"),
    )
    parser.add_argument("--sound-speed", type=float, default=DEFAULT_SOUND_SPEED_MPS)
    parser.add_argument("--shift-min", type=float, default=-200.0)
    parser.add_argument("--shift-max", type=float, default=200.0)
    parser.add_argument("--shift-step", type=float, default=1.0)
    parser.add_argument(
        "--fixed-prior-offset",
        type=float,
        default=0.0,
        help="Fixed global channel offset applied to the prior before scanning residual shift.",
    )
    parser.add_argument("--min-weight", type=float, default=0.15)
    parser.add_argument("--min-stable-fraction", type=float, default=0.5)
    parser.add_argument("--all-usable", action="store_true", help="Use all usable rows instead of only recommended channels.")
    parser.add_argument("--use-raw", action="store_true", help="Use raw observed_dt_ref_s instead of median_smooth_offset_ms.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    obs, prior = load_inputs(args.obs_csv, args.prior_csv)
    prior = prepare_prior(prior)
    obs = prepare_observations(
        obs,
        min_weight=args.min_weight,
        min_stable_fraction=args.min_stable_fraction,
        use_only_recommended=not args.all_usable,
        use_smoothed=not args.use_raw,
    )

    interp_map = make_prior_interpolators(prior)
    prior_channels = prior["channel"].to_numpy(dtype=int)

    scan_df, best_shift_grid, best_metrics_grid, best_pred_grid = scan_shifts(
        obs=obs,
        prior_channels=prior_channels,
        interp_map=interp_map,
        sound_speed=args.sound_speed,
        shift_min=args.shift_min,
        shift_max=args.shift_max,
        shift_step=args.shift_step,
        fixed_prior_offset_ch=args.fixed_prior_offset,
    )

    best_shift_refined = refine_shift_parabolic(scan_df)

    _, best_metrics, best_pred = misfit_for_shift(
        best_shift_refined,
        obs,
        prior_channels,
        interp_map,
        args.sound_speed,
        fixed_prior_offset_ch=args.fixed_prior_offset,
    )

    shifted = evaluate_shifted_prior(
        prior_channels.astype(float),
        best_shift_refined,
        interp_map,
        fixed_prior_offset_ch=args.fixed_prior_offset,
    )

    shifted_prior_df = prior[["channel"]].copy()
    shifted_prior_df["fixed_prior_offset_ch"] = args.fixed_prior_offset
    shifted_prior_df["residual_shift_best_ch"] = best_shift_refined
    shifted_prior_df["total_effective_shift_ch"] = args.fixed_prior_offset + best_shift_refined
    shifted_prior_df["prior_x_m_shifted"] = shifted["prior_x_m"]
    shifted_prior_df["prior_y_m_shifted"] = shifted["prior_y_m"]
    shifted_prior_df["prior_u_m_shifted"] = shifted["prior_u_m"]
    shifted_prior_df["effective_prior_channel"] = shifted["channel_effective"]

    scan_df.to_csv(args.output_dir / "shift_scan_summary.csv", index=False)
    best_pred.to_csv(args.output_dir / "predicted_vs_observed_rows.csv", index=False)
    best_metrics["group"].to_csv(args.output_dir / "group_misfit_summary.csv", index=False)
    shifted_prior_df.to_csv(args.output_dir / "shifted_prior_geometry.csv", index=False)

    fit_metrics = {
        "fixed_prior_offset_ch": float(args.fixed_prior_offset),
        "best_shift_grid_channels": float(best_shift_grid),
        "best_shift_refined_channels": float(best_shift_refined),
        "total_effective_shift_grid_channels": float(args.fixed_prior_offset + best_shift_grid),
        "total_effective_shift_refined_channels": float(args.fixed_prior_offset + best_shift_refined),
        "n_fit_rows": int(len(obs)),
        "shift_min": float(args.shift_min),
        "shift_max": float(args.shift_max),
        "shift_step": float(args.shift_step),
        "use_smoothed": bool(not args.use_raw),
        "use_only_recommended": bool(not args.all_usable),
        **best_metrics["overall"],
    }

    with open(args.output_dir / "fit_metrics.json", "w", encoding="utf-8") as f:
        json.dump(fit_metrics, f, indent=2)

    make_shift_scan_plot(
        scan_df,
        best_shift_refined,
        args.fixed_prior_offset,
        args.output_dir / "shift_scan_curve.png",
    )
    make_path_plot(
        prior,
        interp_map,
        best_shift_refined,
        args.fixed_prior_offset,
        args.output_dir / "prior_vs_shifted_path.png",
    )
    make_obs_pred_plot(best_pred, args.output_dir / "observed_vs_predicted_by_location_anchor.png")
    make_residual_channel_plot(best_pred, args.output_dir / "residual_by_channel.png")
    make_shifted_depth_plot(
        prior,
        interp_map,
        best_shift_refined,
        args.fixed_prior_offset,
        args.output_dir / "depth_profile_prior_vs_shifted.png",
    )

    print(f"Saved outputs to: {args.output_dir}")
    print(f"Rows used in fit: {len(obs)}")
    print(f"Fixed prior offset:     {args.fixed_prior_offset:.3f} channels")
    print(f"Best residual shift:    {best_shift_grid:.3f} channels (grid)")
    print(f"Best residual shift:    {best_shift_refined:.3f} channels (refined)")
    print(f"Total effective shift:  {args.fixed_prior_offset + best_shift_refined:.3f} channels")
    print(f"Weighted RMSE:          {best_metrics['overall']['weighted_rmse_ms']:.3f} ms")
    print(f"Weighted MAE:           {best_metrics['overall']['weighted_mae_ms']:.3f} ms")
    print(f"Median |residual|:      {best_metrics['overall']['median_abs_ms']:.3f} ms")


if __name__ == "__main__":
    main()