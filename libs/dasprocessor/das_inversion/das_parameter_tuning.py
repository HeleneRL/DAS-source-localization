import os
import json
import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from das_cable_inversion import (
    build_observation_table,
    build_prior_geometry,
    linear_fill_to_full_channels,
    choose_control_channels,
    solve_inversion,
    compute_fit_diagnostics,
)


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def weighted_rmse(residuals, weights):
    residuals = np.asarray(residuals, dtype=float)
    weights = np.asarray(weights, dtype=float)
    m = np.isfinite(residuals) & np.isfinite(weights) & (weights > 0)
    if not np.any(m):
        return np.nan
    return np.sqrt(np.average(residuals[m] ** 2, weights=weights[m]))


def percentile_abs(residuals, q):
    residuals = np.asarray(residuals, dtype=float)
    m = np.isfinite(residuals)
    if not np.any(m):
        return np.nan
    return np.percentile(np.abs(residuals[m]), q)


def summarize_geometry(solution):
    prior_xyz = solution["prior_xyz_full"]
    full_xyz = solution["full_xyz_opt"]
    dxyz = full_xyz - prior_xyz
    horiz = np.sqrt(dxyz[:, 0] ** 2 + dxyz[:, 1] ** 2)
    dz = dxyz[:, 2]
    seg = np.linalg.norm(np.diff(full_xyz, axis=0), axis=1)
    return {
        "median_horizontal_shift_m": float(np.median(horiz)),
        "p95_horizontal_shift_m": float(np.percentile(horiz, 95)),
        "max_horizontal_shift_m": float(np.max(horiz)),
        "median_abs_vertical_shift_m": float(np.median(np.abs(dz))),
        "p95_abs_vertical_shift_m": float(np.percentile(np.abs(dz), 95)),
        "segment_spacing_mean_m": float(np.mean(seg)),
        "segment_spacing_std_m": float(np.std(seg)),
    }


def score_run(metrics, weights_dict):
    # Lower is better.
    # Main emphasis: relative timing fit, robust fit, and avoiding implausible geometry.
    return (
        weights_dict["w_rel"] * metrics["weighted_rmse_rel_opt_ms"]
        + weights_dict["w_abs"] * metrics["weighted_rmse_abs_opt_ms"]
        + weights_dict["w_rel_p95"] * metrics["p95_abs_rel_opt_ms"]
        + weights_dict["w_abs_p95"] * metrics["p95_abs_abs_opt_ms"]
        + weights_dict["w_hshift"] * metrics["p95_horizontal_shift_m"]
        + weights_dict["w_vshift"] * metrics["p95_abs_vertical_shift_m"]
        + weights_dict["w_bias"] * metrics["anchor_bias_rms_ms"]
    )


# ------------------------------------------------------------
# Data preparation
# ------------------------------------------------------------

def load_and_prepare(csv_path, channel_offset=0, min_weight=0.8):
    df = pd.read_csv(csv_path)

    obs = build_observation_table(df, channel_offset)
    if "weight" not in obs.columns:
        raise ValueError("Input observations must contain a 'weight' column.")

    # Hard filter: only trust strong arrivals.
    obs = obs[obs["weight"] >= min_weight].copy()

    prior_geom_sparse = build_prior_geometry(df, channel_offset)
    prior_full = linear_fill_to_full_channels(prior_geom_sparse)

    min_ch, max_ch = prior_full["channel"].min(), prior_full["channel"].max()
    obs = obs[(obs["channel_eff"] >= min_ch) & (obs["channel_eff"] <= max_ch)].copy()
    obs = obs[(obs["reference_channel_eff"] >= min_ch) & (obs["reference_channel_eff"] <= max_ch)].copy()
    obs = obs.reset_index(drop=True)

    if len(obs) == 0:
        raise ValueError("No observations remain after filtering. Lower min_weight or inspect the CSV.")

    meta = {
        "origin_lat": safe_float(df.get("enu_origin_lat_deg", pd.Series([np.nan])).dropna().iloc[0]) if "enu_origin_lat_deg" in df else np.nan,
        "origin_lon": safe_float(df.get("enu_origin_lon_deg", pd.Series([np.nan])).dropna().iloc[0]) if "enu_origin_lon_deg" in df else np.nan,
        "origin_h": safe_float(df.get("enu_origin_h_m", pd.Series([np.nan])).dropna().iloc[0]) if "enu_origin_h_m" in df else np.nan,
        "n_rows_raw": int(len(df)),
        "n_obs_used": int(len(obs)),
        "min_weight": float(min_weight),
    }
    return obs, prior_full, meta


# ------------------------------------------------------------
# Single run
# ------------------------------------------------------------

def run_one_config(obs, prior_full, cfg, max_nfev=250):
    control_channels = choose_control_channels(
        prior_full["channel"].values,
        obs["reference_channel_eff"].unique(),
        cfg["control_spacing"],
    )

    solution = solve_inversion(
        obs=obs,
        prior_full=prior_full,
        control_channels=control_channels,
        sound_speed=cfg["sound_speed"],
        channel_spacing=cfg["channel_spacing"],
        abs_scale=cfg["abs_scale"],
        rel_scale=cfg["rel_scale"],
        prior_sigma_xy=cfg["prior_sigma_xy"],
        prior_sigma_z=cfg["prior_sigma_z"],
        curvature_sigma_xy=cfg["curvature_sigma_xy"],
        curvature_sigma_z=cfg["curvature_sigma_z"],
        spacing_sigma=cfg["spacing_sigma"],
        anchor_bias_sigma=cfg["anchor_bias_sigma"],
        huber_delta_abs=cfg["huber_delta_abs"],
        huber_delta_rel=cfg["huber_delta_rel"],
        max_nfev=max_nfev,
    )

    diagnostics = compute_fit_diagnostics(solution)

    obs_abs_res_opt = obs["observed_t_s"].values - diagnostics["pred_abs"]
    obs_rel_res_opt = obs["observed_dt_ref_s"].values - diagnostics["pred_rel"]
    obs_abs_res_prior = obs["observed_t_s"].values - diagnostics["pred_abs_prior"]
    obs_rel_res_prior = obs["observed_dt_ref_s"].values - diagnostics["pred_rel_prior"]
    weights = obs["weight"].values

    geom = summarize_geometry(solution)
    bias_ms = 1000.0 * solution["anchor_bias_s"]

    row = {
        **cfg,
        "n_obs": int(len(obs)),
        "n_control_points": int(len(control_channels)),
        "success": bool(solution["result"].success),
        "status": int(solution["result"].status),
        "message": str(solution["result"].message),
        "cost": float(solution["result"].cost),
        "nfev": int(solution["result"].nfev),
        "rmse_abs_prior_ms": 1000.0 * float(np.sqrt(np.mean(obs_abs_res_prior ** 2))),
        "rmse_abs_opt_ms": 1000.0 * float(np.sqrt(np.mean(obs_abs_res_opt ** 2))),
        "rmse_rel_prior_ms": 1000.0 * float(np.sqrt(np.mean(obs_rel_res_prior ** 2))),
        "rmse_rel_opt_ms": 1000.0 * float(np.sqrt(np.mean(obs_rel_res_opt ** 2))),
        "weighted_rmse_abs_opt_ms": 1000.0 * float(weighted_rmse(obs_abs_res_opt, weights)),
        "weighted_rmse_rel_opt_ms": 1000.0 * float(weighted_rmse(obs_rel_res_opt, weights)),
        "p95_abs_abs_opt_ms": 1000.0 * float(percentile_abs(obs_abs_res_opt, 95)),
        "p95_abs_rel_opt_ms": 1000.0 * float(percentile_abs(obs_rel_res_opt, 95)),
        "anchor_bias_rms_ms": float(np.sqrt(np.mean(bias_ms ** 2))) if len(bias_ms) else np.nan,
        "anchor_bias_max_abs_ms": float(np.max(np.abs(bias_ms))) if len(bias_ms) else np.nan,
        **geom,
    }

    return row, solution, diagnostics


# ------------------------------------------------------------
# Plots
# ------------------------------------------------------------

def plot_metric_vs_param(results_df, param, metric, out_png):
    good = results_df[np.isfinite(results_df[param]) & np.isfinite(results_df[metric])].copy()
    if len(good) == 0:
        return

    plt.figure(figsize=(8, 5))
    x = good[param].values
    y = good[metric].values
    plt.scatter(x, y)

    order = np.argsort(x)
    plt.plot(x[order], y[order], alpha=0.7)
    best_idx = np.argmin(y)
    plt.scatter([x[best_idx]], [y[best_idx]], s=100)
    plt.xlabel(param)
    plt.ylabel(metric)
    plt.title(f"{metric} vs {param}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_top_runs(results_df, out_png, top_n=20):
    good = results_df[np.isfinite(results_df["score"])].sort_values("score").head(top_n).copy()
    if len(good) == 0:
        return

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(good)), good["score"].values, marker="o")
    plt.xlabel("Rank")
    plt.ylabel("Composite score")
    plt.title(f"Top {len(good)} runs by score")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_pair_scatter(results_df, xcol, ycol, ccol, out_png):
    good = results_df[
        np.isfinite(results_df[xcol])
        & np.isfinite(results_df[ycol])
        & np.isfinite(results_df[ccol])
    ].copy()
    if len(good) == 0:
        return

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(good[xcol], good[ycol], c=good[ccol])
    plt.colorbar(sc, label=ccol)
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(f"{ycol} vs {xcol} colored by {ccol}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_best_geometry_comparison(best_solution, out_png):
    prior_xyz = best_solution["prior_xyz_full"]
    full_xyz = best_solution["full_xyz_opt"]
    ctrl_prior = best_solution["control_xyz_prior"]
    ctrl_opt = best_solution["control_xyz_opt"]

    plt.figure(figsize=(9, 7))
    plt.plot(prior_xyz[:, 0], prior_xyz[:, 1], label="Prior cable")
    plt.plot(full_xyz[:, 0], full_xyz[:, 1], label="Best tuned cable")
    plt.scatter(ctrl_prior[:, 0], ctrl_prior[:, 1], s=20, label="Prior control pts")
    plt.scatter(ctrl_opt[:, 0], ctrl_opt[:, 1], s=20, label="Optimized control pts")
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.title("Best tuned geometry: plan view")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_best_residuals(obs, best_diagnostics, out_png_abs, out_png_rel):
    abs_res = 1000.0 * (obs["observed_t_s"].values - best_diagnostics["pred_abs"])
    rel_res = 1000.0 * (obs["observed_dt_ref_s"].values - best_diagnostics["pred_rel"])

    plt.figure(figsize=(8, 5))
    plt.hist(abs_res, bins=70, alpha=0.8)
    plt.xlabel("Absolute residual (ms)")
    plt.ylabel("Count")
    plt.title("Best tuned run: absolute residuals")
    plt.tight_layout()
    plt.savefig(out_png_abs, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(rel_res, bins=70, alpha=0.8)
    plt.xlabel("Relative residual (ms)")
    plt.ylabel("Count")
    plt.title("Best tuned run: relative residuals")
    plt.tight_layout()
    plt.savefig(out_png_rel, dpi=200)
    plt.close()


# ------------------------------------------------------------
# Search grids
# ------------------------------------------------------------

def build_grid_from_args(args):
    # Coarse ranges around the values you gave.
    grid = {
        "sound_speed": args.sound_speed_grid,
        "control_spacing": args.control_spacing_grid,
        "channel_spacing": [args.channel_spacing],
        "abs_scale": [args.abs_scale],
        "rel_scale": [args.rel_scale],
        "prior_sigma_xy": args.prior_sigma_xy_grid,
        "prior_sigma_z": args.prior_sigma_z_grid,
        "curvature_sigma_xy": args.curvature_sigma_xy_grid,
        "curvature_sigma_z": args.curvature_sigma_z_grid,
        "spacing_sigma": args.spacing_sigma_grid,
        "anchor_bias_sigma": args.anchor_bias_sigma_grid,
        "huber_delta_abs": [args.huber_delta_abs],
        "huber_delta_rel": [args.huber_delta_rel],
    }
    return grid


def grid_to_configs(grid):
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    for combo in itertools.product(*vals):
        yield {k: v for k, v in zip(keys, combo)}


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Coarse tuner for DAS cable inversion parameters.")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--channel_offset", type=int, default=0)
    parser.add_argument("--min_weight", type=float, default=0.8, help="Only use arrivals with weight >= this value.")
    parser.add_argument("--max_nfev", type=int, default=250)
    parser.add_argument("--top_n_to_save", type=int, default=20)

    # Fixed unless you want to expand later.
    parser.add_argument("--channel_spacing", type=float, default=1.02)
    parser.add_argument("--abs_scale", type=float, default=0.003)
    parser.add_argument("--rel_scale", type=float, default=0.0015)
    parser.add_argument("--huber_delta_abs", type=float, default=2.0)
    parser.add_argument("--huber_delta_rel", type=float, default=2.0)

    # Coarse grids centered around your current values.
    parser.add_argument("--sound_speed_grid", type=float, nargs="+", default=[1480.0, 1490.0, 1500.0, 1510.0, 1520.0])
    parser.add_argument("--control_spacing_grid", type=int, nargs="+", default=[25, 40, 60, 80])
    parser.add_argument("--prior_sigma_xy_grid", type=float, nargs="+", default=[30.0, 60.0, 90.0])
    parser.add_argument("--curvature_sigma_xy_grid", type=float, nargs="+", default=[4.0, 8.0, 12.0])
    parser.add_argument("--spacing_sigma_grid", type=float, nargs="+", default=[0.04, 0.08, 0.12])
    parser.add_argument("--prior_sigma_z_grid", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    parser.add_argument("--curvature_sigma_z_grid", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    parser.add_argument("--anchor_bias_sigma_grid", type=float, nargs="+", default=[0.01, 0.02, 0.05])

    # Composite score weights.
    parser.add_argument("--score_w_rel", type=float, default=1.0)
    parser.add_argument("--score_w_abs", type=float, default=0.35)
    parser.add_argument("--score_w_rel_p95", type=float, default=0.40)
    parser.add_argument("--score_w_abs_p95", type=float, default=0.10)
    parser.add_argument("--score_w_hshift", type=float, default=0.03)
    parser.add_argument("--score_w_vshift", type=float, default=0.03)
    parser.add_argument("--score_w_bias", type=float, default=0.02)

    args = parser.parse_args()
    ensure_dir(args.output_dir)
    ensure_dir(os.path.join(args.output_dir, "plots"))

    obs, prior_full, meta = load_and_prepare(
        csv_path=args.input_csv,
        channel_offset=args.channel_offset,
        min_weight=args.min_weight,
    )

    score_weights = {
        "w_rel": args.score_w_rel,
        "w_abs": args.score_w_abs,
        "w_rel_p95": args.score_w_rel_p95,
        "w_abs_p95": args.score_w_abs_p95,
        "w_hshift": args.score_w_hshift,
        "w_vshift": args.score_w_vshift,
        "w_bias": args.score_w_bias,
    }

    grid = build_grid_from_args(args)
    configs = list(grid_to_configs(grid))

    print(f"Using {len(obs)} observations after filtering weight >= {args.min_weight}.")
    print(f"Running {len(configs)} parameter combinations...")

    rows = []
    best_solution = None
    best_diagnostics = None
    best_score = np.inf
    failures = []

    for i, cfg in enumerate(configs, start=1):
        print(f"[{i}/{len(configs)}] {cfg}")
        try:
            row, solution, diagnostics = run_one_config(obs, prior_full, cfg, max_nfev=args.max_nfev)
            row["score"] = score_run(row, score_weights)
            rows.append(row)

            if np.isfinite(row["score"]) and row["score"] < best_score:
                best_score = row["score"]
                best_solution = solution
                best_diagnostics = diagnostics
        except Exception as exc:
            fail_row = {**cfg, "error": str(exc)}
            failures.append(fail_row)
            print(f"    FAILED: {exc}")

    results_df = pd.DataFrame(rows)
    failures_df = pd.DataFrame(failures)

    if len(results_df) == 0:
        raise RuntimeError("All tuning runs failed. Check the input data and parameter ranges.")

    results_df = results_df.sort_values("score").reset_index(drop=True)
    results_df.to_csv(os.path.join(args.output_dir, "tuning_results_ranked.csv"), index=False)
    failures_df.to_csv(os.path.join(args.output_dir, "tuning_failures.csv"), index=False)

    with open(os.path.join(args.output_dir, "tuning_setup.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": meta,
                "score_weights": score_weights,
                "grid": grid,
                "n_successful_runs": int(len(results_df)),
                "n_failed_runs": int(len(failures_df)),
            },
            f,
            indent=2,
        )

    best_row = results_df.iloc[0].to_dict()
    pd.DataFrame([best_row]).to_csv(os.path.join(args.output_dir, "best_run_summary.csv"), index=False)
    results_df.head(args.top_n_to_save).to_csv(os.path.join(args.output_dir, "top_runs.csv"), index=False)

    plots_dir = os.path.join(args.output_dir, "plots")
    plot_top_runs(results_df, os.path.join(plots_dir, "plot_top_scores.png"), top_n=min(20, len(results_df)))

    for param in [
        "sound_speed",
        "control_spacing",
        "prior_sigma_xy",
        "curvature_sigma_xy",
        "spacing_sigma",
        "prior_sigma_z",
        "curvature_sigma_z",
        "anchor_bias_sigma",
    ]:
        plot_metric_vs_param(results_df, param, "weighted_rmse_rel_opt_ms", os.path.join(plots_dir, f"weighted_rmse_rel_vs_{param}.png"))
        plot_metric_vs_param(results_df, param, "weighted_rmse_abs_opt_ms", os.path.join(plots_dir, f"weighted_rmse_abs_vs_{param}.png"))
        plot_metric_vs_param(results_df, param, "score", os.path.join(plots_dir, f"score_vs_{param}.png"))

    plot_pair_scatter(
        results_df,
        "sound_speed",
        "weighted_rmse_rel_opt_ms",
        "control_spacing",
        os.path.join(plots_dir, "pair_sound_speed_vs_rel_rmse_colored_control_spacing.png"),
    )
    plot_pair_scatter(
        results_df,
        "prior_sigma_xy",
        "weighted_rmse_rel_opt_ms",
        "curvature_sigma_xy",
        os.path.join(plots_dir, "pair_priorxy_vs_rel_rmse_colored_curvxy.png"),
    )
    plot_pair_scatter(
        results_df,
        "prior_sigma_z",
        "weighted_rmse_rel_opt_ms",
        "curvature_sigma_z",
        os.path.join(plots_dir, "pair_priorz_vs_rel_rmse_colored_curvz.png"),
    )

    if best_solution is not None and best_diagnostics is not None:
        plot_best_geometry_comparison(best_solution, os.path.join(plots_dir, "best_geometry_plan_view.png"))
        plot_best_residuals(
            obs,
            best_diagnostics,
            os.path.join(plots_dir, "best_abs_residual_hist.png"),
            os.path.join(plots_dir, "best_rel_residual_hist.png"),
        )

    print("\nDone.")
    print(f"Successful runs: {len(results_df)}")
    print(f"Failed runs: {len(failures_df)}")
    print("Best run:")
    for k, v in best_row.items():
        print(f"  {k}: {v}")
    print(f"\nResults written to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
