
import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import least_squares

CHANNEL_SPACING_DEFAULT = 1.02
SOUND_SPEED_DEFAULT = 1500.0
CHANNEL_OFFSET_DEFAULT = 0


def weighted_median(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return np.nan
    values = values[mask]
    weights = weights[mask]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights) / np.sum(weights)
    return values[np.searchsorted(cdf, 0.5)]


def huber_scale(x, delta):
    ax = np.abs(x)
    out = np.where(ax <= delta, x, delta * np.sign(x) * np.sqrt(ax / delta))
    return out


def build_prior_geometry(df, channel_offset):
    geom = (
        df.groupby("channel")[["prior_x_m", "prior_y_m", "prior_u_m"]]
        .first()
        .reset_index()
        .rename(columns={"channel": "raw_channel"})
    )
    geom["channel"] = geom["raw_channel"] + channel_offset
    geom = geom.sort_values("channel").reset_index(drop=True)
    return geom[["channel", "prior_x_m", "prior_y_m", "prior_u_m"]]


def linear_fill_to_full_channels(prior_geom):
    full_ch = np.arange(prior_geom["channel"].min(), prior_geom["channel"].max() + 1)
    out = pd.DataFrame({"channel": full_ch})
    for col in ["prior_x_m", "prior_y_m", "prior_u_m"]:
        f = interp1d(
            prior_geom["channel"].values,
            prior_geom[col].values,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        out[col] = f(full_ch)
    return out


def build_observation_table(df, channel_offset):
    obs = df.copy()

    bool_cols = ["use_observation", "passed_snr_threshold", "near_window_edge", "recommended_channel", "base_valid"]
    for c in bool_cols:
        if c in obs.columns and obs[c].dtype == object:
            obs[c] = obs[c].astype(str).str.upper().map({"TRUE": True, "FALSE": False})

    obs["channel_eff"] = obs["channel"] + channel_offset
    obs["reference_channel_eff"] = obs["reference_channel"] + channel_offset
    obs["anchor_id"] = obs["location"].astype(str) + "_a" + obs["anchor_index"].astype(str)

    keep = np.ones(len(obs), dtype=bool)
    if "use_observation" in obs.columns:
        keep &= obs["use_observation"].fillna(False).values.astype(bool)
    if "weight" in obs.columns:
        keep &= np.isfinite(obs["weight"].values)
        keep &= obs["weight"].values > 0

    numeric_needed = [
        "channel_eff", "reference_channel_eff",
        "observed_t_s", "observed_dt_ref_s", "tx_x_m", "tx_y_m", "tx_u_m",
        "weight"
    ]
    for c in numeric_needed:
        obs[c] = pd.to_numeric(obs[c], errors="coerce")
        keep &= np.isfinite(obs[c].values)

    obs = obs.loc[keep].copy().reset_index(drop=True)
    return obs


def choose_control_channels(full_channels, reference_channels, spacing):
    start = int(full_channels.min())
    end = int(full_channels.max())
    ctrl = list(range(start, end + 1, int(spacing)))
    if ctrl[-1] != end:
        ctrl.append(end)
    ctrl = set(ctrl)
    ctrl.add(start)
    ctrl.add(end)
    for rc in reference_channels:
        rc = int(rc)
        nearest = int(round((rc - start) / spacing) * spacing + start)
        nearest = min(max(nearest, start), end)
        ctrl.add(rc)
        ctrl.add(nearest)
    ctrl = np.array(sorted(ctrl), dtype=int)
    return ctrl


def interpolate_curve(ctrl_channels, ctrl_xyz, eval_channels):
    x = CubicSpline(ctrl_channels, ctrl_xyz[:, 0], bc_type="natural")(eval_channels)
    y = CubicSpline(ctrl_channels, ctrl_xyz[:, 1], bc_type="natural")(eval_channels)

    fz = interp1d(
        ctrl_channels,
        ctrl_xyz[:, 2],
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    z = fz(eval_channels)

    return np.column_stack([x, y, z])

def make_channel_lookup(channels):
    channels = np.asarray(channels, dtype=int)
    return {int(ch): i for i, ch in enumerate(channels)}


def maybe_add_latlon(geom_df, origin_lat, origin_lon, origin_h):
    try:
        from pyproj import Transformer
        # local ENU to geodetic through ECEF
        to_ecef = Transformer.from_crs("epsg:4979", "epsg:4978", always_xy=True)
        from_ecef = Transformer.from_crs("epsg:4978", "epsg:4979", always_xy=True)
        lon0, lat0, h0 = origin_lon, origin_lat, origin_h
        x0, y0, z0 = to_ecef.transform(lon0, lat0, h0)

        lat0r = np.deg2rad(lat0)
        lon0r = np.deg2rad(lon0)
        slat, clat = np.sin(lat0r), np.cos(lat0r)
        slon, clon = np.sin(lon0r), np.cos(lon0r)

        R = np.array([
            [-slon, -slat * clon, clat * clon],
            [clon, -slat * slon, clat * slon],
            [0.0, clat, slat],
        ])  # ECEF->ENU

        Rt = R.T  # ENU->ECEF

        enu = geom_df[["x_m", "y_m", "z_m"]].values
        ecef = enu @ Rt + np.array([x0, y0, z0])
        lon, lat, h = from_ecef.transform(ecef[:, 0], ecef[:, 1], ecef[:, 2])
        geom_df["lat_deg"] = lat
        geom_df["lon_deg"] = lon
        geom_df["h_m"] = h
    except Exception as exc:
        warnings.warn(f"Could not compute lat/lon output (pyproj unavailable or failed): {exc}")
    return geom_df


def solve_inversion(
    obs,
    prior_full,
    control_channels,
    sound_speed=1500.0,
    channel_spacing=1.02,
    abs_scale=0.003,
    rel_scale=0.0015,
    prior_sigma_xy=25.0,
    prior_sigma_z=6.0,
    curvature_sigma_xy=3.0,
    curvature_sigma_z=1.5,
    spacing_sigma=0.08,
    anchor_bias_sigma=0.02,
    huber_delta_abs=3.0,
    huber_delta_rel=3.0,
    max_nfev=250,
):
    full_channels = prior_full["channel"].values.astype(int)
    full_lookup = make_channel_lookup(full_channels)

    ctrl_lookup_full_idx = np.array([full_lookup[int(c)] for c in control_channels], dtype=int)
    prior_xyz_full = prior_full[["prior_x_m", "prior_y_m", "prior_u_m"]].values.astype(float)
    prior_xyz_ctrl = prior_xyz_full[ctrl_lookup_full_idx]

    anchors = np.array(sorted(obs["anchor_id"].unique()))
    anchor_to_idx = {a: i for i, a in enumerate(anchors)}

    obs_ch_idx = np.array([full_lookup[int(c)] for c in obs["channel_eff"].values], dtype=int)
    ref_ch_idx = np.array([full_lookup[int(c)] for c in obs["reference_channel_eff"].values], dtype=int)
    anchor_idx = np.array([anchor_to_idx[a] for a in obs["anchor_id"].values], dtype=int)
    weights = obs["weight"].values.astype(float)
    sqrtw = np.sqrt(np.clip(weights, 1e-8, None))

    tx_xyz = obs[["tx_x_m", "tx_y_m", "tx_u_m"]].values.astype(float)
    obs_t_abs = obs["observed_t_s"].values.astype(float)
    obs_t_rel = obs["observed_dt_ref_s"].values.astype(float)

    # initial anchor time biases from prior geometry
    pred_abs_prior = np.linalg.norm(tx_xyz - prior_xyz_full[obs_ch_idx], axis=1) / sound_speed
    init_biases = np.zeros(len(anchors))
    for a_i in range(len(anchors)):
        m = anchor_idx == a_i
        init_biases[a_i] = weighted_median(obs_t_abs[m] - pred_abs_prior[m], weights[m])
        if not np.isfinite(init_biases[a_i]):
            init_biases[a_i] = 0.0

    x0 = np.concatenate([
        np.zeros(len(control_channels)),
        np.zeros(len(control_channels)),
        np.zeros(len(control_channels)),
        init_biases,
    ])

    def unpack(p):
        n = len(control_channels)
        dx = p[0:n]
        dy = p[n:2*n]
        dz = p[2*n:3*n]
        bias = p[3*n:]
        ctrl_xyz = prior_xyz_ctrl + np.column_stack([dx, dy, dz])
        full_xyz = interpolate_curve(control_channels, ctrl_xyz, full_channels)
        return ctrl_xyz, full_xyz, bias

    def residual_vector(p):
        ctrl_xyz, full_xyz, bias = unpack(p)

        xyz_obs = full_xyz[obs_ch_idx]
        xyz_ref = full_xyz[ref_ch_idx]

        pred_abs = np.linalg.norm(tx_xyz - xyz_obs, axis=1) / sound_speed + bias[anchor_idx]
        pred_rel = (
            np.linalg.norm(tx_xyz - xyz_obs, axis=1) -
            np.linalg.norm(tx_xyz - xyz_ref, axis=1)
        ) / sound_speed

        abs_res = (obs_t_abs - pred_abs) / abs_scale
        rel_res = (obs_t_rel - pred_rel) / rel_scale

        abs_res = sqrtw * huber_scale(abs_res, huber_delta_abs)
        rel_res = sqrtw * huber_scale(rel_res, huber_delta_rel)

        # Prior displacement penalty
        dxyz = ctrl_xyz - prior_xyz_ctrl
        prior_pen = np.concatenate([
            dxyz[:, 0] / prior_sigma_xy,
            dxyz[:, 1] / prior_sigma_xy,
            dxyz[:, 2] / prior_sigma_z,
        ])

        # Smoothness / curvature on control points
        d2 = ctrl_xyz[:-2] - 2.0 * ctrl_xyz[1:-1] + ctrl_xyz[2:]
        curv_pen = np.concatenate([
            d2[:, 0] / curvature_sigma_xy,
            d2[:, 1] / curvature_sigma_xy,
            d2[:, 2] / curvature_sigma_z,
        ])

        # Segment-length penalty on full channel-by-channel geometry
        seg = np.linalg.norm(np.diff(full_xyz, axis=0), axis=1)
        spacing_pen = (seg - channel_spacing) / spacing_sigma

        # Weak anchor-bias penalty so absolute times help but do not dominate badly
        bias_pen = bias / anchor_bias_sigma

        return np.concatenate([abs_res, rel_res, prior_pen, curv_pen, spacing_pen, bias_pen])

    result = least_squares(
        residual_vector,
        x0=x0,
        method="trf",
        loss="linear",
        max_nfev=max_nfev,
        verbose=2,
    )

    ctrl_xyz_opt, full_xyz_opt, bias_opt = unpack(result.x)

    out = {
        "result": result,
        "anchors": anchors,
        "anchor_bias_s": bias_opt,
        "control_channels": control_channels,
        "control_xyz_prior": prior_xyz_ctrl,
        "control_xyz_opt": ctrl_xyz_opt,
        "full_channels": full_channels,
        "prior_xyz_full": prior_xyz_full,
        "full_xyz_opt": full_xyz_opt,
        "obs_indices": obs_ch_idx,
        "ref_indices": ref_ch_idx,
        "anchor_idx": anchor_idx,
        "tx_xyz": tx_xyz,
        "weights": weights,
        "obs_t_abs": obs_t_abs,
        "obs_t_rel": obs_t_rel,
        "sound_speed": sound_speed,
    }
    return out


def compute_fit_diagnostics(solution):
    full_xyz = solution["full_xyz_opt"]
    prior_xyz = solution["prior_xyz_full"]
    obs_idx = solution["obs_indices"]
    ref_idx = solution["ref_indices"]
    anchor_idx = solution["anchor_idx"]
    tx_xyz = solution["tx_xyz"]
    bias = solution["anchor_bias_s"]
    sound_speed = solution["sound_speed"]

    pred_abs = np.linalg.norm(tx_xyz - full_xyz[obs_idx], axis=1) / sound_speed + bias[anchor_idx]
    pred_abs_prior = np.linalg.norm(tx_xyz - prior_xyz[obs_idx], axis=1) / sound_speed + bias[anchor_idx]

    pred_rel = (
        np.linalg.norm(tx_xyz - full_xyz[obs_idx], axis=1) -
        np.linalg.norm(tx_xyz - full_xyz[ref_idx], axis=1)
    ) / sound_speed
    pred_rel_prior = (
        np.linalg.norm(tx_xyz - prior_xyz[obs_idx], axis=1) -
        np.linalg.norm(tx_xyz - prior_xyz[ref_idx], axis=1)
    ) / sound_speed

    return {
        "pred_abs": pred_abs,
        "pred_abs_prior": pred_abs_prior,
        "pred_rel": pred_rel,
        "pred_rel_prior": pred_rel_prior,
    }


def save_outputs(obs, prior_full, solution, diagnostics, output_dir, origin_lat, origin_lon, origin_h):
    os.makedirs(output_dir, exist_ok=True)

    full_channels = solution["full_channels"]
    prior_xyz = solution["prior_xyz_full"]
    full_xyz = solution["full_xyz_opt"]

    cable = pd.DataFrame({
        "channel": full_channels,
        "prior_x_m": prior_xyz[:, 0],
        "prior_y_m": prior_xyz[:, 1],
        "prior_z_m": prior_xyz[:, 2],
        "x_m": full_xyz[:, 0],
        "y_m": full_xyz[:, 1],
        "z_m": full_xyz[:, 2],
    })
    cable["dx_m"] = cable["x_m"] - cable["prior_x_m"]
    cable["dy_m"] = cable["y_m"] - cable["prior_y_m"]
    cable["dz_m"] = cable["z_m"] - cable["prior_z_m"]
    cable["horizontal_shift_m"] = np.sqrt(cable["dx_m"]**2 + cable["dy_m"]**2)

    cable = maybe_add_latlon(cable, origin_lat, origin_lon, origin_h)
    cable.to_csv(os.path.join(output_dir, "updated_cable_layout.csv"), index=False)

    ctrl = pd.DataFrame({
        "channel": solution["control_channels"],
        "prior_x_m": solution["control_xyz_prior"][:, 0],
        "prior_y_m": solution["control_xyz_prior"][:, 1],
        "prior_z_m": solution["control_xyz_prior"][:, 2],
        "x_m": solution["control_xyz_opt"][:, 0],
        "y_m": solution["control_xyz_opt"][:, 1],
        "z_m": solution["control_xyz_opt"][:, 2],
    })
    ctrl.to_csv(os.path.join(output_dir, "control_points_optimized.csv"), index=False)

    anchor_bias = pd.DataFrame({
        "anchor_id": solution["anchors"],
        "anchor_bias_s": solution["anchor_bias_s"],
    })
    anchor_bias.to_csv(os.path.join(output_dir, "anchor_biases.csv"), index=False)

    fit = obs.copy()
    fit["predicted_t_abs_s_prior"] = diagnostics["pred_abs_prior"]
    fit["predicted_t_abs_s_opt"] = diagnostics["pred_abs"]
    fit["residual_abs_prior_s"] = fit["observed_t_s"] - fit["predicted_t_abs_s_prior"]
    fit["residual_abs_opt_s"] = fit["observed_t_s"] - fit["predicted_t_abs_s_opt"]

    fit["predicted_dt_ref_s_prior"] = diagnostics["pred_rel_prior"]
    fit["predicted_dt_ref_s_opt"] = diagnostics["pred_rel"]
    fit["residual_dt_ref_prior_s"] = fit["observed_dt_ref_s"] - fit["predicted_dt_ref_s_prior"]
    fit["residual_dt_ref_opt_s"] = fit["observed_dt_ref_s"] - fit["predicted_dt_ref_s_opt"]

    fit.to_csv(os.path.join(output_dir, "observation_fit_diagnostics.csv"), index=False)

    summary = pd.DataFrame({
        "metric": [
            "n_observations",
            "n_control_points",
            "cost",
            "success",
            "nfev",
            "rmse_abs_prior_ms",
            "rmse_abs_opt_ms",
            "rmse_rel_prior_ms",
            "rmse_rel_opt_ms",
            "weighted_rmse_abs_opt_ms",
            "weighted_rmse_rel_opt_ms",
            "median_horizontal_shift_m",
            "p95_horizontal_shift_m",
        ],
        "value": [
            len(obs),
            len(solution["control_channels"]),
            solution["result"].cost,
            bool(solution["result"].success),
            solution["result"].nfev,
            1000.0 * np.sqrt(np.mean(fit["residual_abs_prior_s"]**2)),
            1000.0 * np.sqrt(np.mean(fit["residual_abs_opt_s"]**2)),
            1000.0 * np.sqrt(np.mean(fit["residual_dt_ref_prior_s"]**2)),
            1000.0 * np.sqrt(np.mean(fit["residual_dt_ref_opt_s"]**2)),
            1000.0 * np.sqrt(np.average(fit["residual_abs_opt_s"]**2, weights=fit["weight"])),
            1000.0 * np.sqrt(np.average(fit["residual_dt_ref_opt_s"]**2, weights=fit["weight"])),
            cable["horizontal_shift_m"].median(),
            cable["horizontal_shift_m"].quantile(0.95),
        ]
    })
    summary.to_csv(os.path.join(output_dir, "inversion_summary.csv"), index=False)


def make_plots(obs, solution, diagnostics, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    full_channels = solution["full_channels"]
    prior_xyz = solution["prior_xyz_full"]
    full_xyz = solution["full_xyz_opt"]
    ctrl_prior = solution["control_xyz_prior"]
    ctrl_opt = solution["control_xyz_opt"]
    ctrl_ch = solution["control_channels"]

    # Rebuild TX table
    tx_tbl = (
        obs.groupby("anchor_id")[["tx_x_m", "tx_y_m", "tx_u_m"]]
        .first()
        .reset_index()
    )

    # 1. Plan view
    plt.figure(figsize=(10, 8))
    plt.plot(prior_xyz[:, 0], prior_xyz[:, 1], label="Prior cable")
    plt.plot(full_xyz[:, 0], full_xyz[:, 1], label="Inverted cable")
    plt.scatter(ctrl_prior[:, 0], ctrl_prior[:, 1], s=20, label="Prior control pts")
    plt.scatter(ctrl_opt[:, 0], ctrl_opt[:, 1], s=20, label="Optimized control pts")
    plt.scatter(tx_tbl["tx_x_m"], tx_tbl["tx_y_m"], marker="x", s=70, label="Transmitters")
    for _, row in tx_tbl.iterrows():
        plt.text(row["tx_x_m"], row["tx_y_m"], row["anchor_id"], fontsize=8)
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.title("Cable layout: prior vs inverted")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_plan_view.png"), dpi=200)
    plt.close()

    # 2. Depth/channel
    plt.figure(figsize=(12, 5))
    plt.plot(full_channels, prior_xyz[:, 2], label="Prior z")
    plt.plot(full_channels, full_xyz[:, 2], label="Inverted z")
    plt.scatter(ctrl_ch, ctrl_prior[:, 2], s=15, label="Prior control pts")
    plt.scatter(ctrl_ch, ctrl_opt[:, 2], s=15, label="Optimized control pts")
    plt.xlabel("Channel")
    plt.ylabel("Up (m)")
    plt.title("Depth profile")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_depth_profile.png"), dpi=200)
    plt.close()

    # 3. Horizontal shift
    horiz = np.sqrt((full_xyz[:, 0] - prior_xyz[:, 0])**2 + (full_xyz[:, 1] - prior_xyz[:, 1])**2)
    plt.figure(figsize=(12, 5))
    plt.plot(full_channels, horiz)
    plt.xlabel("Channel")
    plt.ylabel("Horizontal shift (m)")
    plt.title("Horizontal displacement from prior")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_horizontal_shift.png"), dpi=200)
    plt.close()

    # 4. Residual histograms
    abs_prior = obs["observed_t_s"].values - diagnostics["pred_abs_prior"]
    abs_opt = obs["observed_t_s"].values - diagnostics["pred_abs"]
    rel_prior = obs["observed_dt_ref_s"].values - diagnostics["pred_rel_prior"]
    rel_opt = obs["observed_dt_ref_s"].values - diagnostics["pred_rel"]

    plt.figure(figsize=(10, 6))
    plt.hist(1000 * abs_prior, bins=80, alpha=0.5, label="Abs prior")
    plt.hist(1000 * abs_opt, bins=80, alpha=0.5, label="Abs inverted")
    plt.xlabel("Residual (ms)")
    plt.ylabel("Count")
    plt.title("Absolute-time residuals")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_abs_residual_hist.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(1000 * rel_prior, bins=80, alpha=0.5, label="Relative prior")
    plt.hist(1000 * rel_opt, bins=80, alpha=0.5, label="Relative inverted")
    plt.xlabel("Residual (ms)")
    plt.ylabel("Count")
    plt.title("Relative-time residuals")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_rel_residual_hist.png"), dpi=200)
    plt.close()

    # 5. Residual vs channel
    plt.figure(figsize=(12, 5))
    plt.scatter(obs["channel_eff"], 1000 * rel_opt, s=8, alpha=0.35)
    plt.xlabel("Channel")
    plt.ylabel("Relative residual (ms)")
    plt.title("Relative residual after inversion")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_rel_residual_vs_channel.png"), dpi=200)
    plt.close()

    # 6. Observed vs predicted by anchor
    anchor_ids = sorted(obs["anchor_id"].unique())
    ncol = 2
    nrow = int(np.ceil(len(anchor_ids) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 4 * nrow), squeeze=False)
    for ax, aid in zip(axes.ravel(), anchor_ids):
        m = obs["anchor_id"] == aid
        ax.scatter(obs.loc[m, "observed_dt_ref_s"], diagnostics["pred_rel"][m], s=8, alpha=0.5)
        lims = [
            min(obs.loc[m, "observed_dt_ref_s"].min(), diagnostics["pred_rel"][m].min()),
            max(obs.loc[m, "observed_dt_ref_s"].max(), diagnostics["pred_rel"][m].max()),
        ]
        ax.plot(lims, lims)
        ax.set_title(aid)
        ax.set_xlabel("Observed dt_ref (s)")
        ax.set_ylabel("Predicted dt_ref (s)")
    for ax in axes.ravel()[len(anchor_ids):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "plot_observed_vs_predicted_dt_by_anchor.png"), dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Invert DAS cable layout from sweep arrival times.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to inversion_observations.csv")
    parser.add_argument("--output_dir", type=str, default="cable_inversion_output")
    parser.add_argument("--channel_offset", type=int, default=CHANNEL_OFFSET_DEFAULT)
    parser.add_argument("--control_spacing", type=int, default=75)
    parser.add_argument("--sound_speed", type=float, default=SOUND_SPEED_DEFAULT)
    parser.add_argument("--channel_spacing", type=float, default=CHANNEL_SPACING_DEFAULT)
    parser.add_argument("--prior_sigma_xy", type=float, default=25.0)
    parser.add_argument("--prior_sigma_z", type=float, default=6.0)
    parser.add_argument("--curvature_sigma_xy", type=float, default=3.0)
    parser.add_argument("--curvature_sigma_z", type=float, default=1.5)
    parser.add_argument("--spacing_sigma", type=float, default=0.08)
    parser.add_argument("--anchor_bias_sigma", type=float, default=0.02)
    parser.add_argument("--abs_scale", type=float, default=0.003)
    parser.add_argument("--rel_scale", type=float, default=0.0015)
    parser.add_argument("--max_nfev", type=int, default=250)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    origin_lat = float(df["enu_origin_lat_deg"].dropna().iloc[0])
    origin_lon = float(df["enu_origin_lon_deg"].dropna().iloc[0])
    origin_h = float(df["enu_origin_h_m"].dropna().iloc[0])

    obs = build_observation_table(df, args.channel_offset)
    prior_geom_sparse = build_prior_geometry(df, args.channel_offset)
    prior_full = linear_fill_to_full_channels(prior_geom_sparse)

    min_ch, max_ch = prior_full["channel"].min(), prior_full["channel"].max()
    obs = obs[(obs["channel_eff"] >= min_ch) & (obs["channel_eff"] <= max_ch)].copy()
    obs = obs[(obs["reference_channel_eff"] >= min_ch) & (obs["reference_channel_eff"] <= max_ch)].copy()

    control_channels = choose_control_channels(
        prior_full["channel"].values,
        obs["reference_channel_eff"].unique(),
        args.control_spacing
    )

    solution = solve_inversion(
        obs=obs,
        prior_full=prior_full,
        control_channels=control_channels,
        sound_speed=args.sound_speed,
        channel_spacing=args.channel_spacing,
        abs_scale=args.abs_scale,
        rel_scale=args.rel_scale,
        prior_sigma_xy=args.prior_sigma_xy,
        prior_sigma_z=args.prior_sigma_z,
        curvature_sigma_xy=args.curvature_sigma_xy,
        curvature_sigma_z=args.curvature_sigma_z,
        spacing_sigma=args.spacing_sigma,
        anchor_bias_sigma=args.anchor_bias_sigma,
        max_nfev=args.max_nfev,
    )

    diagnostics = compute_fit_diagnostics(solution)
    save_outputs(obs, prior_full, solution, diagnostics, args.output_dir, origin_lat, origin_lon, origin_h)
    make_plots(obs, solution, diagnostics, args.output_dir)

    print("\nDone.")
    print(f"Output written to: {os.path.abspath(args.output_dir)}")
    print(f"Success: {solution['result'].success}")
    print(f"Message: {solution['result'].message}")
    print(f"Cost: {solution['result'].cost:.6f}")
    print(f"Nfev: {solution['result'].nfev}")


if __name__ == "__main__":
    main()
