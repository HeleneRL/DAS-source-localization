import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def latlon_to_local_xy(lat, lon, lat0, lon0):
    """
    Convert lat/lon to local XY in meters using equirectangular approximation.
    Good enough for small areas.
    """
    R = 6371000.0
    lat = np.radians(lat)
    lon = np.radians(lon)
    lat0 = np.radians(lat0)
    lon0 = np.radians(lon0)

    x = (lon - lon0) * np.cos(lat0) * R
    y = (lat - lat0) * R
    return x, y


def cumulative_arclength(x, y):
    ds = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    return s


def project_points_onto_polyline(px, py, x_line, y_line, s_line):
    """
    For each point (px, py), find the nearest projection onto the truth polyline.
    Returns:
        dist: nearest distance (m)
        proj_x, proj_y: projected point coordinates
        proj_s: arclength coordinate of projection on truth polyline
        seg_idx: segment index used
        seg_t: parametric coordinate on segment [0, 1]
    """
    n_pts = len(px)
    dist = np.full(n_pts, np.inf)
    proj_x = np.full(n_pts, np.nan)
    proj_y = np.full(n_pts, np.nan)
    proj_s = np.full(n_pts, np.nan)
    seg_idx = np.full(n_pts, -1, dtype=int)
    seg_t = np.full(n_pts, np.nan)

    for i in range(len(x_line) - 1):
        x1, y1 = x_line[i], y_line[i]
        x2, y2 = x_line[i + 1], y_line[i + 1]
        dx = x2 - x1
        dy = y2 - y1
        seg_len2 = dx * dx + dy * dy

        if seg_len2 == 0:
            t = np.zeros_like(px)
            qx = np.full_like(px, x1, dtype=float)
            qy = np.full_like(py, y1, dtype=float)
        else:
            t = ((px - x1) * dx + (py - y1) * dy) / seg_len2
            t = np.clip(t, 0.0, 1.0)
            qx = x1 + t * dx
            qy = y1 + t * dy

        d = np.sqrt((px - qx) ** 2 + (py - qy) ** 2)
        mask = d < dist

        dist[mask] = d[mask]
        proj_x[mask] = qx[mask]
        proj_y[mask] = qy[mask]
        proj_s[mask] = s_line[i] + t[mask] * np.sqrt(seg_len2)
        seg_idx[mask] = i
        seg_t[mask] = t[mask]

    return dist, proj_x, proj_y, proj_s, seg_idx, seg_t


def interp_along_s(s_query, s_ref, v_ref):
    return np.interp(s_query, s_ref, v_ref)


def get_truth_channel_column(df):
    for col in ["channel", "ch", "Channel", "CHAN", "chan"]:
        if col in df.columns:
            return col
    raise ValueError("Could not find truth channel column. Expected one of: channel, ch, Channel, CHAN, chan")


def main():
    parser = argparse.ArgumentParser(description="Compare estimated cable to truth and diagnose channel/progression mismatch.")
    parser.add_argument("--estimated_csv", required=True, help="Estimated layout CSV, e.g. updated_cable_layout.csv")
    parser.add_argument("--truth_csv", required=True, help="Truth layout CSV, e.g. array-shape.csv")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--channel_offset_est", type=float, default=0.0,
                        help="Optional constant offset applied to estimated channel numbers before comparing to truth.")
    parser.add_argument("--highlight_distance_m", type=float, default=10.0,
                        help="Threshold for highlighting distance outliers")
    parser.add_argument("--highlight_shift_ch", type=float, default=10.0,
                        help="Threshold for highlighting channel-shift outliers")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    est = pd.read_csv(args.estimated_csv)
    truth = pd.read_csv(args.truth_csv)

    if "lat_deg" not in est.columns or "lon_deg" not in est.columns:
        raise ValueError("Estimated CSV must contain lat_deg and lon_deg columns.")

    truth_ch_col = get_truth_channel_column(truth)

    if "lat" not in truth.columns or "lon" not in truth.columns:
        raise ValueError("Truth CSV must contain lat and lon columns.")

    if "channel" not in est.columns:
        raise ValueError("Estimated CSV must contain channel column.")

    est_lat = est["lat_deg"].to_numpy()
    est_lon = est["lon_deg"].to_numpy()
    est_ch = est["channel"].to_numpy().astype(float) + args.channel_offset_est

    truth_lat = truth["lat"].to_numpy()
    truth_lon = truth["lon"].to_numpy()
    truth_ch = truth[truth_ch_col].to_numpy().astype(float)

    lat0 = np.mean(np.concatenate([est_lat, truth_lat]))
    lon0 = np.mean(np.concatenate([est_lon, truth_lon]))

    est_x, est_y = latlon_to_local_xy(est_lat, est_lon, lat0, lon0)
    truth_x, truth_y = latlon_to_local_xy(truth_lat, truth_lon, lat0, lon0)

    s_est = cumulative_arclength(est_x, est_y)
    s_truth = cumulative_arclength(truth_x, truth_y)

    # Project estimated points onto truth polyline
    dist, proj_x, proj_y, proj_s_truth, seg_idx, seg_t = project_points_onto_polyline(
        est_x, est_y, truth_x, truth_y, s_truth
    )

    # Matched truth channel from projected truth arclength
    truth_ch_matched = interp_along_s(proj_s_truth, s_truth, truth_ch)

    # Arc-length mismatch as progression difference after normalizing lengths
    est_frac = s_est / s_est[-1] if s_est[-1] > 0 else np.zeros_like(s_est)
    truth_frac_at_proj = proj_s_truth / s_truth[-1] if s_truth[-1] > 0 else np.zeros_like(proj_s_truth)
    frac_mismatch = truth_frac_at_proj - est_frac

    # Convert progression mismatch to meters on truth and approx channels on truth
    arclength_mismatch_m = frac_mismatch * s_truth[-1]
    truth_channels_per_meter = (truth_ch[-1] - truth_ch[0]) / s_truth[-1] if s_truth[-1] > 0 else 0.0
    progression_shift_ch = arclength_mismatch_m * truth_channels_per_meter

    # Absolute channel shift against matched truth channel
    channel_shift = truth_ch_matched - est_ch

    out = pd.DataFrame({
        "est_index": np.arange(len(est)),
        "est_channel": est["channel"].to_numpy(),
        "est_channel_with_offset": est_ch,
        "est_lat_deg": est_lat,
        "est_lon_deg": est_lon,
        "est_x_m": est_x,
        "est_y_m": est_y,
        "est_s_m": s_est,
        "truth_proj_x_m": proj_x,
        "truth_proj_y_m": proj_y,
        "truth_proj_s_m": proj_s_truth,
        "truth_matched_channel": truth_ch_matched,
        "nearest_distance_m": dist,
        "est_progress_frac": est_frac,
        "truth_progress_frac_at_proj": truth_frac_at_proj,
        "arclength_mismatch_m": arclength_mismatch_m,
        "progression_shift_ch": progression_shift_ch,
        "channel_shift_truth_minus_est": channel_shift,
    })
    out.to_csv(os.path.join(args.output_dir, "channel_progression_diagnostics.csv"), index=False)

    metrics = {
        "n_est_points": len(est),
        "mean_distance_m": float(np.mean(dist)),
        "median_distance_m": float(np.median(dist)),
        "p95_distance_m": float(np.percentile(dist, 95)),
        "max_distance_m": float(np.max(dist)),
        "mean_abs_channel_shift": float(np.mean(np.abs(channel_shift))),
        "median_abs_channel_shift": float(np.median(np.abs(channel_shift))),
        "p95_abs_channel_shift": float(np.percentile(np.abs(channel_shift), 95)),
        "mean_abs_arclength_mismatch_m": float(np.mean(np.abs(arclength_mismatch_m))),
        "median_abs_arclength_mismatch_m": float(np.median(np.abs(arclength_mismatch_m))),
        "p95_abs_arclength_mismatch_m": float(np.percentile(np.abs(arclength_mismatch_m), 95)),
    }
    pd.DataFrame([metrics]).to_csv(os.path.join(args.output_dir, "channel_progression_metrics.csv"), index=False)

    print("Done.")
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")

    # 1) Channel shift vs estimated channel
    plt.figure(figsize=(12, 4))
    plt.plot(est["channel"].to_numpy(), channel_shift, lw=1.5)
    plt.axhline(0, color="k", lw=1)
    plt.axhline(args.highlight_shift_ch, color="r", ls="--", lw=1)
    plt.axhline(-args.highlight_shift_ch, color="r", ls="--", lw=1)
    plt.xlabel("Estimated channel")
    plt.ylabel("Matched truth channel - estimated channel")
    plt.xlim(460,2200)
    plt.title("Channel shift vs channel")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "channel_shift_vs_channel.png"), dpi=200)
    plt.close()

    # 2) Arc-length mismatch vs estimated channel
    plt.figure(figsize=(12, 4))
    plt.plot(est["channel"].to_numpy(), arclength_mismatch_m, lw=1.5)
    plt.axhline(0, color="k", lw=1)
    plt.xlabel("Estimated channel")
    plt.ylabel("Progression mismatch (m along truth)")
    plt.title("Arc-length progression mismatch vs channel")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "arclength_mismatch_vs_channel.png"), dpi=200)
    plt.close()

    # 3) Distance vs channel
    plt.figure(figsize=(12, 4))
    plt.plot(est["channel"].to_numpy(), dist, lw=1.5)
    plt.axhline(args.highlight_distance_m, color="r", ls="--", lw=1)
    plt.xlabel("Estimated channel")
    plt.ylabel("Nearest distance to truth (m)")
    plt.title("Nearest geometric mismatch vs channel")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "distance_vs_channel.png"), dpi=200)
    plt.close()

    # 4) Overlay colored by channel shift
    plt.figure(figsize=(8, 8))
    plt.plot(truth_x, truth_y, "k-", lw=1.5, label="Truth")
    sc = plt.scatter(est_x, est_y, c=channel_shift, s=8, cmap="coolwarm")
    plt.colorbar(sc, label="Matched truth channel - estimated channel")
    plt.title("Estimated geometry colored by channel shift")
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "overlay_channel_shift_colored.png"), dpi=200)
    plt.close()

    # 5) Overlay with only large channel-shift points highlighted
    shift_mask = np.abs(channel_shift) > args.highlight_shift_ch
    plt.figure(figsize=(8, 8))
    plt.plot(truth_x, truth_y, "b-", lw=2, label="Truth")
    plt.plot(est_x, est_y, color="orange", lw=2, label="Estimate")
    if np.any(shift_mask):
        plt.scatter(est_x[shift_mask], est_y[shift_mask], c="red", s=14, label="Large channel mismatch")
    plt.title("Overlay with channel-mismatch outliers highlighted")
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "overlay_channel_shift_outliers.png"), dpi=200)
    plt.close()

    # 6) Quiver plot showing progression mismatch direction
    plt.figure(figsize=(8, 8))
    plt.plot(truth_x, truth_y, "b-", lw=2, label="Truth")
    plt.plot(est_x, est_y, color="orange", lw=2, label="Estimate")
    plt.quiver(
        est_x, est_y,
        proj_x - est_x,
        proj_y - est_y,
        angles="xy", scale_units="xy", scale=1, width=0.0018, alpha=0.8
    )
    plt.title("Estimate → projected truth")
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "overlay_projection_vectors.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    main()