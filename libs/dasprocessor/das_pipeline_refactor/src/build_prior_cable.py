from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymap3d as pm

from common import load_toml, ensure_dir, path_from_cfg

try:
    from scipy.spatial import cKDTree
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def read_boattrack(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"X", "Y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Boattrack CSV is missing columns: {missing}")
    out = pd.DataFrame({"lon": pd.to_numeric(df["X"], errors="coerce"), "lat": pd.to_numeric(df["Y"], errors="coerce")})
    out = out.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    if len(out) < 2:
        raise ValueError("Boattrack must contain at least 2 valid points.")
    return out


def read_cable_estimate(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"lat", "lon", "z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cable estimate CSV is missing columns: {missing}")
    out = pd.DataFrame({
        "lat": pd.to_numeric(df["lat"], errors="coerce"),
        "lon": pd.to_numeric(df["lon"], errors="coerce"),
        "depth": pd.to_numeric(df["z"], errors="coerce"),
    })
    out = out.dropna(subset=["lat", "lon", "depth"]).reset_index(drop=True)
    if len(out) < 1:
        raise ValueError("Cable estimate must contain at least 1 valid point.")
    return out


def latlon_to_local_xy(lat, lon, lat0, lon0):
    e, n, _ = pm.geodetic2enu(lat, lon, 0.0, lat0, lon0, 0.0)
    return np.asarray(e), np.asarray(n)


def local_xy_to_latlon(e, n, lat0, lon0):
    lat, lon, _ = pm.enu2geodetic(e, n, 0.0, lat0, lon0, 0.0)
    return np.asarray(lat), np.asarray(lon)


def remove_duplicate_consecutive_points(x, y, z=None):
    pts = np.column_stack([x, y]) if z is None else np.column_stack([x, y, z])
    keep = np.ones(len(pts), dtype=bool)
    keep[1:] = np.any(np.diff(pts, axis=0) != 0, axis=1)
    pts2 = pts[keep]
    if z is None:
        return pts2[:, 0], pts2[:, 1]
    return pts2[:, 0], pts2[:, 1], pts2[:, 2]


def cumulative_length_2d(x, y):
    seglen = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    return s, seglen


def cumulative_length_3d(x, y, z):
    seglen = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    return s, seglen


def moving_average_centered(y, window):
    if window <= 1:
        return y.copy()
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    ypad = np.pad(y, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(ypad, kernel, mode="valid")


def choose_odd_window_from_distance(s, target_window_m, min_window_samples=3):
    if len(s) < 2:
        return 1
    positive_ds = np.diff(s)
    positive_ds = positive_ds[positive_ds > 0]
    if len(positive_ds) == 0:
        return 1
    window = int(round(target_window_m / np.median(positive_ds)))
    window = max(window, min_window_samples)
    if window % 2 == 0:
        window += 1
    return window


def assign_depth_from_nearest(boat_df: pd.DataFrame, cable_df: pd.DataFrame):
    lat0 = float(np.mean(np.r_[boat_df["lat"].values, cable_df["lat"].values]))
    lon0 = float(np.mean(np.r_[boat_df["lon"].values, cable_df["lon"].values]))

    boat_e, boat_n = latlon_to_local_xy(boat_df["lat"].values, boat_df["lon"].values, lat0, lon0)
    cable_e, cable_n = latlon_to_local_xy(cable_df["lat"].values, cable_df["lon"].values, lat0, lon0)

    boat_xy = np.column_stack([boat_e, boat_n])
    cable_xy = np.column_stack([cable_e, cable_n])

    if HAVE_SCIPY:
        tree = cKDTree(cable_xy)
        _, idx = tree.query(boat_xy, k=1)
    else:
        idx = [int(np.argmin(np.sum((cable_xy - p) ** 2, axis=1))) for p in boat_xy]
        idx = np.asarray(idx)

    out = boat_df.copy()
    out["depth_raw"] = cable_df["depth"].values[idx]
    return out, lat0, lon0


def interpolate_along_polyline_3d(x, y, z, distances):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    distances = np.asarray(distances, dtype=float)

    s, seglen = cumulative_length_3d(x, y, z)
    total_len = s[-1]
    valid = np.where(seglen > 0)[0]
    if len(valid) == 0:
        raise ValueError("All polyline segments have zero length.")

    last_i = valid[-1]
    p0 = np.array([x[last_i], y[last_i], z[last_i]])
    p1 = np.array([x[last_i + 1], y[last_i + 1], z[last_i + 1]])
    last_dir = (p1 - p0) / np.linalg.norm(p1 - p0)

    xi = np.empty_like(distances)
    yi = np.empty_like(distances)
    zi = np.empty_like(distances)

    inside = distances <= total_len
    outside = ~inside

    if np.any(inside):
        d = distances[inside]
        j = np.searchsorted(s, d, side="right") - 1
        j = np.clip(j, 0, len(s) - 2)
        seg_start = s[j]
        seg_end = s[j + 1]
        seg_len = seg_end - seg_start
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (d - seg_start) / seg_len
        t = np.where(seg_len > 0, t, 0.0)

        xi[inside] = x[j] + t * (x[j + 1] - x[j])
        yi[inside] = y[j] + t * (y[j + 1] - y[j])
        zi[inside] = z[j] + t * (z[j + 1] - z[j])

    if np.any(outside):
        extra = distances[outside] - total_len
        end_point = np.array([x[-1], y[-1], z[-1]])
        pts = end_point[None, :] + extra[:, None] * last_dir[None, :]
        xi[outside], yi[outside], zi[outside] = pts[:, 0], pts[:, 1], pts[:, 2]

    return xi, yi, zi, total_len


def main() -> None:
    parser = argparse.ArgumentParser(description="Build channel-indexed prior cable geometry.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_toml(args.config)
    pcfg = cfg["prior_geometry"]
    outdir = ensure_dir(path_from_cfg(cfg, "prior_output_dir"))

    boat_df = read_boattrack(Path(cfg["paths"]["boattrack_csv"]))
    cable_df = read_cable_estimate(Path(cfg["paths"]["cable_estimate_csv"]))
    boat_with_depth, lat0, lon0 = assign_depth_from_nearest(boat_df, cable_df)

    boat_x, boat_y = latlon_to_local_xy(boat_with_depth["lat"].values, boat_with_depth["lon"].values, lat0, lon0)
    cable_x, cable_y = latlon_to_local_xy(cable_df["lat"].values, cable_df["lon"].values, lat0, lon0)

    boat_depth_raw = boat_with_depth["depth_raw"].values.astype(float)
    cable_depth = cable_df["depth"].values.astype(float)

    boat_x, boat_y, boat_depth_raw = remove_duplicate_consecutive_points(boat_x, boat_y, boat_depth_raw)
    boat_s2d, _ = cumulative_length_2d(boat_x, boat_y)

    if bool(pcfg["smooth_depth"]):
        window_samples = choose_odd_window_from_distance(
            boat_s2d, float(pcfg["smooth_window_m"]), int(pcfg["min_window_samples"])
        )
        boat_depth = moving_average_centered(boat_depth_raw, window_samples)
    else:
        boat_depth = boat_depth_raw.copy()

    first_channel = int(pcfg["first_channel"])
    last_channel = int(pcfg["last_channel"])
    channel_spacing_m = float(pcfg["channel_spacing_m"])

    n_channels = last_channel - first_channel + 1
    channel_ids = np.arange(first_channel, last_channel + 1)
    channel_s3d = np.arange(n_channels, dtype=float) * channel_spacing_m

    interp_x, interp_y, interp_depth, total_len_3d = interpolate_along_polyline_3d(
        boat_x, boat_y, boat_depth, channel_s3d
    )
    interp_lat, interp_lon = local_xy_to_latlon(interp_x, interp_y, lat0, lon0)

    interp_df = pd.DataFrame({
        "channel": channel_ids,
        "lat": interp_lat,
        "lon": interp_lon,
        "depth": interp_depth,
        "x_local": interp_x,
        "y_local": interp_y,
        "s_3d": channel_s3d,
    })
    interp_df[["channel", "lat", "lon", "depth"]].to_csv(outdir / "prior_cable_by_channel.csv", index=False)

    cable_x, cable_y, cable_depth = remove_duplicate_consecutive_points(cable_x, cable_y, cable_depth)
    cable_s3d, _ = cumulative_length_3d(cable_x, cable_y, cable_depth)
    cable_lat, cable_lon = local_xy_to_latlon(cable_x, cable_y, lat0, lon0)
    boat_lat_plot, boat_lon_plot = local_xy_to_latlon(boat_x, boat_y, lat0, lon0)

    cable_plot_df = pd.DataFrame({
        "lat": cable_lat, "lon": cable_lon, "depth": cable_depth, "x_local": cable_x, "y_local": cable_y, "s_3d": cable_s3d
    })
    boat_plot_df = pd.DataFrame({"lat": boat_lat_plot, "lon": boat_lon_plot})

    plt.figure(figsize=(10, 8))
    plt.plot(boat_plot_df["lon"], boat_plot_df["lat"], label="Boat track", linewidth=1.2)
    plt.plot(cable_plot_df["lon"], cable_plot_df["lat"], label="Cable estimate", linewidth=1.2)
    plt.plot(interp_df["lon"], interp_df["lat"], label="Interpolated prior", linewidth=2.0)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Prior cable geometry")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "prior_geometry_map.png", dpi=200)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(cable_plot_df["s_3d"], cable_plot_df["depth"], label="Cable estimate", linewidth=1.2)
    plt.plot(interp_df["s_3d"], interp_df["depth"], label="Interpolated prior", linewidth=2.0)
    plt.xlabel("Cumulative 3D distance [m]")
    plt.ylabel("Depth")
    plt.title("Prior cable depth profile")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "prior_geometry_depth.png", dpi=200)
    plt.close()

    print(f"Saved: {outdir / 'prior_cable_by_channel.csv'}")
    print(f"3D polyline length available: {total_len_3d:.2f} m")
    print(f"Length needed for channels: {channel_s3d[-1]:.2f} m")


if __name__ == "__main__":
    main()
