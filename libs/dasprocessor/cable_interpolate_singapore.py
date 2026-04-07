import os
import numpy as np
import pandas as pd
import pymap3d as pm
import matplotlib.pyplot as plt

# Optional fast nearest-neighbor search
try:
    from scipy.spatial import cKDTree
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


# =========================
# USER SETTINGS
# =========================
BOATTRACK_CSV = r"D:\Singapore Data\Cable\boattrack_estimate_hari.csv"
CABLE_ESTIMATE_CSV = r"D:\Singapore Data\array-shape.csv"
OUTPUT_CHANNEL_CSV = r"D:\Singapore Data\Cable\interpolated_channels_from_boattrack.csv"

PLOT_MAP_PNG = r"D:\Singapore Data\Cable\plot_horizontal_map.png"
PLOT_DEPTH_PNG = r"D:\Singapore Data\Cable\plot_depth_profile.png"
PLOT_3D_PNG = r"D:\Singapore Data\Cable\plot_3d_cables.png"
PLOT_H_MAP_DEPTH =  r"D:\Singapore Data\Cable\plot_horizontal_depth.png"

CHANNEL_SPACING_M = 1.02
FIRST_CHANNEL = 348
LAST_CHANNEL = 2267

# Depth smoothing settings
SMOOTH_DEPTH = False
SMOOTH_WINDOW_M = 15.0   # smoothing window in meters along the boat-track polyline
MIN_WINDOW_SAMPLES = 3   # minimum odd-number window size


# =========================
# HELPERS
# =========================
def read_boattrack(path: str) -> pd.DataFrame:
    """
    Reads boattrack CSV with headers: X, Y, Z, Name
    Interprets:
        X = lon
        Y = lat
        Z = altitude (ignored here, since it is always 0)
    """
    df = pd.read_csv(path)

    required = {"X", "Y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Boattrack CSV is missing columns: {missing}")

    out = pd.DataFrame({
        "lon": pd.to_numeric(df["X"], errors="coerce"),
        "lat": pd.to_numeric(df["Y"], errors="coerce"),
    })

    out = out.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    if len(out) < 2:
        raise ValueError("Boattrack must contain at least 2 valid points.")

    return out


def read_cable_estimate(path: str) -> pd.DataFrame:
    """
    Reads cable estimate CSV with headers:
        ch, x, y, z, lat, lon
    Uses lat, lon, z
    """
    df = pd.read_csv(path)

    required = {"lat", "lon", "z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cable estimate CSV is missing columns: {missing}")

    out = pd.DataFrame({
        "lat": pd.to_numeric(df["lat"], errors="coerce"),
        "lon": pd.to_numeric(df["lon"], errors="coerce"),
        "z": pd.to_numeric(df["z"], errors="coerce"),
    })

    # keep ch too if present
    if "ch" in df.columns:
        out["ch"] = pd.to_numeric(df["ch"], errors="coerce")

    out = out.dropna(subset=["lat", "lon", "z"]).reset_index(drop=True)

    if len(out) < 1:
        raise ValueError("Cable estimate must contain at least 1 valid point.")

    return out


def latlon_to_local_xy(lat, lon, lat0, lon0):
    """
    Convert lat/lon to local ENU coordinates (east, north) in meters.
    """
    e, n, _ = pm.geodetic2enu(lat, lon, 0.0, lat0, lon0, 0.0)
    return np.asarray(e), np.asarray(n)


def local_xy_to_latlon(e, n, lat0, lon0):
    """
    Convert local ENU east/north back to lat/lon.
    """
    lat, lon, _ = pm.enu2geodetic(e, n, 0.0, lat0, lon0, 0.0)
    return np.asarray(lat), np.asarray(lon)


def remove_duplicate_consecutive_points(x, y, z=None):
    """
    Remove consecutive duplicate points.
    If z is provided, uniqueness is checked in 3D, otherwise in 2D.
    """
    if z is None:
        pts = np.column_stack([x, y])
    else:
        pts = np.column_stack([x, y, z])

    keep = np.ones(len(pts), dtype=bool)
    keep[1:] = np.any(np.diff(pts, axis=0) != 0, axis=1)
    pts2 = pts[keep]

    if z is None:
        return pts2[:, 0], pts2[:, 1]
    else:
        return pts2[:, 0], pts2[:, 1], pts2[:, 2]


def cumulative_length_2d(x, y):
    dx = np.diff(x)
    dy = np.diff(y)
    seglen = np.sqrt(dx**2 + dy**2)
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    return s, seglen


def cumulative_length_3d(x, y, z):
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)
    seglen = np.sqrt(dx**2 + dy**2 + dz**2)
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    return s, seglen


def moving_average_centered(y, window):
    """
    Centered moving average with edge padding.
    window must be >= 1
    """
    if window <= 1:
        return y.copy()

    pad_left = window // 2
    pad_right = window - 1 - pad_left
    ypad = np.pad(y, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    ys = np.convolve(ypad, kernel, mode="valid")
    return ys


def choose_odd_window_from_distance(s, target_window_m, min_window_samples=3):
    """
    Estimate a window length in samples from cumulative distance.
    Returns an odd integer >= 1.
    """
    if len(s) < 2:
        return 1

    ds = np.diff(s)
    positive_ds = ds[ds > 0]
    if len(positive_ds) == 0:
        return 1

    median_ds = np.median(positive_ds)
    window = int(round(target_window_m / median_ds))
    window = max(window, min_window_samples)
    if window % 2 == 0:
        window += 1
    return window


def assign_depth_from_nearest(boat_df: pd.DataFrame, cable_df: pd.DataFrame):
    """
    For each boattrack point, copy z from the nearest cable-estimate point.
    Nearest search is done in local metric x/y coordinates.
    """
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
        idx = []
        for p in boat_xy:
            d2 = np.sum((cable_xy - p) ** 2, axis=1)
            idx.append(np.argmin(d2))
        idx = np.asarray(idx)

    out = boat_df.copy()
    out["depth_raw"] = cable_df["z"].values[idx]

    return out, lat0, lon0


def interpolate_along_polyline_3d(x, y, z, distances):
    """
    Interpolate 3D points at requested cumulative distances.
    If distance exceeds polyline length, extend along the last valid segment.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    distances = np.asarray(distances, dtype=float)

    if len(x) < 2:
        raise ValueError("Need at least 2 points to interpolate.")

    s, seglen = cumulative_length_3d(x, y, z)
    total_len = s[-1]

    valid = np.where(seglen > 0)[0]
    if len(valid) == 0:
        raise ValueError("All polyline segments have zero length.")

    last_i = valid[-1]
    p0 = np.array([x[last_i], y[last_i], z[last_i]])
    p1 = np.array([x[last_i + 1], y[last_i + 1], z[last_i + 1]])
    last_dir = p1 - p0
    last_dir = last_dir / np.linalg.norm(last_dir)

    xi = np.empty_like(distances, dtype=float)
    yi = np.empty_like(distances, dtype=float)
    zi = np.empty_like(distances, dtype=float)

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
        xi[outside] = pts[:, 0]
        yi[outside] = pts[:, 1]
        zi[outside] = pts[:, 2]

    return xi, yi, zi, total_len


def make_horizontal_plot(boat_df, cable_df, interp_df, out_png):
    plt.figure(figsize=(10, 8))
    plt.plot(boat_df["lon"], boat_df["lat"], label="Initial boat track", linewidth=1.5)
    plt.plot(cable_df["lon"], cable_df["lat"], label="Cable estimate, Mandar", linewidth=1.5)
    plt.plot(interp_df["lon"], interp_df["lat"], label="Interpolated channel position result", linewidth=2.0)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Horizontal cable geometry")
    plt.legend()
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    #plt.close()

def make_horizontal_with_depth(boat_df, cable_df, interp_df, out_png):
    plt.figure(figsize=(10, 8))

    # Background lines (no color)
    plt.plot(
        boat_df["lon"], boat_df["lat"],
        label="Initial boat track", linewidth=1.2, alpha=0.7
    )
    plt.plot(
        cable_df["lon"], cable_df["lat"],
        label="Cable estimate, Mandar", linewidth=1.2, alpha=0.7
    )

    # Colored interpolated cable
    sc = plt.scatter(
        interp_df["lon"], interp_df["lat"],
        c=interp_df["depth"],
        s=12,
        cmap="viridis"   # try "viridis_r" if you want reversed
    )

    # Colorbar
    cbar = plt.colorbar(sc)
    cbar.set_label("Depth / z")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Horizontal cable geometry (colored by depth)")
    plt.legend()
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(out_png, dpi=200)
    # plt.close()  # keep open if using plt.show()


def make_depth_profile_plot(cable_df, interp_df, out_png):
    plt.figure(figsize=(12, 6))
    plt.plot(cable_df["s_3d"], cable_df["depth"], label="Cable estimate, Mandar", linewidth=1.5)
    plt.plot(interp_df["s_3d"], interp_df["depth"], label="Interpolated channel position result", linewidth=2.0)

    plt.xlabel("Cumulative 3D distance [m]")
    plt.ylabel("Depth / z")
    plt.title("Depth profile")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    #plt.close()


def make_3d_plot(cable_df, interp_df, out_png):
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(cable_df["x_local"], cable_df["y_local"], cable_df["depth"],
            label="Cable estimate, Mandar", linewidth=1.5)
    ax.plot(interp_df["x_local"], interp_df["y_local"], interp_df["depth"],
            label="Interpolated channel positions", linewidth=2.0)

    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_zlabel("Depth / z")
    ax.set_title("Full 3D cable geometry")
    ax.legend()

    # Often nicer visually for depth to point downward
    try:
        zmin = min(cable_df["depth"].min(), interp_df["depth"].min())
        zmax = max(cable_df["depth"].max(), interp_df["depth"].max())
        ax.set_zlim(zmax, zmin)
    except Exception:
        pass

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    #plt.close()


# =========================
# MAIN
# =========================
def main():
    print("Reading input files...")
    boat_df = read_boattrack(BOATTRACK_CSV)
    cable_df = read_cable_estimate(CABLE_ESTIMATE_CSV)

    print(f"Boattrack points: {len(boat_df)}")
    print(f"Cable estimate points: {len(cable_df)}")

    print("Assigning depth to boattrack from nearest cable-estimate point...")
    boat_with_depth, lat0, lon0 = assign_depth_from_nearest(boat_df, cable_df)

    # Convert boat and cable data to local metric coordinates
    boat_x, boat_y = latlon_to_local_xy(
        boat_with_depth["lat"].values,
        boat_with_depth["lon"].values,
        lat0,
        lon0
    )

    cable_x, cable_y = latlon_to_local_xy(
        cable_df["lat"].values,
        cable_df["lon"].values,
        lat0,
        lon0
    )

    boat_depth_raw = boat_with_depth["depth_raw"].values.astype(float)
    cable_depth = cable_df["z"].values.astype(float)

    # Remove consecutive duplicates from boat track before smoothing/interpolating
    boat_x, boat_y, boat_depth_raw = remove_duplicate_consecutive_points(
        boat_x, boat_y, boat_depth_raw
    )

    if len(boat_x) < 2:
        raise ValueError("After removing duplicate boat-track points, fewer than 2 points remain.")

    # Smooth depth along boat track
    boat_s2d, _ = cumulative_length_2d(boat_x, boat_y)
    if SMOOTH_DEPTH:
        window_samples = choose_odd_window_from_distance(
            boat_s2d, SMOOTH_WINDOW_M, MIN_WINDOW_SAMPLES
        )
        boat_depth = moving_average_centered(boat_depth_raw, window_samples)
        print(f"Smoothing borrowed depth with centered moving average.")
        print(f"Target window: {SMOOTH_WINDOW_M:.1f} m")
        print(f"Chosen window: {window_samples} samples")
    else:
        boat_depth = boat_depth_raw.copy()
        print("Depth smoothing disabled.")

    # Interpolate channel positions along 3D path
    n_channels = LAST_CHANNEL - FIRST_CHANNEL + 1
    channel_ids = np.arange(FIRST_CHANNEL, LAST_CHANNEL + 1)
    channel_s3d = np.arange(n_channels, dtype=float) * CHANNEL_SPACING_M

    print(f"Interpolating {n_channels} channels from {FIRST_CHANNEL} to {LAST_CHANNEL}...")
    interp_x, interp_y, interp_depth, total_len_3d = interpolate_along_polyline_3d(
        boat_x, boat_y, boat_depth, channel_s3d
    )

    needed_len = channel_s3d[-1]
    print(f"3D polyline length available: {total_len_3d:.2f} m")
    print(f"3D length needed for channels: {needed_len:.2f} m")

    if needed_len > total_len_3d:
        print("Warning: channel requirement exceeds available cable length.")
        print("Extra channels were extended along the final segment direction.")

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

    # Scientist cable for plotting
    cable_x, cable_y, cable_depth = remove_duplicate_consecutive_points(
        cable_x, cable_y, cable_depth
    )
    cable_s3d, _ = cumulative_length_3d(cable_x, cable_y, cable_depth)

    cable_plot_df = pd.DataFrame({
        "lon": local_xy_to_latlon(cable_x, cable_y, lat0, lon0)[1],
        "lat": local_xy_to_latlon(cable_x, cable_y, lat0, lon0)[0],
        "depth": cable_depth,
        "x_local": cable_x,
        "y_local": cable_y,
        "s_3d": cable_s3d,
    })

    # Boat track for map plotting
    boat_lat_plot, boat_lon_plot = local_xy_to_latlon(boat_x, boat_y, lat0, lon0)
    boat_plot_df = pd.DataFrame({
        "lat": boat_lat_plot,
        "lon": boat_lon_plot,
    })

    # Save interpolated channel CSV
    os.makedirs(os.path.dirname(OUTPUT_CHANNEL_CSV), exist_ok=True)
    interp_df[["channel", "lat", "lon", "depth"]].to_csv(OUTPUT_CHANNEL_CSV, index=False)
    print(f"Saved interpolated channel CSV:\n{OUTPUT_CHANNEL_CSV}")

    # Plots
    print("Making plots...")
    make_horizontal_plot(boat_plot_df, cable_plot_df, interp_df, PLOT_MAP_PNG)
    make_depth_profile_plot(cable_plot_df, interp_df, PLOT_DEPTH_PNG)
    make_3d_plot(cable_plot_df, interp_df, PLOT_3D_PNG)
    make_horizontal_with_depth(boat_plot_df,cable_plot_df, interp_df, PLOT_H_MAP_DEPTH)

    print(f"Saved horizontal plot:\n{PLOT_MAP_PNG}")
    print(f"Saved depth profile plot:\n{PLOT_DEPTH_PNG}")
    print(f"Saved 3D plot:\n{PLOT_3D_PNG}")
    print(f'Saved horizontal plot with depth: \n{PLOT_H_MAP_DEPTH}')
    print("Done.")

    plt.show()


if __name__ == "__main__":
    main()