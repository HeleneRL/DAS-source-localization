import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Helpers
# -------------------------

def latlon_to_local_xy(lat, lon, lat0, lon0):
    """
    Convert lat/lon to local tangent plane (meters)
    using equirectangular approximation (good for small areas).
    """
    R = 6371000.0  # Earth radius (m)
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)

    x = (lon_rad - lon0_rad) * np.cos(lat0_rad) * R
    y = (lat_rad - lat0_rad) * R
    return x, y


def nearest_distance_to_polyline(px, py, x_line, y_line):
    """
    Compute nearest distance from points (px,py)
    to a polyline defined by (x_line, y_line).
    """
    distances = []
    nearest_points = []

    for x0, y0 in zip(px, py):
        min_dist = np.inf
        best_pt = (np.nan, np.nan)

        for i in range(len(x_line) - 1):
            x1, y1 = x_line[i], y_line[i]
            x2, y2 = x_line[i + 1], y_line[i + 1]

            dx = x2 - x1
            dy = y2 - y1

            if dx == 0 and dy == 0:
                proj_x, proj_y = x1, y1
            else:
                t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx * dx + dy * dy)
                t = np.clip(t, 0, 1)
                proj_x = x1 + t * dx
                proj_y = y1 + t * dy

            dist = np.sqrt((x0 - proj_x) ** 2 + (y0 - proj_y) ** 2)

            if dist < min_dist:
                min_dist = dist
                best_pt = (proj_x, proj_y)

        distances.append(min_dist)
        nearest_points.append(best_pt)

    return np.array(distances), np.array(nearest_points)


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--estimated_csv", required=True)
    parser.add_argument("--truth_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--highlight_threshold_m", type=float, default=15.0)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------------
    # Load data
    # -------------------------
    est = pd.read_csv(args.estimated_csv)
    truth = pd.read_csv(args.truth_csv)

    # Extract lat/lon
    est_lat = est["lat_deg"].values
    est_lon = est["lon_deg"].values

    truth_lat = truth["lat"].values
    truth_lon = truth["lon"].values

    # -------------------------
    # Convert to local XY (same reference!)
    # -------------------------
    lat0 = np.mean(np.concatenate([est_lat, truth_lat]))
    lon0 = np.mean(np.concatenate([est_lon, truth_lon]))

    est_x, est_y = latlon_to_local_xy(est_lat, est_lon, lat0, lon0)
    truth_x, truth_y = latlon_to_local_xy(truth_lat, truth_lon, lat0, lon0)

    # -------------------------
    # Distance from estimate to truth curve
    # -------------------------
    distances, nearest_pts = nearest_distance_to_polyline(
        est_x, est_y, truth_x, truth_y
    )

    # -------------------------
    # Metrics
    # -------------------------
    metrics = {
        "mean_distance_m": float(np.mean(distances)),
        "median_distance_m": float(np.median(distances)),
        "rmse_distance_m": float(np.sqrt(np.mean(distances ** 2))),
        "p95_distance_m": float(np.percentile(distances, 95)),
        "max_distance_m": float(np.max(distances)),
    }

    print("\n=== Geometry comparison (lat/lon based) ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")

    pd.DataFrame([metrics]).to_csv(
        os.path.join(args.output_dir, "latlon_comparison_metrics.csv"),
        index=False
    )

    # Save per-point distances
    df_out = pd.DataFrame({
        "channel": est.get("channel", np.arange(len(est))),
        "distance_to_truth_m": distances
    })
    df_out.to_csv(
        os.path.join(args.output_dir, "estimate_to_truth_nearest_distance.csv"),
        index=False
    )

    # -------------------------
    # Plot 1: overlay + highlight
    # -------------------------
    plt.figure(figsize=(8, 8))
    plt.plot(truth_x, truth_y, label="Truth", linewidth=2)
    plt.plot(est_x, est_y, label="Estimate", linewidth=2)

    mask = distances > args.highlight_threshold_m
    plt.scatter(est_x[mask], est_y[mask], c="red", s=10, label="High error")

    plt.legend()
    plt.title("Overlay with inconsistencies highlighted")
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.axis("equal")
    plt.grid()
    plt.savefig(os.path.join(args.output_dir, "overlay_inconsistencies.png"), dpi=200)
    plt.close()

    # -------------------------
    # Plot 2: colored by error
    # -------------------------
    plt.figure(figsize=(8, 8))
    sc = plt.scatter(est_x, est_y, c=distances, s=8)
    plt.plot(truth_x, truth_y, "k-", label="Truth")
    plt.colorbar(sc, label="Distance to truth (m)")
    plt.title("Estimate colored by error")
    plt.axis("equal")
    plt.grid()
    plt.savefig(os.path.join(args.output_dir, "overlay_error_colored.png"), dpi=200)
    plt.close()

    # -------------------------
    # Plot 3: quiver arrows
    # -------------------------
    plt.figure(figsize=(8, 8))
    plt.plot(truth_x, truth_y, label="Truth")
    plt.plot(est_x, est_y, label="Estimate")

    plt.quiver(
        est_x, est_y,
        nearest_pts[:, 0] - est_x,
        nearest_pts[:, 1] - est_y,
        angles="xy", scale_units="xy", scale=1, width=0.002
    )

    plt.legend()
    plt.title("Error vectors (estimate → nearest truth)")
    plt.axis("equal")
    plt.grid()
    plt.savefig(os.path.join(args.output_dir, "overlay_quiver_to_truth.png"), dpi=200)
    plt.close()

    # -------------------------
    # Plot 4: distance vs channel
    # -------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(distances)
    plt.title("Distance to truth vs channel index")
    plt.xlabel("Index")
    plt.ylabel("Distance (m)")
    plt.grid()
    plt.savefig(os.path.join(args.output_dir, "channelwise_distance_to_truth.png"), dpi=200)
    plt.close()

    # -------------------------
    # Histogram
    # -------------------------
    plt.figure()
    plt.hist(distances, bins=50)
    plt.title("Distance distribution")
    plt.xlabel("Distance (m)")
    plt.ylabel("Count")
    plt.grid()
    plt.savefig(os.path.join(args.output_dir, "hist_distance_to_truth.png"), dpi=200)
    plt.close()

    # -------------------------
    # CDF
    # -------------------------
    d_sorted = np.sort(distances)
    cdf = np.arange(len(d_sorted)) / len(d_sorted)

    plt.figure()
    plt.plot(d_sorted, cdf)
    plt.title("CDF of distances")
    plt.xlabel("Distance (m)")
    plt.ylabel("CDF")
    plt.grid()
    plt.savefig(os.path.join(args.output_dir, "cdf_distance_to_truth.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    main()