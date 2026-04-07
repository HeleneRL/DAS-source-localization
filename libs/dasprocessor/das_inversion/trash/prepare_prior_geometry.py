from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from pymap3d import geodetic2enu


# Defaults chosen to sit just southwest of the cable area.
DEFAULT_ENU_ORIGIN_LAT_DEG = 1.2160
DEFAULT_ENU_ORIGIN_LON_DEG = 103.8518
DEFAULT_ENU_ORIGIN_H_M = 0.0

CHANNEL_MIN = 348
CHANNEL_MAX = 2267


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare prior cable geometry in a local ENU frame and compute "
            "tangent/normal/arc-length information for inversion."
        )
    )
    parser.add_argument(
        "--prior-csv",
        type=Path,
        default=Path(r"D:\Singapore Data\Cable\interpolated_channels_from_boattrack.csv"),
        help="CSV with columns channel, lat, lon, depth",
    )
    parser.add_argument(
        "--inversion-csv",
        type=Path,
        default=Path(r"D:\Singapore Data\Cable\inversion_observations.csv"),
        help=(
            "Optional inversion dataset created earlier. If present and readable, "
            "the script will reuse its ENU origin columns for consistency."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(r"D:\Singapore Data\Cable\prior_geometry.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--channel-min",
        type=int,
        default=CHANNEL_MIN,
        help="Minimum channel to keep",
    )
    parser.add_argument(
        "--channel-max",
        type=int,
        default=CHANNEL_MAX,
        help="Maximum channel to keep",
    )
    parser.add_argument(
        "--lat0",
        type=float,
        default=DEFAULT_ENU_ORIGIN_LAT_DEG,
        help="ENU origin latitude in degrees",
    )
    parser.add_argument(
        "--lon0",
        type=float,
        default=DEFAULT_ENU_ORIGIN_LON_DEG,
        help="ENU origin longitude in degrees",
    )
    parser.add_argument(
        "--h0",
        type=float,
        default=DEFAULT_ENU_ORIGIN_H_M,
        help="ENU origin height in meters",
    )
    parser.add_argument(
        "--depth-median-window",
        type=int,
        default=7,
        help="Odd median-filter window for depth smoothing",
    )
    parser.add_argument(
        "--depth-mean-window",
        type=int,
        default=21,
        help="Rolling-mean window for depth smoothing",
    )
    return parser.parse_args()


def load_origin_from_inversion_csv(inversion_csv: Path) -> tuple[float, float, float] | None:
    if not inversion_csv.exists():
        return None

    try:
        df = pd.read_csv(inversion_csv, nrows=5)
    except Exception:
        return None

    needed = ["enu_origin_lat_deg", "enu_origin_lon_deg", "enu_origin_h_m"]
    if not all(c in df.columns for c in needed):
        return None

    lat0 = float(df["enu_origin_lat_deg"].iloc[0])
    lon0 = float(df["enu_origin_lon_deg"].iloc[0])
    h0 = float(df["enu_origin_h_m"].iloc[0])
    return lat0, lon0, h0


def ensure_odd_positive(n: int, name: str) -> int:
    if n < 1:
        raise ValueError(f"{name} must be >= 1")
    if n % 2 == 0:
        n += 1
    return n


def smooth_depth(z: pd.Series, median_window: int, mean_window: int) -> pd.Series:
    z1 = z.rolling(window=median_window, center=True, min_periods=1).median()
    z2 = z1.rolling(window=mean_window, center=True, min_periods=1).mean()
    return z2


def compute_derivative(arr: np.ndarray, coord: np.ndarray) -> np.ndarray:
    return np.gradient(arr, coord)


def normalize_xy(dx: np.ndarray, dy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    norm = np.sqrt(dx * dx + dy * dy)
    norm = np.where(norm < 1e-12, 1e-12, norm)
    return dx / norm, dy / norm, norm


def main() -> None:
    args = parse_args()

    lat0_lon0_h0 = load_origin_from_inversion_csv(args.inversion_csv)
    if lat0_lon0_h0 is not None:
        lat0, lon0, h0 = lat0_lon0_h0
    else:
        lat0, lon0, h0 = args.lat0, args.lon0, args.h0

    median_window = ensure_odd_positive(args.depth_median_window, "depth-median-window")
    mean_window = ensure_odd_positive(args.depth_mean_window, "depth-mean-window")

    prior = pd.read_csv(args.prior_csv)

    required_cols = {"channel", "lat", "lon", "depth"}
    missing = required_cols - set(prior.columns)
    if missing:
        raise ValueError(f"Prior CSV missing columns: {sorted(missing)}")

    prior = prior[(prior["channel"] >= args.channel_min) & (prior["channel"] <= args.channel_max)].copy()
    prior = prior.sort_values("channel").reset_index(drop=True)

    if prior.empty:
        raise ValueError("No prior rows remain after channel filtering")

    e_m, n_m, u_m = geodetic2enu(
        prior["lat"].to_numpy(dtype=float),
        prior["lon"].to_numpy(dtype=float),
        prior["depth"].to_numpy(dtype=float),
        lat0,
        lon0,
        h0,
    )

    out = prior.rename(columns={"lat": "prior_lat", "lon": "prior_lon", "depth": "prior_z_m"}).copy()
    out["enu_origin_lat_deg"] = lat0
    out["enu_origin_lon_deg"] = lon0
    out["enu_origin_h_m"] = h0

    out["prior_x_m"] = e_m
    out["prior_y_m"] = n_m
    out["prior_u_m"] = u_m

    out["prior_z_smooth_m"] = smooth_depth(out["prior_z_m"], median_window, mean_window)

    # Horizontal derivatives with respect to channel number.
    ch = out["channel"].to_numpy(dtype=float)
    x = out["prior_x_m"].to_numpy(dtype=float)
    y = out["prior_y_m"].to_numpy(dtype=float)

    dx_dch = compute_derivative(x, ch)
    dy_dch = compute_derivative(y, ch)
    tx_hat, ty_hat, horiz_step_per_ch = normalize_xy(dx_dch, dy_dch)

    out["tangent_x"] = tx_hat
    out["tangent_y"] = ty_hat
    out["normal_x"] = -ty_hat
    out["normal_y"] = tx_hat
    out["horizontal_step_m_per_channel"] = horiz_step_per_ch

    # Arc length based on 3D prior geometry and on horizontal projection.
    z = out["prior_u_m"].to_numpy(dtype=float)
    ds3 = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
    dsh = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)

    arc3 = np.zeros(len(out), dtype=float)
    arch = np.zeros(len(out), dtype=float)
    arc3[1:] = np.cumsum(ds3)
    arch[1:] = np.cumsum(dsh)

    out["cum_dist_3d_m"] = arc3
    out["cum_dist_horizontal_m"] = arch

    # Curvature proxy from tangent direction changes.
    dtx = compute_derivative(out["tangent_x"].to_numpy(dtype=float), ch)
    dty = compute_derivative(out["tangent_y"].to_numpy(dtype=float), ch)
    out["curvature_proxy_per_channel"] = np.sqrt(dtx * dtx + dty * dty)

    # Useful indices for downstream optimization.
    out["channel_index_zero_based"] = np.arange(len(out), dtype=int)

    # Diagnostics.
    if out["channel"].duplicated().any():
        dupes = out.loc[out["channel"].duplicated(), "channel"].tolist()
        raise ValueError(f"Duplicate channels in prior geometry: {dupes[:10]}")

    expected = np.arange(int(out["channel"].iloc[0]), int(out["channel"].iloc[-1]) + 1)
    missing_channels = sorted(set(expected) - set(out["channel"].to_numpy(dtype=int)))
    if missing_channels:
        print(f"Warning: prior geometry has {len(missing_channels)} missing channels within range.")
        print(f"First few missing channels: {missing_channels[:10]}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    print(f"Saved: {args.output_csv}")
    print(f"Rows: {len(out)}")
    print(f"Channel range: {int(out['channel'].min())} .. {int(out['channel'].max())}")
    print(f"ENU origin used: lat0={lat0:.8f}, lon0={lon0:.8f}, h0={h0:.3f}")
    print(f"3D prior arc length: {out['cum_dist_3d_m'].iloc[-1]:.3f} m")
    print(f"Horizontal prior arc length: {out['cum_dist_horizontal_m'].iloc[-1]:.3f} m")
    print(
        "Mean horizontal step per channel: "
        f"{out['horizontal_step_m_per_channel'].mean():.4f} m/channel"
    )


if __name__ == "__main__":
    main()
