# libs/dasprocessor/bearing_tools.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import json
import math
import numpy as np

from .constants import get_run
from .saveandload import load_cable_geometry
from .gnss import interpolate_coordinates  # uses cable geometry distances via GPS


# ---------- Basics & helpers ----------

SAMPLE_RATE = 25_000.0  # Hz
EARTH_R = 6_378_000.0   # m (consistent with your gnss.py default)
DEG = 180.0 / math.pi


def _normalize_arrivals(d: Mapping[str, Mapping[str, Union[int, float]]]
                        ) -> Dict[int, Dict[int, int]]:
    """
    Convert a JSON-loaded dict with string keys to int->int->int.
    """
    out: Dict[int, Dict[int, int]] = {}
    for ch_s, inner in d.items():
        ch = int(ch_s)
        out[ch] = {int(pk_s): int(v) for pk_s, v in inner.items()}
    return out


def load_merged_arrivals(path: Union[str, Path]
                         ) -> Dict[int, Dict[int, int]]:
    """
    Load merged peaks JSON and return {channel: {packet_idx: sample_index}}.
    """
    path = Path(path)
    with path.open("r") as fh:
        raw = json.load(fh)
    return _normalize_arrivals(raw)


def seconds_from_samples(samples: Union[int, float, np.ndarray]) -> np.ndarray:
    """
    Convert sample index(es) to seconds at the global DAS sample rate.
    """
    return np.asarray(samples, dtype=float) / SAMPLE_RATE


def clamp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def wrap_bearing_deg(b: float) -> float:
    """Wrap bearing to [0, 360) deg."""
    b = b % 360.0
    return b if b >= 0 else b + 360.0


# ---------- Subarray construction ----------

def build_subarrays(center_channels: Sequence[int],
                    aperture_len: int,
                    run_number: int) -> Dict[int, List[int]]:
    """
    Build subarrays around each center channel.

    A subarray is: center ± (aperture_len-1)/2 channels.
    Caps to valid channel indices for the run.

    Returns
    -------
    {center_channel: [ch1, ch2, ...]}  (inclusive, ascending)
    """
    if aperture_len <= 0 or aperture_len % 2 == 0:
        raise ValueError("aperture_len must be a positive odd integer.")

    run = get_run("2024-05-03", run_number)
    ch_count = int(run["channel_count"])

    half = (aperture_len - 1) // 2
    subarrays: Dict[int, List[int]] = {}
    for c in center_channels:
        start = max(0, c - half)
        stop = min(ch_count - 1, c + half)
        subarrays[c] = list(range(start, stop + 1))
    return subarrays


# ---------- Channel geometry ----------

def _bearing_deg_from_two_gps(p1: Tuple[float, float],
                              p2: Tuple[float, float]) -> float:
    """
    Bearing (deg from North, clockwise) from p1 (lat,lon) to p2 (lat,lon).
    Uses great-circle initial bearing formula.
    """
    lat1 = math.radians(p1[0])
    lat2 = math.radians(p2[0])
    dlon = math.radians(p2[1] - p1[1])
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    brng = math.atan2(y, x) * DEG
    return wrap_bearing_deg(brng)


def _local_xy_m(lat: np.ndarray, lon: np.ndarray,
                lat0: float, lon0: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Equirectangular local projection (meters) around (lat0,lon0).
    Returns (x_east_m, y_north_m).
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat0r = math.radians(lat0)
    dx = (np.radians(lon - lon0) * math.cos(lat0r)) * EARTH_R
    dy = (np.radians(lat - lat0)) * EARTH_R
    return dx, dy


def channel_gps_for_run(run_number: int
                        ) -> np.ndarray:
    """
    Return per-channel GPS (lat, lon, alt) as array shape (channel_count, 3).
    Uses cable layout and interpolates along distance with channel spacing.
    """
    run = get_run("2024-05-03", run_number)
    ch_count = int(run["channel_count"])
    ch_spacing = float(run["channel_distance"])  # meters

    # Load raw cable polyline (lat, lon, alt)
    cable_gps = load_cable_geometry('../resources/cable-layout.json')  # (N,3)

    # Compute cumulative distance along cable polyline in gnss.py:
    # We reuse interpolate_coordinates which expects distances and knowns.
    known_latlonalt = np.asarray(cable_gps, dtype=float)
    known_dists = np.zeros(known_latlonalt.shape[0])
    # Derive distances via the same method gnss.distance_gps uses:
    # (We could re-import it, but interpolate_coordinates only needs
    #  known_distances to be monotonically increasing.)
    # Approximate cumulative distance in meters (Cartesian chord lengths):
    from .gnss import to_cartesian
    cart = to_cartesian(known_latlonalt[:, 0], known_latlonalt[:, 1], known_latlonalt[:, 2])
    seg = np.sqrt(np.sum(np.diff(cart, axis=0) ** 2, axis=1))
    known_dists[1:] = np.cumsum(seg)

    wanted_dists = np.arange(ch_count) * ch_spacing
    gps = interpolate_coordinates(wanted_dists,
                                  known_dists,
                                  known_latlonalt,
                                  input_cs="GPS",
                                  output_cs="GPS")
    # gps shape (ch_count, 3) with [lat, lon, alt]
    return gps


def subarray_centers_and_headings(subarrays: Mapping[int, Sequence[int]],
                                  gps_per_channel: np.ndarray
                                  ) -> Dict[int, Dict[str, Union[float, Tuple[float, float]]]]:
    """
    For each subarray, compute:
      - center GPS (lat, lon) of the subarray (mean of endpoints),
      - heading (deg from North) as the bearing from first to last element.

    Returns
    -------
    {center_channel: {"center_lat": float, "center_lon": float, "heading_deg": float}}
    """
    out: Dict[int, Dict[str, Union[float, Tuple[float, float]]]] = {}
    for c, chans in subarrays.items():
        if len(chans) < 2:
            # heading undefined for single-element
            lat_c = float(gps_per_channel[c, 0])
            lon_c = float(gps_per_channel[c, 1])
            out[c] = {"center_lat": lat_c, "center_lon": lon_c, "heading_deg": float("nan")}
            continue

        ch_first = chans[0]
        ch_last = chans[-1]
        lat1, lon1 = gps_per_channel[ch_first, 0], gps_per_channel[ch_first, 1]
        lat2, lon2 = gps_per_channel[ch_last, 0], gps_per_channel[ch_last, 1]

        # center point = midpoint of endpoints (good enough at subarray scale)
        center_lat = float(0.5 * (lat1 + lat2))
        center_lon = float(0.5 * (lon1 + lon2))
        heading_deg = _bearing_deg_from_two_gps((lat1, lon1), (lat2, lon2))

        out[c] = {"center_lat": center_lat,
                  "center_lon": center_lon,
                  "heading_deg": heading_deg}
    return out


# ---------- Bearing estimation from TDOA slope ----------

def _principal_axis_angle_deg(xs: np.ndarray, ys: np.ndarray) -> float:
    """
    PCA-like direction of a cloud (angle from North, clockwise).
    Inputs are local x=east, y=north (meters).
    """
    X = np.column_stack([xs - xs.mean(), ys - ys.mean()])
    # First principal component
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    vx, vy = Vt[0, 0], Vt[0, 1]  # direction in (x_east, y_north)
    # Bearing from North, clockwise: atan2(x, y)
    return wrap_bearing_deg(math.degrees(math.atan2(vx, vy)))


def _project_onto_axis(xs: np.ndarray, ys: np.ndarray, bearing_deg: float) -> np.ndarray:
    """
    Project (x_east, y_north) onto an axis with given bearing (deg from North).
    Returns scalar coordinate s (meters) along that axis.
    """
    theta = math.radians(bearing_deg)
    ux, uy = math.sin(theta), math.cos(theta)  # unit vector (east, north)
    return xs * ux + ys * uy


def _linear_fit_with_outlier_reject(x: np.ndarray, y: np.ndarray,
                                    max_z: float = 3.5
                                    ) -> Tuple[float, float, np.ndarray]:
    """
    Robust-ish linear fit y = a + b*x.
    Two-pass: LS, then remove points with |residual| > max_z * MAD, refit.
    Returns (a, b, mask_used).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan"), float("nan"), mask

    b, a = np.polyfit(x[mask], y[mask], 1)  # returns slope, intercept
    resid = y[mask] - (a + b * x[mask])
    mad = np.median(np.abs(resid - np.median(resid))) + 1e-12
    keep = np.abs(resid) <= max_z * 1.4826 * mad  # 1.4826 ≈ MAD->σ
    if keep.sum() >= 2:
        b, a = np.polyfit(x[mask][keep], y[mask][keep], 1)
        final_mask = mask.copy()
        final_mask[mask] = keep
        return a, b, final_mask
    else:
        return a, b, mask


def estimate_bearings_for_packets(
    arrivals: Mapping[int, Mapping[int, int]],
    subarrays: Mapping[int, Sequence[int]],
    gps_per_channel: np.ndarray,
    packet_indices: Iterable[int],
    run_number: int,
    speed_of_sound: float = 1500.0,
    min_fraction_present: float = 0.5,
    use_pca_heading: bool = False,
) -> Dict[int, Dict[int, Optional[Dict[str, Union[float, int, Tuple[float, float]]]]]]:
    """
    Estimate bearings per subarray and packet via TDOA slope.

    For each subarray (center channel key) and packet index k:
      - Gather arrival times for channels in the subarray for packet k
      - Convert to seconds
      - Compute subarray heading (deg from North):
          * If use_pca_heading=False: bearing from first->last element
          * If use_pca_heading=True: principal axis of channel positions
      - Project channel positions into local ENU, then onto the axis
      - Fit t(s) = t0 + g*s with outlier rejection
      - Convert slope g (s/m) -> incidence angle θ: cos θ = c * g
      - Return two ambiguous bearings (α ± θ), plus diagnostics

    Only returns an estimate if the number of valid channels >=
    ceil(min_fraction_present * subarray_size). Otherwise value is None.

    Returns
    -------
    results: Dict[
        center_channel,
        Dict[
            packet_index,
            None | {
              "bearing_deg_pair": (float, float),  # (α-θ, α+θ), wrapped
              "alpha_deg": float,                  # subarray axis bearing
              "theta_deg": float,                  # incidence
              "g_s_per_m": float,                  # slope
              "n_used": int,                       # channels in fit
              "center_lat": float,
              "center_lon": float
            }
        ]
    ]
    """
    run = get_run("2024-05-03", run_number)
    # reference for local projection: overall mean of all subarray centers
    centers_meta = subarray_centers_and_headings(subarrays, gps_per_channel)
    if len(centers_meta) == 0:
        return {}
    ref_lat = float(np.mean([m["center_lat"] for m in centers_meta.values()]))
    ref_lon = float(np.mean([m["center_lon"] for m in centers_meta.values()]))

    results: Dict[int, Dict[int, Optional[Dict[str, Union[float, int, Tuple[float, float]]]]]] = {}

    for c, chans in subarrays.items():
        # subarray heading
        if use_pca_heading:
            # PCA-based heading
            lats = gps_per_channel[np.array(chans), 0]
            lons = gps_per_channel[np.array(chans), 1]
            xs, ys = _local_xy_m(lats, lons, ref_lat, ref_lon)
            alpha = _principal_axis_angle_deg(xs, ys)
        else:
            # endpoint-based heading
            ch_first, ch_last = chans[0], chans[-1]
            p1 = (float(gps_per_channel[ch_first, 0]), float(gps_per_channel[ch_first, 1]))
            p2 = (float(gps_per_channel[ch_last, 0]), float(gps_per_channel[ch_last, 1]))
            alpha = _bearing_deg_from_two_gps(p1, p2)

        center_lat = float(centers_meta[c]["center_lat"])
        center_lon = float(centers_meta[c]["center_lon"])

        # Precompute projected scalar coordinate s for each channel in subarray
        lats = gps_per_channel[np.array(chans), 0]
        lons = gps_per_channel[np.array(chans), 1]
        xs, ys = _local_xy_m(lats, lons, ref_lat, ref_lon)
        s_axis = _project_onto_axis(xs, ys, alpha)  # meters along axis

        # Packet loop
        results[c] = {}
        sub_size = len(chans)
        min_required = int(math.ceil(min_fraction_present * sub_size))

        for k in packet_indices:
            # Collect arrivals for this packet
            times = []
            ss = []
            for idx, ch in enumerate(chans):
                t_samp = arrivals.get(ch, {}).get(k, None)
                if t_samp is None:
                    continue
                times.append(t_samp / SAMPLE_RATE)
                ss.append(s_axis[idx])

            if len(times) < min_required:
                results[c][k] = None
                continue

            t = np.asarray(times, float)
            s = np.asarray(ss, float)

            # Fit t(s) with simple outlier rejection
            a, g, mask = _linear_fit_with_outlier_reject(s, t, max_z=3.5)
            n_used = int(np.isfinite(mask).sum() if mask.dtype == bool else len(t))

            if not np.isfinite(g):
                results[c][k] = None
                continue

            # g = dt/ds (s per meter). Convert to incidence θ via cos θ = c * g.
            cg = speed_of_sound * g
            cg = float(clamp(np.array([cg]), -1.0, 1.0)[0])
            theta = math.degrees(math.acos(cg))

            # Two ambiguous bearings about the axis
            beta_minus = wrap_bearing_deg(alpha - theta)
            beta_plus = wrap_bearing_deg(alpha + theta)

            results[c][k] = {
                "bearing_deg_pair": (beta_minus, beta_plus),
                "alpha_deg": alpha,
                "theta_deg": theta,
                "g_s_per_m": float(g),
                "n_used": int(len(t)),
                "center_lat": center_lat,
                "center_lon": center_lon,
            }

    return results
