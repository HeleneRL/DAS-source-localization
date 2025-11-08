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
    pkg_dir = Path(__file__).resolve().parent
    cable_gps = load_cable_geometry(str((pkg_dir / "../resources/cable-layout.json").resolve()))

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
    gps = gps[:, [1, 0, 2]] # swap to (lat, lon, alt), all was flipped 
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

'''
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

            # g = dt/ds (s per meter). Convert to incidence via |cos δ| = |c*g|.
            # Use ABS to keep δ in [0, 90] so bearings are around the AXIS, not the NORMAL.
            cg = speed_of_sound * g
            cg_abs = float(min(1.0, max(0.0, abs(cg))))
            theta = math.degrees(math.acos(cg_abs))  # δ in [0, 90]

            # Two ambiguous bearings about the axis (head-tail ambiguity only)
            beta_minus = wrap_bearing_deg(alpha - theta)
            beta_plus  = wrap_bearing_deg(alpha + theta)


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
'''
def estimate_bearings_for_packets(
    arrivals: Mapping[int, Mapping[int, int]],
    subarrays: Mapping[int, Sequence[int]],
    gps_per_channel: np.ndarray,
    packet_indices: Iterable[int],
    run_number: int,
    speed_of_sound: float = 1500.0,
    min_fraction_present: float = 0.5,
    use_pca_heading: bool = False,
    debug: bool = True,
    *,
    time_gate_s: Optional[float] = None,   # e.g. 1.5 -> keep arrivals within ±1.5 s of subarray median
    use_linear_spacing: bool = False       # True -> use ideal spacing (index * channel_distance) instead of GPS projection
) -> Dict[int, Dict[int, Optional[Dict[str, Union[float, int, Tuple[float, float]]]]]]:
    """
    Estimate bearings per subarray and packet via TDOA slope.

    Bearing convention: degrees from North, clockwise (0°=North, 90°=East).
    Lat/lon are always handled as (lat, lon).

    New options:
      - time_gate_s: reject per-subarray arrivals whose time differs from the
        subarray median by more than ±time_gate_s (helps avoid mis-associated packets).
      - use_linear_spacing: use ideal along-array spacing from channel indices
        and run["channel_distance"] instead of GPS-projected positions.
    """
    # ---- quick global checks ----
    if debug:
        print("\n[DEBUG] estimate_bearings_for_packets() — global checks")
        print(f"  speed_of_sound       = {speed_of_sound} m/s")
        print(f"  min_fraction_present = {min_fraction_present}")
        print(f"  use_pca_heading      = {use_pca_heading}")
        print(f"  time_gate_s          = {time_gate_s}")
        print(f"  use_linear_spacing   = {use_linear_spacing}")
        print(f"  gps_per_channel shape: {getattr(gps_per_channel, 'shape', None)} (expected (N,3) as (lat, lon, alt))")
        pk_list = list(packet_indices) if not isinstance(packet_indices, range) else [packet_indices.start]
        print(f"  packet_indices       = {pk_list[:10]}{'...' if len(pk_list)>10 else ''}")
        try:
            ex_idx = [0, 1] if gps_per_channel.shape[0] > 1 else [0]
            for ii in ex_idx:
                lat_i, lon_i = gps_per_channel[ii, 0], gps_per_channel[ii, 1]
                print(f"    ch {ii:4d}: lat={lat_i:.6f}, lon={lon_i:.6f}")
        except Exception as e:
            print(f"  [WARN] Cannot preview gps_per_channel: {e}")

    run = get_run("2024-05-03", run_number)
    ch_dist = float(run["channel_distance"])

    # reference for local projection: overall mean of all subarray centers
    centers_meta = subarray_centers_and_headings(subarrays, gps_per_channel)
    if len(centers_meta) == 0:
        if debug:
            print("  [DEBUG] No subarrays provided; returning empty dict.")
        return {}

    ref_lat = float(np.mean([m["center_lat"] for m in centers_meta.values()]))
    ref_lon = float(np.mean([m["center_lon"] for m in centers_meta.values()]))

    if debug:
        print(f"  ENU reference: ref_lat={ref_lat:.6f}, ref_lon={ref_lon:.6f}")

    results: Dict[int, Dict[int, Optional[Dict[str, Union[float, int, Tuple[float, float]]]]]] = {}

    # ---- per-subarray loop ----
    for c, chans in subarrays.items():
        if debug:
            print(f"\n[DEBUG] Subarray center={c}, size={len(chans)}")
            print(f"  channels: {chans}")

        # subarray heading
        if use_pca_heading:
            # PCA-based heading
            lats = gps_per_channel[np.array(chans), 0]
            lons = gps_per_channel[np.array(chans), 1]
            xs, ys = _local_xy_m(lats, lons, ref_lat, ref_lon)
            alpha = _principal_axis_angle_deg(xs, ys)
            if debug:
                print(f"  heading (PCA) alpha={alpha:.3f}° (deg from North CW)")
        else:
            # endpoint-based heading from first->last element (lat,lon)
            ch_first, ch_last = chans[0], chans[-1]
            p1 = (float(gps_per_channel[ch_first, 0]), float(gps_per_channel[ch_first, 1]))  # (lat, lon)
            p2 = (float(gps_per_channel[ch_last, 0]),  float(gps_per_channel[ch_last, 1]))   # (lat, lon)
            alpha = _bearing_deg_from_two_gps(p1, p2)
            if debug:
                print(f"  endpoint heading alpha={alpha:.3f}° from ch {ch_first} -> {ch_last}")
                print(f"    p1(lat,lon)={p1}, p2(lat,lon)={p2}")
                xs_full, ys_full = _local_xy_m(
                    np.array([p1[0], p2[0]]), np.array([p1[1], p2[1]]), ref_lat, ref_lon
                )
                dx, dy = xs_full[1] - xs_full[0], ys_full[1] - ys_full[0]
                axis_bearing_from_ENU = wrap_bearing_deg(math.degrees(math.atan2(dx, dy)))
                print(f"    ENU delta approx: dx={dx:.2f} m (east), dy={dy:.2f} m (north) -> bearing_ENU={axis_bearing_from_ENU:.3f}°")
                if abs(((axis_bearing_from_ENU - alpha + 180) % 360) - 180) > 5:
                    print("    [WARN] ENU-based bearing and geodetic bearing differ by >5°. Check lat/lon order or reference.")

        center_lat = float(centers_meta[c]["center_lat"])
        center_lon = float(centers_meta[c]["center_lon"])

        # Along-axis coordinate s for each channel
        if use_linear_spacing:
            # Ideal evenly spaced coordinates centred at zero (in meters)
            idx = np.arange(len(chans)) - (len(chans) - 1) / 2.0
            s_axis = idx * ch_dist
            if debug:
                print(f"  s_axis (LINEAR) range: [{np.min(s_axis):.2f}, {np.max(s_axis):.2f}] m")
        else:
            # GPS-projected coordinates along alpha
            lats = gps_per_channel[np.array(chans), 0]
            lons = gps_per_channel[np.array(chans), 1]
            xs, ys = _local_xy_m(lats, lons, ref_lat, ref_lon)
            s_axis = _project_onto_axis(xs, ys, alpha)  # meters along axis
            if debug:
                print(f"  s_axis (GPS) range: [{np.min(s_axis):.2f}, {np.max(s_axis):.2f}] m")
                if len(s_axis) >= 3:
                    ds = np.diff(s_axis)
                    frac_pos = np.mean(ds > 0)
                    frac_neg = np.mean(ds < 0)
                    print(f"  s_axis monotonicity hint: frac_pos={frac_pos:.2f}, frac_neg={frac_neg:.2f}")

        # Packet loop
        results[c] = {}
        sub_size = len(chans)
        min_required = int(math.ceil(min_fraction_present * sub_size))

        for k in packet_indices:
            # Gather arrivals
            times = []
            ss = []
            present_channels = 0
            for idx, ch in enumerate(chans):
                t_samp = arrivals.get(ch, {}).get(k, None)
                if t_samp is None:
                    continue
                present_channels += 1
                times.append(t_samp / SAMPLE_RATE)   # seconds
                ss.append(s_axis[idx])

            if debug:
                print(f"    packet {k}: detections {present_channels}/{sub_size} (need >= {min_required})")

            if present_channels < min_required:
                results[c][k] = None
                if debug:
                    print("      -> insufficient detections, skipping")
                continue

            t = np.asarray(times, float)
            s = np.asarray(ss, float)

            # --- Time gating to remove mis-associated channels (optional) ---
            if time_gate_s is not None and t.size >= 3:
                t_med = float(np.median(t))
                keep = np.abs(t - t_med) <= float(time_gate_s)
                if debug:
                    n_kept = int(np.sum(keep))
                    print(f"      time-gate ±{time_gate_s:.2f}s: kept {n_kept}/{t.size}")
                t = t[keep]
                s = s[keep]
                if t.size < min_required:
                    results[c][k] = None
                    if debug:
                        print("      -> after time-gating, too few points; skipping")
                    continue

            # Fit t(s) with simple outlier rejection
            a, g, mask = _linear_fit_with_outlier_reject(s, t, max_z=3.5)
            n_used = int(np.isfinite(mask).sum() if mask.dtype == bool else len(t))

            if debug:
                corr = np.corrcoef(s, t)[0, 1] if len(s) >= 2 else float("nan")
                print(f"      fit: g={g:.6g} s/m, intercept a={a:.6g}, used={n_used}, corr(s,t)≈{corr:.3f}")

            if not np.isfinite(g):
                results[c][k] = None
                if debug:
                    print("      -> invalid slope g, skipping")
                continue

            # g = dt/ds (s/m). Convert to incidence via |cos θ| = |c*g|.
            cg = speed_of_sound * g
            cg_abs = float(min(1.0, max(0.0, abs(cg))))
            theta = math.degrees(math.acos(cg_abs))  # θ ∈ [0,90]
            if debug:
                print(f"      c*g={cg:.6g} (abs={cg_abs:.6g}) => theta={theta:.3f}°")
                if abs(cg) > 1.0 + 1e-3:
                    print("      [WARN] |c*g| > 1 ⇒ unphysical slope. Check detections / geometry / c.")

            # Two ambiguous bearings about the axis (head-tail ambiguity only)
            beta_minus = wrap_bearing_deg(alpha - theta - 90)
            beta_plus  = wrap_bearing_deg(alpha + theta + 90)

            if debug:
                print(f"      alpha={alpha:.3f}°, bearings: alpha-θ={beta_minus:.3f}°, alpha+θ={beta_plus:.3f}°")

            results[c][k] = {
                "bearing_deg_pair": (beta_minus, beta_plus),
                "alpha_deg": float(alpha),
                "theta_deg": float(theta),
                "g_s_per_m": float(g),
                "n_used": int(n_used),
                "center_lat": center_lat,
                "center_lon": center_lon,
            }

    return results




# ---------- Bearing-ray building & intersections ----------

from itertools import product
from typing import NamedTuple


class Ray(NamedTuple):
    """Bearing ray anchored at a lat/lon point."""
    lat: float
    lon: float
    bearing_deg: float
    weight: float = 1.0   # optional weighting (e.g., n_used, SNR-derived)


def _ray_normal_form(x0: float, y0: float, bearing_deg: float) -> Tuple[float, float, float]:
    """
    In ENU (x east, y north), convert a ray (x0,y0, bearing) to line normal form:
        n_x * x + n_y * y + c = 0, where ||n|| = 1
    The unit normal is perpendicular to the unit direction.
    """
    th = math.radians(bearing_deg)
    ux, uy = math.sin(th), math.cos(th)          # unit direction (east, north)
    nx, ny = -uy, ux                             # rotate +90: normal
    c = -(nx * x0 + ny * y0)
    return nx, ny, c


def intersect_bearing_rays_enu(
    rays: Sequence[Ray],
    ref_lat: float,
    ref_lon: float,
    weights: Optional[Sequence[float]] = None,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Least-squares intersection of >=2 bearing rays in a local ENU frame.

    Parameters
    ----------
    rays : list of Ray(lat,lon,bearing_deg,weight)
    ref_lat, ref_lon : reference for ENU projection
    weights : optional weights per ray (overrides Ray.weight if provided)

    Returns
    -------
    {
      "x_m": float, "y_m": float,           # ENU solution in meters
      "lat": float, "lon": float,           # back-projected to GPS
      "residuals": np.ndarray,              # signed perpendicular distances (m)
      "rms_residual": float,
      "A": np.ndarray, "b": np.ndarray,     # design and rhs (for diagnostics)
      "weights": np.ndarray
    }
    """
    if len(rays) < 2:
        raise ValueError("Need at least two rays to intersect.")

    # Project anchors to ENU
    lats = np.array([r.lat for r in rays], float)
    lons = np.array([r.lon for r in rays], float)
    xs, ys = _local_xy_m(lats, lons, ref_lat, ref_lon)

    # Build normal equations (weighted)
    A_rows = []
    b_vals = []
    w = np.array([r.weight for r in rays], float) if weights is None else np.asarray(weights, float)
    w = np.clip(w, 1e-6, np.inf)

    for (x0, y0, r, wi) in zip(xs, ys, rays, w):
        nx, ny, c = _ray_normal_form(x0, y0, r.bearing_deg)
        A_rows.append([math.sqrt(wi) * nx, math.sqrt(wi) * ny])
        b_vals.append(-math.sqrt(wi) * c)

    A = np.asarray(A_rows, float)
    b = np.asarray(b_vals, float)

    # Solve min ||A [x;y] - b||
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    x_hat, y_hat = float(sol[0]), float(sol[1])

    # Residuals: signed perpendicular distance for each constraint
    res = (A @ sol - b) / np.sqrt(np.sum(A**2, axis=1))
    rms = float(np.sqrt(np.mean(res**2)))

    # Back-project to GPS
    # invert _local_xy_m approximately (small domain)
    lat = ref_lat + (y_hat / EARTH_R) * (180.0 / math.pi)
    lon = ref_lon + (x_hat / (EARTH_R * math.cos(math.radians(ref_lat)))) * (180.0 / math.pi)

    return {
        "x_m": x_hat,
        "y_m": y_hat,
        "lat": lat,
        "lon": lon,
        "residuals": res,
        "rms_residual": rms,
        "A": A,
        "b": b,
        "weights": w,
    }


def _rays_from_bearing_results_for_packet(
    packet_results_for_subarrays: Mapping[int, Dict[str, Union[float, Tuple[float, float]]]],
    pick: Optional[Mapping[int, int]] = None,
) -> Tuple[List[Ray], Dict[int, Tuple[float, float]]]:
    """
    Build rays for ONE packet from estimate_bearings_for_packets() output.

    Parameters
    ----------
    packet_results_for_subarrays : {center_ch: {
        "bearing_deg_pair": (bmin, bplus),
        "alpha_deg": ..., "theta_deg": ...,
        "center_lat": ..., "center_lon": ..., "n_used": ...
    }}
    pick : optional dict {center_ch: 0 or 1} to pick which of (α-θ, α+θ)

    Returns
    -------
    rays : list[Ray]
    bearing_pairs : {center_ch: (bmin, bplus)}  # returned for disambiguation
    """
    rays: List[Ray] = []
    bearing_pairs: Dict[int, Tuple[float, float]] = {}

    for c, info in packet_results_for_subarrays.items():
        if info is None:
            continue
        bmin, bplus = info["bearing_deg_pair"]  # type: ignore[index]
        clat = float(info["center_lat"])        # type: ignore[index]
        clon = float(info["center_lon"])        # type: ignore[index]
        wt = float(info.get("n_used", 1))       # type: ignore[arg-type]

        bearing_pairs[c] = (float(bmin), float(bplus))

        choice = None if pick is None else pick.get(c, None)
        if choice is None:
            # leave choice to the disambiguation routine
            continue
        chosen_b = (bmin, bplus)[int(choice)]
        rays.append(Ray(lat=clat, lon=clon, bearing_deg=float(chosen_b), weight=wt))

    return rays, bearing_pairs


def _enumerate_ray_choices(
    bearing_pairs: Mapping[int, Tuple[float, float]],
    centers_order: Optional[Sequence[int]] = None,
) -> Iterable[Tuple[Dict[int, int], List[Ray]]]:
    """
    Generate all 2^N combinations of picking (α-θ) vs (α+θ) for N subarrays.
    Yields (pick_map, rays_list).
    """
    keys = list(centers_order) if centers_order is not None else list(bearing_pairs.keys())
    for bits in product([0, 1], repeat=len(keys)):
        pick = {c: b for c, b in zip(keys, bits)}
        rays: List[Ray] = [
            Ray(lat=np.nan, lon=np.nan, bearing_deg=0.0)  # placeholder; replaced below
        ]
        rays.clear()
        for c, b in zip(keys, bits):
            bmin, bplus = bearing_pairs[c]
            rays.append(Ray(lat=float("nan"), lon=float("nan"), bearing_deg=(bmin, bplus)[b]))
        yield pick, rays


def best_intersection_from_bearing_pairs(
    packet_results_for_subarrays: Mapping[int, Dict[str, Union[float, Tuple[float, float]]]],
    ref_lat: float,
    ref_lon: float,
) -> Dict[str, Union[float, Dict[int, int], np.ndarray]]:
    """
    Try all combinations of (α±θ) across available subarrays for ONE packet,
    choose the combination that minimizes the LS intersection RMS residual.

    Returns
    -------
    {
      "lat": float, "lon": float,
      "x_m": float, "y_m": float,
      "rms_residual": float,
      "pick": {center_ch: 0/1},        # chosen branch per subarray
      "residuals": np.ndarray
    }

    Notes
    -----
    - Requires at least two subarrays with non-None info.
    - Weights rays by n_used.
    - If N is large, 2^N grows fast. For N>12, consider a greedy or RANSAC strategy.
    """
    # Collect centers, bearing pairs, centers' lat/lon and weights
    centers: List[int] = []
    pairs: List[Tuple[float, float]] = []
    lats: List[float] = []
    lons: List[float] = []
    wts: List[float] = []

    for c, info in packet_results_for_subarrays.items():
        if info is None:
            continue
        centers.append(int(c))
        pairs.append( (float(info["bearing_deg_pair"][0]), float(info["bearing_deg_pair"][1])) )  # type: ignore[index]
        lats.append(float(info["center_lat"]))   # type: ignore[index]
        lons.append(float(info["center_lon"]))   # type: ignore[index]
        wts.append(float(info.get("n_used", 1))) # type: ignore[arg-type]

    if len(centers) < 2:
        return {"lat": float("nan"), "lon": float("nan"),
                "x_m": float("nan"), "y_m": float("nan"),
                "rms_residual": float("inf"),
                "pick": {}, "residuals": np.array([])}

    # Precompute ENU anchors
    xs, ys = _local_xy_m(np.array(lats), np.array(lons), ref_lat, ref_lon)

    best = {"rms_residual": float("inf")}
    # Brute-force all 2^N choices (ok for small N)
    for bits in product([0, 1], repeat=len(centers)):
        bearings = [pairs[i][bits[i]] for i in range(len(centers))]
        rays = [Ray(lat=lats[i], lon=lons[i], bearing_deg=bearings[i], weight=wts[i])
                for i in range(len(centers))]

        sol = intersect_bearing_rays_enu(rays, ref_lat, ref_lon)
        if sol["rms_residual"] < best["rms_residual"]:
            best = {
                "lat": sol["lat"], "lon": sol["lon"],
                "x_m": sol["x_m"], "y_m": sol["y_m"],
                "rms_residual": sol["rms_residual"],
                "pick": {centers[i]: bits[i] for i in range(len(centers))},
                "residuals": sol["residuals"],
            }
    return best


def positions_from_bearing_results(
    bearing_results: Mapping[int, Mapping[int, Optional[Dict[str, Union[float, Tuple[float, float]]]]]],
    packet_indices: Iterable[int],
    ref_lat: Optional[float] = None,
    ref_lon: Optional[float] = None,
) -> Dict[int, Dict[str, Union[float, Dict[int, int], np.ndarray]]]:
    """
    For each packet index k:
      - collect subarray bearing pairs from `estimate_bearings_for_packets(...)`
      - disambiguate by trying all (α±θ) combinations
      - compute best LS intersection in ENU
    Returns {packet_idx: result_dict} where result_dict matches
    best_intersection_from_bearing_pairs(...) output.

    If ref_lat/lon are not provided, uses the mean of subarray centers per packet.
    """
    out: Dict[int, Dict[str, Union[float, Dict[int, int], np.ndarray]]] = {}
    Ks = list(packet_indices)
    for k in Ks:
        # Gather subarray infos for this packet
        per_sub = {c: info_k for c, per_k in bearing_results.items()
                   if (info_k := per_k.get(k, None)) is not None}
        if len(per_sub) < 2:
            out[k] = {"lat": float("nan"), "lon": float("nan"),
                      "x_m": float("nan"), "y_m": float("nan"),
                      "rms_residual": float("inf"),
                      "pick": {}, "residuals": np.array([])}
            continue

        # Pick reference if not provided
        if ref_lat is None or ref_lon is None:
            lats = [float(info["center_lat"]) for info in per_sub.values()]  # type: ignore[index]
            lons = [float(info["center_lon"]) for info in per_sub.values()]  # type: ignore[index]
            rlat = float(np.mean(lats))
            rlon = float(np.mean(lons))
        else:
            rlat, rlon = float(ref_lat), float(ref_lon)

        out[k] = best_intersection_from_bearing_pairs(per_sub, rlat, rlon)
    return out

# ---------- Caching: per-run channel GPS ----------

_GPS_CACHE: Dict[int, np.ndarray] = {}

def get_cached_channel_gps_for_run(run_number: int) -> np.ndarray:
    """
    Cached version of channel_gps_for_run(run_number).
    """
    if run_number not in _GPS_CACHE:
        _GPS_CACHE[run_number] = channel_gps_for_run(run_number)
    return _GPS_CACHE[run_number]


# ---------- Packet selection (coverage-based) ----------

def packet_coverage_counts(
    arrivals: Mapping[int, Mapping[int, int]],
    subarrays: Mapping[int, Sequence[int]],
    packet_indices: Optional[Iterable[int]] = None
) -> Dict[int, int]:
    """
    Count how many subarrays have >=1 detection (in any element) for each packet.

    Returns
    -------
    {packet_idx: count_of_subarrays_with_at_least_one_detection}
    """
    # build set of packets to consider
    if packet_indices is None:
        pk_all = set()
        for ch_map in arrivals.values():
            pk_all.update(ch_map.keys())
        packet_indices = sorted(pk_all)

    coverage: Dict[int, int] = {int(k): 0 for k in packet_indices}
    for _, chans in subarrays.items():
        present = set()
        for ch in chans:
            present.update(arrivals.get(ch, {}).keys())
        for k in present:
            if k in coverage:
                coverage[int(k)] += 1
    return coverage


def select_packets_by_coverage(
    arrivals: Mapping[int, Mapping[int, int]],
    subarrays: Mapping[int, Sequence[int]],
    min_subarrays: int = 2,
    limit: Optional[int] = None,
    packet_indices: Optional[Iterable[int]] = None
) -> List[int]:
    """
    Return packet indices that are seen by at least `min_subarrays` subarrays,
    sorted by coverage (desc). Optional `limit` restricts how many to return.
    """
    cov = packet_coverage_counts(arrivals, subarrays, packet_indices)
    ranked = sorted([k for k, n in cov.items() if n >= min_subarrays],
                    key=lambda k: cov[k], reverse=True)
    return ranked if limit is None else ranked[:limit]


# ---------- Auto heading choice (endpoint vs PCA) ----------

def auto_heading_selector(
    subarrays: Mapping[int, Sequence[int]],
    gps_per_channel: np.ndarray,
    r2_threshold: float = 0.98
) -> Dict[int, bool]:
    """
    Decide per-subarray whether PCA-based heading should be used.

    Heuristic:
      - Fit a straight line to ENU points.
      - Compute R^2 of that fit.
      - If R^2 < r2_threshold => use PCA (return True), else endpoint-based (False).

    Returns
    -------
    {center_channel: use_pca_heading_bool}
    """
    out: Dict[int, bool] = {}
    # Reference for local projection: mean of subarray centers
    centers_meta = subarray_centers_and_headings(subarrays, gps_per_channel)
    ref_lat = float(np.mean([m["center_lat"] for m in centers_meta.values()])) if centers_meta else 0.0
    ref_lon = float(np.mean([m["center_lon"] for m in centers_meta.values()])) if centers_meta else 0.0

    for c, chans in subarrays.items():
        lats = gps_per_channel[np.array(chans), 0]
        lons = gps_per_channel[np.array(chans), 1]
        xs, ys = _local_xy_m(lats, lons, ref_lat, ref_lon)

        # Fit y = a + b x, compute R^2
        if len(xs) < 2:
            out[c] = False
            continue
        b, a = np.polyfit(xs, ys, 1)
        yhat = a + b * xs
        ss_res = np.sum((ys - yhat)**2)
        ss_tot = np.sum((ys - np.mean(ys))**2) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        out[c] = (r2 < r2_threshold)
    return out


# ---------- Build rays for a chosen packet (given a pick-map) ----------

def rays_for_packet_with_pick(
    bearing_results: Mapping[int, Mapping[int, Optional[Dict[str, Union[float, Tuple[float, float]]]]]],
    packet_index: int,
    pick_map: Mapping[int, int]
) -> List[Ray]:
    """
    Build rays for ONE packet using a fixed pick (0→α-θ, 1→α+θ) per subarray.

    Parameters
    ----------
    bearing_results: output of estimate_bearings_for_packets(...)
    packet_index: which packet to extract
    pick_map: {center_channel: 0 or 1}

    Returns
    -------
    list of Ray(lat, lon, bearing_deg, weight=n_used)
    """
    rays: List[Ray] = []
    per_sub = {c: info_k for c, per_k in bearing_results.items()
               if (info_k := per_k.get(packet_index, None)) is not None}
    for c, info in per_sub.items():
        bmin, bplus = info["bearing_deg_pair"]  # type: ignore[index]
        clat = float(info["center_lat"])        # type: ignore[index]
        clon = float(info["center_lon"])        # type: ignore[index]
        wt   = float(info.get("n_used", 1))     # type: ignore[arg-type]
        choice = int(pick_map.get(int(c), 0))
        chosen_b = (bmin, bplus)[choice]
        rays.append(Ray(lat=clat, lon=clon, bearing_deg=float(chosen_b), weight=wt))
    return rays


# ---------- Simple trajectory smoothing in ENU (no SoS/weights) ----------

def smooth_positions_enu(
    positions_per_packet: Mapping[int, Dict[str, Union[float, Dict[int, int], np.ndarray]]],
    ref_lat: Optional[float] = None,
    ref_lon: Optional[float] = None,
    window: int = 5,     # should be odd
    poly: int = 2        # used only if scipy.savgol_filter is available
) -> Dict[int, Dict[str, float]]:
    """
    Smooth packet-wise positions (lat/lon) in ENU with a simple filter.

    - Projects to ENU around (ref_lat, ref_lon). If None, uses mean of valid points.
    - Tries Savitzky–Golay if scipy is available; otherwise uses moving average.
    - Returns a dict {packet_idx: {"x_m":..., "y_m":..., "lat":..., "lon":...}}.

    Notes: This does NOT estimate velocities; purely positional smoothing.
    """
    # Collect valid points
    ks = sorted(positions_per_packet.keys())
    pts = [(k, positions_per_packet[k]) for k in ks
           if np.isfinite(positions_per_packet[k].get("lat", np.nan)) and
              np.isfinite(positions_per_packet[k].get("lon", np.nan))]
    if not pts:
        return {}

    lat_arr = np.array([float(p[1]["lat"]) for p in pts])
    lon_arr = np.array([float(p[1]["lon"]) for p in pts])
    k_arr   = np.array([int(p[0]) for p in pts])

    # Reference
    if ref_lat is None or ref_lon is None:
        ref_lat = float(np.mean(lat_arr))
        ref_lon = float(np.mean(lon_arr))

    # ENU
    x, y = _local_xy_m(lat_arr, lon_arr, ref_lat, ref_lon)

    # Smoothing
    try:
        from scipy.signal import savgol_filter
        if window % 2 == 0:
            window += 1
        x_s = savgol_filter(x, window_length=max(window, poly + 3), polyorder=min(poly, window - 1))
        y_s = savgol_filter(y, window_length=max(window, poly + 3), polyorder=min(poly, window - 1))
    except Exception:
        # Fallback: simple moving average
        w = max(1, window)
        ker = np.ones(w) / w
        # pad at edges (reflect)
        def movavg(z):
            pad = w // 2
            zp = np.pad(z, pad, mode='reflect')
            return np.convolve(zp, ker, mode='valid')
        x_s = movavg(x)
        y_s = movavg(y)

    # Back to lat/lon
    lat_s = ref_lat + (y_s / EARTH_R) * (180.0 / math.pi)
    lon_s = ref_lon + (x_s / (EARTH_R * math.cos(math.radians(ref_lat)))) * (180.0 / math.pi)

    # Reassemble dict keyed by packet indices we had
    out: Dict[int, Dict[str, float]] = {}
    for kk, xs, ys, lats, lons in zip(k_arr, x_s, y_s, lat_s, lon_s):
        out[int(kk)] = {"x_m": float(xs), "y_m": float(ys),
                        "lat": float(lats), "lon": float(lons)}
    return out


# ---------- GeoJSON exporters (points, rays, path) ----------

def geojson_feature_point(lat: float, lon: float, props: Optional[dict] = None) -> dict:
    return {"type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": props or {}}

def geojson_feature_line(coords_lonlat: List[Tuple[float, float]], props: Optional[dict] = None) -> dict:
    return {"type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords_lonlat},
            "properties": props or {}}

def geojson_subarray_centers(
    centers_meta: Mapping[int, Mapping[str, float]]
) -> dict:
    """
    Make a FeatureCollection with subarray centers.
    """
    feats = []
    for c, m in centers_meta.items():
        feats.append(geojson_feature_point(m["center_lat"], m["center_lon"],
                                           {"center_channel": int(c),
                                            "heading_deg": float(m["heading_deg"]) if np.isfinite(m["heading_deg"]) else None}))
    return {"type": "FeatureCollection", "features": feats}

def geojson_rays_for_packet(
    bearing_results: Mapping[int, Mapping[int, Optional[Dict[str, Union[float, Tuple[float, float]]]]]],
    packet_index: int,
    pick_map: Mapping[int, int],
    ray_length_m: float = 1_000.0
) -> dict:
    """
    Emit rays (line segments) of fixed length from subarray centers for a packet.
    """
    feats = []
    per_sub = {c: info_k for c, per_k in bearing_results.items()
               if (info_k := per_k.get(packet_index, None)) is not None}
    for c, info in per_sub.items():
        clat = float(info["center_lat"])   # type: ignore[index]
        clon = float(info["center_lon"])   # type: ignore[index]
        bmin, bplus = info["bearing_deg_pair"]  # type: ignore[index]
        chosen = (bmin, bplus)[int(pick_map.get(int(c), 0))]

        # endpoint at ray_length_m along chosen bearing
        th = math.radians(float(chosen))
        # ENU step
        dx = ray_length_m * math.sin(th)
        dy = ray_length_m * math.cos(th)
        # to lon/lat
        lat2 = clat + (dy / EARTH_R) * (180.0 / math.pi)
        lon2 = clon + (dx / (EARTH_R * math.cos(math.radians(clat)))) * (180.0 / math.pi)

        feats.append(geojson_feature_line([(clon, clat), (lon2, lat2)],
                                          {"center_channel": int(c),
                                           "bearing_deg": float(chosen)}))
    return {"type": "FeatureCollection", "features": feats}

def geojson_positions_path(
    positions_per_packet: Mapping[int, Mapping[str, float]]
) -> dict:
    """
    Emit a FeatureCollection with a single LineString for the estimated track
    (sorted by packet index), plus point features per packet.
    """
    ks = sorted([k for k in positions_per_packet.keys()
                 if np.isfinite(positions_per_packet[k].get("lat", np.nan)) and
                    np.isfinite(positions_per_packet[k].get("lon", np.nan))])
    if not ks:
        return {"type": "FeatureCollection", "features": []}
    coords = [(positions_per_packet[k]["lon"], positions_per_packet[k]["lat"]) for k in ks]  # type: ignore[index]
    feats = [geojson_feature_line(coords, {"name": "estimated_track"})]
    for k in ks:
        feats.append(geojson_feature_point(positions_per_packet[k]["lat"],  # type: ignore[index]
                                           positions_per_packet[k]["lon"],  # type: ignore[index]
                                           {"packet": int(k)}))
    return {"type": "FeatureCollection", "features": feats}
