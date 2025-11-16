import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

'''
this module provides functions to compute GPS positions for channels along a cable layout defined in a GeoJSON file.
it accounts for missing altitude data by interpolating values, and allows for flexible origin placement for channel 0.


'''



def _deg2rad(x: float) -> float:
    return x * math.pi / 180.0

def _horiz_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Fast local metric using equirectangular approximation; accurate for short segments.
    """
    lat0 = _deg2rad((lat1 + lat2) * 0.5)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(lat0)
    dx = (lon2 - lon1) * m_per_deg_lon
    dy = (lat2 - lat1) * m_per_deg_lat
    return math.hypot(dx, dy)

def _interp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)

def _interpolate_alts_inplace(alts: List[float], treat_zero_as_missing: bool = True) -> None:
    """
    Fills runs of missing alts (None or 0 if flagged) by linear interpolation.
    Ends get nearest non-missing value.
    """
    n = len(alts)
    is_missing = [False] * n
    for i, z in enumerate(alts):
        if z is None:
            is_missing[i] = True
        elif treat_zero_as_missing and abs(z) < 1e-9:
            is_missing[i] = True

    # find indices of valid alts
    valid = [i for i, miss in enumerate(is_missing) if not miss]
    if not valid:
        # nothing to interpolate; leave zeros/None as-is
        return

    # fill leading
    first_valid = valid[0]
    for i in range(0, first_valid):
        alts[i] = alts[first_valid]

    # fill trailing
    last_valid = valid[-1]
    for i in range(last_valid + 1, n):
        alts[i] = alts[last_valid]

    # fill interior gaps
    j = 0
    while j < len(valid) - 1:
        a = valid[j]
        b = valid[j + 1]
        za, zb = alts[a], alts[b]
        gap = b - a
        if gap > 1:
            for k in range(1, gap):
                t = k / gap
                alts[a + k] = _interp(za, zb, t)
        j += 1

# ---------- sample polyline and assign channels ----------

def _load_polyline_lonlatalt(geojson_path: Path) -> List[Tuple[float, float, Optional[float]]]:
    """
    Returns a single list of (lon, lat, alt) concatenating all parts of MultiLineString.
    If alt missing, returns None.
    """
    with geojson_path.open("r", encoding="utf-8") as f:
        gj = json.load(f)

    feats = gj.get("features", [])
    if not feats:
        raise ValueError("GeoJSON has no features.")
    geom = feats[0].get("geometry", {})
    if geom.get("type") != "MultiLineString":
        raise ValueError(f"Expected MultiLineString, got {geom.get('type')}")

    coords_multi: List[List[List[float]]] = geom.get("coordinates", [])
    pts: List[Tuple[float, float, Optional[float]]] = []
    for line in coords_multi:
        for c in line:
            if len(c) >= 3:
                lon, lat, alt = float(c[0]), float(c[1]), float(c[2])
            else:
                lon, lat = float(c[0]), float(c[1])
                alt = None
            pts.append((lon, lat, alt))
    if len(pts) < 2:
        raise ValueError("Not enough polyline points.")
    return pts

def _build_chainage(pts_llz: List[Tuple[float, float, Optional[float]]]) -> Tuple[List[float], List[float], List[Optional[float]], List[float]]:
    """
    Returns (lats, lons, alts, cumdist_m) along the polyline.
    """
    lons = [p[0] for p in pts_llz]
    lats = [p[1] for p in pts_llz]
    alts = [p[2] for p in pts_llz]

    cum = [0.0]

    for i in range(1, len(pts_llz)):
            dxy = _horiz_distance_m(lats[i-1], lons[i-1], lats[i], lons[i])
            dz  = alts[i] - alts[i-1]
            cum.append(cum[-1] + math.hypot(dxy, dz))
    return lats, lons, alts, cum

def _nearest_chainage_to(lats: List[float], lons: List[float], cum: List[float], lat0: float, lon0: float) -> float:
    """
    Returns the chainage (distance along the polyline) of the closest vertex to (lat0, lon0).
    (Vertex-only nearest; fine for a quick anchor.)
    """
    best_i = 0
    best_d = float("inf")
    for i, (la, lo) in enumerate(zip(lats, lons)):
        d = _horiz_distance_m(lat0, lon0, la, lo)
        if d < best_d:
            best_d = d
            best_i = i
    return cum[best_i]

def _sample_polyline(lats: List[float], lons: List[float], alts: List[float], cum: List[float], chainage: float) -> Tuple[float, float, float]:
    """
    Linearly interpolate lat/lon/alt at a given chainage (meters).
    """
    if chainage <= cum[0]:
        return lats[0], lons[0], alts[0]
    if chainage >= cum[-1]:
        return lats[-1], lons[-1], alts[-1]
    # find segment
    lo, hi = 0, len(cum) - 1
    # binary search
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if cum[mid] <= chainage:
            lo = mid
        else:
            hi = mid
    t = (chainage - cum[lo]) / max(1e-12, (cum[hi] - cum[lo]))
    lat = _interp(lats[lo], lats[hi], t)
    lon = _interp(lons[lo], lons[hi], t)
    alt = _interp(alts[lo], alts[hi], t)
    return lat, lon, alt

def _check_spacing_block(out: Dict[int, List[float]], start: int, n: int = 6):
    for k in range(n):
        a = out[start + k]
        b = out[start + k + 1]
        dxy = _horiz_distance_m(a[0], a[1], b[0], b[1])
        dz  = b[2] - a[2]
        d3  = math.hypot(dxy, dz)
        print(f"ch{start+k}->{start+k+1}: {d3:.4f} m")

def compute_channel_positions(
    geojson_path: str | Path,
    channel_count: int,
    channel_distance: float,
    origin: str = "start",
    origin_offset_m: float = 0.0,
    nearest_to: Optional[Tuple[float, float]] = None,
    interpolate_missing_alts: bool = True
) -> Dict[int, List[float]]:
    """
    Build {channel_number: [lat, lon, alt]} for a linear array laid along a cable polyline.

    Args
    ----
    geojson_path : path to the cable-layout.json (MultiLineString; coords = [lon, lat, alt])
    channel_count : total channels (e.g., 1200)
    channel_distance : spacing in meters (e.g., 1.02)
    origin : one of {"start", "end", "nearest", "chainage"}
        "start"   -> channel 0 at start of polyline
        "end"     -> channel 0 at end of polyline
        "nearest" -> channel 0 at vertex nearest to `nearest_to=(lat, lon)`
        "chainage"-> channel 0 at absolute chainage = origin_offset_m
    origin_offset_m : extra offset (meters) added after origin selection
        (use negative to move towards start, positive towards end)
    nearest_to : (lat, lon) used when origin == "nearest"
    interpolate_missing_alts : if True, treat alt==0 as missing and interpolate

    Returns
    -------
    dict: {channel_number: [lat, lon, alt]}
          lat/lon in degrees, alt in meters. Keys are ints.
    """
    geojson_path = Path(geojson_path)
    pts = _load_polyline_lonlatalt(geojson_path)

    lons = [p[0] for p in pts]
    lats = [p[1] for p in pts]
    alts = [0.0 if (p[2] is None) else float(p[2]) for p in pts]

    if interpolate_missing_alts:
        _interpolate_alts_inplace(alts, treat_zero_as_missing=True)

    # Rebuild pts with filled alts so BOTH chainage and sampling see the same alts
    pts_filled = list(zip(lons, lats, alts))  # (lon, lat, alt_filled)

    # ---------- Build cumulative distance AFTER filling alts ----------
    # IMPORTANT: your _build_chainage(...) must compute *3D* cumdist:
    #    d3 = hypot(_horiz_distance_m(...), dz)
    lats, lons, alts, cum = _build_chainage(pts_filled)

    total_len = cum[-1]


    # pick chainage of channel 0
    if origin == "start":
        chain0 = 0.0
    elif origin == "end":
        chain0 = total_len
    elif origin == "nearest":
        if nearest_to is None:
            raise ValueError("origin='nearest' requires nearest_to=(lat, lon)")
        lat0, lon0 = nearest_to
        chain0 = _nearest_chainage_to(lats, lons, cum, lat0=lat0, lon0=lon0)
    elif origin == "chainage":
        chain0 = float(origin_offset_m)
        origin_offset_m = 0.0  # already applied
    else:
        raise ValueError("origin must be one of {'start','end','nearest','chainage'}")

    chain0 = max(0.0, min(total_len, chain0 + origin_offset_m))

    # build each channel position
    out: Dict[int, List[float]] = {}
    for ch in range(channel_count):
        s = max(0.0, min(total_len, chain0 + ch * channel_distance))
        lat, lon, alt = _sample_polyline(lats, lons, alts, cum, s)
        out[ch] = [lat, lon, alt]

    return out
