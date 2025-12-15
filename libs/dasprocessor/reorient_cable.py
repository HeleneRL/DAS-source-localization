import json
import numpy as np
from pymap3d import geodetic2enu, enu2geodetic

# ----------------------------------------------------------------------
# INPUT / OUTPUT PATHS
# ----------------------------------------------------------------------
cable_in = r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\B_4\channel_pos_geo.json"
segments_in = r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\segment_orientations.json"

# Simple channel->[lat,lon,depth]
cable_out_json = r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\B_4\channel_pos_geo_reoriented.json"

# Full GeoJSON polyline, for geojson.io
cable_out_geojson = r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\B_4\channel_pos_geo_reoriented.geojson"

# ----------------------------------------------------------------------
# REORIENTATION RANGE
# ----------------------------------------------------------------------
# We only re-draw channels in [REORIENT_START_CH, REORIENT_END_CH].
# Everything below REORIENT_START_CH and above REORIENT_END_CH is kept as-is.
REORIENT_START_CH = 1
REORIENT_END_CH   = 400   # inclusive


# ----------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------
def load_cable(path):
    """Load channel_pos_geo.json as a sorted list of channels and arrays of lat, lon, depth."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    channels = sorted(int(k) for k in data.keys())
    min_ch = min(channels)
    max_ch = max(channels)

    expected = list(range(min_ch, max_ch + 1))
    if channels != expected:
        print("[WARNING] Channels are not contiguous; this script assumes contiguous indices.")
        print(f"  Found: {channels[:10]} ... {channels[-10:]}")

    n = max_ch - min_ch + 1
    lat = np.zeros(n)
    lon = np.zeros(n)
    depth = np.zeros(n)

    for ch in channels:
        lat[ch - min_ch], lon[ch - min_ch], depth[ch - min_ch] = data[str(ch)]

    return min_ch, max_ch, lat, lon, depth


def load_segment_orientations(path):
    """
    Load segment_orientations.json and extract:
      centers (channel index, float)
      theta_deg (continuous orientation estimate)
    Uses 'theta_cont' if present, otherwise falls back to 'theta_deg_norm' or 'theta_deg'.
    """
    with open(path, "r", encoding="utf-8") as f:
        seg_list = json.load(f)

    centers = []
    thetas = []

    for seg in seg_list:
        if not seg.get("success", False):
            continue


        if "theta_deg_continuous" in seg:
            th = seg["theta_deg_continuous"]
        elif "theta_deg_norm" in seg:
            th = seg["theta_deg_norm"]
        elif "theta_deg" in seg:
            th = seg["theta_deg"]
        else:
            continue


        rmin = seg.get("receiver_min")
        rmax = seg.get("receiver_max")
        if rmin is None or rmax is None:
            continue

        center = 0.5 * (rmin + rmax)
        centers.append(center)
        thetas.append(th)

    if not centers:
        raise ValueError("No valid segments with success=True and angle fields found in segment_orientations.json")

    centers = np.array(centers, dtype=float)
    thetas = np.array(thetas, dtype=float)

    order = np.argsort(centers)
    centers = centers[order]
    thetas = thetas[order]

    return centers, thetas


def wrap_angle_deg(angle):
    """Wrap angle to [-180, 180) degrees."""
    return (angle + 180.0) % 360.0 - 180.0


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    # 1) Load cable geometry
    min_ch, max_ch, lat, lon, depth = load_cable(cable_in)
    n = max_ch - min_ch + 1

    print(f"Loaded cable with channels {min_ch}–{max_ch} (n={n})")

    # 2) Sanity for reorientation range
    if REORIENT_START_CH < min_ch or REORIENT_END_CH > max_ch:
        raise ValueError(
            f"Reorientation range [{REORIENT_START_CH}, {REORIENT_END_CH}] "
            f"not inside cable channel range [{min_ch}, {max_ch}]"
        )

    # 3) ENU reference: channel 0 lat/lon, alt0=0 (as requested, depth ignored)
    if min_ch != 0:
        print(f"[WARNING] Minimum channel index is {min_ch}, expected 0. Using that as reference anyway.")
    idx0 = 0 - min_ch if min_ch == 0 else 0
    lat0 = float(lat[idx0])
    lon0 = float(lon[idx0])
    alt0 = 0.0

    # 4) Convert all channel coords to ENU (x,y), keep alt=0
    x = np.zeros(n)
    y = np.zeros(n)
    z_enu = np.zeros(n)

    for i in range(n):
        phi = lat[i]
        lam = lon[i]
        e, n_, u = geodetic2enu(phi, lam, alt0, lat0, lon0, alt0)
        x[i] = e
        y[i] = n_
        z_enu[i] = u

    # 5) Original step lengths in horizontal plane
    L = np.sqrt(np.diff(x)**2 + np.diff(y)**2)

    # 6) Load segment orientation data
    centers, thetas_deg = load_segment_orientations(segments_in)
    print("Loaded segment orientations for centers at:", centers)
    print("Angles (deg):", thetas_deg)

    thetas_deg = np.array([wrap_angle_deg(t) for t in thetas_deg])

    # 7) Build desired theta per channel index (in degrees)
    ch_indices = np.arange(min_ch, max_ch + 1, dtype=float)

    theta_deg_desired = np.empty_like(ch_indices)
    theta_deg_desired[:] = np.nan

    min_c = centers.min()
    max_c = centers.max()

    for i, ch in enumerate(ch_indices):
        if ch < min_c:
            theta_deg_desired[i] = thetas_deg[0]
        elif ch > max_c:
            theta_deg_desired[i] = thetas_deg[-1]
        else:
            theta_deg_desired[i] = np.interp(ch, centers, thetas_deg)

    theta_rad_desired = np.deg2rad(theta_deg_desired)

    # 8) Hard redraw only for channels in [REORIENT_START_CH, REORIENT_END_CH]
    x_new = x.copy()
    y_new = y.copy()

    start_idx = REORIENT_START_CH - min_ch     # index of channel REORIENT_START_CH
    end_idx   = REORIENT_END_CH   - min_ch     # index of channel REORIENT_END_CH

    print(f"Keeping channels {min_ch}–{REORIENT_START_CH-1} fixed.")
    print(f"Rebuilding channels {REORIENT_START_CH}–{REORIENT_END_CH} from DAS orientations.")
    print(f"Keeping channels {REORIENT_END_CH+1}–{max_ch} fixed.")

    # We will overwrite steps from channel CH to CH+1 for CH in [REORIENT_START_CH, REORIENT_END_CH-1]
    # i is the index of the 'from' channel in x_new, y_new.
    # L[i] is the horizontal step length from i -> i+1.
    # Orientation used for step from ch to ch+1 is theta(ch).
    last_step_index = min(end_idx - 1, n - 2)

    for i in range(start_idx, last_step_index + 1):
        ch = min_ch + i  # channel index for this step
        step_len = L[i]
        th = theta_rad_desired[ch - min_ch]  # theta for this channel
        x_new[i + 1] = x_new[i] + step_len * np.cos(th)
        y_new[i + 1] = y_new[i] + step_len * np.sin(th)

    # Channels < REORIENT_START_CH and > REORIENT_END_CH remain at original x,y

    # 9) Convert ENU back to geodetic, keep original depth
    lat_new = np.zeros_like(lat)
    lon_new = np.zeros_like(lon)

    for i in range(n):
        e = x_new[i]
        n_ = y_new[i]
        u = 0.0
        phi, lam, alt = enu2geodetic(e, n_, u, lat0, lon0, alt0)
        lat_new[i] = phi
        lon_new[i] = lam

    # 10) Per-channel JSON: "ch": [lat, lon, depth]
    cable_out_dict = {}
    for i, ch in enumerate(range(min_ch, max_ch + 1)):
        cable_out_dict[str(ch)] = [
            float(lat_new[i]),
            float(lon_new[i]),
            float(depth[i]),
        ]

    with open(cable_out_json, "w", encoding="utf-8") as f:
        json.dump(cable_out_dict, f, indent=2)

    print("\nSaved reoriented per-channel JSON to:")
    print(f"  {cable_out_json}")

    # 11) GeoJSON MultiLineString for geojson.io
    # Coordinates: [lon, lat, depth]
    coords = [
        [float(lon_new[i]), float(lat_new[i]), float(depth[i])]
        for i in range(n)
    ]

    # Approximate 3D length using ENU + depth
    dx = np.diff(x_new)
    dy = np.diff(y_new)
    dz = np.diff(depth)  # depth in meters (positive down)
    seg_lengths = np.sqrt(dx**2 + dy**2 + dz**2)
    shape_len = float(seg_lengths.sum())

    geojson_fc = {
        "type": "FeatureCollection",
        "name": "TBS_Fjordlab_Tether_Reoriented",
        "crs": {
            "type": "name",
            "properties": {
                "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
            }
        },
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "fid_1": 1.0,
                    "Label": "SDP Tether",
                    "Location": "On Seabed",
                    "Shape__Len": shape_len,
                },
                "geometry": {
                    "type": "MultiLineString",
                    "coordinates": [
                        coords  # single polyline
                    ],
                },
            }
        ],
    }

    with open(cable_out_geojson, "w", encoding="utf-8") as f:
        json.dump(geojson_fc, f, indent=2)

    print("\nSaved reoriented cable GeoJSON to:")
    print(f"  {cable_out_geojson}")


if __name__ == "__main__":
    main()
