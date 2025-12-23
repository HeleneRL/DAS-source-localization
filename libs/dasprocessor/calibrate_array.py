import json
from datetime import datetime
import numpy as np
from scipy.optimize import least_squares
from pymap3d import geodetic2enu

''' 

OBS! This script is currently not in use, it is not needed for the main processing.
It was used to calibrate array orientation segments based on known speaker positions and recorded arrival times, but the results were not great and 
the method was abandoned. The likely reason is the whole multipath issue that was not handled here.


No other scripts depend on this one.


'''

























# ----------------------------------------------------------------------
# USER INPUT: FILES
# ----------------------------------------------------------------------

# Speaker JSON file (list of [lat, lon, datetime_iso, packet_index])
speaker_json = r"C:\Users\helen\Documents\PythonProjects\my-project\interp_debug.json"

# Receiver JSON file with arrival times (channel -> {packet_index: sample_idx})
receiver_json = r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\B_4\peaks-merged-all_hilbert_channels.json"

# Channel positions JSON ("channel" -> [lat, lon, depth])
channel_pos_json = r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\B_4\channel_pos_geo.json"

# Output summary JSON
summary_out = r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\segment_orientations.json"

# ----------------------------------------------------------------------
# USER INPUT: SPEAKER RANGES (MULTIPLE STRAIGHT SECTIONS)
# ----------------------------------------------------------------------
# Each tuple is (min_packet_index, max_packet_index), inclusive.
speaker_ranges = [
    (2,7),
    (49, 59)
]

# ----------------------------------------------------------------------
# USER INPUT: RECEIVER SEGMENT SWEEP
# ----------------------------------------------------------------------

receiver_start = 70    # first segment start channel
receiver_stop = 310    # last *start* channel (stop is exclusive in range)
receiver_step = 20     # e.g. 70–80, 80–90, ...

# ----------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------
c = 1475.0     # m/s (sound speed)
fs = 25000.0   # Hz (sampling rate)
d = 1.02       # hydrophone spacing (m)

# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------
def parse_time(tstr):
    """Parse ISO time with or without microseconds."""
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(tstr, fmt)
        except ValueError:
            pass
    raise ValueError(f"Time format not recognized: {tstr}")

def residuals(params, measurements, speaker_dict, d, z_h, c, ref_h_index):
    """
    Nonlinear least-squares residuals:
    params = [x0, y0, theta, dt_global]
    residual = t_rec - t_model
    """
    x0, y0, theta, dt = params
    cosT = np.cos(theta)
    sinT = np.sin(theta)

    res = np.zeros(len(measurements))

    for k, (hid, sid, t_rec) in enumerate(measurements):
        # hydrophone position (linear array model)
        offset = hid - ref_h_index
        hx = x0 + offset * d * cosT
        hy = y0 + offset * d * sinT
        hz = z_h

        sx, sy, sz, t_tx = speaker_dict[sid]

        dx = sx - hx
        dy = sy - hy
        dz = sz - hz
        dist = np.sqrt(dx * dx + dy * dy + dz * dz)

        t_model = t_tx + dist / c + dt
        res[k] = t_rec - t_model

    return res

def unwrap_orientations(segment_results):
    """
    Take the list of segment results (each with theta_deg_norm) and add a
    'theta_deg_continuous' field that enforces continuity along the cable.
    We assume segments are ordered by receiver_min.
    """
    # Filter successful segments and sort by receiver_min
    ok_segments = [s for s in segment_results if s.get("success", False)]
    ok_segments.sort(key=lambda s: s["receiver_min"])

    prev = None
    for seg in ok_segments:
        theta = seg["theta_deg_norm"]  # initial in [-180, 180]

        if prev is None:
            seg["theta_deg_continuous"] = theta
            prev = theta
            continue

        cand = theta
        # Adjust by +/- 180 multiples to stay close to prev
        # (avoid jumps larger than ~90 degrees)
        while cand - prev > 90.0:
            cand -= 180.0
        while cand - prev < -90.0:
            cand += 180.0

        seg["theta_deg_continuous"] = cand
        prev = cand

    # Put back into original list (they share objects, so already updated)
    return segment_results

# ----------------------------------------------------------------------
# LOAD STATIC DATA: SPEAKERS, RECEIVERS, CHANNEL POSITIONS
# ----------------------------------------------------------------------
with open(speaker_json, "r", encoding="utf-8") as f:
    speakers = json.load(f)  # list of [lat, lon, datetime_iso, packet_index]

with open(receiver_json, "r", encoding="utf-8") as f:
    receivers_all = json.load(f)  # dict: "channel" -> {packet_index: sample_idx}

with open(channel_pos_json, "r", encoding="utf-8") as f:
    channel_pos = json.load(f)  # dict: "channel" -> [lat, lon, depth]

# Convert channel_pos keys to int if needed
if isinstance(next(iter(channel_pos.keys())), str):
    channel_pos = {int(k): v for k, v in channel_pos.items()}

# ----------------------------------------------------------------------
# PREPARE SPEAKERS (MULTIPLE RANGES)
# ----------------------------------------------------------------------
selected_speakers = []
for smin, smax in speaker_ranges:
    subset = [s for s in speakers if smin <= s[3] <= smax]
    selected_speakers.extend(subset)

if not selected_speakers:
    raise ValueError("No speakers found in ANY of the configured ranges.")

# Sort by packet index
selected_speakers.sort(key=lambda s: s[3])

# Reference ENU origin = first speaker in combined list
lat0, lon0, t0_str, _ = selected_speakers[0]
alt0 = 0.0
t0 = parse_time(t0_str)

speaker_dict = {}  # sid -> (x, y, z, t_rel)
for lat, lon, t_str, sid in selected_speakers:
    x, y, z = geodetic2enu(lat, lon, 0.0, lat0, lon0, alt0)
    t_abs = parse_time(t_str)
    t_rel = (t_abs - t0).total_seconds()
    speaker_dict[int(sid)] = (x, y, 0.0, t_rel)

print("\nUsing speakers from these ranges:")
for a, b in speaker_ranges:
    print(f"  {a}–{b}")
print(f"Total speakers used: {len(speaker_dict)}  (IDs {min(speaker_dict)}–{max(speaker_dict)})")

# ----------------------------------------------------------------------
# LOOP OVER RECEIVER SEGMENTS
# ----------------------------------------------------------------------
segment_results = []

for receiver_min in range(receiver_start, receiver_stop, receiver_step):
    receiver_max = receiver_min + receiver_step

    ch_min = receiver_min
    ch_max = receiver_max

    if ch_min not in channel_pos or ch_max not in channel_pos:
        print(f"\n[Segment {receiver_min}-{receiver_max}] Skipping: missing depth info for {ch_min} or {ch_max}")
        segment_results.append({
            "receiver_min": receiver_min,
            "receiver_max": receiver_max,
            "success": False,
            "reason": "missing_depth",
        })
        continue

    depth_min = channel_pos[ch_min][2]
    depth_max = channel_pos[ch_max][2]
    z_h = 0.5 * (depth_min + depth_max) + 30.0  # your depth rule

    print(f"\n=== Segment {receiver_min}-{receiver_max} ===")
    print(f"Segment depth z_h = {z_h:.3f} m (avg({depth_min:.3f}, {depth_max:.3f}) + 30)")

    # Select receivers in this interval
    selected_receivers = {
        int(rid): rec for rid, rec in receivers_all.items()
        if receiver_min <= int(rid) <= receiver_max
    }

    if not selected_receivers:
        print(f"No receivers found in range [{receiver_min}, {receiver_max}], skipping.")
        segment_results.append({
            "receiver_min": receiver_min,
            "receiver_max": receiver_max,
            "success": False,
            "reason": "no_receivers",
        })
        continue

    # Sorted hydrophone IDs (int)
    hydro_ids = sorted(selected_receivers.keys())
    ref_h_index = 0  # index in hydro_ids list used as reference hydrophone

    print(f"Using {len(hydro_ids)} receivers (IDs {hydro_ids[0]}–{hydro_ids[-1]})")

    # Build measurements: list of (hydro_index, speaker_id, t_rec_seconds)
    measurements_list = []

    for hi, hid in enumerate(hydro_ids):
        rec_dict = selected_receivers[hid]  # dict: "speaker_id_str" -> sample_index
        for sid in speaker_dict.keys():
            sample_idx = rec_dict.get(str(sid))  # may be missing
            if sample_idx is None:
                continue
            t_rec = sample_idx / fs
            measurements_list.append((hi, sid, t_rec))

    if not measurements_list:
        print("No measurements after filtering; skipping segment.")
        segment_results.append({
            "receiver_min": receiver_min,
            "receiver_max": receiver_max,
            "success": False,
            "reason": "no_measurements",
        })
        continue

    measurements = np.array(
        measurements_list,
        dtype=[("hid", int), ("sid", int), ("t_rec", float)]
    )

    print(f"Number of measurements: {len(measurements)}")

    # Initial guess
    all_sx = [v[0] for v in speaker_dict.values()]
    all_sy = [v[1] for v in speaker_dict.values()]
    x0_guess = float(np.mean(all_sx))
    y0_guess = float(np.mean(all_sy))
    theta_guess = 0.0
    dt_guess = 0.0

    p0 = np.array([x0_guess, y0_guess, theta_guess, dt_guess])

    # Run nonlinear least-squares
    try:
        result = least_squares(
            residuals,
            p0,
            args=(measurements, speaker_dict, d, z_h, c, ref_h_index),
            method="lm",  # Levenberg–Marquardt
        )
        success = result.success
        msg = result.message
    except Exception as e:
        print(f"Optimization error for segment {receiver_min}-{receiver_max}: {e}")
        segment_results.append({
            "receiver_min": receiver_min,
            "receiver_max": receiver_max,
            "success": False,
            "reason": "exception",
            "exception": str(e),
        })
        continue

    print("Optimization success:", success)
    print("Message:", msg)

    x0_est, y0_est, theta_est, dt_est = result.x
    theta_deg = float(np.degrees(theta_est))
    theta_deg_norm = ((theta_deg + 180.0) % 360.0) - 180.0  # in [-180, 180]

    print(f"Estimated theta = {theta_deg:.3f} deg (normalized: {theta_deg_norm:.3f} deg)")
    print(f"Estimated depth z_h = {z_h:.3f} m")

    segment_results.append({
        "receiver_min": receiver_min,
        "receiver_max": receiver_max,
        "success": bool(success),
        "message": str(msg),
        "theta_deg": theta_deg,
        "theta_deg_norm": theta_deg_norm,
        "z_h": float(z_h),
        "n_measurements": int(len(measurements)),
    })

# ----------------------------------------------------------------------
# POST-PROCESS: ENFORCE CONTINUITY (UNWRAP THETA ALONG CABLE)
# ----------------------------------------------------------------------
segment_results = unwrap_orientations(segment_results)

# ----------------------------------------------------------------------
# SAVE SUMMARY JSON
# ----------------------------------------------------------------------
with open(summary_out, "w", encoding="utf-8") as f:
    json.dump(segment_results, f, indent=2)

print(f"\nSaved segment summary to:\n  {summary_out}")

# ----------------------------------------------------------------------
# PRINT ORIENTATION TABLE
# ----------------------------------------------------------------------
print("\nSummary of orientations (degrees from East axis, CCW):")
print(f"{'Segment':>12}  {'theta_deg':>10}  {'theta_norm':>11}  {'theta_cont':>11}  {'depth z_h':>10}  {'ok?':>4}  {'Nmeas':>6}")
for seg in sorted(segment_results, key=lambda s: s["receiver_min"]):
    if not seg.get("success", False):
        continue
    rmin = seg["receiver_min"]
    rmax = seg["receiver_max"]
    th  = seg["theta_deg"]
    thn = seg["theta_deg_norm"]
    thc = seg.get("theta_deg_continuous", float("nan"))
    zh  = seg["z_h"]
    nm  = seg["n_measurements"]
    print(f"{rmin:4d}-{rmax:<4d}  {th:10.3f}  {thn:11.3f}  {thc:11.3f}  {zh:10.3f}  {'yes':>4}  {nm:6d}")
