import argparse
import json
import os
from pathlib import Path
from pymap3d import enu2geodetic, geodetic2enu
import matplotlib.pyplot as plt

import numpy as np


# your own helpers
from dasprocessor.channel_gps import compute_channel_positions
from dasprocessor.doa_v2 import fit_doa, cone_radian_from_slope, cone_plane_intersection

# ---------------- CONFIG ----------------

FS = 25_000.0           # Hz
C_SOUND = 1475.0        # m/s
SOURCE_DEPTH = -30.0    # ENU z of source plane
ANGLE_UNCERTAINTY_DEG = 5.0

MAX_EXTRA_DIST_1 = 10.0  # m (first residual threshold)
MAX_EXTRA_DIST_2 = 5.0   # m (second residual threshold)

MIN_CHANNELS_FOR_DOA = 3   # or whatever you consider safe

XY_WINDOW = (0.0, 700.0, -500.0, 500.0)  # (E_min, E_max, N_min, N_max)
N_SAMPLES_CONE = 300

CABLE_LAYOUT_FILE = (
    r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\suspected_cable-layout.json"
)
PEAKS_FILE = (
    r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\B_4\peaks-reordered_all_hilbert_channels.json"
)

#which channel is the first in the water (0-based index)
N_SKIP = 23          # 0..22 are on land / not in water
channel_distance = 1.02
CHANNEL_OFFSET = N_SKIP * channel_distance  # m


DEBUG_PLOTS = False  # choose packet index to visualize, or None to disable
DEBUG_PACKET_IDX = 0

def debug_plot_doa_stage(
    positions_enu: np.ndarray,
    times_sec: np.ndarray,
    residuals: np.ndarray,
    model,
    axis_vec: np.ndarray,
    packet_idx: int,
    start_channel: int,
    stage_label: str,
    c_sound: float = C_SOUND,
) -> None:
    """
    Debug plot: along-array absolute position vs. arrival time with fitted line.

    positions_enu : (N,3) ENU positions of the *used* channels
    times_sec     : (N,) arrival times (seconds)
    residuals     : (N,) residuals from this fit
    model         : sklearn regressor (LinearRegression or RANSACRegressor)
    axis_vec      : (3,) array axis returned by fit_doa for this stage
    packet_idx    : int, which packet we are plotting
    start_channel : int, subarray start (for title)
    stage_label   : "stage 1", "stage 2", "final", etc.
    """

    if positions_enu.size == 0:
        return

    # Flatten array in vertical (use ENU but set z = 0)
    pos_flat = positions_enu.copy()
    pos_flat[:, 2] = 0.0

    # Ensure axis_vec is unit length
    axis_vec = np.asarray(axis_vec, dtype=float)
    norm_axis = np.linalg.norm(axis_vec)
    if norm_axis != 0.0:
        axis_vec /= norm_axis

    # Absolute along-array coordinate (no centering), then start at 0 m
    s = pos_flat @ axis_vec           # (N,)
    s = s - s.min()                   # make first used channel ≈ 0 m

    X_feat = s.reshape(-1, 1)

    # Predict times from the regression model
    if hasattr(model, "predict"):
        t_fit = model.predict(X_feat)
    else:
        t_fit = model(s)
    t_fit = np.asarray(t_fit).ravel()

    # Get slope dt/ds for angle annotation
    if hasattr(model, "estimator_"):  # RANSACRegressor
        slope = float(model.estimator_.coef_[0])
        intercept = float(model.estimator_.intercept_)
    else:                             # plain LinearRegression
        slope = float(model.coef_[0])
        intercept = float(model.intercept_)

    # Convert slope to angle w.r.t. array axis
    axis_dot_n = c_sound * slope
    axis_dot_n = float(np.clip(axis_dot_n, -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(axis_dot_n)))

    textstr = (
        f"slope = {slope:.3e} s/m\n"
        f"angle = ±{angle_deg:.1f}°"
    )

    # Sort for a nice continuous line
    order = np.argsort(s)
    s_sorted = s[order]
    t_sorted = times_sec[order]
    t_fit_sorted = t_fit[order]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.scatter(s_sorted, t_sorted, s=30, label="data", color="orange")
    ax.plot(s_sorted, t_fit_sorted, linewidth=2, label="fit")

    # # Highlight worst residual
    # if residuals is not None and len(residuals) == len(s):
    #     worst_idx = int(np.argmax(np.abs(residuals)))
    #     ax.scatter(s[worst_idx], times_sec[worst_idx],
    #                s=60, marker="x", label="max residual")

    ax.set_xlabel(f"Along-array position(m), distance from channel {start_channel}")
    ax.set_ylabel("Arrival time (s)")
    ax.set_title(
        f"Arrival times for packet {packet_idx} for subarray with start channel {start_channel}\n"
        f"{stage_label} fit"
    )
    ax.grid(True, alpha=0.3)

    # Put the annotation box in a corner depending on slope sign
    if slope >= 0:
        x = 0.98
        ha = "right"
    else:
        x = 0.02
        ha = "left"

    ax.text(
        x, 0.02, textstr,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment=ha,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    ax.legend()
    plt.tight_layout()
    plt.show()



# ---------------- CORE PER-PACKET LOGIC ----------------


def process_packet(
    packet_idx: int,
    start_channel: int,
    array_length: int,
    geodetic_channel_positions: dict[int, np.ndarray],
    packet_arrivals: dict,
) -> tuple[dict, dict]:
    """
    Process one packet:
      - build subarray for given start_channel & array_length
      - run DOA fit with two-stage cleaning
      - compute cone-plane intersection (inner/nominal/outer)
      - return:
          ellipse_entry: dict with inner/nominal/outer points (lists)
          info_entry   : dict with all diagnostic info
    """

    # ---------------- basic bookkeeping ----------------
    packet_key = str(packet_idx)
    all_channel_arrivals = packet_arrivals.get(packet_key, None)
    if all_channel_arrivals is None:
        # No arrivals at all for this packet
        ellipse_entry = {
            "inner_points": [],
            "nominal_points": [],
            "outer_points": [],
        }
        info_entry = {
            "valid": False,
            "packet_idx": packet_idx,
            "start_channel": start_channel,
            "array_length": array_length,
            "reason": "packet not present in peaks file",
        }
        return ellipse_entry, info_entry

    # Build subarray arrivals dict using INT channel keys
    arrivals_dict = {
        ch: all_channel_arrivals[str(ch)]
        for ch in range(start_channel, start_channel + array_length)
        if str(ch) in all_channel_arrivals
    }

    channels = sorted(arrivals_dict.keys())
    channels_available = len(channels)

    if channels_available < MIN_CHANNELS_FOR_DOA:
        # Not enough channels even to try
        ellipse_entry = {
            "inner_points": [],
            "nominal_points": [],
            "outer_points": [],
        }
        info_entry = {
            "valid": False,
            "packet_idx": packet_idx,
            "start_channel": start_channel,
            "array_length": array_length,
            "channels_requested": array_length,
            "channels_available": channels_available,
            "reason": "not enough channels in packet",
        }
        return ellipse_entry, info_entry

    # Arrival times (samples -> seconds)
    times_packet = np.array([arrivals_dict[ch] for ch in channels], dtype=float)
    times_sec = times_packet / FS

    # ---------------- positions in ENU ----------------
    # reference = channel 0 geodetic
    ref = geodetic_channel_positions[0].copy()
    ref[2] = 0.0  # ocean level (for ENU transform)

    channel_positions_enu = []
    for ch in channels:
        lat, lon, alt = geodetic_channel_positions[ch]
        e, n, u = geodetic2enu(lat, lon, alt, ref[0], ref[1], ref[2])
        channel_positions_enu.append([e, n, u])
    channel_positions_enu = np.array(channel_positions_enu, dtype=float)

    # ---------------- DOA fitting with cleaning ----------------
    try:
        # 1st fit
        slope, residuals, model, axis = fit_doa(times_sec, channel_positions_enu)

        if DEBUG_PLOTS and packet_idx == DEBUG_PACKET_IDX:
            debug_plot_doa_stage(
                positions_enu=channel_positions_enu,
                times_sec=times_sec,
                residuals=residuals,
                model=model,
                axis_vec=axis,
                packet_idx=packet_idx,
                start_channel=start_channel,
                stage_label="Stage 1 (all channels with ToA)",
            )

        # first-pass residual mask
        tau_1 = MAX_EXTRA_DIST_1 / C_SOUND
        mask_1 = np.abs(residuals) < tau_1

        channels_inlier = [ch for ch, keep in zip(channels, mask_1) if keep]
        positions_inlier = channel_positions_enu[mask_1]
        times_inlier = times_sec[mask_1]

        if len(times_inlier) < MIN_CHANNELS_FOR_DOA:
            raise RuntimeError("Too few inliers after first cleaning")

        # 2nd fit on inliers
        slope_refit, residuals_refit, model_refit, axis_refit = fit_doa(times_inlier, positions_inlier)

        if DEBUG_PLOTS and  packet_idx == DEBUG_PACKET_IDX:
            debug_plot_doa_stage(
                positions_enu=positions_inlier,
                times_sec=times_inlier,
                residuals=residuals_refit,
                model=model_refit,
                axis_vec=axis_refit,
                packet_idx=packet_idx,
                start_channel=start_channel,
                stage_label="Stage 2 (after 10m cleaning)",
            )

        # second-pass mask
        tau_2 = MAX_EXTRA_DIST_2 / C_SOUND
        mask_refit = np.abs(residuals_refit) < tau_2

        channels_inlier2 = [ch for ch, keep in zip(channels_inlier, mask_refit) if keep]
        positions_inlier2 = positions_inlier[mask_refit]
        times_inlier2 = times_inlier[mask_refit]
        residuals_refit2 = residuals_refit[mask_refit]

        if len(times_inlier2) < MIN_CHANNELS_FOR_DOA:
            raise RuntimeError("Too few inliers after second cleaning")

        slope_final, residuals_final, model_final, axis_final = fit_doa(
            times_inlier2, positions_inlier2
        )

        if DEBUG_PLOTS and packet_idx == DEBUG_PACKET_IDX:
            debug_plot_doa_stage(
                positions_enu=positions_inlier2,
                times_sec=times_inlier2,
                residuals=residuals_final,
                model=model_final,
                axis_vec=axis_final,
                packet_idx=packet_idx,
                start_channel=start_channel,
                stage_label="Stage 3 (Final fit after 5 m cleaning)",
            )

    except Exception as exc:
        # Any failure in DOA → mark packet invalid but still record some info
        ellipse_entry = {
            "inner_points": [],
            "nominal_points": [],
            "outer_points": [],
        }
        info_entry = {
            "valid": False,
            "packet_idx": packet_idx,
            "start_channel": start_channel,
            "array_length": array_length,
            "channels_requested": array_length,
            "channels_available": channels_available,
            "reason": f"DOA fit failed: {exc!s}",
        }
        return ellipse_entry, info_entry
    

    # ---------------- angle + residual stats ----------------
    cone_angle = cone_radian_from_slope(slope_final, speed=C_SOUND)
    cone_angle_deg = float(np.degrees(cone_angle))

    if cone_angle_deg <= 90.0:
        angle_off_axis = cone_angle_deg
        side = "+axis (first to last)"
    else:
        angle_off_axis = 180.0 - cone_angle_deg
        side = "-axis (last to first)"

    residual_rms = float(np.sqrt(np.mean(residuals_final**2)))
    residual_max_abs = float(np.max(np.abs(residuals_final)))

    # ---------------- cone-plane intersection ----------------
    inner_points, nominal_points, outer_points = cone_plane_intersection(
        positions_inlier2,
        slope=slope_final,
        source_depth=SOURCE_DEPTH,
        array_axis=axis_final,
        angle_uncertainty_deg=ANGLE_UNCERTAINTY_DEG,
        speed=C_SOUND,
        n_samples=N_SAMPLES_CONE,
        xy_window=XY_WINDOW,
    )

    # convert to plain lists for JSON
    ellipse_entry = {
        "inner_points": inner_points.tolist(),
        "nominal_points": nominal_points.tolist(),
        "outer_points": outer_points.tolist(),
    }

    info_entry = {
        "valid": True,
        "packet_idx": packet_idx,
        "start_channel": start_channel,
        "array_length": array_length,
        "channels_requested": array_length,
        "channels_available": channels_available,
        "channels_used": len(channels_inlier2),
        "slope_final": float(slope_final),
        "angle_deg": cone_angle_deg,
        "angle_off_axis_deg": float(angle_off_axis),
        "side": side,
        "n_inliers_stage1": int(len(times_inlier)),
        "n_inliers_stage2": int(len(times_inlier2)),
        "residual_rms": residual_rms,
        "residual_max_abs": residual_max_abs,
        "ellipse_has_points": bool(len(nominal_points) > 0),
    }

    return ellipse_entry, info_entry


# ---------------- MAIN ----------------


def main() -> None:
    print("starting DOA computation...")

    parser = argparse.ArgumentParser(description="compute DOA for all packets")
    parser.add_argument(
        "--start-channel",
        type=int,
        default=100,
        help="Start channel of subarray",
    )
    parser.add_argument(
        "--array-length",
        type=int,
        default=30,
        help="Length of the subarray",
    )
    args = parser.parse_args()

    start_channel = args.start_channel
    array_length = args.array_length



    # #1) Load geodetic positions (once)
    # geodetic_channel_positions = compute_channel_positions(
    #     CABLE_LAYOUT_FILE,
    #     channel_count=1200,
    #     channel_distance=1.02,
    #     origin_offset_m=CHANNEL_OFFSET,
    # )

        # 1) Compute channel positions in geodetic and (optionally) save them
    channel_pos_all = compute_channel_positions(
        CABLE_LAYOUT_FILE,
        channel_count=1200-N_SKIP,
        channel_distance=1.02
    )

    from typing import Dict, List
    channel_pos_geo: Dict[int, List[float]] = {}


    # place 0–23 at first coordinate (same as wet_positions[0])
    first_lat, first_lon, first_alt = channel_pos_all[0]
    for ch in range(N_SKIP + 1):     # 0..23 inclusive
        channel_pos_geo[ch] = [first_lat, first_lon, first_alt]

    # fill the rest using wet_positions shifted by 23
    for ch in range(N_SKIP + 1, 1200-N_SKIP):
        channel_pos_geo[ch] = channel_pos_all[ch - (N_SKIP + 1)]

    # channel_position_path = r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\B_4\channel_pos_geo_adjusted.json"

    # with open(channel_position_path, "r", encoding="utf-8") as f:
    #     channel_pos_geo_raw = json.load(f)

    # # Convert keys to integers
    # geodetic_channel_positions = {int(ch): pos for ch, pos in channel_pos_geo_raw.items()}

    # Now valid:
    #enu_ref = enu_reference_from_channels(channel_pos_geo, channel_idx=0)


    # 2) Load arrivals for all packets and all channels (once)
    with open(PEAKS_FILE, "r") as file:
        packet_arrivals = json.load(file)

    # 3) Loop over all packets
    ellipse_dict: dict[str, dict] = {}
    info_dict: dict[str, dict] = {}

    packet_keys = sorted(packet_arrivals.keys(), key=int)

    for pkt_key in packet_keys:
        pkt_idx = int(pkt_key)
        ellipse_entry, info_entry = process_packet(
            pkt_idx,
            start_channel,
            array_length,
            channel_pos_geo,
            packet_arrivals,
        )
        ellipse_dict[str(pkt_idx)] = ellipse_entry
        info_dict[str(pkt_idx)] = info_entry

        print(
            f"packet {pkt_idx:4d}: "
            f"valid={info_entry['valid']} "
            f"angle={info_entry.get('angle_deg', None)}"
        )

    # 4) Save aggregated JSONs
    script_dir = Path(__file__).resolve().parent

    # New target folders
    ellipse_dir = (script_dir / ".." / "resources" / "new_cable_subarray_ellipses").resolve()
    info_dir    = (script_dir / ".." / "resources" / "new_cable_subarray_info").resolve()

    # Ensure folders exist
    os.makedirs(ellipse_dir, exist_ok=True)
    os.makedirs(info_dir, exist_ok=True)

    # Construct filenames
    ellipse_path = ellipse_dir / f"ellipse_bands_start_ch_{start_channel}_arrlen_{array_length}.json"
    info_path    = info_dir    / f"doa_info_start_ch_{start_channel}_arrlen_{array_length}.json"

    # Save JSON files
    with ellipse_path.open("w", encoding="utf-8") as f:
        json.dump(ellipse_dict, f, indent=2)

    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info_dict, f, indent=2)

    print(f"Saved ellipse bands to: {ellipse_path}")
    print(f"Saved DOA info to:     {info_path}")

if __name__ == "__main__":
    main()
