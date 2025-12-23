import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, peak_prominences, hilbert
import json
import math
from mpl_toolkits.mplot3d import Axes3D



'''

Altough this is called debugging.py, it contains functions that are actually used in processing, mostly for debugging and visulization, but deleting this 
script would create issues, so dont delete it.


several other scripts depend on this one at this stage.


'''












def cone_plane_intersection_enu(center_enu, axis_enu, psi_rad, z_source, n_points=360):
    """
    Compute approximate ENU points on the intersection of:
      - a cone (apex at center_enu, axis axis_enu, opening angle psi_rad)
      - a horizontal plane U = z_source

    Returns
    -------
    pts : (M, 3) array of ENU points on the intersection curve.
    """
    C = np.asarray(center_enu, dtype=float)
    axis = np.asarray(axis_enu, dtype=float)
    axis /= np.linalg.norm(axis)

    # Build orthonormal basis {axis, b1, b2} where b1,b2 span the plane ⟂ axis
    tmp = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(tmp, axis)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    b1 = np.cross(axis, tmp)
    b1 /= np.linalg.norm(b1)
    b2 = np.cross(axis, b1)
    b2 /= np.linalg.norm(b2)

    z0 = C[2]
    tan_psi = np.tan(psi_rad)

    pts = []
    phis = np.linspace(0.0, 2*np.pi, n_points, endpoint=False)

    for phi in phis:
        # for a given azimuth around the axis, the cone direction is:
        # dir = axis + tan(psi) * (cos phi * b1 + sin phi * b2), up to scale
        k = np.cos(phi) * b1[2] + np.sin(phi) * b2[2]
        denom = axis[2] + tan_psi * k
        if abs(denom) < 1e-9:
            continue  # almost parallel to plane; skip

        # solve for t so that z = z_source:
        # z_source = z0 + t * (axis_z + tan_psi * k)
        t = (z_source - z0) / denom
        if t <= 0:
            # we usually only care about "forward" from the apex
            continue

        # position on the cone at this t and phi
        r_perp = t * tan_psi * (np.cos(phi) * b1 + np.sin(phi) * b2)
        p = C + t * axis + r_perp
        pts.append(p)

    if len(pts) == 0:
        return np.empty((0, 3))
    return np.vstack(pts)



def visualize_doa_fit(positions_enu, times, residuals, direction_a, direction_b, channels, model, title="DOA fit"):
    X, Y, Z = positions_enu[:,0], positions_enu[:,1], positions_enu[:,2]

    # Normalize direction - direction is the propagation direction
    d_prop_a = np.array(direction_a, dtype=float)
    d_prop_a /= np.linalg.norm(d_prop_a) if np.linalg.norm(d_prop_a) != 0 else 1.0

    d_prop_b = np.array(direction_b, dtype=float)
    d_prop_b /= np.linalg.norm(d_prop_b) if np.linalg.norm(d_prop_b) != 0 else 1.0



    fig = plt.figure(figsize=(14,5))

    # ---------------- 3D geometry + direction ----------------
    ax = fig.add_subplot(1,2,1, projection='3d')
    p = ax.scatter(X, Y, Z, c=times, cmap='viridis')
    fig.colorbar(p, ax=ax, label="arrival time")

    center = np.array([X.mean(), Y.mean(), Z.mean()])

    
    L = 5
    
    # # Direction TOWARD source (ambiguous)
    # ax.plot(
    #     [center[0], center[0] + L * d_prop_a[0]],
    #     [center[1], center[1] + L * d_prop_a[1]],
    #     [center[2], center[2] + L * d_prop_a[2]],
    #     color="r",
    #     lw=3,
    #     label="toward source option A"
    # )

    # # Direction TOWARD source (ambiguous)
    # ax.plot(
    #     [center[0], center[0] + L * d_prop_b[0]],
    #     [center[1], center[1] + L * d_prop_b[1]],
    #     [center[2], center[2] + L * d_prop_b[2]],
    #     color="b",
    #     lw=3,
    #     label="toward source option B"
    # )

    ax.set_title("Channel geometry + arrival times")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_zlabel("Up (m)")

     # ---------------- along-array coordinate vs arrival time + fit ----------------
    ax2 = fig.add_subplot(1, 2, 2)

    # Reproduce the same horizontalization + axis as in fit_doa
    pos_flat = positions_enu.copy()
    pos_flat[:, 2] = 0.0

    axis_vec = pos_flat[-1] - pos_flat[0]
    axis_norm = np.linalg.norm(axis_vec)
    if axis_norm != 0:
        axis_vec /= axis_norm

    # Scalar coordinate along array (this is exactly s in fit_doa)
    s = pos_flat @ axis_vec            # shape (N,)
    X_feat = s.reshape(-1, 1)          # (N, 1), as used in fit_doa

    # Predicted times from the regression model
    if hasattr(model, "predict"):
        t_fit = model.predict(X_feat)
    else:
        # Fallback in case model is a simple callable in some experiment
        t_fit = model(s)
    t_fit = np.asarray(t_fit).ravel()

        # Get slope (dt/ds) for annotation
    if hasattr(model, "estimator_"):  # RANSAC
        slope = float(model.estimator_.coef_[0])
        intercept = float(model.estimator_.intercept_)
    else:  # plain LinearRegression
        slope = float(model.coef_[0])
        intercept = float(model.intercept_)
    
    c = 1475.0
    axis_dot_n = c * slope
    axis_dot_n = float(np.clip(axis_dot_n, -1.0, 1.0))
    angle_deg = np.degrees(np.arccos(axis_dot_n))

    textstr = (
        f"slope = {slope:.3e} s/m\n"
        f"angle = ±{angle_deg:.1f}°"
)


    # Sort along s for a nice continuous line
    order = np.argsort(s)
    s_sorted = s[order]
    times_sorted = times[order]
    t_fit_sorted = t_fit[order]

    # Plot data + fit
    ax2.scatter(s_sorted, times_sorted, s=30, label="data")
    ax2.plot(s_sorted, t_fit_sorted, linewidth=2, label="fit")

    # Highlight largest residual point (optional but nice)
    worst_idx = int(np.argmax(np.abs(residuals)))
    ax2.scatter(s[worst_idx], times[worst_idx], s=60, marker='x', label="max residual")

    if slope >= 0:
        # Put the box bottom-right
        x = 0.98
        ha = 'right'
    else:
        # Put the box bottom-left
        x = 0.02
        ha = 'left'

    y = 0.02   # stays at bottom of plot

    ax2.text(
        x, y, textstr,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        horizontalalignment=ha,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
    )

    ax2.set_xlabel("Along-array coordinate s (m)")
    ax2.set_ylabel("Arrival time (s)")
    ax2.set_title("Arrival time vs along-array coordinate + fitted line")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()




def distance_3d(point1, point2):
    """
    Computes the 3D distance in meters between two points given as
    (latitude, longitude, depth), where depth is in meters below sea surface.
    """
    # Unpack points
    lat1, lon1, depth1 = point1
    lat2, lon2, depth2 = point2

    # Mean latitude in radians
    mean_lat = math.radians((lat1 + lat2) / 2.0)

    # Convert degree differences to meters
    dy = (lat2 - lat1) * 111_320           # meters per degree latitude
    dx = (lon2 - lon1) * 111_320 * math.cos(mean_lat)
    dz = depth2 - depth1                    # depth difference in meters

    # 3D Euclidean distance
    distance = math.sqrt(dx**2 + dy**2 + dz**2)
    return distance

def find_distance_tx_to_channels(channel_id, source_tx_id):
    """
    Load channel positions and source TX positions from JSON files,
    then compute and print the distance between a specific channel and source TX.
    """
    filename = r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\B_4\channel_pos_geo.json"
    with open(filename, 'r') as file:
        channel_pos_geo = json.load(file)
    # Example points from the channel positions


    filename_2 = r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\B_4\source_tx_pos.json"
    with open(filename_2, 'r') as file:
        source_tx_positions = json.load(file)

    p1 = (channel_pos_geo[channel_id][0], channel_pos_geo[channel_id][1], channel_pos_geo[channel_id][2])  # (lat, lon, alt)
    p2 = (source_tx_positions[source_tx_id][0], source_tx_positions[source_tx_id][1], -30)  # (lat, lon, alt)

    # print(f"Point 1 (Channel {channel_id}): {p1}")
    # print(f"Point 2 (Source TX {source_tx_id}): {p2}")

    distance = distance_3d(p1, p2)
    print(f"Distance between Channel {channel_id} and Source TX {source_tx_id}: {distance:.4f} m")
    return distance




def depth_correct_timestamp(packet_idx, channel_idx, t_sample, fs, c=1475.0):
    """
    Correct a timestamp for vertical propagation effects using real 3D geometry.

    Inputs
    ------
    packet_idx : int
        Index of the transmitted packet (source position).
    channel_idx : int
        DAS channel index.
    t_sample : float
        Raw timestamp in samples.
    fs : float
        Sampling frequency.
    c : float
        Sound speed [m/s].

    Returns
    -------
    t_corr : float
        Depth-corrected timestamp in seconds.
    """

    # --- Load geo positions ---
    with open(r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\B_4\channel_pos_geo.json","r") as f:
        ch_geo = json.load(f)

    with open(r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\B_4\source_tx_pos.json","r") as f:
        tx_geo = json.load(f)

    # --- Get channel position (lat, lon, depth) ---
    if isinstance(ch_geo, list):
        lat_c, lon_c, z_c = ch_geo[channel_idx]
    else:  # dict with string or int keys
        ch_entry = ch_geo.get(str(channel_idx), ch_geo.get(channel_idx))
        lat_c, lon_c, z_c = ch_entry

    # --- Get source position (lat, lon); fix depth at -30 m ---
    if isinstance(tx_geo, list):
        lat_s, lon_s = tx_geo[packet_idx][:2]
    else:  # dict with string or int keys
        tx_entry = tx_geo.get(str(packet_idx), tx_geo.get(packet_idx))
        lat_s, lon_s = tx_entry[:2]

    z_s = -30.0  # known source depth (m)

    # --- Convert to local Cartesian distances (approx ENU) ---
    mean_lat = math.radians((lat_c + lat_s) / 2.0)
    dx = (lon_c - lon_s) * 111_320 * math.cos(mean_lat)
    dy = (lat_c - lat_s) * 111_320
    dz = z_c - z_s

    # Full slant range
    R = math.sqrt(dx*dx + dy*dy + dz*dz)
    if R < 1e-6:
        # Degenerate: just return raw time in seconds
        return t_sample / fs

    # Vertical direction cosine (component of ray along vertical)
    n_z = dz / R

    # Vertical travel-time contribution for this channel
    # (simple model: time associated with depth component)
    t_vertical = dz * n_z / c  # [s]

    # Raw time in seconds
    t_raw = t_sample / fs

    # Corrected time (remove vertical term)
    t_corr = t_raw - t_vertical

    return t_corr





def _xcorr_normalized(x, h):
    """valid-mode, normalized cross-correlation."""
    xc = correlate(x, h, mode='valid')
    m = np.max(np.abs(xc))
    return xc / m if m != 0 else xc




def plot_channel_corr_and_peaks(rx_col, preamble, peak_properties,
                                targets=None, tol=None, fs=25_000,
                                zoom_center=None, zoom_halfwidth=500,
                                title_prefix=""):
    """
    Draw normalized correlation, overlay detected peaks (height & prominence),
    and optional target windows (target ± tol).

    rx_col          : 1-D array with one channel of DAS samples (already filtered)
    preamble        : 1-D array (same one you pass to detector)
    peak_properties : dict you pass to find_peaks (prominence, height, distance)
    targets         : 1-D array of expected packet sample indices (SAME timebase as peaks),
                      e.g. target = med_first + np.arange(N)*sequence_period
                      (but shifted to your slice’s origin if needed)
    tol             : integer (samples). If not None, draw target+tol bands.
    zoom_center     : center sample of a zoom window (in correlation index coords)
    zoom_halfwidth  : half-width of the zoom window (samples)
    """
    # 1) correlation (normalized)
    xc = _xcorr_normalized(rx_col, preamble)

    # 2) run the exact same peak finder you use in production
    from scipy.signal import find_peaks
    pk_idx, pk_props = find_peaks(xc, **peak_properties)

    # compute prominences explicitly (useful to see real values)
    prom, left_bases, right_bases = peak_prominences(xc, pk_idx)

    # 3) choose plotting range
    n = len(xc)
    if zoom_center is None:
        lo, hi = 0, n
    else:
        lo = max(0, int(zoom_center - zoom_halfwidth))
        hi = min(n, int(zoom_center + zoom_halfwidth))

    xs = np.arange(lo, hi)
    xcv = xc[lo:hi]

    # make the plot
    plt.figure(figsize=(11, 4))
    plt.plot(xs, xcv, label='normalized corr')
    # overlay peaks in range
    in_rng = (pk_idx >= lo) & (pk_idx < hi)
    pk = pk_idx[in_rng]
    if pk.size:
        plt.plot(pk, xc[pk], "o", label="peaks")
        # annotate height and prominence for a few nearest peaks (avoid clutter)
        for i, k in enumerate(pk[:20]):  # cap annotations
            txt = f"h={xc[k]:.2f}, p={prom[np.where(pk_idx==k)[0][0]]:.2f}"
            plt.annotate(txt, (k, xc[k]), xytext=(5, 8),
                         textcoords='offset points', fontsize=8)

    # 4) overlay target windows (target±tol) if provided
    if targets is not None and tol is not None:
        # IMPORTANT: targets must be in the SAME coordinate system as correlation indices
        # If your targets are in RAW sample indices, convert: target_corr = target_raw - PRE
        for t in targets:
            t_corr = t
            if lo <= t_corr < hi:
                plt.axvspan(t_corr-tol, t_corr + tol, alpha=0.15, color='gray')
                plt.axvline(t_corr, ls='--', alpha=0.4, label='target' if t == targets[0] else None)
                print(f"Target position (corr idx): {t_corr}")

    plt.xlabel("correlation index (samples)")
    plt.ylabel("normalized corr")
    plt.title(f"{title_prefix} corr & peaks")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



def plot_channel_corr_with_selected(
    rx_col,
    preamble,
    selected_peaks_map,
    *,
    targets=None,
    tol=None,
    fs=25_000,
    targets_are_raw=False,
    zoom_center=None,
    zoom_halfwidth=800,
    title_prefix="",
    annotate=True
):
    """
    Plot normalized matched-filter correlation for ONE channel and overlay:
      - targets (dashed line) + optional ±tol band, and
      - selected peaks (those that actually qualified and were saved).

    Parameters
    ----------
    rx_col : 1-D array
        One DAS channel (already filtered).
    preamble : 1-D array
        The reference preamble.
    selected_peaks_map : dict[int -> int]
        {packet_id : corr_index} for this channel (from mypeaks[ch]).
        NOTE: corr_index is in *correlation* coordinates.
    targets : 1-D array, optional
        Expected packet positions. If `targets_are_raw=False`, they are
        assumed to already be in correlation indices. If raw, set
        `targets_are_raw=True`.
    tol : int, optional
        Tolerance window (samples). If given, draw target ± tol band(s).
    fs : int
        Sample rate (for axis labels only).
    targets_are_raw : bool
        If True, convert targets to correlation coordinates by subtracting PRE.
    zoom_center : int, optional
        Center index (correlation coords) for zoom.
    zoom_halfwidth : int
        Half-width for zoom window (samples).
    title_prefix : str
        Title prefix (e.g., "ch 148").
    annotate : bool
        If True, annotate selected peaks with packet id and value.
    """
    PRE = len(preamble) - 1

    # 1) correlation (normalized)
    xc = _xcorr_normalized(rx_col, preamble)
    xc = np.abs(hilbert(xc))          # amplitude envelope
    n = len(xc)

    # 2) convert targets if provided
    if targets is not None:
        targets = np.asarray(targets, dtype=int)
        if targets_are_raw:
            targets = targets - PRE  # convert to correlation indices

    # 3) pick plotting window
    if zoom_center is None:
        lo, hi = 0, n
    else:
        lo = max(0, int(zoom_center - zoom_halfwidth))
        hi = min(n, int(zoom_center + zoom_halfwidth))

    xs = np.arange(lo, hi)
    xcv = xc[lo:hi]

    # 4) plot
    plt.figure(figsize=(11, 4))
    plt.plot(xs, xcv, label="normalized corr")

    # 5) targets + tol bands
    if targets is not None:
        first_drawn = True
        for t in targets:
            if lo <= t < hi:
                if tol is not None:
                    plt.axvspan(t - tol, t + tol, alpha=0.12, label="target±tol" if first_drawn else None)
                plt.axvline(t, ls="--", alpha=0.5, label="target" if first_drawn else None)
                first_drawn = False

    # 6) selected peaks (packet_id -> corr_idx)
    if selected_peaks_map:
        pkt_ids = sorted(selected_peaks_map.keys())
        sel_idx = np.array([selected_peaks_map[p] for p in pkt_ids], dtype=int)
        in_rng = (sel_idx >= lo) & (sel_idx < hi)
        if np.any(in_rng):
            plt.plot(sel_idx[in_rng], xc[sel_idx[in_rng]], "o", label="selected peaks")
            if annotate:
                for pid, idx in zip(np.array(pkt_ids)[in_rng], sel_idx[in_rng]):
                    plt.annotate(f"pkt {pid}\n{idx}", (idx, xc[idx]),
                                 xytext=(6, 8), textcoords="offset points", fontsize=8)

    plt.xlabel("correlation index (samples)")
    plt.ylabel("normalized corr")
    plt.title(f"{title_prefix} corr + selected peaks (fs={fs} Hz)")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def timestamp_differenct_2_distance(speed_m_s, fs, timestamp_1, timestamp_2):
    """
    Compute distance difference (m) between two timestamps,
    given a propagation speed (m/s).
    """
    delta_t = (timestamp_2 - timestamp_1) / fs # in seconds
    distance_diff = speed_m_s * delta_t
    return distance_diff

def main()->None:
    d_76 = find_distance_tx_to_channels("76", 50)
    d_115 = find_distance_tx_to_channels("115", 50)
    d_255 = find_distance_tx_to_channels("255", 50)
    d_338 = find_distance_tx_to_channels("338", 50)


    d_76_115_theory = abs(d_76 - d_115)
    d_255_338_theory = abs(d_255 - d_338)
    print("Theoretical Distance difference between Channel 76 and Channel 115 for Source TX 50:", d_76_115_theory)
    print("Theoretical Distance difference between Channel 255 and Channel 338 for Source TX 50:", d_255_338_theory)

    print

    
    filename = r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\B_4\peaks-reordered_all_hilbert_channels.json"
    with open(filename, 'r') as file:
        packet_channel_timestamps = json.load(file)
    
    timestamp_76 = packet_channel_timestamps["50"]["76"]
    timestamp_115 = packet_channel_timestamps["50"]["115"]
    timestamp_255 = packet_channel_timestamps["50"]["255"]
    timestamp_338 = packet_channel_timestamps["50"]["338"]

    speed = 1475.0  # m/s
    fs = 25000  # Hz

    dd_76_115 = np.abs(timestamp_differenct_2_distance(speed, fs, timestamp_76, timestamp_115))
    dd_255_338 = np.abs(timestamp_differenct_2_distance(speed, fs, timestamp_255, timestamp_338))

    print("Measured distance (extra travel) difference between Channel 76 and Channel 115 for Packet 50:", dd_76_115)
    print("Measured distance (extra travel) difference between Channel 255 and Channel 338 for Packet 50:", dd_255_338) 

    






if __name__ == "__main__":
    main()