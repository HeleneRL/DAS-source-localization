import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, peak_prominences, hilbert
import json
import math
from mpl_toolkits.mplot3d import Axes3D




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

    print(f"Point 1 (Channel {channel_id}): {p1}")
    print(f"Point 2 (Source TX {source_tx_id}): {p2}")

    distance = distance_3d(p1, p2)
    print(f"Distance between Channel {channel_id} and Source TX {source_tx_id}: {distance:.4f} m")
    return distance






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