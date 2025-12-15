import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor
import argparse
from pymap3d import enu2geodetic, geodetic2enu


from dasprocessor.debugging import  visualize_doa_fit, depth_correct_timestamp
from dasprocessor.channel_gps import compute_channel_positions
import json
import os



def fit_doa(times, channel_positions_enu):
    """
    Fit DOA from arrival times and channel positions (ENU).

    """

    times = np.asarray(times, dtype=float)
    channel_positions_enu = np.asarray(channel_positions_enu, dtype=float).copy()


    # # Array axis (horizontal) from endpoints
    # axis = channel_positions_enu[-1] - channel_positions_enu[0]
    # axis /= np.linalg.norm(axis)

    # Center positions
    center = channel_positions_enu.mean(axis=0)           
    P = channel_positions_enu - center                     

    # SVD to find principal direction
    U, S, Vt = np.linalg.svd(P, full_matrices=False)
    axis = Vt[0]                                   

    # Normalize axis, should be unit vector already from SVD but just in case
    axis = axis / np.linalg.norm(axis)

    # enforce a consistent direction (e.g. roughly from first to last)
    if np.dot(axis, channel_positions_enu[-1] - channel_positions_enu[0]) < 0:
        axis = -axis
   

    s = channel_positions_enu @ axis
    s = s - s[0]        # make first sensor s=0
    X = s.reshape(-1, 1)
    y = times

    # 3) Robust or plain 1D regression: t = a * s + b
   
    base = LinearRegression()
    model = RANSACRegressor(
        base,
        min_samples=3,
        max_trials=200,
    )
    model.fit(X, y)
    slope = float(model.estimator_.coef_[0])   # dt/ds

    # Time residuals from this 1D model
    residuals = y - model.predict(X)


    return slope, residuals, model, axis


def cone_radian_from_slope(slope, speed=1475.0):
    axis_dot_n = speed * slope
    axis_dot_n = float(np.clip(axis_dot_n, -1.0, 1.0))
    angle_rad = np.arccos(axis_dot_n)

    return angle_rad



def orthonormal_basis_from_axis(axis):
    """
    Given a unit vector 'axis', build an orthonormal basis (e1, e2, e3),
    where e1 = axis, and e2, e3 span the perpendicular plane.
    """
    axis = np.asarray(axis, dtype=float)

    # Pick a vector not parallel to axis
    if abs(axis[2]) < 0.9:
        tmp = np.array([0.0, 0.0, 1.0])
    else:
        tmp = np.array([1.0, 0.0, 0.0])

    e2 = np.cross(axis, tmp)
    e2 = e2 / np.linalg.norm(e2)
    e3 = np.cross(axis, e2)

    return axis, e2, e3

def cone_plane_intersection(
    channel_positions_enu,
    slope,
    source_depth,
    array_axis,
    angle_uncertainty_deg,
    speed=1475.0,
    n_samples=360,
    theta_eps_deg=0.5,   # avoid exactly 0 or 180
    max_range_m=None,    
    xy_window=None,      
):
    """
    Compute intersection curves between the DOA cone (from a 1D array)
    """

    channel_positions_enu = np.asarray(channel_positions_enu, dtype=float)
    array_ref = channel_positions_enu.mean(axis=0)

    # --- Nominal angle between array axis and propagation direction ---
    theta_nominal = cone_radian_from_slope(slope, speed=speed)  # radians
    theta_nominal_deg = np.degrees(theta_nominal)

    # Clamp nominal angle away from 0 and 180 to avoid exact degeneracy
    theta_nominal_deg = np.clip(
        theta_nominal_deg, theta_eps_deg, 180.0 - theta_eps_deg
    )

    # Inner/outer cone angles
    inner_deg = np.clip(
        theta_nominal_deg - angle_uncertainty_deg,
        theta_eps_deg,
        180.0 - theta_eps_deg,
    )
    outer_deg = np.clip(
        theta_nominal_deg + angle_uncertainty_deg,
        theta_eps_deg,
        180.0 - theta_eps_deg,
    )

    theta_inner   = np.radians(inner_deg)
    theta_outer   = np.radians(outer_deg)
    theta_nominal = np.radians(theta_nominal_deg)

    # --- Local orthonormal basis: e1 = array_axis ---
    e1, e2, e3 = orthonormal_basis_from_axis(array_axis)

    def intersect_for_theta(theta_val: float) -> np.ndarray:
        """
        Given a cone half-angle theta_val (radians) around e1,
        intersect with z = source_depth plane and return ENU points.
        Applies max_range_m / xy_window filters if given.
        """

        phi = np.linspace(0.0, 2 * np.pi, n_samples, endpoint=False)
        cos_theta = np.cos(theta_val)
        sin_theta = np.sin(theta_val)

        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        # PROPAGATION directions on cone:
        n_prop = (
            cos_theta * e1[None, :]
            + sin_theta * (
                cos_phi[:, None] * e2[None, :]
                + sin_phi[:, None] * e3[None, :]
            )
        )

        # Start with directions FROM array TO source
        n_vecs = -n_prop

        dz = source_depth - array_ref[2]
        eps = 1e-8

        def try_with(n_vecs_local):
            n_z = n_vecs_local[:, 2]
            valid = np.abs(n_z) > eps
            if not np.any(valid):
                return None

            R = np.empty_like(n_z)
            R[valid] = dz / n_z[valid]
            valid = valid & (R > 0)
            if not np.any(valid):
                return None

            pts = array_ref[None, :] + R[valid, None] * n_vecs_local[valid, :]

            # --- NEW: apply range filter (max distance from array_ref) ---
            if max_range_m is not None:
                d = np.linalg.norm(pts - array_ref[None, :], axis=1)
                mask = d <= max_range_m
                pts = pts[mask]
                if pts.shape[0] == 0:
                    return None

            # --- NEW: apply ENU window filter ---
            if xy_window is not None:
                E_min, E_max, N_min, N_max = xy_window
                E = pts[:, 0]
                N = pts[:, 1]
                mask = (
                    (E >= E_min) & (E <= E_max) &
                    (N >= N_min) & (N <= N_max)
                )
                pts = pts[mask]
                if pts.shape[0] == 0:
                    return None

            return pts

        # Try current hemisphere
        pts = try_with(n_vecs)
        if pts is not None:
            return pts

        # Try flipped hemisphere
        pts_flipped = try_with(-n_vecs)
        if pts_flipped is not None:
            return pts_flipped

        return np.empty((0, 3))

    # --- Compute intersections for inner, nominal, outer cones ---
    inner_points   = intersect_for_theta(theta_inner)
    nominal_points = intersect_for_theta(theta_nominal)
    outer_points   = intersect_for_theta(theta_outer)

    return inner_points, nominal_points, outer_points



from sklearn.cluster import KMeans
"""

def fit_doa_multiline(
    times,
    channel_positions_enu,
    n_lines_target=3,
    min_points_per_line=3,
    alpha_s=0.2,  # weight of proximity in s (0..1)
):

    times = np.asarray(times, float)
    channel_positions_enu = np.asarray(channel_positions_enu, float)

    # --- compute array axis via SVD (same as fit_doa) ---
    center = channel_positions_enu.mean(axis=0)
    P = channel_positions_enu - center
    _, _, Vt = np.linalg.svd(P, full_matrices=False)
    axis = Vt[0] / np.linalg.norm(Vt[0])

    # make direction consistent
    if np.dot(axis, channel_positions_enu[-1] - channel_positions_enu[0]) < 0:
        axis = -axis

    # along-array coordinate s
    s = channel_positions_enu @ axis
    s = s - s.min()
    X = s.reshape(-1, 1)
    y = times
    n = len(y)

    # fallback if too few points
    if n < n_lines_target * min_points_per_line:
        lr = LinearRegression().fit(X, y)
        slope = float(lr.coef_[0])
        intercept = float(lr.intercept_)
        residuals = y - lr.predict(X)

        class SimpleModel:
            def __init__(self, a, b):
                self.coef_ = np.array([a])
                self.intercept_ = b
            def predict(self, X_in):
                X_in = np.asarray(X_in).reshape(-1, 1)
                return X_in[:, 0] * self.coef_[0] + self.intercept_

        model = SimpleModel(slope, intercept)
        extra = {
            "n_lines": 1,
            "lines": [{"slope": slope, "intercept": intercept, "points": list(range(n))}],
            "s": s.tolist(),
            "axis": axis.tolist(),
        }
        return slope, residuals, model, extra

    # --- rough global slope m0 for residuals ---
    lr0 = LinearRegression().fit(X, y)
    m0 = float(lr0.coef_[0])

    # feature space: proximity + vertical offset
    s_norm = (s - s.mean()) / (s.std() + 1e-9)
    residual = y - m0 * s
    features = np.column_stack([alpha_s * s_norm, residual])

    # choose k = n_lines_target but cap if not enough points
    max_k = n // min_points_per_line
    k = min(n_lines_target, max_k)
    if k < 1:
        k = 1

    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    labels = kmeans.fit_predict(features)

    slopes = []
    intercepts = []
    lines = []

    for c in range(k):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue

        lr = LinearRegression().fit(X[idx], y[idx])
        m = float(lr.coef_[0])
        b = float(lr.intercept_)
        slopes.append(m)
        intercepts.append(b)
        lines.append({"slope": m, "intercept": b, "points": idx.tolist()})

    slopes = np.array(slopes, float)
    intercepts = np.array(intercepts, float)
    k_eff = len(lines)



    if k_eff == 0:
        # safety fallback: just use global line
        slope_final = float(lr0.coef_[0])
        intercept_final = float(lr0.intercept_)
    else:
        sizes = np.asarray([len(L["points"]) for L in lines])

        # --- choose slope closest to the MEDIAN slope ---
        median_slope = float(np.median(slopes))

        # optionally, ignore tiny clusters if you want:
        # big_enough = sizes >= min_points_per_line
        # candidate_slopes = slopes[big_enough]
        # if not np.any(big_enough):
        #     candidate_slopes = slopes
        #     big_enough = np.ones_like(sizes, dtype=bool)
        # median_slope = float(np.median(candidate_slopes))

        # index of cluster whose slope is closest to the median
        best_idx = int(np.argmin(np.abs(slopes - median_slope)))

        slope_final = float(slopes[best_idx])

        # intercept from that cluster's own points
        idx_pts = np.array(lines[best_idx]["points"], int)
        intercept_final = float(np.mean(y[idx_pts] - slope_final * s[idx_pts]))

    # # ---------- NEW PART: pick dominant cluster ----------
    # sizes = np.asarray([len(L["points"]) for L in lines])
    # best_idx = int(np.argmax(sizes))     # index of largest cluster

    # slope_final = float(slopes[best_idx])

    # # intercept from that cluster's own points
    # idx_pts = np.array(lines[best_idx]["points"], int)
    # intercept_final = float(np.mean(y[idx_pts] - slope_final * s[idx_pts]))

    # ------------------------------------------------------

    class SimpleModel:
        def __init__(self, a, b):
            self.coef_ = np.array([a])
            self.intercept_ = b
        def predict(self, X_in):
            X_in = np.asarray(X_in).reshape(-1, 1)
            return X_in[:, 0] * self.coef_[0] + self.intercept_

    model_final = SimpleModel(slope_final, intercept_final)
    residuals_final = y - model_final.predict(X)

    extra = {
        "n_lines": int(k_eff),
        "lines": lines,
        "dominant_line_index": best_idx,
        "s": s.tolist(),
        "axis": axis.tolist(),
    }

    return slope_final, residuals_final, model_final, extra

"""
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import numpy as np


def fit_doa_multiline(
    times,
    channel_positions_enu,
    n_lines_target=3,
    min_points_per_line=3,
    alpha_s=0.2,     # weight of proximity in s (0..1)
    n_refine_iter=5, # how many residual-based refinement steps
    sound_speed=1475.0,
):
   


    times = np.asarray(times, float)
    channel_positions_enu = np.asarray(channel_positions_enu, float)

    # ---- Compute array axis via SVD (same as fit_doa) ----
    center = channel_positions_enu.mean(axis=0)
    P = channel_positions_enu - center
    _, _, Vt = np.linalg.svd(P, full_matrices=False)
    axis = Vt[0] / np.linalg.norm(Vt[0])

    # make direction consistent
    if np.dot(axis, channel_positions_enu[-1] - channel_positions_enu[0]) < 0:
        axis = -axis

    # ---- Along-array coordinate ----
    s = channel_positions_enu @ axis
    s = s - s.min()
    X = s.reshape(-1, 1)
    y = times
    n = len(y)

    slope_max = 1.0 / sound_speed  # physical max |dt/ds|

    # ---- Too few points -> single line ----
    if n < n_lines_target * min_points_per_line:
        lr = LinearRegression().fit(X, y)
        slope = float(lr.coef_[0])
        intercept = float(lr.intercept_)
        residuals = y - lr.predict(X)

        class SimpleModel:
            def __init__(self, a, b):
                self.coef_ = np.array([a])
                self.intercept_ = b
            def predict(self, X_in):
                X_in = np.asarray(X_in).reshape(-1, 1)
                return X_in[:, 0] * self.coef_[0] + self.intercept_

        model = SimpleModel(slope, intercept)
        extra = {
            "n_lines": 1,
            "lines": [{"slope": slope, "intercept": intercept, "points": list(range(n))}],
            "s": s.tolist(),
            "axis": axis.tolist(),
        }
        return slope, residuals, model, extra

    # ==========================================================
    # Stage 1: ROUGH CLUSTERING
    # ==========================================================
    lr0 = LinearRegression().fit(X, y)
    m0 = float(lr0.coef_[0])

    s_norm = (s - s.mean()) / (s.std() + 1e-9)
    residual0 = y - m0 * s
    features = np.column_stack([alpha_s * s_norm, residual0])

    max_k = n // min_points_per_line
    k = min(n_lines_target, max_k)
    if k < 1:
        k = 1

    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    labels = kmeans.fit_predict(features)

    # ==========================================================
    # Stage 2: RESIDUAL-BASED REFINEMENT WITH SLOPE SANITY CHECK
    # ==========================================================
    for _ in range(n_refine_iter):
        lines = []
        slopes = []
        intercepts = []

        for c in range(k):
            idx = np.where(labels == c)[0]
            if idx.size < min_points_per_line:
                continue

            # initial fit for this cluster
            lr = LinearRegression().fit(X[idx], y[idx])
            m = float(lr.coef_[0])
            b = float(lr.intercept_)

            # ---- physical sanity check ----
            if abs(m) > slope_max:
                # Re-fit using central 2–3 points in s for this cluster
                s_c = s[idx]
                y_c = y[idx]
                order_c = np.argsort(s_c)
                mid = len(order_c) // 2
                # take 3 central points if possible, else 2
                start = max(0, mid - 1)
                end = min(len(order_c), mid + 2)
                idx_mid = idx[order_c[start:end]]

                if idx_mid.size >= 2:
                    lr_mid = LinearRegression().fit(
                        s[idx_mid].reshape(-1, 1), y[idx_mid]
                    )
                    m = float(lr_mid.coef_[0])
                    b = float(lr_mid.intercept_)

                # still unphysical? clamp slope to physical max and
                # force line through median of cluster
                if abs(m) > slope_max:
                    m = np.sign(m) * slope_max * 0.99  # stay just below limit
                    s_med = float(np.median(s[idx]))
                    t_med = float(np.median(y[idx]))
                    b = t_med - m * s_med

            slopes.append(m)
            intercepts.append(b)
            lines.append({"slope": m, "intercept": b, "points": idx.tolist()})

        if len(lines) == 0:
            labels = np.zeros(n, int)
            break

        slopes = np.array(slopes)
        intercepts = np.array(intercepts)

        # ---- reassign each point based ONLY on residual to each line ----
        new_labels = np.zeros(n, int)
        for i in range(n):
            res_to_each = [
                abs(y[i] - (slopes[j] * s[i] + intercepts[j]))
                for j in range(len(lines))
            ]
            new_labels[i] = int(np.argmin(res_to_each))

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

    # ---- Final refit with refined labels ----
    lines = []
    slopes = []
    intercepts = []

    unique_labels = np.unique(labels)
    for c in unique_labels:
        idx = np.where(labels == c)[0]
        if idx.size < 2:
            continue
        lr = LinearRegression().fit(X[idx], y[idx])
        m = float(lr.coef_[0])
        b = float(lr.intercept_)

        # apply slope clamp again for safety
        if abs(m) > slope_max:
            m = np.sign(m) * slope_max * 0.99
            s_med = float(np.median(s[idx]))
            t_med = float(np.median(y[idx]))
            b = t_med - m * s_med

        slopes.append(m)
        intercepts.append(b)
        lines.append({"slope": m, "intercept": b, "points": idx.tolist()})

    slopes = np.array(slopes)
    intercepts = np.array(intercepts)

    # ==========================================================
    # Stage 3: Choose DOA slope from physically valid slopes
    # ==========================================================
    if slopes.size == 0:
        slope_final = float(lr0.coef_[0])
        intercept_final = float(lr0.intercept_)
    else:
        # keep only physically valid slopes
        valid = np.abs(slopes) <= slope_max * 1.01
        if np.any(valid):
            slopes_valid = slopes[valid]
            median_slope = float(np.median(slopes_valid))
            # index among all lines that is closest to this median
            best_idx = int(np.argmin(np.abs(slopes - median_slope)))
        else:
            median_slope = float(np.median(slopes))
            best_idx = int(np.argmin(np.abs(slopes - median_slope)))

        slope_final = float(slopes[best_idx])
        idx_pts = np.array(lines[best_idx]["points"], int)
        intercept_final = float(np.mean(y[idx_pts] - slope_final * s[idx_pts]))

    class SimpleModel:
        def __init__(self, a, b):
            self.coef_ = np.array([a])
            self.intercept_ = b
        def predict(self, X_in):
            X_in = np.asarray(X_in).reshape(-1, 1)
            return X_in[:, 0] * self.coef_[0] + self.intercept_

    model_final = SimpleModel(slope_final, intercept_final)
    residuals_final = y - model_final.predict(X)

    extra = {
        "n_lines": len(lines),
        "lines": lines,
        "s": s.tolist(),
        "axis": axis.tolist(),
    }

    return slope_final, residuals_final, model_final, extra











def main() -> None:
    print("starting DOA computation...")

    parser = argparse.ArgumentParser(description="compute and visualize DOA from packet arrivals")
    parser.add_argument("--packet", type=int, default=0, help="Packet index to process")
    parser.add_argument("--start-channel", type=int, default=100, help="Start channel  of subarray")
    parser.add_argument("--array-length", type=int, default=25, help="Length of the subarray")
    args = parser.parse_args()

    from dasprocessor.plot.source_track import load_source_points_for_run, _tx_datetimes_for_run, _interp_positions_at_times
    from pathlib import Path
    import matplotlib.pyplot as plt
    


    fs = 25000  # Sampling frequency
    c = 1475.0  # m/s in water


    # 1. Load geodetic positions
    geodetic_channel_positions = compute_channel_positions(
        r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\cable-layout.json",
        channel_count=1200,
        channel_distance=1.02
    )

    # 2. Load arrivals for all packets and all channels
    peaks_file = r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\B_4\peaks-reordered_all_hilbert_channels.json"

    with open(
        peaks_file,
        'r'
    ) as file:
        packet_arrivals = json.load(file)



    packet_idx = args.packet
    all_channel_arrivals = packet_arrivals[str(args.packet)]

    # Build dict with int keys
    arrivals_dict = {
        ch: all_channel_arrivals[str(ch)]
        for ch in range(args.start_channel, args.start_channel + args.array_length)
        if str(ch) in all_channel_arrivals
    }

    # Channels present in THIS packet (already ints)
    channels = sorted(arrivals_dict.keys())

    # Use int keys here
    times_packet = np.array([arrivals_dict[ch] for ch in channels])

    
    # 2. Convert geodetics → ENU
    ref = geodetic_channel_positions.get(0)
    ref[2] = 0.0  # set ref altitude to zero, ocean level, the 0 channel is at 10 meters depth, mostly for plotting

    print(f"Reference position for ENU conversion: lat={ref[0]}, lon={ref[1]}, alt={ref[2]}")


    channel_positions_enu = []
    for ch in channels:
        channel_positions_enu.append(
            geodetic2enu(
                geodetic_channel_positions[ch][0],
                geodetic_channel_positions[ch][1],
                geodetic_channel_positions[ch][2],
                ref[0],
                ref[1],
                ref[2]
            )
        )
    channel_positions_enu = np.array(channel_positions_enu, dtype=float)

    # 3. Depth-correct arrival times, only for validation purposes
    # times_corr_sec = []
    # for ch, t in zip(channels, times_packet):
    #     t_corr = depth_correct_timestamp(packet_idx, ch, t, fs)
    #     times_corr_sec.append(t_corr)

    # times_corr_sec = np.array(times_corr_sec)


    
    times_sec = times_packet / fs
    

    # 4. Fit DOA
    slope, residuals, model, axis = fit_doa(times_sec, channel_positions_enu)

    max_extra_dist_1 = 10.0  # m
    tau_1 = max_extra_dist_1 / c  # ~6.8e-3 s
    mask_1 = np.abs(residuals) < tau_1

    channels_inlier = [ch for ch, keep in zip(channels, mask_1) if keep]
    positions_inlier = channel_positions_enu[mask_1]
    times_inlier = times_sec[mask_1]

    # Re-fit DOA using only inliers
    slope_refit, residuals_refit, model_refit, axis_refit = fit_doa(times_inlier, positions_inlier)

    

    # --- second-pass masking based on refit residuals ---

    max_extra_dist_2 = 5.0  # m
    tau_2 = max_extra_dist_2 / c  # ~3.4e-3 s
    mask_refit = np.abs(residuals_refit) < tau_2

    channels_inlier2   = [ch for ch, keep in zip(channels_inlier, mask_refit) if keep]
    positions_inlier2  = positions_inlier[mask_refit]
    times_inlier2      = times_inlier[mask_refit]
    residuals_refit2   = residuals_refit[mask_refit]


    # Optionally refit a third time on the doubly-cleaned data
    slope_final, residuals_final, model_final, axis_final = fit_doa(times_inlier2, positions_inlier2)

    print(f"Final slope after cleaning: {slope_final:.3e} s/m")


    cone_angle= cone_radian_from_slope(slope_final, speed=c)

    if np.degrees(cone_angle) <= 90:
        angle_off_axis = np.degrees(cone_angle)
        side = "+axis (first to last)"
    else:
        angle_off_axis = 180.0 - np.degrees(cone_angle)
        side = "-axis (last to first)"

    print(f"Angle between propagation and array axis: {np.degrees(cone_angle):.1f}°")
    print(f"Wave is {angle_off_axis:.1f}° off the axis, arriving from the {side} end.")


    # --- Compute ellipse at source depth ---
    source_depth = -30.0  # for example, ENU z = -50 m

    uncertainty_deg = 5.0  # e.g., 5 degrees uncertainty

    inner_points, nominal_points, outer_points  = cone_plane_intersection(
        positions_inlier2,
        slope=slope_final,
        source_depth=source_depth,
        speed=1475.0,
        n_samples=300,   # finer sampling if you want
        array_axis=axis_final,
        angle_uncertainty_deg=uncertainty_deg,  # e.g., 5 degrees uncertainty
        xy_window=(0, 700, -500, 500),
    )


    #dump the points into a json file

    savepath = Path(__file__).resolve().parent / f"../resources/subarray_ellipses/ellipse_points_start_ch_{args.start_channel}_pkt_{args.packet}_arrlength_{args.array_length}_unc_{uncertainty_deg}.json"
    savepath = savepath.resolve()
    os.makedirs(savepath.parent, exist_ok=True)

    ellipse_dict = {
        "inner_points": inner_points.tolist(),
        "nominal_points": nominal_points.tolist(),
        "outer_points": outer_points.tolist(),
    }
    
    with open(savepath, "w") as f:
        json.dump(ellipse_dict, f)

    print(f"Saved ellipse points to {savepath}")

    

















if __name__ == "__main__":
    main()  
