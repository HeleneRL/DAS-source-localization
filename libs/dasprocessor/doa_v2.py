import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor
import argparse
from pymap3d import enu2geodetic, geodetic2enu


from dasprocessor.debugging import visualize_doa_fit
from dasprocessor.channel_gps import compute_channel_positions
import json
from dasprocessor.doa_results_io import DoaResult, append_doa_result
import os



def fit_doa(times, positions_enu, use_ransac=True):
    """
    Fit DOA from arrival times and channel positions (ENU).
    """

    times = np.asarray(times, dtype=float)
    positions_enu = np.asarray(positions_enu, dtype=float).copy()

    # --- Use ONLY horizontal coordinates for DOA estimation ---
    positions_enu[:, 2] = 0.0

    # 1) Array axis (horizontal) from endpoints
    axis = positions_enu[-1] - positions_enu[0]
    axis /= np.linalg.norm(axis)

    # 2) Scalar coordinate along the array axis
    s = positions_enu @ axis          # shape (N,)
    X = s.reshape(-1, 1)
    y = times

    # 3) Robust or plain 1D regression: t = a * s + b
    if use_ransac:
        base = LinearRegression()
        model = RANSACRegressor(
            base,
            min_samples=4,
            max_trials=200,
        )
        model.fit(X, y)
        slope = float(model.estimator_.coef_[0])   # dt/ds
    else:
        model = LinearRegression().fit(X, y)
        slope = float(model.coef_[0])


    # Time residuals from this 1D model
    residuals = y - model.predict(X)


    return slope, residuals, model


def bearing_angle_from_slope(slope, speed=1475.0):
    axis_dot_n = speed * slope
    axis_dot_n = float(np.clip(axis_dot_n, -1.0, 1.0))
    angle_deg = np.arccos(axis_dot_n)

    return angle_deg


def temporal_smooth(times_prev, times_curr, times_next):
    """Smooth timestamps for better DOA stability."""
    return 0.33 * times_prev + 0.34 * times_curr + 0.33 * times_next


def median_packet_smooth(times_list):
    """Apply median filter across packets."""
    return np.median(np.stack(times_list, axis=0), axis=0)








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

    
    times_sec = times_packet / fs
    

    # 4. Fit DOA
    slope, residuals, model = fit_doa(times_sec, channel_positions_enu)


    max_extra_dist_1 = 10.0  # m
    tau_1 = max_extra_dist_1 / c  # ~6.8e-3 s
    mask_1 = np.abs(residuals) < tau_1

    channels_inlier = [ch for ch, keep in zip(channels, mask_1) if keep]
    positions_inlier = channel_positions_enu[mask_1]
    times_inlier = times_sec[mask_1]

    # Re-fit DOA using only inliers
    slope_refit, residuals_refit, model_refit = fit_doa(times_inlier, positions_inlier)

    

    # --- second-pass masking based on refit residuals ---

    max_extra_dist_2 = 5.0  # m
    tau_2 = max_extra_dist_2 / c  # ~3.4e-3 s
    mask_refit = np.abs(residuals_refit) < tau_2

    channels_inlier2   = [ch for ch, keep in zip(channels_inlier, mask_refit) if keep]
    positions_inlier2  = positions_inlier[mask_refit]
    times_inlier2      = times_inlier[mask_refit]
    residuals_refit2   = residuals_refit[mask_refit]


    # Optionally refit a third time on the doubly-cleaned data
    slope_final, residuals_final, model_final = fit_doa(times_inlier2, positions_inlier2)

    print(f"Final slope after cleaning: {slope_final:.3e} s/m")


    bearing_angle= bearing_angle_from_slope(slope_final, speed=c)

    print(f"Bearing (array-relative) for packet {packet_idx}: ±{np.degrees(bearing_angle):.1f} deg")

    flat_pos = positions_inlier2.copy()
    flat_pos[:, 2] = 0.0

    mid_array_axis = flat_pos[len(flat_pos)//2] - flat_pos[len(flat_pos)//2+1]
    mid_array_axis /= np.linalg.norm(mid_array_axis)

    print(f"Mid array axis: {mid_array_axis}")
    direction_1 = np.cos(bearing_angle)*mid_array_axis + np.sin(bearing_angle)*np.cross(np.array([0,0,1]), mid_array_axis)
    direction_2 = np.cos(bearing_angle)*mid_array_axis - np.sin(bearing_angle)*np.cross(np.array([0,0,1]), mid_array_axis)
    direction_source_A = direction_1/np.linalg.norm(direction_1)
    direction_source_B = direction_2/np.linalg.norm(direction_2)
 

    center_lat, center_lon, _ = enu2geodetic(flat_pos[len(flat_pos)//2][0], flat_pos[len(flat_pos)//2][1],0, ref[0], ref[1], ref[2])
 


    doa_result = DoaResult(
        packet=packet_idx,
        center_lat=center_lat,
        center_lon=center_lon,
        dir_A_enu=direction_source_A.tolist(),
        dir_B_enu=direction_source_B.tolist(),
        channels_min=int(min(channels_inlier)),
        channels_max=int(max(channels_inlier)),
        n_channels=len(channels_inlier),
    )

    savepath = Path(__file__).resolve().parent / f"../resources/B_4/DOA_results-{args.start_channel}-{args.start_channel + args.array_length}.json"
    savepath = savepath.resolve()
    os.makedirs(savepath.parent, exist_ok=True)


    append_doa_result(savepath, doa_result)
    print(f"Appended DOA result for packet {packet_idx} to {savepath}")



  
    #5. Visualize

    # axis = channel_positions_enu[-1] - channel_positions_enu[0]
    # axis /= np.linalg.norm(axis)

    # # project each channel position onto the axis
    # proj = channel_positions_enu @ axis  # 1D coordinate along-array

    # diffs = np.diff(proj)
    # print("mean spacing along local axis:", np.mean(diffs))

    # plt.figure()
    # plt.plot(proj, times_sec, 'o-')
    # plt.xlabel("Along-array coordinate (m)")
    # plt.ylabel("Arrival time (s)")
    # plt.title(f"Packet {packet_idx} arrival times vs array position before cleaning")
    # plt.show()


    # axis_1 = positions_inlier2[-1]-positions_inlier2[0]
    # axis_1 /= np.linalg.norm(axis_1)

    # proj_inlier = positions_inlier @ axis_1
    # plt.figure()
    # plt.plot(proj_inlier, times_inlier, 'o-')
    # plt.xlabel("Along-array coordinate (m)")
    # plt.ylabel("Arrival time (s)")
    # plt.title(f"Packet {packet_idx} arrival times vs array position after first cleaning")
    # plt.show()

    # axis_2 = positions_inlier2[-1]-positions_inlier2[0]
    # axis_2 /= np.linalg.norm(axis_2)

    # proj_inlier2 = positions_inlier2 @ axis_2
    # plt.figure()
    # plt.plot(proj_inlier2, times_inlier2, 'o-')
    # plt.xlabel("Along-array coordinate (m)")
    # plt.ylabel("Arrival time (s)")
    # plt.title(f"Packet {packet_idx} arrival times vs array position after second cleaning")
    # plt.show()


    
    # visualize_doa_fit(
    #     channel_positions_enu,
    #     times_sec,
    #     residuals,
    #     direction_,
    #     left_right_ambiguity(direction, channel_positions_enu),
    #     channels,
    #     model,
    #     title=f"Packet {packet_idx} first pass"
    # )

    # visualize_doa_fit(
    #     positions_inlier,
    #     times_inlier,
    #     residuals_refit,
    #     direction_refit,
    #     left_right_ambiguity(direction_refit, positions_inlier),
    #     channels_inlier,
    #     model_refit,
    #     title=f"Packet {packet_idx} second pass"
    # )

#     visualize_doa_fit(
#         positions_inlier2,
#         times_inlier2,
#         residuals_refit2,
#         direction_source_A,
#         direction_source_B,
#         channels_inlier2,
#         model_final,
#         title=f"Packet {packet_idx} final fit"
# )



if __name__ == "__main__":
    main()  
