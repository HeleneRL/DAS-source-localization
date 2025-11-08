# libs/dasprocessor/scripts/plot_packet_bearings.py
from pathlib import Path
import json
import math
import folium

from dasprocessor.bearing_tools import (
    load_merged_arrivals,
    build_subarrays,
    get_cached_channel_gps_for_run,
    subarray_centers_and_headings,
    estimate_bearings_for_packets,
)
from dasprocessor.constants import get_run

from dasprocessor.scripts.plot_run_source_track import load_source_points_for_run, build_source_track_layer, build_transmission_points_layer

csv_path = Path(r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\source-position.csv")
run_number = 2


SPEED_OF_SOUND = 1500.0  # m/s
G_MAX = 1.0 / SPEED_OF_SOUND  # s/m, physical max slope (|dt/ds| <= 1/c)

def endpoints_centered(lat, lon, bearing_deg, length_m):
    """Return (lat1, lon1), (lat2, lon2) for a segment of given length,
    centered at (lat, lon), oriented by bearing."""
    R = 6_378_000.0
    th = math.radians(bearing_deg)
    ux, uy = math.sin(th), math.cos(th)
    half = 0.5 * length_m
    dx1, dy1 = -half * ux, -half * uy
    dx2, dy2 =  half * ux,  half * uy
    lat1 = lat + (dy1 / R) * (180.0 / math.pi)
    lon1 = lon + (dx1 / (R * math.cos(math.radians(lat)))) * (180.0 / math.pi)
    lat2 = lat + (dy2 / R) * (180.0 / math.pi)
    lon2 = lon + (dx2 / (R * math.cos(math.radians(lat)))) * (180.0 / math.pi)
    return (lat1, lon1), (lat2, lon2)

def endpoint_from(lat, lon, bearing_deg, length_m):
    """End point starting at (lat,lon) along bearing for length_m."""
    R = 6_378_000.0
    th = math.radians(bearing_deg)
    dx = length_m * math.sin(th)
    dy = length_m * math.cos(th)
    lat2 = lat + (dy / R) * (180.0 / math.pi)
    lon2 = lon + (dx / (R * math.cos(math.radians(lat)))) * (180.0 / math.pi)
    return (lat2, lon2)

def main():
    # ---------- Inputs ----------
    run_number = 2
    centers = [119,122, 125, 128,203, 206, 209, 212, 263, 266, 269, 272, 347, 350, 353, 356]
    aperture_len = 15                # odd
    packet_index = 80              # packet to visualize
    ray_length_m = 400.0            # length of each bearing ray from center
    min_fraction_present = 0.2      # stricter requirement for robustness

    # Path to your merged arrivals JSON (under package resources)
    pkg_dir = Path(__file__).resolve().parent.parent.parent          # .../libs
    res_dir = (pkg_dir / "resources" / "B_4").resolve()       # adjust band if needed
    merged_json = res_dir / "peaks-merged-run2.json"
    print(f"Using merged JSON: {merged_json}")

    # ---------- Load data & geometry ----------
    arrivals = load_merged_arrivals(merged_json)
    subarrays = build_subarrays(centers, aperture_len, run_number)
    gps = get_cached_channel_gps_for_run(run_number)
    centers_meta = subarray_centers_and_headings(subarrays, gps)

    # ---------- Estimate bearings for the chosen packet ----------
    bearing_results = estimate_bearings_for_packets(
        arrivals=arrivals,
        subarrays=subarrays,
        gps_per_channel=gps,
        packet_indices=[packet_index],
        run_number=run_number,
        min_fraction_present=min_fraction_present,
        use_pca_heading=False,  # keep endpoint-based heading for now
        debug=True,
        time_gate_s=1.5,          # <-- NEW: try 1.0–2.0 s
        use_linear_spacing=True,  # <-- NEW: sanity-check against GPS projection
        speed_of_sound=1440.0 
    )

    # Post-filter & diagnostics per subarray
    qualified = {}
    failed = {}  # center -> reason string

    # For reason “insufficient detections” we check coverage directly
    def has_sufficient_detections(center):
        chans = subarrays[center]
        need = math.ceil(min_fraction_present * len(chans))
        count = 0
        for ch in chans:
            if packet_index in arrivals.get(ch, {}):
                count += 1
        return count >= need, count, need

    for c in centers:
        info = bearing_results.get(c, {}).get(packet_index, None)
        enough, have, need = has_sufficient_detections(c)
        if not enough:
            failed[c] = f"insufficient detections ({have}/{need})"
            continue
        if info is None:
            failed[c] = "fit failed / not enough inliers"
            continue
        g = float(info.get("g_s_per_m", float("nan")))
        if not math.isfinite(g) or abs(g) > 1.1 * G_MAX:  # unphysical slope
            failed[c] = f"unphysical slope |g|={abs(g):.4g} (> 1/c≈{G_MAX:.4g})"
            continue
        # good
        qualified[c] = info

    print(f"\nPacket {packet_index}: {len(qualified)} qualified, {len(failed)} failed.")
    if failed:
        for c, reason in failed.items():
            print(f"  - center {c}: {reason}")

    if not qualified:
        print("No valid subarray bearings to plot. Exiting.")
        return

    # ---------- Map ----------
    # Center map on mean of subarray centers (all, not just qualified, for context)
    lat0 = float(sum(m["center_lat"] for m in centers_meta.values()) / len(centers_meta))
    lon0 = float(sum(m["center_lon"] for m in centers_meta.values()) / len(centers_meta))
    m = folium.Map(location=[lat0, lon0], zoom_start=15, tiles="OpenStreetMap")

    # a) Plot subarray centers and centered physical aperture segments
    run = get_run("2024-05-03", run_number)
    aperture_length_m = (aperture_len - 1) * float(run["channel_distance"])

    for c in centers:
        meta = centers_meta[c]
        lat = float(meta["center_lat"])
        lon = float(meta["center_lon"])
        hdg = float(meta["heading_deg"])

        # center marker (green if qualified, gray if failed)
        color = "#2ca02c" if c in qualified else "#7f7f7f"
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color=color,
            fill=True,
            fill_opacity=0.9,
            popup=f"center={c}, heading={hdg:.2f}°, aperture={aperture_length_m:.1f} m",
        ).add_to(m)

        # physical aperture segment, centered
        (lat1, lon1), (lat2, lon2) = endpoints_centered(lat, lon, hdg, aperture_length_m)
        folium.PolyLine([[lat1, lon1], [lat2, lon2]],
                        weight=4, opacity=0.9, color=color).add_to(m)

    # b) Plot BOTH ambiguous bearing rays for each qualified subarray
    #    (each ray starts at the subarray center and goes out ray_length_m)
    for c, info in qualified.items():
        clat = float(info["center_lat"])
        clon = float(info["center_lon"])
        bmin, bplus = info["bearing_deg_pair"]  # α-θ, α+θ

        for b, col in [(bmin, "#1f77b4"), (bplus, "#d62728")]:
            lat2, lon2 = endpoint_from(clat, clon, float(b), ray_length_m)
            folium.PolyLine(
                [[clat, clon], [lat2, lon2]],
                weight=3,
                opacity=0.9,
                color=col,
                popup=f"center={c}, bearing={float(b):.2f}°"
            ).add_to(m)



    tx_layer = build_transmission_points_layer(csv_path, run_number, label_every=10,
                                           name=f"TX positions (run {run_number})")
    tx_layer.add_to(m)

    
    folium.LayerControl().add_to(m)

    out_path = Path(__file__).with_name(f"packet_{packet_index}_bearings_run{run_number}.html")
    m.save(str(out_path))
    print(f"\nMap saved to: {out_path.resolve()}")

if __name__ == "__main__":
    main()
