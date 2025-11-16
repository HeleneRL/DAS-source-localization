# libs/dasprocessor/scripts/plot_everything_map.py
from __future__ import annotations

import json
from pathlib import Path
import folium
import numpy as np  # needed by center_for_gps_or_run

from dasprocessor.plot.map_layers import (
    add_cable_layout_layer,
    add_channel_positions_layer,
    build_source_track_layer,
    build_transmission_points_layer,
    build_doa_layer_from_results,
)
# IMPORTANT: load the boat track points (lat, lon, dt) from here:
from dasprocessor.plot.source_track import load_source_points_for_run

from dasprocessor.channel_gps import compute_channel_positions
from dasprocessor.constants import get_run


def center_for_gps_or_run(gps: dict, run_number: int, date_str="2024-05-03"):
    """Return map center (lat, lon) from GPS dictionary or run metadata."""
    if gps:
        # Sort channels and pick the middle one
        keys = sorted(gps.keys())
        mid_key = keys[len(keys) // 2]
        mid_entry = gps[mid_key]
        lat, lon = mid_entry[0], mid_entry[1]  # assuming [lat, lon, alt]
        return float(lat), float(lon)
    
    # fallback to metadata
    run = get_run(date_str, run_number)
    return float(run.get("map_center_lat", 0.0)), float(run.get("map_center_lon", 0.0))


def main():
    run_number = 2
    geojson_path = Path(r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\cable-layout.json")
    csv_path = Path(r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\source-position.csv")


    path_list = [
    Path(r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\B_4\DOA_results-75-105.json"),
    Path(r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\B_4\DOA_results-100-130.json"),
    Path(r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\B_4\DOA_results-150-180.json"),
    Path(r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\B_4\DOA_results-180-210.json"),
    Path(r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\B_4\DOA_results-220-250.json"),
    Path(r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\B_4\DOA_results-245-275.json"),
    Path(r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\B_4\DOA_results-328-358.json"),
]

    channel_pos_geo = compute_channel_positions(geojson_path, channel_count=1200, channel_distance=1.02)


    filename = r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\B_4\channel_pos_geo.json"

    # Open the file in write mode ('w') and use json.dump() to write the dictionary
    with open(filename, 'w') as f:
        json.dump(channel_pos_geo, f, indent=4) # indent=4 for pretty-printing with 4 spaces

    print(f"Dictionary successfully saved to {filename}")

    lat0, lon0 = center_for_gps_or_run(channel_pos_geo, run_number)
    with open(r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\B_4\consecutive_peaks_v2.json", 'r') as file:
        consecutive_channel_dict = json.load(file)

    m = folium.Map(location=[lat0, lon0], zoom_start=15, tiles="OpenStreetMap")

    # Cable, channels, subarrays
    add_cable_layout_layer(m, geojson_path, name="Cable layout", color="#07035E", marker_every=25)
    add_channel_positions_layer(m, channel_pos_geo, name="Channels", color="#0A047D", draw_every=5)
    #add_subarray_centers_layer(m, centers, aperture_len, run_number, name="Subarrays", color="#1f77b4")
    
    # ✅ Boat track (expects 3-tuples: lat, lon, dt)
    points = load_source_points_for_run(csv_path, run_number)
    build_source_track_layer(points, name=f"Boat track (run {run_number})").add_to(m)

    # ✅ TX positions
    build_transmission_points_layer(csv_path, run_number, label_every=10, name="TX positions").add_to(m)

    packets = 50  # only show DOA for this packet (set to None for all)

    for el in path_list:
        print(f"Processing array DOA results from {el}")
        print(type(el))
        if el.exists():
            with el.open("r") as f:
                doa_results_small = json.load(f)

            # All packets:
            build_doa_layer_from_results(m, doa_results_small, name=f"DOA {el.stem[12:-1]}", packet_filter=packets).add_to(m)
        else:
            print(f"No DOA results file found at {el}")

   

    
    
    


    folium.LayerControl().add_to(m)
    out = Path(__file__).with_name(f"run{run_number}_map_overview.html")
    m.save(str(out))
    print(f"Map saved to: {out.resolve()}")


if __name__ == "__main__":
    main()



    

    