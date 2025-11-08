# libs/dasprocessor/scripts/plot_everything_map.py
from __future__ import annotations

from pathlib import Path
import folium
import numpy as np  # needed by center_for_gps_or_run

from dasprocessor.plot.map_layers import (
    add_cable_layout_layer,
    add_channel_positions_layer,
    add_subarray_centers_layer,
    build_source_track_layer,
    build_transmission_points_layer,
)
# IMPORTANT: load the boat track points (lat, lon, dt) from here:
from dasprocessor.plot.source_track import load_source_points_for_run

from dasprocessor.bearing_tools import get_cached_channel_gps_for_run
from dasprocessor.constants import get_run


def center_for_gps_or_run(gps: np.ndarray, run_number: int, date_str="2024-05-03"):
    """Return map center (lat, lon) from GPS array or run metadata."""
    if gps.size > 0:
        mid = gps[gps.shape[0] // 2]
        return float(mid[0]), float(mid[1])
    run = get_run(date_str, run_number)
    return float(run.get("map_center_lat", 0.0)), float(run.get("map_center_lon", 0.0))


def main():
    run_number = 2
    geojson_path = Path(r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\cable-layout.json")
    csv_path = Path(r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\source-position.csv")
    centers = [119, 122, 125, 128, 203, 206, 209, 212, 263, 266, 269, 272, 347, 350, 353, 356]
    aperture_len = 15

    gps = get_cached_channel_gps_for_run(run_number)
    lat0, lon0 = center_for_gps_or_run(gps, run_number)

    m = folium.Map(location=[lat0, lon0], zoom_start=15, tiles="OpenStreetMap")

    # Cable, channels, subarrays
    add_cable_layout_layer(m, geojson_path, name="Cable layout", color="#111111", marker_every=25)
    add_channel_positions_layer(m, gps, name="Channels", color="#cc3300", draw_every=5)
    add_subarray_centers_layer(m, centers, aperture_len, run_number, name="Subarrays", color="#1f77b4")

    # ✅ Boat track (expects 3-tuples: lat, lon, dt)
    points = load_source_points_for_run(csv_path, run_number)
    build_source_track_layer(points, name=f"Boat track (run {run_number})").add_to(m)

    # ✅ TX positions
    build_transmission_points_layer(csv_path, run_number, label_every=10, name="TX positions").add_to(m)

    folium.LayerControl().add_to(m)
    out = Path(__file__).with_name(f"run{run_number}_map_overview.html")
    m.save(str(out))
    print(f"Map saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
