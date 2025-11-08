from pathlib import Path
import json
import math
import folium

from dasprocessor.bearing_tools import (
    build_subarrays,
    get_cached_channel_gps_for_run,
    subarray_centers_and_headings,
)

from dasprocessor.constants import get_run


def main():
    run_number = 2
    centers = [112, 208, 268, 552]
    aperture_len = 9

    subarrays = build_subarrays(centers, aperture_len, run_number)
    gps = get_cached_channel_gps_for_run(run_number)
    meta = subarray_centers_and_headings(subarrays, gps)

    # center the map on the mean of centers
    lat0 = sum(m["center_lat"] for m in meta.values()) / len(meta)
    lon0 = sum(m["center_lon"] for m in meta.values()) / len(meta)

    m = folium.Map(location=[lat0, lon0], zoom_start=15, tiles="OpenStreetMap")

        # helper: draw a short arrow along heading
    def endpoints_centered(lat, lon, bearing_deg, length_m):
        """Return (lat1, lon1), (lat2, lon2) for a segment of given length,
        centered at (lat, lon), oriented by bearing."""
        R = 6378000.0
        th = math.radians(bearing_deg)
        # unit direction in ENU
        ux, uy = math.sin(th), math.cos(th)
        half = 0.5 * length_m
        # ENU offsets
        dx1, dy1 = -half * ux, -half * uy
        dx2, dy2 =  half * ux,  half * uy
        # ENU -> lat/lon (small-angle)
        lat1 = lat + (dy1 / R) * (180.0 / math.pi)
        lon1 = lon + (dx1 / (R * math.cos(math.radians(lat)))) * (180.0 / math.pi)
        lat2 = lat + (dy2 / R) * (180.0 / math.pi)
        lon2 = lon + (dx2 / (R * math.cos(math.radians(lat)))) * (180.0 / math.pi)
        return (lat1, lon1), (lat2, lon2)
    
        # True physical aperture extent = (aperture_len - 1) * channel_distance
    run = get_run("2024-05-03", run_number)
    aperture_length_m = (aperture_len - 1) * float(run["channel_distance"])



    for c in centers:
        lat = float(meta[c]["center_lat"])
        lon = float(meta[c]["center_lon"])
        hdg = float(meta[c]["heading_deg"])

        # point
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color="#1f77b4",
            fill=True,
            fill_opacity=0.9,
            popup=f"center={c}, heading={hdg:.2f}°, length={aperture_length_m:.1f} m",
        ).add_to(m)

        # centered segment with physical aperture length
        (lat1, lon1), (lat2, lon2) = endpoints_centered(lat, lon, hdg, aperture_length_m)
        folium.PolyLine([[lat1, lon1], [lat2, lon2]], weight=4, opacity=0.9).add_to(m)

    out_path = Path(__file__).with_name("subarray_centers_run2.html")
    m.save(str(out_path))
    print(f"Map saved to: {out_path.resolve()}")

if __name__ == "__main__":
    main()



