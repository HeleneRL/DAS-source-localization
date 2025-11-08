from pathlib import Path
import json
import math

from dasprocessor.bearing_tools import (
    build_subarrays,
    get_cached_channel_gps_for_run,
    subarray_centers_and_headings,
    geojson_subarray_centers,
)

def fmt_deg(d):
    return f"{d:7.3f}" if math.isfinite(d) else "   nan"

def main():
    # --- inputs ---
    run_number = 2
    centers = [112, 208, 268, 552]
    aperture_len = 9  # must be odd

    # 1) Build subarrays
    subarrays = build_subarrays(centers, aperture_len, run_number)

    # 2) Per-channel GPS (cached)
    gps = get_cached_channel_gps_for_run(run_number)

    # 3) Center lat/lon + heading
    meta = subarray_centers_and_headings(subarrays, gps)

    # Pretty print
    print("Center  |  Lat (deg)   Lon (deg)   Heading (deg from North)")
    print("--------+---------------------------------------------------")
    for c in centers:
        m = meta[c]
        print(f"{c:6d} | {fmt_deg(m['center_lat'])}   {fmt_deg(m['center_lon'])}       {fmt_deg(m['heading_deg'])}")

    # 4) Save GeoJSON (next to this script)
    out_path = Path(__file__).with_name("subarray_centers_run2.geojson")
    geo = geojson_subarray_centers(meta)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(geo, f, indent=2)
    print(f"\nGeoJSON saved to: {out_path.resolve()}")

if __name__ == "__main__":
    main()
