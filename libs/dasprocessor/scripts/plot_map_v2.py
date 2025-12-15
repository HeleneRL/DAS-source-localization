# libs/dasprocessor/scripts/plot_map_v2.py
from __future__ import annotations

import json
from pathlib import Path
import argparse
from typing import List

import folium
import numpy as np  # needed by center_for_gps_or_run

from dasprocessor.plot.map_layers_v2 import (
    add_cable_layout_layer,
    add_channel_positions_layer,
    add_subarray_layer,
    add_track_points_layer,
    add_ellipse_layer,
    enu_reference_from_channels,
    build_source_track_layer,
    build_transmission_points_layer,
    build_doa_layer_from_results,
)

from dasprocessor.plot.source_track import load_source_points_for_run
from dasprocessor.channel_gps import compute_channel_positions
from dasprocessor.constants import get_run, get_trial_day_metadata


# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------

RUN_NUMBER = 2

# Cable / array config
CHANNEL_COUNT = 1200
CHANNEL_SPACING_M = 1.02

# Subarray config (channels, not meters)
#SUBARRAY_START_CHANNELS = [100, 130, 245, 265]
SUBARRAY_START_CHANNELS = [100, 120, 245, 265]
SUBARRAY_ARRAY_LENGTH = 30  # number of channels in each subarray

# DOA & ellipse viz config
PACKET_TO_SHOW_ELLIPSES =  60   # None → don't show ellipses 
PACKET_TO_SHOW_TRACK = None   # None → full track
PACKET_FILTER_FOR_DOA = 50   # None → all packets in DOA JSON

ANGLE_UNCERTAINTY_DEG = 5.0  # just for layer naming / info

# Optional: write channel positions geo JSON for debugging/other tools
WRITE_CHANNEL_POS_GEO = True

#which channel is the first in the water (0-based index)
N_SKIP = 23          # 0..22 are on land / not in water
channel_distance = 1.02
CHANNEL_OFFSET = -N_SKIP * channel_distance  # m

MAP_CENTER_LAT = 63.4406
MAP_CENTER_LON = 10.3518

MAP_OUTPUT_DIR = Path(
    r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\new_cable_map_plots"
)


# --------------------------------------------------------------------
# PATH HELPERS
# --------------------------------------------------------------------

def get_paths() -> dict[str, Path]:
    """
    Return a dictionary of all important paths, built relative to the repo.
    Assumes this file lives at .../my-project/libs/dasprocessor/scripts/.
    """
    script_dir = Path(__file__).resolve().parent          # .../libs/dasprocessor/scripts
    libs_dir = script_dir.parent.parent                   # .../libs
    resources_root = libs_dir / "resources"               # .../libs/resources

    paths = {
        "resources_root": resources_root,
        "cable_layout": resources_root / "suspected_cable-layout.json",
        "source_csv": resources_root / "source-position.csv",
        "b4_dir": resources_root / "B_4",
        "subarray_ellipses": resources_root / "new_cable_subarray_ellipses",
        "subarray_info": resources_root / "new_cable_subarray_info",
        "track_estimate": resources_root / "new_cable_track_estimate_from_doa.json",
    }
    return paths




# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------

def main() -> None:
    print("Starting DOA / map visualization...")

        # ------------------------------------------------------------------
    # CLI OVERRIDES
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Plot DOA / track / ellipses on map.")
    parser.add_argument(
        "--packet",
        type=int,
        default=None,
        help="Packet index to use for *all* views (ellipses, track, DOA).",
    )
    parser.add_argument(
        "--packet-ellipses",
        type=int,
        default=None,
        help="Packet for ellipse bands (overrides --packet).",
    )
    parser.add_argument(
        "--packet-track",
        type=int,
        default=None,
        help="Packet for DOA track points (overrides --packet).",
    )
    parser.add_argument(
        "--packet-doa",
        type=int,
        default=None,
        help="Packet for DOA rays (overrides --packet).",
    )
    args = parser.parse_args()

    # Resolve effective packet numbers
    pkt_ellipses = (
        args.packet_ellipses
        if args.packet_ellipses is not None
        else (args.packet if args.packet is not None else PACKET_TO_SHOW_ELLIPSES)
    )
    pkt_track = (
        args.packet_track
        if args.packet_track is not None
        else (args.packet if args.packet is not None else PACKET_TO_SHOW_TRACK)
    )
    pkt_doa = (
        args.packet_doa
        if args.packet_doa is not None
        else (args.packet if args.packet is not None else PACKET_FILTER_FOR_DOA)
    )

    print(f"Using packets: ellipses={pkt_ellipses}, track={pkt_track}, doa={pkt_doa}")



    paths = get_paths()
    cable_layout_path = paths["cable_layout"]
    source_csv_path = paths["source_csv"]
    b4_dir = paths["b4_dir"]
    subarray_ellipses_dir = paths["subarray_ellipses"]
    track_estimate_path = paths["track_estimate"]

    # 1) Compute channel positions in geodetic and (optionally) save them
    channel_pos_all = compute_channel_positions(
        cable_layout_path,
        channel_count=CHANNEL_COUNT-N_SKIP,
        channel_distance=CHANNEL_SPACING_M
    )

    from typing import Dict, List
    channel_pos_geo: Dict[int, List[float]] = {}


    # place 0–23 at first coordinate (same as wet_positions[0])
    first_lat, first_lon, first_alt = channel_pos_all[0]
    for ch in range(N_SKIP + 1):     # 0..23 inclusive
        channel_pos_geo[ch] = [first_lat, first_lon, first_alt]

    # fill the rest using wet_positions shifted by 23
    for ch in range(N_SKIP + 1, CHANNEL_COUNT-N_SKIP):
        channel_pos_geo[ch] = channel_pos_all[ch - (N_SKIP + 1)]


    
    # channel_position_path = r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\B_4\channel_pos_geo_adjusted.json"

    # with open(channel_position_path, "r", encoding="utf-8") as f:
    #     channel_pos_geo_raw = json.load(f)

    # # Convert keys to integers
    # channel_pos_geo = {int(ch): pos for ch, pos in channel_pos_geo_raw.items()}

    # Now valid:
    enu_ref = enu_reference_from_channels(channel_pos_geo, channel_idx=0)



    if WRITE_CHANNEL_POS_GEO:
        out_geo = b4_dir / "channel_pos_geo.json"
        out_geojson = b4_dir / "channel_pos_coords.json"
        out_geo.parent.mkdir(parents=True, exist_ok=True)
        with out_geo.open("w", encoding="utf-8") as f:
            json.dump(channel_pos_geo, f, indent=2)
        print(f"Channel positions JSON saved to {out_geo}")
         # 11) GeoJSON MultiLineString for geojson.io
        # Coordinates: [lon, lat, depth]
        coords = []
        for ch in range(N_SKIP, CHANNEL_COUNT-N_SKIP):
            lat, lon, depth = channel_pos_geo[ch]
            coords.append([lon, lat, depth])  # depth negative in GeoJSON  

        geojson_fc = {
            "type": "FeatureCollection",
            "name": "TBS_Fjordlab_Tether_Reoriented",
            "crs": {
                "type": "name",
                "properties": {
                    "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
                }
            },
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "fid_1": 1.0,
                        "Label": "SDP Tether",
                        "Location": "On Seabed",
                    },
                    "geometry": {
                        "type": "MultiLineString",
                        "coordinates": [
                            coords  # single polyline
                        ],
                    },
                }
            ],
        }

        with open(out_geojson, "w", encoding="utf-8") as f:
            json.dump(geojson_fc, f, indent=2)

  

    # 3) Base map
    m = folium.Map(location=[MAP_CENTER_LAT, MAP_CENTER_LON], zoom_start=16, tiles="OpenStreetMap")

    # 4) Cable + channel layers
    add_cable_layout_layer(
        m,
        cable_layout_path,
        name="Cable layout",
        color="#07035E",
        marker_every=25,
    )
    add_channel_positions_layer(
        m,
        channel_pos_geo,
        name="Channels",
        color="#0A047D",
        draw_every=5,
    )

    # 5) Boat track and TX positions (from GPS / source log)
    # source_points = load_source_points_for_run(source_csv_path, RUN_NUMBER)
    # build_source_track_layer(
    #     source_points,
    #     name=f"Boat track (run {RUN_NUMBER})",
    # ).add_to(m)

    build_transmission_points_layer(
        source_csv_path,
        RUN_NUMBER,
        label_every=10,
        name="TX positions",
    ).add_to(m)

    # 6) Ellipse uncertainty bands for a single packet (optional)
    if pkt_ellipses is not None:
        pkt_key = str(pkt_ellipses)
        for start_ch in SUBARRAY_START_CHANNELS:
            ellipse_path = (
                subarray_ellipses_dir
                / f"ellipse_bands_start_ch_{start_ch}_arrlen_{SUBARRAY_ARRAY_LENGTH}.json"
            )
            if not ellipse_path.exists():
                print(f"[WARN] Ellipse file not found: {ellipse_path}")
                continue

            with ellipse_path.open("r", encoding="utf-8") as f:
                ellipse_data = json.load(f)

            if pkt_key not in ellipse_data:
                print(f"[WARN] Packet {pkt_key} not in {ellipse_path.name}")
                continue

            ellipse_entry = ellipse_data[pkt_key]  # dict with inner/nominal/outer


            add_ellipse_layer(m, ellipse_entry, enu_ref=enu_ref,name=f"Ellipse start_ch={start_ch}, pkt={pkt_key}, unc={ANGLE_UNCERTAINTY_DEG}°")




    # 7) Estimated track points from DOA
    if track_estimate_path.exists():
        with track_estimate_path.open("r", encoding="utf-8") as f:
            track_dict = json.load(f)

        add_track_points_layer(m, track_dict, enu_ref=enu_ref, name="Estimated track", color="#33FF57", packet=pkt_track)

    else:
        print(f"[INFO] No track estimate file found at {track_estimate_path}")


    # 8) Subarray visualization

    add_subarray_layer(
        m,
        start_channels=SUBARRAY_START_CHANNELS,
        array_length=SUBARRAY_ARRAY_LENGTH,
        channels_gps=channel_pos_geo

    )




    # # 9) DOA results layer(s)
    # doa_results_paths = [
    #     b4_dir / "DOA_results-100-131.json",
    #     # add more here if you want
    # ]

    # for doa_path in doa_results_paths:
    #     print(f"Processing array DOA results from {doa_path}")
    #     if doa_path.exists():
    #         with doa_path.open("r", encoding="utf-8") as f:
    #             doa_results_small = json.load(f)

    #         # Example: packet_filter takes an int or None
    #         build_doa_layer_from_results(
    #             m,
    #             doa_results_small,
    #             name=f"DOA {doa_path.stem}",   # e.g. "DOA DOA_results-100-131"
    #             packet_filter=PACKET_FILTER_FOR_DOA,
    #         ).add_to(m)
    #     else:
    #         print(f"[WARN] No DOA results file found at {doa_path}")

    # 10) Finalize
    folium.LayerControl().add_to(m)

   

    # -------------------------------------------------------------
    # OUTPUT MAP FILE
    # -------------------------------------------------------------
    # A filename that reflects subarrays, array length, and packet.
    start_ch_str = "-".join(str(ch) for ch in SUBARRAY_START_CHANNELS)

    pkt_str = f"pkt_{args.packet}" if args.packet is not None else "pkt_all"

    filename = (
        f"map_startch_{start_ch_str}"
        f"_arrlen_{SUBARRAY_ARRAY_LENGTH}"
        f"_{pkt_str}.html"
    )

    out = MAP_OUTPUT_DIR / filename
    m.save(str(out))

    print(f"\nMap saved to:\n  {out.resolve()}\n")



if __name__ == "__main__":
    main()
