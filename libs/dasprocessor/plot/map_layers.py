# libs/dasprocessor/plot/map_layers.py
from __future__ import annotations

from pathlib import Path
import json
import math
import numpy as np
import folium

from typing import Dict, Any, List, Tuple


from dasprocessor.constants import get_run


# from dasprocessor.bearing_tools import (
#     build_subarrays,
#     subarray_centers_and_headings,
# )


from dasprocessor.channel_gps import compute_channel_positions

from dasprocessor.channel_segment_heading import channel_heading_and_center



DATE_STR = "2024-05-03"


def endpoints_centered(lat: float, lon: float, bearing_deg: float, length_m: float):
    """Return (lat1, lon1), (lat2, lon2) for a short segment centered on (lat, lon)."""
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


# ---------------------------------------------------------------------------
# CABLE
# ---------------------------------------------------------------------------

def add_cable_layout_layer(
    m: folium.Map,
    geojson_path: Path,
    name: str = "Cable layout (raw)",
    color: str = "#111111",
    show: bool = True,
    marker_every: int = 0,
) -> folium.FeatureGroup:
    """Add the raw cable layout from GeoJSON."""
    layer = folium.FeatureGroup(name=name, show=show)
    with geojson_path.open("r", encoding="utf-8") as fh:
        gj = json.load(fh)

    folium.GeoJson(
        gj,
        name=name,
        style_function=lambda _: {"color": color, "weight": 3, "opacity": 0.8},
        show=show,
    ).add_to(layer)

    # Optional sparse vertex markers
    if marker_every > 0:
        coords = []
        for feat in gj.get("features", []):
            geom = feat.get("geometry", {})
            gtype = geom.get("type", "")
            seqs = geom.get("coordinates", [])
            if gtype == "MultiLineString":
                for seg in seqs:
                    coords += seg
            elif gtype == "LineString":
                coords += seqs
        for i, pt in enumerate(coords[::marker_every]):
            lon, lat = float(pt[0]), float(pt[1])
            alt = float(pt[2]) if len(pt) > 2 else 0.0
            folium.CircleMarker(
                [lat, lon],
                radius=3,
                color=color,
                fill=True,
                fill_opacity=0.9,
                tooltip=f"Vertex {i*marker_every}, alt={alt:.1f} m",
            ).add_to(layer)

    layer.add_to(m)
    return layer


# ---------------------------------------------------------------------------
# CHANNEL POSITIONS
# ---------------------------------------------------------------------------

def add_channel_positions_layer(
    m: folium.Map,
    gps: dict,
    name: str = "Channel GPS (interp)",
    color: str = "#15b9e2",
    show: bool = True,
    draw_every: int = 5,
    label_every: int = 0,
) -> folium.FeatureGroup:
    """Plot all channel GPS positions as a polyline + optional thinned markers."""
    layer = folium.FeatureGroup(name=name, show=show)
    if not gps:
        layer.add_to(m)
        return layer

    items = sorted(gps.items())  # List[Tuple[int, List[float]]]
    coords = [(float(v[0]), float(v[1])) for _, v in items]
    folium.PolyLine(coords, color=color, weight=2, opacity=0.8).add_to(layer)

    for pos in range(0, len(items), draw_every):
        ch, vals = items[pos]
        lat = float(vals[0])
        lon = float(vals[1])
        alt = float(vals[2]) if len(vals) > 2 else float("nan")

        folium.CircleMarker(
            [lat, lon],
            radius=2,
            color=color,
            fill=True,
            fill_opacity=0.9,
            tooltip=f"Ch {ch} (alt={alt:.1f} m)" if np.isfinite(alt) else f"Ch {ch}",
        ).add_to(layer)

        if label_every and (pos % label_every == 0):
            folium.map.Marker(
                [lat, lon],
                icon=folium.DivIcon(
                    icon_size=(40, 12),
                    icon_anchor=(0, 0),
                    html=f'<div style="font-size:10px; color:{color}; font-weight:bold;">{ch}</div>',
                ),
            ).add_to(layer)

    layer.add_to(m)
    return layer


# ---------------------------------------------------------------------------
# SUBARRAY CENTERS + HEADINGS
# ---------------------------------------------------------------------------



# def add_subarray_centers_layer(
#     m: folium.Map,
#     centers: list[int],
#     aperture_len: int,
#     run_number: int,
#     name: str = "Subarray centers + headings",
#     color: str = "#1f77b4",
#     show: bool = True,
# ) -> folium.FeatureGroup:
#     """Add markers + heading lines for all subarrays."""
#     layer = folium.FeatureGroup(name=name, show=show)
#     subarrays = build_subarrays(centers, aperture_len, run_number)
#     cable_path = Path(r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\cable-layout.json")
#     gps = compute_channel_positions(cable_path, channel_count=1200, channel_distance=1.02)
#     meta = {}
#     for center_ch, ch_list in subarrays.items():
#         if not ch_list:
#             # skip empty lists (or raise if that's unexpected)
#             continue

#         first_ch = ch_list[0]
#         last_ch  = ch_list[-1]

#         heading_deg, (lat_center, lon_center, _alt_center) = channel_heading_and_center(
#             first_ch, last_ch, gps
#         )

#         meta[center_ch] = {
#             "center_lat": float(lat_center),
#             "center_lon": float(lon_center),
#             "heading_deg": float(heading_deg),
#         }
#     run = get_run(DATE_STR, run_number)
#     aperture_length_m = (aperture_len - 1) * float(run["channel_distance"])

#     for c in centers:
#         info = meta.get(c)
#         if not info:
#             continue
#         lat, lon, hdg = float(info["center_lat"]), float(info["center_lon"]), float(info["heading_deg"])
#         folium.CircleMarker(
#             [lat, lon],
#             radius=4,
#             color=color,
#             fill=True,
#             fill_opacity=0.9,
#             popup=f"center={c}, heading={hdg:.2f}°, length={aperture_length_m:.1f} m",
#         ).add_to(layer)
#         if math.isfinite(hdg):
#             (lat1, lon1), (lat2, lon2) = endpoints_centered(lat, lon, hdg, aperture_length_m)
#             folium.PolyLine([[lat1, lon1], [lat2, lon2]], color=color, weight=4, opacity=0.9).add_to(layer)

#     layer.add_to(m)
#     return layer


# ---------------------------------------------------------------------------
# BOAT TRACK + TX POINTS
# ---------------------------------------------------------------------------

from dasprocessor.plot.source_track import (
    load_source_points_for_run,
    build_source_track_layer,
    build_transmission_points_layer,
)
# These are already great and return FeatureGroups


__all__ = [
    "add_cable_layout_layer",
    "add_channel_positions_layer",
    #"add_subarray_centers_layer",
    "build_source_track_layer",
    "build_transmission_points_layer",
]


# ---------------------------------------------------------------------------
# DOA RAYS
# ---------------------------------------------------------------------------




import folium
import numpy as np
from typing import Iterable, Optional, Union



from pymap3d import enu2geodetic, geodetic2enu


def _get_field(obj, name, default=None):
    """Helper: support both dicts and dataclass-like objects."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)

def build_doa_layer_from_results(
    fmap: folium.Map,
    doa_results: Iterable,
    name: str = "DOA estimates",
    packet_filter: Optional[Union[int, Iterable[int]]] = None,
    ray_length_m: float = 300.0,
    color_A: str = "#fefefe",  # blue-ish
    color_B: str = "#d70d0d",  # green-ish
) -> folium.FeatureGroup:
    """
    Build a folium layer with DOA rays for one, many, or all packets.

    Parameters
    ----------
    fmap : folium.Map
        Existing folium map to attach CRS / bounds context.
        (Not strictly required for geometry, but kept for symmetry with other helpers.)
    doa_results : Iterable
        List of DOA result dicts or DoaResult dataclass instances.
        Each entry must contain at least:
          - packet (int)
          - center_lat (float)
          - center_lon (float)
          - dir_A_enu (Sequence[float])  # [E, N, U]
          - dir_B_enu (Sequence[float])
          - channels_min (int)
          - channels_max (int)
          - n_channels (int)
    name : str
        Layer name for the folium LayerControl.
    packet_filter : None | int | Iterable[int]
        - None  -> include all packets
        - int   -> include only that packet
        - iterable of ints -> include only packets in that collection
    ray_length_m : float
        Length of the DOA rays in meters (projected from center in ENU).
    color_A, color_B : str
        Colors for DOA A and DOA B rays.

    Returns
    -------
    folium.FeatureGroup
        The feature group containing all DOA markers/lines.
    """
    fg = folium.FeatureGroup(name=name, show=True)

    # Normalize packet_filter
    if packet_filter is None:
        def use_packet(_): return True
    elif isinstance(packet_filter, int):
        def use_packet(p): return p == packet_filter
    else:
        packet_set = set(packet_filter)
        def use_packet(p): return p in packet_set

    for res in doa_results:
        packet = _get_field(res, "packet")
        if packet is None or not use_packet(packet):
            continue

        center_lat = _get_field(res, "center_lat")
        center_lon = _get_field(res, "center_lon")
        dir_A = np.asarray(_get_field(res, "dir_A_enu"), dtype=float)
        dir_B = np.asarray(_get_field(res, "dir_B_enu"), dtype=float)

        channels_min = _get_field(res, "channels_min")
        channels_max = _get_field(res, "channels_max")
        n_channels = _get_field(res, "n_channels")

        if center_lat is None or center_lon is None:
            # Skip malformed entries
            continue

        # ENU offsets for endpoints (center is at ENU origin here)
        end_A_enu = ray_length_m * dir_A
        end_B_enu = ray_length_m * dir_B

        # Convert ENU back to geodetic (lat, lon, alt). Assume alt = 0 at center.
        lat_A, lon_A, _ = enu2geodetic(
            end_A_enu[0], end_A_enu[1], 0,
            center_lat, center_lon, 0.0
        )
        lat_B, lon_B, _ = enu2geodetic(
            end_B_enu[0], end_B_enu[1], 0,
            center_lat, center_lon, 0.0
        )

        # Popup text with packet and channel info
        common_info = (
            f"Packet: {packet}<br>"
            f"Channels: {channels_min}–{channels_max} (n={n_channels})"
        )

        # --- Ray A ---
        folium.PolyLine(
            locations=[(center_lat, center_lon), (lat_A, lon_A)],
            color=color_A,
            weight=3,
            opacity=0.9,
            tooltip=f"DOA A (packet {packet})",
            popup=folium.Popup(common_info + "<br>Ray: A", max_width=250),
        ).add_to(fg)

        # --- Ray B ---
        folium.PolyLine(
            locations=[(center_lat, center_lon), (lat_B, lon_B)],
            color=color_B,
            weight=3,
            opacity=0.9,
            tooltip=f"DOA B (packet {packet})",
            popup=folium.Popup(common_info + "<br>Ray: B", max_width=250),
        ).add_to(fg)

        # Optional: small marker at the center
        folium.CircleMarker(
            location=(center_lat, center_lon),
            radius=3,
            color="#000000",
            fill=True,
            fill_opacity=0.8,
            tooltip=f"DOA center (packet {packet})",
        ).add_to(fg)

    return fg

