# libs/dasprocessor/plot/map_layers_v2.py
from __future__ import annotations

from pathlib import Path
import json
import math
from typing import Dict, Any, List, Tuple, Iterable, Optional, Union

import folium
import numpy as np
from pymap3d import enu2geodetic  # geodetic2enu not needed here

from dasprocessor.plot.source_track import (
    load_source_points_for_run,     # re-exported via __all__
    build_source_track_layer,       # re-exported via __all__
    build_transmission_points_layer # re-exported via __all__
)


# ---------------------------------------------------------------------------
# ENU REFERENCE HELPERS
# ---------------------------------------------------------------------------

def enu_reference_from_channels(
    channels_gps: dict,
    channel_idx: int = 0,
) -> tuple[float, float, float]:
    """
    Derive ENU reference (lat, lon, alt) from a channel_gps dictionary.

    By default, uses channel 0 and sets altitude to 0.0, which should match
    the reference used when computing ENU coordinates elsewhere:

        ref = geodetic_channel_positions[0]; ref[2] = 0.0

    channels_gps is expected to map channel index -> [lat, lon, alt].
    """
    if channel_idx not in channels_gps:
        raise KeyError(f"Channel {channel_idx} not found in channels_gps")

    lat, lon, alt = channels_gps[channel_idx]
    return float(lat), float(lon), 0.0  # force ENU alt reference to 0.0


# ---------------------------------------------------------------------------
# SMALL GEO UTILS
# ---------------------------------------------------------------------------

def endpoints_centered(lat: float, lon: float, bearing_deg: float, length_m: float):
    """
    Return (lat1, lon1), (lat2, lon2) for a short segment centered on (lat, lon).

    Not currently used in this module, but can be handy for visualising headings.
    """
    R = 6_378_000.0
    th = math.radians(bearing_deg)
    ux, uy = math.sin(th), math.cos(th)
    half = 0.5 * length_m
    dx1, dy1 = -half * ux, -half * uy
    dx2, dy2 = half * ux, half * uy

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

    # Optional sparse vertex markers along the cable geometry
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
                tooltip=f"Vertex {i * marker_every}, alt={alt:.1f} m",
            ).add_to(layer)

    layer.add_to(m)
    return layer


# ---------------------------------------------------------------------------
# CHANNEL POSITIONS
# ---------------------------------------------------------------------------

def add_channel_positions_layer(
    m: folium.Map,
    channels_gps: dict,
    name: str = "Channel GPS (interp)",
    color: str = "#15b9e2",
    show: bool = True,
    draw_every: int = 5,
    label_every: int = 100,
) -> folium.FeatureGroup:
    """
    Plot all channel GPS positions as a polyline + optional thinned markers.

    channels_gps is expected to map channel index -> [lat, lon, alt].
    """
    layer = folium.FeatureGroup(name=name, show=show)
    if not channels_gps:
        layer.add_to(m)
        return layer

    items = sorted(channels_gps.items())  # List[Tuple[int, List[float]]]
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
                    html=(
                        f'<div style="font-size:10px; color:{color}; '
                        f'font-weight:bold;">ch: {ch}</div>'
                    ),
                ),
            ).add_to(layer)

    layer.add_to(m)
    return layer


# ---------------------------------------------------------------------------
# TRACK POINTS (FROM DOA CENTROIDS)
# ---------------------------------------------------------------------------

def add_track_points_layer(
    m: folium.Map,
    track_points: dict,
    enu_ref: tuple[float, float, float],
    name: str = "Track points",
    color: str = "#FF5733",
    show: bool = True,
    packet: int | None = None,
) -> folium.FeatureGroup:
    """
    Add boat track points (from DOA centroid estimate) as circle markers.

    track_points is a dict mapping packet_key (str) -> list of [E, N, U].
    enu_ref is (ref_lat, ref_lon, ref_alt) used when ENU was created.

    If packet is None → plot all packets.
    If packet is an int → only plot that packet's centroids.
    """
    layer = folium.FeatureGroup(name=name, show=show)

    ref_lat, ref_lon, ref_alt = enu_ref

    # Decide which packets to show
    if packet is None:
        pkt_items = track_points.items()
    else:
        key = str(packet)
        if key in track_points:
            pkt_items = [(key, track_points[key])]
        else:
            print(
                f"[add_track_points_layer] Packet {packet} not in track_points, "
                "nothing to plot."
            )
            pkt_items = []

    # Plot the points
    for pkt_key, enu_list in pkt_items:
        for enu_coord in enu_list:
            centroid = np.asarray(enu_coord, float)
            lat, lon, alt = enu2geodetic(
                centroid[0],
                centroid[1],
                centroid[2],
                ref_lat,
                ref_lon,
                ref_alt,
            )
            # Draw your circle marker
            folium.CircleMarker(
                [lat, lon],
                radius=3,
                color=color,
                fill=True,
                fill_opacity=0.9,
            ).add_to(layer)

            # Add always-visible text label
            folium.map.Marker(
                [lat, lon],
                icon=folium.DivIcon(
                    html=f"""<div style="font-size: 12px; color: green; 
                                        transform: translate(10px, -10px);
                                        white-space: nowrap;">
                                {pkt_key}
                            </div>"""
                )
            ).add_to(layer)

    layer.add_to(m)
    return layer


# ---------------------------------------------------------------------------
# SUBARRAYS
# ---------------------------------------------------------------------------

def add_subarray_layer(
    m: folium.Map,
    start_channels: List[int],
    array_length: int,
    channels_gps: dict,
    name: str = "Subarrays",
    color_even: str = "#FF633F",   # color for indices 0,2,4,...
    color_odd: str = "#B10000",    # color for indices 1,3,5,...
    show: bool = True,
) -> folium.FeatureGroup:
    """
    Visualise subarrays as sets of channels along the main cable.

    start_channels : list of first-channel indices for each subarray.
    array_length   : number of channels in each subarray.
    channels_gps   : channel index -> [lat, lon, alt].

    """

    print("Subarray layer start_channels =", start_channels)  # <--- add this
    layer = folium.FeatureGroup(name=name, show=show)

    for idx, start_ch in enumerate(start_channels):
        # Pick color based on whether this is 0th, 1st, 2nd, 3rd,... element
        color = color_even if (idx % 2 == 0) else color_odd

        channels_in_subarray = [start_ch + i for i in range(array_length)]

        for ch in channels_in_subarray:
            if ch not in channels_gps:
                continue
            lat, lon = channels_gps[ch][0], channels_gps[ch][1]
            folium.CircleMarker(
                [lat, lon],
                radius=4,
                color=color,
                fill=True,
                fill_opacity=0.9,
                popup=(
                    f"Subarray #{idx}  start={start_ch}, length={array_length}, "
                    f"channel={ch}"
                ),
            ).add_to(layer)

    layer.add_to(m)
    return layer


# ---------------------------------------------------------------------------
# RE-EXPORTED BOAT TRACK + TX POINTS
# ---------------------------------------------------------------------------

# build_source_track_layer and build_transmission_points_layer are imported above
# and simply re-exported here for convenience.

# ---------------------------------------------------------------------------
# DOA RAYS
# ---------------------------------------------------------------------------

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
    color_A: str = "#fefefe",
    color_B: str = "#d70d0d",
) -> folium.FeatureGroup:
    """
    Build a folium layer with DOA rays for one, many, or all packets.

    doa_results entries must contain at least:
      - packet (int)
      - center_lat (float)
      - center_lon (float)
      - dir_A_enu (Sequence[float])  # [E, N, U]
      - dir_B_enu (Sequence[float])
      - channels_min (int)
      - channels_max (int)
      - n_channels (int)
    """
    fg = folium.FeatureGroup(name=name, show=True)

    # Normalise packet_filter into a predicate
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
            end_A_enu[0], end_A_enu[1], 0.0,
            center_lat, center_lon, 0.0,
        )
        lat_B, lon_B, _ = enu2geodetic(
            end_B_enu[0], end_B_enu[1], 0.0,
            center_lat, center_lon, 0.0,
        )

        # Popup text with packet and channel info
        common_info = (
            f"Packet: {packet}<br>"
            f"Channels: {channels_min}–{channels_max} (n={n_channels})"
        )

        # Ray A
        folium.PolyLine(
            locations=[(center_lat, center_lon), (lat_A, lon_A)],
            color=color_A,
            weight=3,
            opacity=0.9,
            tooltip=f"DOA A (packet {packet})",
            popup=folium.Popup(common_info + "<br>Ray: A", max_width=250),
        ).add_to(fg)

        # Ray B
        folium.PolyLine(
            locations=[(center_lat, center_lon), (lat_B, lon_B)],
            color=color_B,
            weight=3,
            opacity=0.9,
            tooltip=f"DOA B (packet {packet})",
            popup=folium.Popup(common_info + "<br>Ray: B", max_width=250),
        ).add_to(fg)

        # Center marker
        folium.CircleMarker(
            location=(center_lat, center_lon),
            radius=3,
            color="#000000",
            fill=True,
            fill_opacity=0.8,
            tooltip=f"DOA center (packet {packet})",
        ).add_to(fg)

        # Optional: precomputed ellipse in lat/lon
        ellipse = _get_field(res, "ellipse_latlon", None)
        if ellipse:
            folium.PolyLine(
                locations=[(lat, lon) for lat, lon in ellipse],
                color="#ff8800",
                weight=2,
                opacity=0.6,
                tooltip=f"DOA cone intersection (packet {packet})",
            ).add_to(fg)

    return fg


# ---------------------------------------------------------------------------
# ELLIPSE / UNCERTAINTY BANDS (FROM ENU)
# ---------------------------------------------------------------------------

def add_band_between_lines(
    m: folium.Map,
    inner_latlon: List[Tuple[float, float]],
    outer_latlon: List[Tuple[float, float]],
    color: str = "#E62A15",
    fill_opacity: float = 0.3,
    name: str = "uncertainty band",
    show: bool = True,
) -> folium.FeatureGroup:
    """
    Fill the area between two boundary polylines defined in (lat, lon).

    inner_latlon, outer_latlon: lists of (lat, lon) along inner and outer boundaries.
    """
    poly_coords: List[List[float]] = []

    # Outer boundary
    for lat, lon in outer_latlon:
        poly_coords.append([lat, lon])

    # Inner boundary reversed
    for lat, lon in reversed(inner_latlon):
        poly_coords.append([lat, lon])

    # Close polygon explicitly (Leaflet will close, but this is clearer)
    if poly_coords and poly_coords[0] != poly_coords[-1]:
        poly_coords.append(poly_coords[0])

    layer = folium.FeatureGroup(name=name, show=show)

    folium.Polygon(
        locations=poly_coords,
        color=color,
        weight=1,
        fill=True,
        fill_color=color,
        fill_opacity=fill_opacity,
        tooltip=name,
    ).add_to(layer)

    layer.add_to(m)
    return layer


def add_ellipse_layer(
    m: folium.Map,
    enu_points: dict,
    enu_ref: tuple[float, float, float],
    name: str = "Ellipse band",
    color: str = "#E62A15",
    show: bool = True,
) -> folium.FeatureGroup:
    """
    Add an ellipse / cone intersection uncertainty band from ENU points.

    enu_points is expected to be a dict:
      {
        "inner_points":   [[E, N, U], ...],
        "nominal_points": [[E, N, U], ...],  # currently not drawn
        "outer_points":   [[E, N, U], ...],
      }

    enu_ref is (ref_lat, ref_lon, ref_alt) used when ENU was created.
    """
    inner_points = np.asarray(enu_points.get("inner_points", []), dtype=float)
    outer_points = np.asarray(enu_points.get("outer_points", []), dtype=float)

    ref_lat, ref_lon, ref_alt = enu_ref

    # Convert ENU -> geodetic
    llh_inner: List[Tuple[float, float, float]] = []
    for el in inner_points:
        lat, lon, alt = enu2geodetic(el[0], el[1], el[2], ref_lat, ref_lon, ref_alt)
        llh_inner.append((lat, lon, alt))

    llh_outer: List[Tuple[float, float, float]] = []
    for el in outer_points:
        lat, lon, alt = enu2geodetic(el[0], el[1], el[2], ref_lat, ref_lon, ref_alt)
        llh_outer.append((lat, lon, alt))

    print(f"Adding {len(llh_inner)} ellipse points to map layer '{name}'")

    layer = add_band_between_lines(
        m,
        [(lat, lon) for (lat, lon, alt) in llh_inner],
        [(lat, lon) for (lat, lon, alt) in llh_outer],
        color=color,
        name=name,
        show=show,
    )

    return layer


# ---------------------------------------------------------------------------
# EXPORTS
# ---------------------------------------------------------------------------

__all__ = [
    "enu_reference_from_channels",
    "add_cable_layout_layer",
    "add_channel_positions_layer",
    "add_subarray_layer",
    "add_track_points_layer",
    "build_source_track_layer",
    "build_transmission_points_layer",
    "build_doa_layer_from_results",
    "add_ellipse_layer",
    "add_band_between_lines",
]
