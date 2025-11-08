# libs/dasprocessor/plot/map_layers.py
from __future__ import annotations

from pathlib import Path
import json
import math
import numpy as np
import folium

from dasprocessor.constants import get_run
from dasprocessor.bearing_tools import (
    build_subarrays,
    get_cached_channel_gps_for_run,
    subarray_centers_and_headings,
)





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
    gps: np.ndarray,
    name: str = "Channel GPS (interp)",
    color: str = "#cc3300",
    show: bool = True,
    draw_every: int = 5,
    label_every: int = 0,
) -> folium.FeatureGroup:
    """Plot all channel GPS positions as a polyline + optional thinned markers."""
    layer = folium.FeatureGroup(name=name, show=show)
    if gps.size == 0:
        layer.add_to(m)
        return layer

    coords = [(float(row[0]), float(row[1])) for row in gps]
    folium.PolyLine(coords, color=color, weight=2, opacity=0.8).add_to(layer)

    for idx in range(0, gps.shape[0], draw_every):
        lat, lon = gps[idx, 0], gps[idx, 1]
        alt = gps[idx, 2] if gps.shape[1] > 2 else np.nan
        folium.CircleMarker(
            [lat, lon],
            radius=2,
            color=color,
            fill=True,
            fill_opacity=0.9,
            tooltip=f"Ch {idx} (alt={alt:.1f} m)" if np.isfinite(alt) else f"Ch {idx}",
        ).add_to(layer)
        if label_every and (idx % label_every == 0):
            folium.map.Marker(
                [lat, lon],
                icon=folium.DivIcon(
                    icon_size=(40, 12),
                    icon_anchor=(0, 0),
                    html=f'<div style="font-size:10px; color:{color}; font-weight:bold;">{idx}</div>',
                ),
            ).add_to(layer)

    layer.add_to(m)
    return layer


# ---------------------------------------------------------------------------
# SUBARRAY CENTERS + HEADINGS
# ---------------------------------------------------------------------------

def add_subarray_centers_layer(
    m: folium.Map,
    centers: list[int],
    aperture_len: int,
    run_number: int,
    name: str = "Subarray centers + headings",
    color: str = "#1f77b4",
    show: bool = True,
) -> folium.FeatureGroup:
    """Add markers + heading lines for all subarrays."""
    layer = folium.FeatureGroup(name=name, show=show)
    subarrays = build_subarrays(centers, aperture_len, run_number)
    gps = get_cached_channel_gps_for_run(run_number)
    meta = subarray_centers_and_headings(subarrays, gps)
    run = get_run(DATE_STR, run_number)
    aperture_length_m = (aperture_len - 1) * float(run["channel_distance"])

    for c in centers:
        info = meta.get(c)
        if not info:
            continue
        lat, lon, hdg = float(info["center_lat"]), float(info["center_lon"]), float(info["heading_deg"])
        folium.CircleMarker(
            [lat, lon],
            radius=4,
            color=color,
            fill=True,
            fill_opacity=0.9,
            popup=f"center={c}, heading={hdg:.2f}°, length={aperture_length_m:.1f} m",
        ).add_to(layer)
        if math.isfinite(hdg):
            (lat1, lon1), (lat2, lon2) = endpoints_centered(lat, lon, hdg, aperture_length_m)
            folium.PolyLine([[lat1, lon1], [lat2, lon2]], color=color, weight=4, opacity=0.9).add_to(layer)

    layer.add_to(m)
    return layer


# ---------------------------------------------------------------------------
# BOAT TRACK + TX POINTS
# ---------------------------------------------------------------------------

from dasprocessor.plot.source_track import (
    load_source_points_for_run,
    build_source_track_layer,
    build_transmission_points_layer,
)
# These are already great and return FeatureGroups, so just import & reexport.


__all__ = [
    "add_cable_layout_layer",
    "add_channel_positions_layer",
    "add_subarray_centers_layer",
    "build_source_track_layer",
    "build_transmission_points_layer",
]
