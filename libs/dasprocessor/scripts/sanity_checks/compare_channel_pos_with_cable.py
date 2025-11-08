# libs/dasprocessor/scripts/plot_run_cable_vs_channels.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import json
import numpy as np
import folium

from dasprocessor.bearing_tools import channel_gps_for_run
from dasprocessor.constants import get_run

DATE_STR = "2024-05-03"  # keep consistent with your constants


def _load_cable_layout_geojson(path: Path) -> np.ndarray:
    """
    Load your cable-layout GeoJSON (FeatureCollection with MultiLineString)
    and return an (N, 3) array in (lat, lon, alt) order.
    """
    with path.open("r", encoding="utf-8") as fh:
        gj = json.load(fh)

    coords_all = []
    for feat in gj.get("features", []):
        geom = feat.get("geometry", {})
        gtype = geom.get("type", "")
        if gtype == "MultiLineString":
            # coordinates: [ [ [lon, lat, alt], ... ], [ ... ] ]
            for line in geom.get("coordinates", []):
                for pt in line:
                    lon, lat = float(pt[0]), float(pt[1])
                    alt = float(pt[2]) if len(pt) > 2 else 0.0
                    coords_all.append((lat, lon, alt))
        elif gtype == "LineString":
            for pt in geom.get("coordinates", []):
                lon, lat = float(pt[0]), float(pt[1])
                alt = float(pt[2]) if len(pt) > 2 else 0.0
                coords_all.append((lat, lon, alt))
        else:
            # ignore other geometry types
            continue

    return np.asarray(coords_all, dtype=float) if coords_all else np.empty((0, 3), dtype=float)


def build_cable_layout_layer(
    geojson_path: Path,
    name: str = "Cable layout (raw)",
    color: str = "#111111",
    show: bool = True,
    marker_every: int = 0  # 0 = no per-vertex markers; N to drop a small marker every Nth vertex
) -> folium.FeatureGroup:
    """
    Turn your cable layout polyline vertices into a Folium layer:
    a polyline plus (optional) sparse vertex markers.
    """
    layer = folium.FeatureGroup(name=name, show=show)
    cable = _load_cable_layout_geojson(geojson_path)
    if cable.size == 0:
        return layer

    coords = [(float(lat), float(lon)) for (lat, lon, _alt) in cable]
    folium.PolyLine(coords, color=color, weight=3, opacity=0.8).add_to(layer)

    if marker_every and marker_every > 0:
        for i in range(0, cable.shape[0], marker_every):
            lat, lon, alt = map(float, cable[i])
            folium.CircleMarker(
                location=[lat, lon],
                radius=3,
                color=color,
                fill=True,
                fill_opacity=0.9,
                tooltip=f"Layout vertex {i} (alt={alt:.1f} m)",
            ).add_to(layer)

    return layer


def build_channel_positions_layer(
    run_number: int,
    name: str = "Channel GPS (interp)",
    color: str = "#cc3300",
    show: bool = True,
    draw_every: int = 5,   # thin markers to keep the map snappy
    label_every: int = 0,  # label every Nth channel (0 = none)
) -> folium.FeatureGroup:
    """
    Re-usable layer: interpolated per-channel GPS from channel_gps_for_run.
    """
    layer = folium.FeatureGroup(name=name, show=show)
    gps = channel_gps_for_run(run_number)  # (ch_count, 3) -> lat, lon, alt
    if gps.size == 0:
        return layer

    # polyline through all channels

    coords_line = [(float(row[0]), float(row[1])) for row in gps]

    if len(coords_line) >= 2:
        folium.PolyLine(coords_line, color=color, weight=2, opacity=0.8).add_to(layer)

    # markers (thinned)
    idxs = np.arange(gps.shape[0])
    mask = (idxs % max(1, draw_every)) == 0
    for idx in idxs[mask]:
        lat, lon, alt = map(float, gps[idx])
        folium.CircleMarker(
            location=[lat, lon],
            radius=2,
            color=color,
            fill=True,
            fill_opacity=0.9,
            tooltip=f"Ch {idx}  (lat={lat:.6f}, lon={lon:.6f}, alt={alt:.1f} m)",
        ).add_to(layer)

        if label_every and (idx % label_every == 0):
            folium.map.Marker(
                [lat, lon],
                icon=folium.DivIcon(
                    icon_size=(48, 14),
                    icon_anchor=(0, 0),
                    html=(f'<div style="font-size:10px; color:{color}; '
                          f'font-weight:bold; text-shadow:-1px 0 #fff,0 1px #fff,1px 0 #fff,0 -1px #fff;">{idx}</div>')
                ),
            ).add_to(layer)

    return layer


def _center_for_run(run_number: int) -> Tuple[float, float]:
    gps = channel_gps_for_run(run_number)
    if gps.size:
        mid = gps[gps.shape[0] // 2]
        return float(mid[0]), float(mid[1])
    run = get_run(DATE_STR, run_number)
    return float(run.get("map_center_lat", 0.0)), float(run.get("map_center_lon", 0.0))


def make_map_compare(
    run_number: int,
    geojson_path: Path,
    zoom_start: int = 15,
    tiles: str = "OpenStreetMap",
) -> folium.Map:
    """
    Build a Folium map with both layers so you can sanity-check the alignment.
    """
    lat0, lon0 = _center_for_run(run_number)
    m = folium.Map(location=[lat0, lon0], zoom_start=zoom_start, tiles=tiles)

    # Raw layout from file (black) + interpolated channel GPS (orange/red)
    build_cable_layout_layer(geojson_path, name="Cable layout (raw)", color="#111111", show=True, marker_every=25).add_to(m)
    build_channel_positions_layer(run_number, name=f"Channels (run {run_number})", color="#cc3300",
                                  show=True, draw_every=5, label_every=0).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def main():
    run_number = 2
    geojson_path = Path(r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\cable-layout.json")
    m = make_map_compare(run_number, geojson_path)
    out_path = Path(__file__).with_name(f"run{run_number}_cable_vs_channels.html")
    m.save(str(out_path))
    print(f"Map saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
