# libs/dasprocessor/scripts/plot_run_channel_positions.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import folium

from dasprocessor.constants import get_run
from dasprocessor.bearing_tools import channel_gps_for_run  # returns (lat, lon, alt) per channel

DATE_STR = "2024-05-03"  # keep consistent with your constants


def build_channel_positions_layer(
    run_number: int = 2,
    name: str = "Channel positions",
    color: str = "#cc3300",
    show: bool = True,
    marker_radius: int = 2,
    label_every: int = 0,  # 0 = no labels; otherwise label every Nth channel
    draw_every: int = 1,   # 1 = draw all; 5 = draw every 5th channel, etc.
    add_polyline: bool = True,
) -> folium.FeatureGroup:
    """
    Build a Folium FeatureGroup for per-channel positions for the given run.

    Parameters
    ----------
    run_number : int
        Run number to pull channel geometry for.
    name : str
        Layer name for Folium's layer control.
    color : str
        Hex color for markers/polyline.
    show : bool
        Whether the layer is visible by default.
    marker_radius : int
        CircleMarker radius in pixels.
    label_every : int
        If >0, places a small text label for every Nth channel index.
    draw_every : int
        If >1, subsamples channels for markers/labels to reduce weight.
    add_polyline : bool
        If True, draws a thin polyline through all channel coordinates.

    Returns
    -------
    folium.FeatureGroup
    """
    layer = folium.FeatureGroup(name=name, show=show)

    # Pull per-channel (lat, lon, alt)
    gps = channel_gps_for_run(run_number)  # shape (ch_count, 3) -> [lat, lon, alt]
    if gps.size == 0:
        return layer

    # Polyline (full resolution, independent of draw_every so the path looks continuous)
    if add_polyline and gps.shape[0] >= 2:
        coords_line = [(float(lat), float(lon)) for (lat, lon) in gps[:, :2]]
        folium.PolyLine(coords_line, color=color, weight=2, opacity=0.8).add_to(layer)

    # Markers (optionally thinned)
    ch_indices = np.arange(gps.shape[0])
    if draw_every > 1:
        mask = (ch_indices % draw_every) == 0
    else:
        mask = np.ones_like(ch_indices, dtype=bool)

    for idx in ch_indices[mask]:
        lat, lon, alt = map(float, gps[idx])
        folium.CircleMarker(
            location=[lat, lon],
            radius=marker_radius,
            color=color,
            fill=True,
            fill_opacity=0.9,
            tooltip=f"Ch {idx}  (lat={lat:.6f}, lon={lon:.6f}, alt={alt:.1f} m)",
        ).add_to(layer)

        if label_every > 0 and (idx % label_every) == 0:
            folium.map.Marker(
                [lat, lon],
                icon=folium.DivIcon(
                    icon_size=(48, 14),
                    icon_anchor=(0, 0),
                    html=(
                        f'<div style="font-size:10px; color:{color}; '
                        f'font-weight:bold; text-shadow: -1px 0 #fff, 0 1px #fff, '
                        f'1px 0 #fff, 0 -1px #fff;">{idx}</div>'
                    ),
                ),
            ).add_to(layer)

    return layer


def _center_for_run(run_number: int) -> Tuple[float, float]:
    """
    Choose a map center near mid-channel.
    """
    gps = channel_gps_for_run(run_number)
    if gps.size == 0:
        raise ValueError(f"No channel GPS for run {run_number}")
    mid = gps[gps.shape[0] // 2]
    return float(mid[0]), float(mid[1])


def make_map_with_channels(
    run_number: int,
    tiles: str = "OpenStreetMap",
    zoom_start: int = 15,
    draw_every: int = 5,   # default to lighten the page if many channels
    label_every: int = 0,
) -> folium.Map:
    """
    Convenience: build a Folium map centered on the cable with a channel layer.
    """
    lat0, lon0 = _center_for_run(run_number)
    m = folium.Map(location=[lat0, lon0], zoom_start=zoom_start, tiles=tiles)

    chan_layer = build_channel_positions_layer(
        run_number=run_number,
        name=f"Channels (run {run_number})",
        color="#cc3300",
        show=True,
        marker_radius=2,
        label_every=label_every,
        draw_every=draw_every,
        add_polyline=True,
    )
    chan_layer.add_to(m)

    folium.LayerControl().add_to(m)
    return m


def main():
    run_number = 2

    m = make_map_with_channels(
        run_number=run_number,
        tiles="OpenStreetMap",
        zoom_start=15,
        draw_every=5,   # tune depending on channel_count
        label_every=100 # label every 100th channel; set 0 for none
    )

    out_path = Path(__file__).with_name(f"run{run_number}_channels.html")
    m.save(str(out_path))
    print(f"Map saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
