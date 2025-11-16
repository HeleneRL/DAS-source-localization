# libs/dasprocessor/scripts/plot_run_source_track.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import csv
from datetime import datetime, timedelta
import numpy as np 


import folium

from dasprocessor.constants import get_run

DATE_STR = "2024-05-03"  # experiment date in your constants


def _parse_iso_dt(s: str) -> datetime:
    # CSV uses e.g. "2024-05-03 12:22:09.123456"
    # Python accepts space in fromisoformat
    return datetime.fromisoformat(s)


def _run_datetime_bounds(run_number: int) -> Tuple[datetime, datetime]:
    """Build absolute datetime bounds for the run from constants."""
    run = get_run(DATE_STR, run_number)
    (h1, m1, s1), (h2, m2, s2) = run["time_range"]
    start = datetime.fromisoformat(f"{DATE_STR} {h1:02d}:{m1:02d}:{s1:02d}")
    stop  = datetime.fromisoformat(f"{DATE_STR} {h2:02d}:{m2:02d}:{s2:02d}")
    return start, stop


def load_source_points_for_run(csv_path: Path, run_number: int) -> List[Tuple[float, float, datetime]]:
    """
    Load (lat, lon, dt) points from the CSV for the given run's time window.
    """
    start_dt, stop_dt = _run_datetime_bounds(run_number)
    out: List[Tuple[float, float, datetime]] = []

    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                dt = _parse_iso_dt(row["datetime"])
                if not (start_dt <= dt <= stop_dt):
                    continue
                lat = float(row["lat"])
                lon = float(row["lon"])
                out.append((lat, lon, dt))
            except Exception:
                # skip malformed rows silently
                continue
    return out

def _tx_datetimes_for_run(run_number: int) -> list[datetime]:
    """
    Build the 128 transmission datetimes for the run from constants:
    sequence_start + i * (sequence_period / sample_rate), i=0..sequence_count-1
    """
    run = get_run(DATE_STR, run_number)
    (h, m, s) = run["sequence_start"]
    base = datetime.fromisoformat(f"{DATE_STR} {h:02d}:{m:02d}:{s:02d}")

    seq_count = int(run["sequence_count"])
    dt_per = float(run["sequence_period"]) / float(run["sample_rate"])  # seconds between transmissions

    return [base + timedelta(seconds=i * dt_per) for i in range(seq_count)]


def _interp_positions_at_times(
    points: list[tuple[float, float, datetime]],
    query_times: list[datetime]
) -> list[tuple[float, float, datetime, int]]:
    """
    Linearly interpolate (lat, lon) at the query_times.
    Returns list of (lat, lon, dt, idx), where idx is 0..len(query_times)-1.
    Only keeps queries within the time span of 'points'.
    """
    if not points:
        return []

    # Sort inputs by time
    pts = sorted(points, key=lambda t: t[2])
    t0 = np.array([p[2].timestamp() for p in pts], dtype=float)
    lat = np.array([p[0] for p in pts], dtype=float)
    lon = np.array([p[1] for p in pts], dtype=float)

    out = []
    tq = np.array([q.timestamp() for q in query_times], dtype=float)
    # Keep only queries within the covered interval
    mask = (tq >= t0[0]) & (tq <= t0[-1])
    if not np.any(mask):
        return []

    tq_in = tq[mask]
    lat_q = np.interp(tq_in, t0, lat)
    lon_q = np.interp(tq_in, t0, lon)

    j = 0
    for i, keep in enumerate(mask):
        if not keep:
            continue
        out.append((float(lat_q[j]), float(lon_q[j]), query_times[i], i))
        j += 1

    return out



def build_source_track_layer(points: List[Tuple[float, float, datetime]], name: str = "Boat track (run)") -> folium.FeatureGroup:
    """
    Turn a list of (lat, lon, dt) into a Folium FeatureGroup with a line + markers.
    """
    layer = folium.FeatureGroup(name=name, show=True)
    if not points:
        return layer

    # Sort by time to ensure proper path order
    pts_sorted = sorted(points, key=lambda t: t[2])
    coords = [(lat, lon) for (lat, lon, _) in pts_sorted]

    # Polyline for the path
    folium.PolyLine(coords, color="#2ca02c", weight=3, opacity=0.9).add_to(layer)

    # Optional sparse markers: first and last, plus maybe every Nth
    first_lat, first_lon, first_dt = pts_sorted[0]
    last_lat, last_lon, last_dt = pts_sorted[-1]
    folium.Marker([first_lat, first_lon],
                  icon=folium.Icon(color="green", icon="play"),
                  tooltip=f"Start: {first_dt}").add_to(layer)
    folium.Marker([last_lat, last_lon],
                  icon=folium.Icon(color="darkred", icon="stop"),
                  tooltip=f"End: {last_dt}").add_to(layer)

    return layer

def build_transmission_points_layer(
    csv_path: Path,
    run_number: int,
    label_every: int = 10,
    name: str = "TX positions"
) -> folium.FeatureGroup:
    """
    Build a Folium layer with 128 transmission points (one per signal) for the run.
    Labels every `label_every`th point with the packet index.
    """
    layer = folium.FeatureGroup(name=name, show=True)

    # Boat track points restricted to the run time window
    src_points = load_source_points_for_run(csv_path, run_number)
    if not src_points:
        return layer

    # Transmission times from constants
    q_times = _tx_datetimes_for_run(run_number)

    # Interpolate lat/lon at those times
    tx_pts = _interp_positions_at_times(src_points, q_times)
    if not tx_pts:
        # Layer stays empty if the time windows don't overlap enough
        return layer

    # Draw: small circles; label each 'label_every' point
    coords_line = []
    for (lat, lon, dt, idx) in tx_pts:
        coords_line.append((lat, lon))
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color="#e47e11",
            fill=True,
            fill_opacity=0.9,
            tooltip=f"Packet {idx} @ {dt}",
        ).add_to(layer)

        if label_every > 0 and idx % label_every == 0:
            folium.map.Marker(
                [lat, lon],
                icon=folium.DivIcon(
                    icon_size=(40, 12),
                    icon_anchor=(0, 0),
                    html=f'<div style="font-size:10px; color:#0033cc; font-weight:bold;">{idx}</div>',
                ),
            ).add_to(layer)

    # Optional: polyline through the 128 points (thin)
    if len(coords_line) >= 2:
        folium.PolyLine(coords_line, color="#e47e11", weight=2, opacity=0.6).add_to(layer)

    return layer



def main():
    run_number = 2
    csv_path = Path(r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\source-position.csv")

    # Base map centered near the middle of the run window
    points = load_source_points_for_run(csv_path, run_number)
    if not points:
        print("No source points found for this run/time window. "
              "Double-check the CSV and run time_range in constants.")
        return
    mid = points[len(points)//2]
    lat0, lon0 = float(mid[0]), float(mid[1])
    m = folium.Map(location=[lat0, lon0], zoom_start=15, tiles="OpenStreetMap")

    # Full boat track
    track_layer = build_source_track_layer(points, name=f"Boat track (run {run_number})")
    track_layer.add_to(m)

    # NEW: 128 TX points with labels every 10th
    tx_layer = build_transmission_points_layer(csv_path, run_number, label_every=10,
                                               name=f"TX positions (run {run_number})")
    tx_layer.add_to(m)

    folium.LayerControl().add_to(m)
    out_path = Path(__file__).with_name(f"run{run_number}_track_and_tx.html")
    m.save(str(out_path))
    print(f"Map saved to: {out_path.resolve()}")



if __name__ == "__main__":
    main()
