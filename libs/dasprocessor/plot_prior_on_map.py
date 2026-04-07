from __future__ import annotations

import pandas as pd
import folium
from folium.plugins import MeasureControl
from pathlib import Path
import math


# ============================================================
# USER SETTINGS
# ============================================================
TX_FILE = Path(r"D:\Singapore Data\transmission_times_sweeps_with_tx_positions.csv")
CABLE_FILE = Path(r"D:\Singapore Data\Cable\interpolated_channels_from_boattrack.csv")
OUTPUT_HTML = Path(r"D:\Singapore Data\cable_tx_map.html")

CHANNEL_LABEL_STEP = 50  # mark every 50th channel
MAP_TILES = "OpenStreetMap"  # can change to "CartoDB positron"
TX_DEPTH_M = -3.0  # constant transmitter depth


# ============================================================
# HELPERS
# ============================================================
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    return df


def load_table(path: Path) -> pd.DataFrame:
    """
    Reads either comma-separated or tab-separated text robustly.
    """
    try:
        df = pd.read_csv(path)
        if len(df.columns) == 1:
            df = pd.read_csv(path, sep="\t")
    except Exception:
        df = pd.read_csv(path, sep="\t")
    return clean_columns(df)


def get_map_center(cable_df: pd.DataFrame, tx_df: pd.DataFrame) -> tuple[float, float]:
    lats = []
    lons = []

    if "lat" in cable_df.columns and "lon" in cable_df.columns:
        lats.extend(pd.to_numeric(cable_df["lat"], errors="coerce").dropna().tolist())
        lons.extend(pd.to_numeric(cable_df["lon"], errors="coerce").dropna().tolist())

    for col in ["tx_lat_peak1", "tx_lat_peak2"]:
        if col in tx_df.columns:
            lats.extend(pd.to_numeric(tx_df[col], errors="coerce").dropna().tolist())

    for col in ["tx_lon_peak1", "tx_lon_peak2"]:
        if col in tx_df.columns:
            lons.extend(pd.to_numeric(tx_df[col], errors="coerce").dropna().tolist())

    if not lats or not lons:
        raise ValueError("Could not determine map center from the data.")

    return sum(lats) / len(lats), sum(lons) / len(lons)


def safe_float(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def nearest_cable_point(channel_value: int, cable_df: pd.DataFrame) -> pd.Series | None:
    """
    Returns the row in cable_df with channel closest to channel_value.
    """
    if cable_df.empty:
        return None
    idx = (cable_df["channel"] - channel_value).abs().idxmin()
    return cable_df.loc[idx]


def make_tx_popup(row: pd.Series, sweep_num: int) -> str:
    lat = row.get(f"tx_lat_peak{sweep_num}", "")
    lon = row.get(f"tx_lon_peak{sweep_num}", "")
    t = row.get(f"tx_time_peak{sweep_num}", "")
    utc_peak = row.get(f"utc_peak{sweep_num}", "")
    location = row.get("location", "")
    ref_ch = row.get("reference_channel", "")

    return (
        f"<b>{location} - Sweep {sweep_num}</b><br>"
        f"Reference channel: {ref_ch}<br>"
        f"Peak time: {utc_peak}<br>"
        f"Chosen TX time: {t}<br>"
        f"TX lat: {lat}<br>"
        f"TX lon: {lon}<br>"
        f"TX depth: {TX_DEPTH_M} m"
    )


def add_text_marker(map_obj, lat, lon, text, color="black", font_size="10pt"):
    """
    Adds a simple always-visible text label.
    """
    html = f"""
    <div style="
        font-size: {font_size};
        color: {color};
        white-space: nowrap;
        font-weight: bold;
        text-shadow:
            -1px -1px 0 white,
             1px -1px 0 white,
            -1px  1px 0 white,
             1px  1px 0 white;
    ">{text}</div>
    """
    folium.Marker(
        location=[lat, lon],
        icon=folium.DivIcon(html=html)
    ).add_to(map_obj)


# ============================================================
# MAIN
# ============================================================
def main():
    # ----------------------------
    # Load data
    # ----------------------------
    tx_df = load_table(TX_FILE)
    cable_df = load_table(CABLE_FILE)

    # Standardize numeric columns
    for col in ["channel", "lat", "lon", "depth"]:
        if col in cable_df.columns:
            cable_df[col] = pd.to_numeric(cable_df[col], errors="coerce")

    for col in [
        "reference_channel",
        "tx_lat_peak1", "tx_lon_peak1",
        "tx_lat_peak2", "tx_lon_peak2",
        "tx_depth_m"
    ]:
        if col in tx_df.columns:
            tx_df[col] = pd.to_numeric(tx_df[col], errors="coerce")

    cable_df = cable_df.dropna(subset=["channel", "lat", "lon"]).sort_values("channel").reset_index(drop=True)

    # ----------------------------
    # Create map
    # ----------------------------
    center_lat, center_lon = get_map_center(cable_df, tx_df)

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=16,
        tiles=MAP_TILES,
        control_scale=True
    )

    folium.TileLayer("CartoDB positron").add_to(m)
    folium.TileLayer("CartoDB Voyager").add_to(m)
    folium.TileLayer("Esri.WorldImagery", name="Esri Satellite", attr="Esri").add_to(m)

    m.add_child(MeasureControl())

    # ----------------------------
    # Feature groups
    # ----------------------------
    fg_cable = folium.FeatureGroup(name="Cable line", show=True)
    fg_channel_labels = folium.FeatureGroup(name=f"Every {CHANNEL_LABEL_STEP}th channel", show=True)
    fg_reference = folium.FeatureGroup(name="Reference channels", show=True)
    fg_tx1 = folium.FeatureGroup(name="Transmission positions - Sweep 1", show=True)
    fg_tx2 = folium.FeatureGroup(name="Transmission positions - Sweep 2", show=True)

    # ----------------------------
    # Plot cable line
    # ----------------------------
    cable_points = cable_df[["lat", "lon"]].values.tolist()

    folium.PolyLine(
        cable_points,
        color="blue",
        weight=4,
        opacity=0.85,
        tooltip="Cable"
    ).add_to(fg_cable)

    # ----------------------------
    # Mark every 50th channel
    # ----------------------------
    min_channel = int(cable_df["channel"].min())
    max_channel = int(cable_df["channel"].max())

    # We mark channels divisible by CHANNEL_LABEL_STEP
    label_channels = cable_df[cable_df["channel"] % CHANNEL_LABEL_STEP == 0].copy()

    # If you want to always include first/last, uncomment this block:
    # endpoints = cable_df.iloc[[0, -1]]
    # label_channels = pd.concat([label_channels, endpoints]).drop_duplicates(subset=["channel"])

    for _, row in label_channels.iterrows():
        ch = int(row["channel"])
        lat = row["lat"]
        lon = row["lon"]
        depth = row["depth"] if "depth" in row and not pd.isna(row["depth"]) else None

        popup_txt = f"<b>Channel {ch}</b>"
        if depth is not None:
            popup_txt += f"<br>Depth: {depth:.2f} m"

        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color="darkblue",
            fill=True,
            fill_opacity=1.0,
            popup=folium.Popup(popup_txt, max_width=250),
            tooltip=f"Ch {ch}"
        ).add_to(fg_channel_labels)

        label = f"{ch}"
        if depth is not None:
            label += f" ({depth:.1f} m)"
        add_text_marker(fg_channel_labels, lat, lon, label, color="darkblue", font_size="9pt")

    # ----------------------------
    # Mark reference channels for each location
    # ----------------------------
    ref_rows = tx_df.dropna(subset=["reference_channel"]).copy()

    for _, row in ref_rows.iterrows():
        ref_ch = int(row["reference_channel"])
        loc_name = str(row["location"])

        cable_row = nearest_cable_point(ref_ch, cable_df)
        if cable_row is None:
            continue

        lat = float(cable_row["lat"])
        lon = float(cable_row["lon"])
        depth = float(cable_row["depth"]) if "depth" in cable_row and not pd.isna(cable_row["depth"]) else math.nan

        popup_txt = (
            f"<b>{loc_name}</b><br>"
            f"Reference channel: {ref_ch}<br>"
            f"Cable lat: {lat:.8f}<br>"
            f"Cable lon: {lon:.8f}<br>"
        )
        if not math.isnan(depth):
            popup_txt += f"Depth: {depth:.2f} m"

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_txt, max_width=300),
            tooltip=f"{loc_name} ref ch {ref_ch}",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(fg_reference)

        add_text_marker(
            fg_reference,
            lat,
            lon,
            f"{loc_name}\nref {ref_ch}",
            color="red",
            font_size="10pt"
        )

    # ----------------------------
    # Plot transmission positions: sweep 1
    # ----------------------------
    for _, row in tx_df.iterrows():
        lat = safe_float(row.get("tx_lat_peak1"))
        lon = safe_float(row.get("tx_lon_peak1"))
        if lat is None or lon is None:
            continue

        location_name = str(row.get("location", ""))
        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color="green",
            fill=True,
            fill_opacity=0.9,
            popup=folium.Popup(make_tx_popup(row, 1), max_width=320),
            tooltip=f"{location_name} sweep 1"
        ).add_to(fg_tx1)

        add_text_marker(
            fg_tx1,
            lat,
            lon,
            f"{location_name} s1",
            color="green",
            font_size="10pt"
        )

    # ----------------------------
    # Plot transmission positions: sweep 2
    # ----------------------------
    for _, row in tx_df.iterrows():
        lat = safe_float(row.get("tx_lat_peak2"))
        lon = safe_float(row.get("tx_lon_peak2"))
        if lat is None or lon is None:
            continue

        location_name = str(row.get("location", ""))
        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color="purple",
            fill=True,
            fill_opacity=0.9,
            popup=folium.Popup(make_tx_popup(row, 2), max_width=320),
            tooltip=f"{location_name} sweep 2"
        ).add_to(fg_tx2)

        add_text_marker(
            fg_tx2,
            lat,
            lon,
            f"{location_name} s2",
            color="purple",
            font_size="10pt"
        )

    # ----------------------------
    # Add groups and controls
    # ----------------------------
    fg_cable.add_to(m)
    fg_channel_labels.add_to(m)
    fg_reference.add_to(m)
    fg_tx1.add_to(m)
    fg_tx2.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # Fit map to cable bounds
    if len(cable_points) > 0:
        m.fit_bounds(cable_points)

    # Save
    m.save(str(OUTPUT_HTML))
    print(f"Saved map to: {OUTPUT_HTML}")


if __name__ == "__main__":
    main()