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
EST_CABLE_FILE = Path(r"D:\Singapore Data\Cable\interpolated_channels_from_boattrack.csv")
TRUE_CABLE_FILE = Path(r"D:\Singapore Data\array-shape.csv")
OUTPUT_HTML = Path(r"D:\Singapore Data\cable_tx_map_with_true_layout.html")

CHANNEL_LABEL_STEP = 50
MAP_TILES = "OpenStreetMap"
TX_DEPTH_M = -3.0  # constant transmitter depth


# ============================================================
# HELPERS
# ============================================================
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    return df


def load_table(path: Path) -> pd.DataFrame:
    """
    Reads comma- or tab-separated text robustly.
    """
    try:
        df = pd.read_csv(path)
        if len(df.columns) == 1:
            df = pd.read_csv(path, sep="\t")
    except Exception:
        df = pd.read_csv(path, sep="\t")
    return clean_columns(df)


def get_map_center(*dfs_and_cols) -> tuple[float, float]:
    lats = []
    lons = []

    for df, lat_col, lon_col in dfs_and_cols:
        if df is None or df.empty:
            continue
        if lat_col in df.columns and lon_col in df.columns:
            lats.extend(pd.to_numeric(df[lat_col], errors="coerce").dropna().tolist())
            lons.extend(pd.to_numeric(df[lon_col], errors="coerce").dropna().tolist())

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
    ">{text.replace(chr(10), '<br>')}</div>
    """
    folium.Marker(
        location=[lat, lon],
        icon=folium.DivIcon(html=html)
    ).add_to(map_obj)


def add_cable_line(feature_group, cable_df, color, tooltip_text):
    cable_points = cable_df[["lat", "lon"]].values.tolist()
    folium.PolyLine(
        cable_points,
        color=color,
        weight=4,
        opacity=0.85,
        tooltip=tooltip_text
    ).add_to(feature_group)
    return cable_points


def add_channel_markers(feature_group, cable_df, step, color, label_prefix="", depth_col="depth"):
    """
    Mark every `step`-th channel with popup and text.
    """
    label_channels = cable_df[cable_df["channel"] % step == 0].copy()

    for _, row in label_channels.iterrows():
        ch = int(row["channel"])
        lat = float(row["lat"])
        lon = float(row["lon"])
        depth = None
        if depth_col in row.index and not pd.isna(row[depth_col]):
            depth = float(row[depth_col])

        popup_txt = f"<b>{label_prefix}Channel {ch}</b>"
        if depth is not None:
            popup_txt += f"<br>Depth: {depth:.2f} m"

        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color=color,
            fill=True,
            fill_opacity=1.0,
            popup=folium.Popup(popup_txt, max_width=250),
            tooltip=f"{label_prefix}Ch {ch}"
        ).add_to(feature_group)

        label = f"{ch}"
        if depth is not None:
            label += f" ({depth:.1f} m)"
        add_text_marker(feature_group, lat, lon, label, color=color, font_size="9pt")


def add_reference_markers(feature_group, tx_df, cable_df, marker_color, text_color, cable_name, depth_col="depth"):
    """
    Plot reference channels from tx_df onto a given cable dataframe.
    """
    ref_rows = tx_df.dropna(subset=["reference_channel"]).copy()

    for _, row in ref_rows.iterrows():
        ref_ch = int(row["reference_channel"])
        loc_name = str(row["location"])

        cable_row = nearest_cable_point(ref_ch, cable_df)
        if cable_row is None:
            continue

        lat = float(cable_row["lat"])
        lon = float(cable_row["lon"])
        depth = None
        if depth_col in cable_row.index and not pd.isna(cable_row[depth_col]):
            depth = float(cable_row[depth_col])

        popup_txt = (
            f"<b>{loc_name}</b><br>"
            f"Cable: {cable_name}<br>"
            f"Reference channel: {ref_ch}<br>"
            f"Cable lat: {lat:.8f}<br>"
            f"Cable lon: {lon:.8f}<br>"
        )
        if depth is not None:
            popup_txt += f"Depth: {depth:.2f} m"

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_txt, max_width=320),
            tooltip=f"{loc_name} ref ch {ref_ch} ({cable_name})",
            icon=folium.Icon(color=marker_color, icon="info-sign")
        ).add_to(feature_group)

        add_text_marker(
            feature_group,
            lat,
            lon,
            f"{loc_name}\nref {ref_ch}",
            color=text_color,
            font_size="10pt"
        )


# ============================================================
# MAIN
# ============================================================
def main():
    # ----------------------------
    # Load data
    # ----------------------------
    tx_df = load_table(TX_FILE)
    est_cable_df = load_table(EST_CABLE_FILE)
    true_cable_df = load_table(TRUE_CABLE_FILE)

    # ----------------------------
    # Standardize numeric columns
    # ----------------------------
    for col in ["channel", "lat", "lon", "depth"]:
        if col in est_cable_df.columns:
            est_cable_df[col] = pd.to_numeric(est_cable_df[col], errors="coerce")

    # true cable has ch and z instead of channel and depth
    if "ch" in true_cable_df.columns:
        true_cable_df = true_cable_df.rename(columns={"ch": "channel"})
    if "z" in true_cable_df.columns:
        true_cable_df = true_cable_df.rename(columns={"z": "depth"})

    for col in ["channel", "lat", "lon", "depth"]:
        if col in true_cable_df.columns:
            true_cable_df[col] = pd.to_numeric(true_cable_df[col], errors="coerce")

    for col in [
        "reference_channel",
        "tx_lat_peak1", "tx_lon_peak1",
        "tx_lat_peak2", "tx_lon_peak2",
        "tx_depth_m"
    ]:
        if col in tx_df.columns:
            tx_df[col] = pd.to_numeric(tx_df[col], errors="coerce")

    est_cable_df = est_cable_df.dropna(subset=["channel", "lat", "lon"]).sort_values("channel").reset_index(drop=True)
    true_cable_df = true_cable_df.dropna(subset=["channel", "lat", "lon"]).sort_values("channel").reset_index(drop=True)

    # ----------------------------
    # Create map
    # ----------------------------
    center_lat, center_lon = get_map_center(
        (est_cable_df, "lat", "lon"),
        (true_cable_df, "lat", "lon"),
        (tx_df, "tx_lat_peak1", "tx_lon_peak1"),
        (tx_df, "tx_lat_peak2", "tx_lon_peak2"),
    )

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
    fg_est_cable = folium.FeatureGroup(name="Estimated cable line", show=True)
    fg_est_channel_labels = folium.FeatureGroup(name=f"Estimated cable: every {CHANNEL_LABEL_STEP}th channel", show=True)
    fg_est_reference = folium.FeatureGroup(name="Estimated cable: reference channels", show=True)

    fg_true_cable = folium.FeatureGroup(name="True cable line", show=True)
    fg_true_channel_labels = folium.FeatureGroup(name=f"True cable: every {CHANNEL_LABEL_STEP}th channel", show=True)
    fg_true_reference = folium.FeatureGroup(name="True cable: reference channels", show=True)

    fg_tx1 = folium.FeatureGroup(name="Transmission positions - Sweep 1", show=True)
    fg_tx2 = folium.FeatureGroup(name="Transmission positions - Sweep 2", show=True)

    # ----------------------------
    # Plot estimated cable
    # ----------------------------
    est_cable_points = add_cable_line(
        fg_est_cable,
        est_cable_df,
        color="blue",
        tooltip_text="Estimated cable"
    )

    add_channel_markers(
        fg_est_channel_labels,
        est_cable_df,
        step=CHANNEL_LABEL_STEP,
        color="darkblue",
        label_prefix="Estimated "
    )

    add_reference_markers(
        fg_est_reference,
        tx_df,
        est_cable_df,
        marker_color="red",
        text_color="red",
        cable_name="Estimated cable"
    )

    # ----------------------------
    # Plot true cable
    # ----------------------------
    true_cable_points = add_cable_line(
        fg_true_cable,
        true_cable_df,
        color="orange",
        tooltip_text="True cable"
    )

    add_channel_markers(
        fg_true_channel_labels,
        true_cable_df,
        step=CHANNEL_LABEL_STEP,
        color="orange",
        label_prefix="True "
    )

    add_reference_markers(
        fg_true_reference,
        tx_df,
        true_cable_df,
        marker_color="orange",
        text_color="darkorange",
        cable_name="True cable"
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
    fg_est_cable.add_to(m)
    fg_est_channel_labels.add_to(m)
    fg_est_reference.add_to(m)

    fg_true_cable.add_to(m)
    fg_true_channel_labels.add_to(m)
    fg_true_reference.add_to(m)

    fg_tx1.add_to(m)
    fg_tx2.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # ----------------------------
    # Fit bounds
    # ----------------------------
    all_points = []
    all_points.extend(est_cable_points)
    all_points.extend(true_cable_points)

    if all_points:
        m.fit_bounds(all_points)

    # ----------------------------
    # Save
    # ----------------------------
    m.save(str(OUTPUT_HTML))
    print(f"Saved map to: {OUTPUT_HTML}")


if __name__ == "__main__":
    main()