from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


# =========================
# USER SETTINGS
# =========================
SWEEP_FILE = Path(r"D:\Singapore Data\transmission_times_sweeps.txt")
GPS_FILE = Path(r"D:\Singapore Data\tx gps\20260107-114125 - Lf comms track das 20260107.txt")
OUTPUT_FILE = Path(r"D:\Singapore Data\transmission_times_sweeps_with_tx_positions.csv")

TX_DEPTH_M = -3.0  # constant transmitter depth


# =========================
# TIME PARSING
# =========================
def parse_datetime(dt_str: str) -> datetime:
    """
    Parse timestamps that may or may not include fractional seconds.
    Examples:
        2026-01-07 04:36:36.999
        2026-01-07 04:36:24
        2026-01-07 04:00:14.220
    """
    dt_str = dt_str.strip()
    formats = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            pass
    raise ValueError(f"Could not parse datetime: {dt_str}")


# =========================
# GPS LOADING
# =========================
def load_transmitter_gps(gps_file: Path) -> list[dict]:
    """
    Load only transmitter rows (type == 'T') from the GPS file.
    Returns a list of dicts sorted by timestamp.
    """
    tx_points = []

    with gps_file.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_type = row["type"].strip()
            if row_type != "T":
                continue

            try:
                tx_points.append(
                    {
                        "time": parse_datetime(row["date time"]),
                        "latitude": float(row["latitude"]),
                        "longitude": float(row["longitude"]),
                        "altitude_m": float(row["altitude(m)"]) if row["altitude(m)"] else None,
                    }
                )
            except Exception as e:
                print(f"Skipping bad GPS row: {row}")
                print(f"Reason: {e}")

    tx_points.sort(key=lambda x: x["time"])
    return tx_points


def find_first_point_at_or_after(target_time: datetime, tx_points: list[dict]) -> dict | None:
    """
    Return the first transmitter GPS point with time >= target_time.
    Returns None if no such point exists.
    """
    for point in tx_points:
        if point["time"] >= target_time:
            return point
    return None


# =========================
# MAIN PROCESSING
# =========================
def process_sweeps(sweep_file: Path, gps_file: Path, output_file: Path) -> None:
    tx_points = load_transmitter_gps(gps_file)

    if not tx_points:
        raise RuntimeError("No transmitter GPS points (type 'T') were found in the GPS file.")

    with sweep_file.open("r", newline="", encoding="utf-8-sig") as f_in:
        reader = csv.DictReader(f_in)
        input_rows = list(reader)
        input_fields = reader.fieldnames or []

    extra_fields = [
        "tx_depth_m",
        "tx_time_peak1",
        "tx_lat_peak1",
        "tx_lon_peak1",
        "tx_altitude_m_peak1",
        "tx_time_peak2",
        "tx_lat_peak2",
        "tx_lon_peak2",
        "tx_altitude_m_peak2",
    ]

    output_fields = input_fields + extra_fields
    output_rows = []

    for row in input_rows:
        try:
            peak1_time = parse_datetime(row["utc_peak1"])
            peak2_time = parse_datetime(row["utc_peak2"])
        except Exception as e:
            print(f"Skipping row due to bad peak time: {row}")
            print(f"Reason: {e}")
            continue

        p1 = find_first_point_at_or_after(peak1_time, tx_points)
        p2 = find_first_point_at_or_after(peak2_time, tx_points)

        out_row = dict(row)
        out_row["tx_depth_m"] = TX_DEPTH_M

        if p1 is not None:
            out_row["tx_time_peak1"] = p1["time"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            out_row["tx_lat_peak1"] = f"{p1['latitude']:.8f}"
            out_row["tx_lon_peak1"] = f"{p1['longitude']:.8f}"
            out_row["tx_altitude_m_peak1"] = "" if p1["altitude_m"] is None else f"{p1['altitude_m']:.3f}"
        else:
            out_row["tx_time_peak1"] = ""
            out_row["tx_lat_peak1"] = ""
            out_row["tx_lon_peak1"] = ""
            out_row["tx_altitude_m_peak1"] = ""

        if p2 is not None:
            out_row["tx_time_peak2"] = p2["time"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            out_row["tx_lat_peak2"] = f"{p2['latitude']:.8f}"
            out_row["tx_lon_peak2"] = f"{p2['longitude']:.8f}"
            out_row["tx_altitude_m_peak2"] = "" if p2["altitude_m"] is None else f"{p2['altitude_m']:.3f}"
        else:
            out_row["tx_time_peak2"] = ""
            out_row["tx_lat_peak2"] = ""
            out_row["tx_lon_peak2"] = ""
            out_row["tx_altitude_m_peak2"] = ""

        output_rows.append(out_row)

    with output_file.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Done. Wrote output to:\n{output_file}")


if __name__ == "__main__":
    process_sweeps(SWEEP_FILE, GPS_FILE, OUTPUT_FILE)