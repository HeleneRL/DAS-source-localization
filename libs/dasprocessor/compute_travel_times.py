from __future__ import annotations

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List, Optional

from dasprocessor.constants import get_run  


'''

NOT CURRENTLY IN USE!

This script computes travel times for DAS data based on arrival times stored in a JSON file.

used for debugging, at an early stage, not part of any pipeline.









no other scripts depend on this one.

'''
















DATE_STR = "2024-05-03"  


def _parse_iso_dt(s: str) -> datetime:
    """Parse 'YYYY-MM-DD HH:MM:SS.ffffff' strings from the CSV."""
    return datetime.fromisoformat(s)


def load_source_points(csv_path: Path) -> List[Tuple[float, float, datetime]]:
    """
    Load all (lat, lon, datetime) points from the source-position CSV.
    Not strictly needed for travel-time computation, but handy to have.
    """
    out: List[Tuple[float, float, datetime]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                dt = _parse_iso_dt(row["datetime"])
                lat = float(row["lat"])
                lon = float(row["lon"])
                out.append((lat, lon, dt))
            except Exception:
                continue
    return out


def build_tx_sample_indices(
    date_str: str,
    run_number: int,
) -> Tuple[List[int], float]:
    """
    For a given run, build the transmit *sample index* for each packet.

    We work on the DAS time axis in samples:

        sample 0 ........
                 ^
                 start of DAS file

    For a particular run we know:
      - run["time_range"][0]: wall-clock time of the run start
      - run["offset_in_samples"]: sample index in DAS file where the run starts
      - run["sequence_start"]: wall-clock time of the first transmission
      - run["sequence_period"]: separation between packets, in *samples*
      - run["sample_rate"]: DAS sample rate, Hz

    So:

      t_run_start_wall  = time_range[0]   (datetime)
      t_seq_start_wall  = sequence_start  (datetime)
      dt_sec            = (t_seq_start_wall - t_run_start_wall).total_seconds()
      dt_samples        = dt_sec * sample_rate

      tx0_sample = offset_in_samples + dt_samples
      txk_sample = tx0_sample + k * sequence_period
    """
    run = get_run(date_str, run_number)

    fs = float(run["sample_rate"])  # Hz
    offset_in_samples = int(run["offset_in_samples"])
    seq_period = int(run["sequence_period"])
    seq_count = int(run["sequence_count"])

    # Wall-clock start of the run
    (h0, m0, s0), _ = run["time_range"]
    run_start_dt = datetime.fromisoformat(
        f"{date_str} {h0:02d}:{m0:02d}:{s0:02d}"
    )

    # Wall-clock time of first transmission
    (hs, ms, ss) = run["sequence_start"]
    seq_start_dt = datetime.fromisoformat(
        f"{date_str} {hs:02d}:{ms:02d}:{ss:02d}"
    )

    # Time between run start and first transmission, in seconds and samples
    dt_sec = (seq_start_dt - run_start_dt).total_seconds()
    dt_samples = int(round(dt_sec * fs))

    # Sample index of first transmission
    tx0_sample = offset_in_samples + dt_samples

    # One index per packet
    tx_samples = [tx0_sample + i * seq_period for i in range(seq_count)]

    return tx_samples, fs


def load_arrival_times_json(json_path: Path) -> Dict[str, Dict[str, int]]:
    """
    Load the JSON with arrival times.
    Structure assumed: { "channel": { "packet": arrival_sample, ... }, ... }.
    Keys are strings in the file.
    """
    with json_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data


def average_travel_time_for_interval(
    arrivals: Dict[str, Dict[str, int]],
    tx_samples: List[int],
    fs: float,
    channel: int,
    pkt_start: int,
    pkt_end: int,
) -> Tuple[Optional[float], List[Tuple[int, float]]]:
    """
    Compute the average travel time (seconds) for a given channel and
    packet index interval [pkt_start, pkt_end] (inclusive).

    Returns:
      (average_time_seconds or None if no valid packets,
       list of (packet_index, travel_time_seconds) for all packets used)
    """
    ch_key = str(channel)
    if ch_key not in arrivals:
        raise ValueError(f"Channel {channel} not found in arrival JSON")

    ch_arrivals = arrivals[ch_key]

    times_sec: List[Tuple[int, float]] = []

    for k in range(pkt_start, pkt_end + 1):
        pkt_key = str(k)
        if pkt_key not in ch_arrivals:
            # missing arrival for this packet → skip
            continue
        if k >= len(tx_samples):
            # No TX time for this index (shouldn't normally happen)
            continue

        arrival_sample = int(ch_arrivals[pkt_key])
        tx_sample = int(tx_samples[k])

        travel_samples = arrival_sample - tx_sample
        travel_sec = travel_samples / fs
        times_sec.append((k, travel_sec))

    if not times_sec:
        return None, []

    avg = sum(t for _, t in times_sec) / len(times_sec)
    return avg, times_sec


def main():
    # --- user parameters ---
    run_number = 2
    channel = 160                 # e.g. sensor right at the crossing
    interval1 = (60, 70)           # inclusive
    interval2 = (70, 80)          # inclusive

    json_path = Path(r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\B_4\peaks-merged-all_hilbert_channels.json")          
    src_csv_path = Path(r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources\source-position.csv")     
    # ------------------------

    # Load data
    print("Loading arrival times JSON...")
    arrivals = load_arrival_times_json(json_path)

    print("Loading source-position CSV (optional)...")
    _ = load_source_points(src_csv_path)  # not used in the calc, but available

    print(f"Building TX sample indices for run {run_number} ...")
    tx_samples, fs = build_tx_sample_indices(DATE_STR, run_number)

    # Interval 1
    avg1, list1 = average_travel_time_for_interval(
        arrivals, tx_samples, fs, channel, interval1[0], interval1[1]
    )

    # Interval 2
    avg2, list2 = average_travel_time_for_interval(
        arrivals, tx_samples, fs, channel, interval2[0], interval2[1]
    )

    print(f"\nSample rate fs = {fs:.1f} Hz")

    def pretty(interval, avg, vals):
        a, b = interval
        if avg is None:
            print(f"Packets {a}-{b}: no valid arrivals.")
            return
        print(f"Packets {a}-{b}:")
        print(f"  packets used: {[k for (k, _) in vals]}")
        print(f"  avg travel time: {avg:.6f} s")
        print(f"  per-packet times (s):")
        for k, t in vals:
            print(f"    pkt {k:3d}: {t:.6f}")

    print("\n=== Interval 1 ===")
    pretty(interval1, avg1, list1)

    print("\n=== Interval 2 ===")
    pretty(interval2, avg2, list2)


if __name__ == "__main__":
    main()
