# libs/dasprocessor/scripts/find_good_packets.py
from pathlib import Path
from dasprocessor.bearing_tools import (
    load_merged_arrivals,
    build_subarrays,
    get_cached_channel_gps_for_run,
    packet_coverage_counts,
    select_packets_by_coverage,
)

def main():
    # ---------- Inputs ----------
    run_number = 2
    centers = [112, 124, 208, 268, 352, 364]
    aperture_len = 23
    band = "B_4"  # adjust if needed
    min_subarrays = 6   # how many subarrays must have detections for a packet to count
    limit = 100         # how many top packets to show

    # ---------- Load arrivals ----------
    pkg_dir = Path(__file__).resolve().parent.parent.parent
    merged_json = (pkg_dir / "resources" / band / f"peaks-merged-run{run_number}.json").resolve()
    print(f"Using merged arrivals file:\n  {merged_json}\n")

    arrivals = load_merged_arrivals(merged_json)
    gps = get_cached_channel_gps_for_run(run_number)
    subarrays = build_subarrays(centers, aperture_len, run_number)

    # ---------- Compute coverage ----------
    coverage = packet_coverage_counts(arrivals, subarrays)
    ranked_packets = select_packets_by_coverage(
        arrivals, subarrays, min_subarrays=min_subarrays, limit=limit
    )

    # ---------- Display ----------
    if not ranked_packets:
        print(f"No packets found that are visible in ≥{min_subarrays} subarrays.")
        return

    print(f"Packets visible in at least {min_subarrays} subarrays:\n")
    print(f"{'Packet':>8} | {'#Subarrays':>11}")
    print("-" * 25)
    for k in ranked_packets:
        print(f"{k:8d} | {coverage[k]:11d}")

    print(f"\nTotal candidate packets: {len(ranked_packets)}")

if __name__ == "__main__":
    main()
