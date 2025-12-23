#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import Set, Dict, Any, List



'''

this script finds packet indices that are marked as valid and have ellipse points in all subarray info/ellipse JSON files. 

mostly useful for filtering packets for further processing or analysis.

should be combined with the plotting scripts to visualize only the qualified packets... 



no other scripts depend on this one directly, but you need to make sure to only plot packets that are actually qualified. if else you will get errors.



'''


def load_packet_set_for_subarray(info_path: Path, ellipse_path: Path) -> Set[int]:
    """
    For one subarray:
      - load DOA info JSON
      - load ellipse bands JSON
      - return the set of packet indices that:
          * have info['valid'] == True
          * have info['ellipse_has_points'] == True
          * have non-empty 'nominal_points' in ellipse JSON
    """
    with info_path.open("r", encoding="utf-8") as f:
        info_dict: Dict[str, Any] = json.load(f)

    with ellipse_path.open("r", encoding="utf-8") as f:
        ellipse_dict: Dict[str, Any] = json.load(f)

    good_packets: Set[int] = set()

    for pkt_key, pkt_info in info_dict.items():
        # Only consider packets marked valid and with ellipse_has_points
        if not pkt_info.get("valid", False):
            continue
        if not pkt_info.get("ellipse_has_points", False):
            continue

        # Make sure the packet exists in ellipse file too
        ell_entry = ellipse_dict.get(pkt_key)
        if ell_entry is None:
            continue

        nominal = ell_entry.get("nominal_points", [])
        if not nominal:  # empty list -> no ellipse points
            continue

        # Use JSON key as packet index
        try:
            pkt_idx = int(pkt_key)
        except ValueError:
            # fallback: use stored packet_idx if key is weird
            pkt_idx = int(pkt_info.get("packet_idx"))

        good_packets.add(pkt_idx)

    return good_packets


def collect_json_files(folder: Path, prefix: str) -> List[Path]:
    """
    Return a sorted list of JSON files in 'folder' whose name starts with 'prefix'.
    """
    return sorted(
        p for p in folder.glob("*.json") if p.name.startswith(prefix)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Find packet indices that are valid and have ellipse points "
            "in all subarray info/ellipse JSON files."
        )
    )
    parser.add_argument(
        "info_folder",
        type=str,
        help="Folder containing doa_info_*.json files (subarray_info).",
    )
    parser.add_argument(
        "ellipse_folder",
        type=str,
        help="Folder containing ellipse_bands_*.json files (subarray_ellipses).",
    )
    args = parser.parse_args()

    info_folder = Path(args.info_folder).resolve()
    ellipse_folder = Path(args.ellipse_folder).resolve()

    if not info_folder.is_dir():
        raise NotADirectoryError(f"info_folder is not a directory: {info_folder}")
    if not ellipse_folder.is_dir():
        raise NotADirectoryError(f"ellipse_folder is not a directory: {ellipse_folder}")

    # Collect JSON files
    info_files = collect_json_files(info_folder, prefix="doa_info_")
    ellipse_files = collect_json_files(ellipse_folder, prefix="ellipse_bands_")

    if not info_files:
        raise RuntimeError(f"No doa_info_*.json files found in {info_folder}")
    if not ellipse_files:
        raise RuntimeError(f"No ellipse_bands_*.json files found in {ellipse_folder}")

    # Simple pairing by sorted order (names share start_ch & arrlen pattern)
    if len(info_files) != len(ellipse_files):
        raise RuntimeError(
            f"Number of info files ({len(info_files)}) != number of ellipse files ({len(ellipse_files)})"
        )

    print("Found subarrays:")
    for inf, ell in zip(info_files, ellipse_files):
        print(f"  info:    {inf.name}")
        print(f"  ellipse: {ell.name}")
        print("")

    # For each subarray: compute the set of "good" packets
    good_sets: List[Set[int]] = []
    for inf, ell in zip(info_files, ellipse_files):
        good = load_packet_set_for_subarray(inf, ell)
        print(f"{inf.name}: {len(good)} packets are valid AND have ellipse points")
        good_sets.append(good)

    if not good_sets:
        print("\nNo subarrays processed.")
        return

    # Intersection across all subarrays:
    common_good = set.intersection(*good_sets)

    if not common_good:
        print("\nNo packet index is valid+ellipse in ALL subarrays.")
        return

    print("\nPackets valid+ellipse in ALL subarrays:")
    for pkt_idx in sorted(common_good):
        print(pkt_idx)


if __name__ == "__main__":
    main()
