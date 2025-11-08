"""
-------------------
Merge multiple peaks-*.json files (packet detections) into one combined map.

Each input JSON must be of the form:
    { "channel": { "packet_index": sample_index, ... }, ... }

Output:
    A single JSON containing all channels merged together:
        arrivals[channel][packet_index] = sample_index
"""

import json
from pathlib import Path
from collections import defaultdict


def load_and_merge_arrivals(files):
    """
    Merge many peaks-*.json files into one map:
    arrivals[channel][packet_idx] = sample_index
    """
    arrivals = defaultdict(dict)  # channel -> {packet_idx: sample}
    for f in files:
        print(f"Loading {f.name} ...")
        with open(f, "r") as fh:
            data = json.load(fh)

        for ch_str, pk_map in data.items():
            ch = int(ch_str)
            for pk_str, sample in pk_map.items():
                pk = int(pk_str)
                # Keep earliest sample if duplicates appear
                if pk not in arrivals[ch]:
                    arrivals[ch][pk] = int(sample)
                else:
                    arrivals[ch][pk] = min(arrivals[ch][pk], int(sample))
    return dict(arrivals)


def main():
    # --- Configure paths ---
    root = (Path(__file__).resolve().parent.parent / "resources" / "B_4")

    # List all your run-2 JSON files here
    files = [
        root / "peaks-100-112-run2.json",
        root / "peaks-112-124-run2.json",
        root / "peaks-124-136-run2.json",
        root / "peaks-196-208-run2.json",
        root / "peaks-208-220-run2.json",
        root / "peaks-256-268-run2.json",
        root / "peaks-268-280-run2.json",
        root / "peaks-340-352-run2.json",
        root / "peaks-352-364-run2.json",
        root / "peaks-364-376-run2.json",
    ]

    # --- Merge ---
    arrivals = load_and_merge_arrivals(files)
    print(f"\n✅ Merged {len(arrivals)} channels")

    # --- Save merged file ---
    merged_path = root / "peaks-merged-run2.json"
    with open(merged_path, "w") as fh:
        json.dump(
            {str(ch): {str(pk): int(sm) for pk, sm in pks.items()} for ch, pks in arrivals.items()},
            fh,
            indent=2,
        )

    print(f"💾 Saved merged file to:\n{merged_path}")


if __name__ == "__main__":
    main()
