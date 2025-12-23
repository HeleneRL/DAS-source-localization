#!/usr/bin/env python
import json
from pathlib import Path
from functools import reduce

import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union


'''
right now this script is a stand alone script to find centroids of intersection regions from ellipse bands for multiple subarrays.
It reads in ellipse band data from multiple subarrays (JSON files), computes the intersection of the bands for each packet, and extracts the centroids of the intersection regions.
The resulting centroids are saved to a JSON file.

this functuonality should eventually be integrated into a larger processing pipeline for DAS data analysis.

running this script requires having precomputed ellipse band data for multiple subarrays, stored in JSON files in a specified directory.

several paths are hardcoded and may need to be adjusted.



'''











SOURCE_DEPTH = -30.0   # obs obs



def band_polygon_enu(inner_points, outer_points):
    """
    inner_points, outer_points: arrays (Ni, 3), (No, 3) in ENU.
    Returns a *valid* Shapely Polygon in ENU (E, N), or None.
    """
    inner_points = np.asarray(inner_points, float)
    outer_points = np.asarray(outer_points, float)

    if inner_points.size == 0 or outer_points.size == 0:
        return None

    coords = []

    # outer boundary forward
    for e, n, _ in outer_points:
        coords.append((e, n))

    # inner boundary reversed
    for e, n, _ in inner_points[::-1]:
        coords.append((e, n))

    # close polygon
    if coords[0] != coords[-1]:
        coords.append(coords[0])

    if len(coords) < 4:
        return None

    poly = Polygon(coords)

    if not poly.is_valid:
        # print("Polygon invalid:", explain_validity(poly))  # debugging if you want
        poly = poly.buffer(0)  # attempt to clean

        if not poly.is_valid:
            # still broken → give up on this band
            return None

    return poly

def shapely_polygons_from_geom(geom):
    """Return list of Polygon objects from a Shapely geometry (Polygon/MultiPolygon/etc.)."""
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    # GeometryCollection etc.
    return [g for g in geom.geoms if isinstance(g, Polygon)]


def main():
    # Folders – adjust paths if needed
    base = Path(r"C:\Users\helen\Documents\PythonProjects\my-project\libs\resources")
    info_folder = base / "new_cable_subarray_info"
    ellipse_folder = base / "new_cable_subarray_ellipses"

    # Collect matching info / ellipse files
    info_files = sorted(info_folder.glob("doa_info_start_ch_*_arrlen_*.json"))
    ellipse_files = sorted(ellipse_folder.glob("ellipse_bands_start_ch_*_arrlen_*.json"))

    if len(info_files) != len(ellipse_files):
        raise RuntimeError("info_files and ellipse_files count mismatch")

    print("Subarrays used for track reconstruction:")
    for inf, ell in zip(info_files, ellipse_files):
        print(f"  info:    {inf.name}")
        print(f"  ellipse: {ell.name}")
    print("")

    # Load all ellipse dicts in memory
    ellipse_dicts = []
    for ell_path in ellipse_files:
        with ell_path.open("r", encoding="utf-8") as f:
            ellipse_dicts.append(json.load(f))

    # Find common packet keys across all subarrays
    packet_key_sets = [set(d.keys()) for d in ellipse_dicts]
    common_packets = sorted(set.intersection(*packet_key_sets), key=int)

    print(f"Common packets across all subarrays: {len(common_packets)}")

    track = {}  # packet_idx (str) -> [E, N, U]

    for pkt_key in common_packets:
        pkt_idx = int(pkt_key)

        # Build band polygon for each subarray
        band_polys = []
        for ell_dict in ellipse_dicts:
            entry = ell_dict.get(pkt_key, None)
            if entry is None:
                band_polys = []
                break

            inner = entry.get("inner_points", [])
            outer = entry.get("outer_points", [])
            poly = band_polygon_enu(inner, outer)
            if poly is None or poly.is_empty:
                band_polys = []
                break
            band_polys.append(poly)

        if not band_polys:
            continue

        # All are valid polygons now, but we can still be extra-safe:
        for i, p in enumerate(band_polys):
            if not p.is_valid:
                band_polys[i] = p.buffer(0)
                if not band_polys[i].is_valid:
                    band_polys = []
                    break

        if not band_polys:
            continue

        # Now intersection should be robust
        inter = reduce(lambda a, b: a.intersection(b), band_polys)
        if inter.is_empty:
            continue


       
        polys = shapely_polygons_from_geom(inter)
        if not polys:
            continue

        polys_sorted = sorted(polys, key=lambda p: p.area, reverse=True)

        # Take up to two largest regions
        selected_polys = polys_sorted[:2]

        centroid_list = []
        for poly in selected_polys:
            E, N = poly.centroid.coords[0]
            centroid_list.append([float(E), float(N), float(SOURCE_DEPTH)])

        # Store ALL selected centroids for this packet
        track[pkt_key] = centroid_list


    print(f"Track points estimated for {len(track)} packets.")

    # Save to JSON
    out_path = base / "new_cable_track_estimate_from_doa.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(track, f, indent=2)

    print(f"Saved track estimate to: {out_path}")


if __name__ == "__main__":
    main()
