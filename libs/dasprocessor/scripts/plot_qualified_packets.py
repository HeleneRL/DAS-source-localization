from __future__ import annotations
import subprocess
import sys  

#  List of packets to plot maps for
# run find_qualified_packets.py and make sure you only include actually qualified packets in this list, or you will get errors

QUALIFIED_PACKETS = [
    46, 48, 49,
    50, 51, 52, 53, 54, 55, 56, 57, 58, 60,
]




QUALIFIED_PACKETS_Gr2 = [ 
    0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 
    16, 17, 18, 19, 21, 22, 23, 24, 25, 26
]
    

QUALIFIED_PACKETS_all = [ 
    0, 1, 2, 6, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 25, 26,
    27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
    39, 40, 46, 47, 48, 49,
    50, 51, 52, 53, 55, 56, 57, 58, 60, 61,
    62, 63, 64, 65, 66, 67
]


QUALIFIED_PACKETS_test_gr = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    50, 51, 52, 53, 54, 55, 56, 57, 58,
    60, 61, 66
]









def main():
    for p in QUALIFIED_PACKETS:
        print(f"=== Building map for packet {p} ===")
        subprocess.run(
            [
                sys.executable,                 # <-- use current interpreter
                "-m",
                "dasprocessor.scripts.plot_map_v2",
                "--packet",
                str(p),
            ],
            check=True,
        )

if __name__ == "__main__":
    main()
