# libs/dasprocessor/doa_results_io.py
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence


@dataclass
class DoaResult:
    packet: int
    center_lat: float
    center_lon: float
    dir_A_enu: Sequence[float]   # [E, N, U]
    dir_B_enu: Sequence[float]   # [E, N, U]
    channels_min: int
    channels_max: int
    n_channels: int


def append_doa_result(path: str | Path, result: DoaResult) -> None:
    """
    Append a DOA result to a JSON file storing a list of results.
    Creates the file if it doesn't exist.
    """
    path = Path(path)

    if path.exists():
        with path.open("r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(asdict(result))

    with path.open("w") as f:
        json.dump(data, f, indent=2)
