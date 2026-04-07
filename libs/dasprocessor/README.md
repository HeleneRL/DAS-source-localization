# Singapore DAS bulk detector

Files:
- `singapore_das_config.toml` – editable config with paths, anchors, and detector settings.
- `run_das_bulk_detector.py` – bulk matched-filter script.

## What it does
For each location and each anchor time:
1. Builds a 3.5–4.5 kHz, 5 s LFM reference.
2. Reads only the raw time window needed for a ±2 s search around the anchor.
3. Uses matched filtering on channels `0:3000`.
4. Saves one best peak per channel per anchor.

## Important detail
A ±2 s **search window** does **not** mean a 4 s raw-data read. To search candidate chirp start times in `[anchor-2 s, anchor+2 s]`, the script must read:

`search_before + search_after + chirp_duration = 2 + 2 + 5 = 9 s`

This is why the script reads a 9 s raw window and then the valid matched-filter output spans the intended 4 s search interval.

## Run
From the folder containing the files:

```bash
python run_das_bulk_detector.py --config singapore_das_config.toml
```

## Output columns
Main outputs are CSV files with one row per `(location, anchor, channel)` including:
- `peak_global_sample`
- `peak_time_s_from_sequence_start`
- `peak_time_utc`
- `peak_time_local_sg`
- `peak_raw_envelope`
- `peak_prominence_raw`
- `snr_like`
- `passed_snr_threshold`
- `near_window_edge`

## Notes
- `snr_like` is a robust score based on `(peak - median) / MAD` of the envelope in the local search window.
- `near_window_edge = true` is a warning flag that the best peak landed very close to the search-boundary.
- If you want to save only channels above a confidence threshold, filter on `passed_snr_threshold` after the run.
