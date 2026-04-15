# DAS Inversion Pipeline (Refactored)

This folder contains a cleaned, modular version of the DAS arrival-to-cable-inversion pipeline.

## What changed

This refactor keeps the same core logic as the original codebase:

1. detect arrivals from DAS data,
2. build a channel trust map,
3. attach transmitter positions,
4. build a prior cable geometry,
5. merge everything into `inversion_observations.csv`,
6. invert the cable geometry.

But it improves several things:

- one shared config file for all stages,
- consistent output layout and filenames,
- better detector quality metrics,
- observation weighting uses both **pick-level quality** and **channel-level trust**,
- one main inversion script instead of two overlapping versions,
- more comments and clearer function boundaries.

## Folder layout

```text
config/
  pipeline_config.toml

src/
  common.py
  detector_bulk.py
  build_transmitter_positions.py
  build_prior_cable.py
  build_trust_map.py
  build_inversion_dataset.py
  invert_cable.py
  run_pipeline.py

outputs/
  raw_detections/
  transmitter/
  prior_geometry/
  trust/
  inversion_dataset/
  inversion/
```

## Main scripts

### `src/detector_bulk.py`
Reads DAS HDF5 sequences and writes per-channel arrival detections.

New pick-quality fields include:
- `peak_ratio_best_to_second`
- `peak_width_samples`
- `peak_width_ms`
- `timing_quality_score`
- `pick_quality_score`

These are intended to describe:
- contrast,
- ambiguity,
- peak sharpness,
- overall pick confidence.

### `src/build_trust_map.py`
Builds per-location and global trust summaries from the detector CSV.

### `src/build_transmitter_positions.py`
Matches sweep timestamps to transmitter GPS positions.

### `src/build_prior_cable.py`
Interpolates a channel-indexed prior cable geometry from the boat-track and cable estimate.

### `src/build_inversion_dataset.py`
Merges arrivals, transmitter positions, trust summaries, and prior geometry into one inversion table.

This version uses:
- detector validity flags,
- detector quality score,
- trust score,
- recommendation flags,

to build a more continuous observation weight.

### `src/invert_cable.py`
Runs the cable inversion.

This combines the best parts of the original inversion scripts:
- robust weighted inversion,
- control-point selection by spacing or by quality,
- output geometry, diagnostics, and plots.

### `src/run_pipeline.py`
Convenience wrapper to run the whole chain in order.



### `src/make_thesis_plots.py`
Builds a final thesis-ready figure set from the inversion outputs.

It is designed as a **post-inversion visualization step** rather than part of the solver.
That means you can rerun it freely without recomputing the inversion.

Main features:
- channel labels every 100 channels on plan/depth plots,
- labeled transmitters in plan view,
- observed-vs-predicted plots using a **high-confidence subset**,
- side-by-side prior vs inverted timing plots,
- residual envelope plots instead of unreadable full scatter clouds,
- uncertainty tube around the inverted cable derived from residual misfit.






## Typical use

From this folder:

```bash
python src/run_pipeline.py --config config/pipeline_config.toml
```

Or run individual steps:

```bash
python src/detector_bulk.py --config config/pipeline_config.toml
python src/build_transmitter_positions.py --config config/pipeline_config.toml
python src/build_prior_cable.py --config config/pipeline_config.toml
python src/build_trust_map.py --config config/pipeline_config.toml
python src/build_inversion_dataset.py --config config/pipeline_config.toml
python src/invert_cable.py --config config/pipeline_config.toml
python src/make_thesis_plots.py --config config/pipeline_config.toml
```

## Dependencies

Core:
- numpy
- pandas
- scipy
- matplotlib
- h5py
- pymap3d

Optional:
- pyproj (for lat/lon output on inverted cable)
- tomli on Python < 3.11

## Notes

- This refactor is meant to be a **clean operational baseline**, not a full research framework.
- The biggest design change is that the inversion dataset now uses **pick-level quality** more explicitly instead of relying almost entirely on channel-level trust.
- I removed parameter-tuning/reporting/QC side scripts from the core pipeline because they are downstream consumers, not necessary to produce the inversion result itself.


## Final thesis-style plots

After the inversion is finished, run:

```bash
python src/make_thesis_plots.py --config config/pipeline_config.toml
```

By default this reads:
- `outputs/inversion_dataset/inversion_observations.csv` via the configured inversion-dataset path,
- inversion outputs from the configured inversion output directory,
- the focalization / truth-like geometry from `cable_estimate_csv` in the config.

It writes a new folder inside the inversion output directory:

```text
thesis_plots/
```

with files such as:
- `figure_planview_tube.png`
- `figure_depth_profile.png`
- `figure_observed_vs_predicted_prior_vs_inverted.png`
- `figure_relative_residual_envelope.png`
- `figure_fit_histograms_highconf.png`
- `figure_horizontal_shift.png`
- `figure_distance_to_focalization_vs_channel.png`
- `figure_control_quality_and_points.png`

Useful optional flags:

```bash
python src/make_thesis_plots.py   --config config/pipeline_config.toml   --min_weight 0.8   --label_every 100
```

### README notes worth adding on your side

These are the user-facing details I would explicitly document:
- The script is a **post-inversion** step and does not change the solution.
- `--min_weight` controls the high-confidence subset used in the observed-vs-predicted and histogram plots.
- `--label_every` controls how often channels are annotated on geometry plots.
- The uncertainty tube is a **diagnostic visualization derived from relative residual misfit**, not a formal posterior confidence interval.
- The focalization / truth-like geometry is taken from `cable_estimate_csv` unless overridden with `--truth_csv`.
- If the user wants all plots regenerated after styling changes, only `make_thesis_plots.py` needs to be rerun.

- This refactor is meant to be a **clean operational baseline**, not a full research framework.
- The biggest design change is that the inversion dataset now uses **pick-level quality** more explicitly instead of relying almost entirely on channel-level trust.
- I removed parameter-tuning/reporting/QC side scripts from the core pipeline because they are downstream consumers, not necessary to produce the inversion result itself.

