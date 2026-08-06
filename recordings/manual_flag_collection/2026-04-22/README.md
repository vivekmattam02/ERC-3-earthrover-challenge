# Manual Flag Collection - 2026-04-22

This folder contains manual rover data collection runs captured through the local SDK recorder during no-GPS testing.

## Runs

- `run_01_1521.h5`
  - summary: `run_01_1521.summary.json`
  - telemetry rows: `1829`
  - front frames: `1817`
  - rear frames: `0`

- `run_02_1537.h5`
  - summary: `run_02_1537.summary.json`
  - telemetry rows: `1019`
  - front frames: `1019`
  - rear frames: `0`

- `run_03_1551.h5`
  - summary: `run_03_1551.summary.json`
  - telemetry rows: `1304`
  - front frames: `1291`
  - rear frames: `0`

## Notes

- These runs were recorded with `scripts/record_sdk_session.py`.
- GPS was not usable in this session; these files are intended for visual/manual data analysis.
- Manual teleop commands were not recorded, so `controls` is expected to be empty.
