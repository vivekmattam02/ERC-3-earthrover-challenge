# ERC-3 EarthRover Challenge

[![NYU](https://img.shields.io/badge/NYU-EarthRover-57068C)](#erc-3-earthrover-challenge)
[![Indoor](https://img.shields.io/badge/Indoor-MBRA%20%2B%20Corridor%20Graph-1f6feb)](#indoor-track)
[![Outdoor](https://img.shields.io/badge/Outdoor-LogoNav%20%2B%20OSM%20Routing-0a7f5a)](#outdoor-track)
[![Docs](https://img.shields.io/badge/Docs-Index-c26d00)](docs/INDEX.md)

This repository contains the indoor, outdoor, and marathon navigation code used
for the NYU EarthRover Challenge project.

> **Current no-GPS status:** The repository also contains a rough-terrain,
> front-camera teach-and-repeat branch. Its recording and route-preparation
> pipeline works, but recent physical tests did **not** demonstrate a reliable
> autonomous repeat. Treat it as supervised experimental work, not a
> competition-ready deployment path. The current evidence is in
> [`No-GPS Field Trial - Findings`](obsidian_vault/01%20Source%20of%20Truth/No-GPS%20Field%20Trial%20-%20Findings.md).
> Start with the shorter [Current No-GPS - Read This First](obsidian_vault/00%20Home/Current%20No-GPS%20-%20Read%20This%20First.md) card.

The stack is deliberately split:

- **Indoor:** CosPlace-style corridor localization, temporal stabilization, graph
  progression, MBRA as the local controller, and depth veto logic.
- **Outdoor:** SDK mission checkpoints, optional OpenStreetMap route expansion,
  LogoNav as the local controller, and a layered safety envelope around it.

The main research ingredients behind the project are:

- [CosPlace](https://github.com/gmberton/CosPlace) for visual place recognition
- [MBRA / Model-Based Re-Annotation](https://model-base-reannotation.github.io/) for short-horizon learned visual control ideas
- [LogoNav](https://openreview.net/forum?id=9DyLaIHqrD) for outdoor learned navigation
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) for monocular depth estimation
- [SegFormer](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512) for semantic segmentation experiments
- [OpenStreetMap](https://www.openstreetmap.org/) for pedestrian-route expansion

## Competition Results

- **Indoor:** 8 / 11 checkpoints reached
- **Outdoor:** one full success, roughly 50% success overall
- **Marathon:** one checkpoint, then the rover toppled during a bad transition

The full race-day writeup is in [competition_results.tex](competition_results.tex).

## Quick Start

```bash
cd ERC-3-earthrover-challenge
cp earth-rovers-sdk/.env.sample earth-rovers-sdk/.env
python3 verify_workspace.py
```

Then:

1. activate the local `erv` environment,
2. place the required model weights in their expected directories,
3. verify the SDK bridge and camera feed,
4. read [docs/INDEX.md](docs/INDEX.md) and the relevant current source-of-truth note,
5. read [No-GPS Field Trial - Findings](obsidian_vault/01%20Source%20of%20Truth/No-GPS%20Field%20Trial%20-%20Findings.md) before rough-terrain repeat work,
6. choose the indoor, outdoor, or no-GPS experimental entrypoint.

## Main Components

| Area | Key files |
|---|---|
| Indoor runtime | [`live_indoor_runtime.py`](live_indoor_runtime.py) |
| Legacy indoor variants | [`legacy/README.md`](legacy/README.md) |
| Outdoor runtime | [`live_outdoor_runtime.py`](live_outdoor_runtime.py) |
| Indoor localization and graph logic | [`src/corridor_localizer.py`](src/corridor_localizer.py), [`src/temporal_localization.py`](src/temporal_localization.py), [`src/graph_planner.py`](src/graph_planner.py) |
| Controllers | [`src/mbra_controller.py`](src/mbra_controller.py), [`src/adaptive_pursuit_controller.py`](src/adaptive_pursuit_controller.py), [`src/outdoor_logonav_controller.py`](src/outdoor_logonav_controller.py), [`src/outdoor_gps_controller.py`](src/outdoor_gps_controller.py) |
| Safety and perception | [`src/depth_estimator.py`](src/depth_estimator.py), [`src/outdoor_traversability.py`](src/outdoor_traversability.py), [`src/semantic_risk_estimator.py`](src/semantic_risk_estimator.py), [`src/imu_safety.py`](src/imu_safety.py), [`src/vision_safety_monitor.py`](src/vision_safety_monitor.py) |
| Tools and diagnostics | [`scripts/`](scripts/), [`earth-rovers-sdk/`](earth-rovers-sdk/) |

### Indoor stack

- Visual place recognition and temporal stabilization:
  [`src/corridor_localizer.py`](src/corridor_localizer.py),
  [`src/temporal_localization.py`](src/temporal_localization.py)
- Corridor graph planning and checkpoint progression:
  [`src/graph_planner.py`](src/graph_planner.py)
- MBRA controller wrapper:
  [`src/mbra_controller.py`](src/mbra_controller.py)
- Indoor runtime entrypoints:
  [`live_indoor_runtime.py`](live_indoor_runtime.py)
- Superseded indoor experiments:
  [`legacy/README.md`](legacy/README.md)

### Outdoor stack

- Outdoor runtime:
  [`live_outdoor_runtime.py`](live_outdoor_runtime.py)
- LogoNav wrapper:
  [`src/outdoor_logonav_controller.py`](src/outdoor_logonav_controller.py)
- GPS controller:
  [`src/outdoor_gps_controller.py`](src/outdoor_gps_controller.py)
- OSM route expansion:
  [`src/osm_router.py`](src/osm_router.py)
- SDK / mission checkpoint interface:
  [`src/earthrover_interface.py`](src/earthrover_interface.py),
  [`earth-rovers-sdk/`](earth-rovers-sdk/)

### Safety and perception

- Depth estimator:
  [`src/depth_estimator.py`](src/depth_estimator.py)
- Outdoor traversability:
  [`src/outdoor_traversability.py`](src/outdoor_traversability.py)
- Semantic risk estimation:
  [`src/semantic_risk_estimator.py`](src/semantic_risk_estimator.py)
- Vision safety monitor:
  [`src/vision_safety_monitor.py`](src/vision_safety_monitor.py)
- IMU safety:
  [`src/imu_safety.py`](src/imu_safety.py)

### Scripts and evaluation

- Preflight:
  [`scripts/preflight_marathon.py`](scripts/preflight_marathon.py)
- Localization diagnostics:
  [`scripts/diagnose_localization.py`](scripts/diagnose_localization.py)
- Traversability calibration:
  [`scripts/calibrate_traversability.py`](scripts/calibrate_traversability.py)
- Semantic probes:
  [`scripts/probe_semantics.py`](scripts/probe_semantics.py),
  [`scripts/semantic_second_pass.py`](scripts/semantic_second_pass.py)

## Read This First

- [No-GPS Field Trial - Findings](obsidian_vault/01%20Source%20of%20Truth/No-GPS%20Field%20Trial%20-%20Findings.md) — current no-GPS field truth
- [no_gps_route_repeat_story.tex](no_gps_route_repeat_story.tex) — full no-GPS technical report
- [CLAUDE.md](CLAUDE.md) — historical indoor-system reference
- [docs/INDEX.md](docs/INDEX.md) — documentation map
- [guide.md](guide.md) — practical run guide
- [final_system_overview.tex](final_system_overview.tex) — one-document architecture overview

## Repository Layout

| Path | Purpose |
|---|---|
| [`src/`](src/) | shared controllers, localization, safety, and runtime modules |
| [`scripts/`](scripts/) | diagnostics, calibration, probes, and preflight utilities |
| [`earth-rovers-sdk/`](earth-rovers-sdk/) | browser/FastAPI bridge for the rover |
| [`mbra_repo/`](mbra_repo/) | optional MBRA / LogoNav workspace used only by explicit MBRA runs |
| [`mbra_repo_1/`](mbra_repo_1/) | historical image-memory research snapshot; not an active runtime dependency |
| [`legacy/`](legacy/) | superseded indoor runtime and controller snapshots |
| [`third_party/Depth-Anything-V2/`](third_party/Depth-Anything-V2/) | vendored depth dependency |
| [`models/`](models/) | local model checkpoint directory |
| [`docs/`](docs/) | technical reports, notes, and project documentation |

## Indoor Track

The indoor track is a known-corridor checkpoint problem. The current indoor
runtime is built around:

- corridor localization from recorded data,
- temporal stabilization over recent VPR outputs,
- graph progression over exact checkpoint steps,
- MBRA as the local controller,
- depth safety as a veto layer.

Main entrypoints:

- [`live_indoor_runtime.py`](live_indoor_runtime.py)

The MBRA-first and heavy-recovery forks are retained under
[`legacy/`](legacy/) for provenance only. New runs must use the maintained
runtime above.

Core indoor modules:

- [`src/corridor_localizer.py`](src/corridor_localizer.py)
- [`src/temporal_localization.py`](src/temporal_localization.py)
- [`src/graph_planner.py`](src/graph_planner.py)
- [`src/navigation_runtime.py`](src/navigation_runtime.py)
- [`src/mbra_controller.py`](src/mbra_controller.py)

## No-GPS Rough-Terrain Branch

This branch reuses visual localization and graph progression for a manually
taught route. The live runner uses the **front camera only**. The relevant
entrypoints are [`scripts/record_sdk_session.py`](scripts/record_sdk_session.py),
[`scripts/prepare_manual_route.py`](scripts/prepare_manual_route.py),
[`scripts/run_prepared_route.py`](scripts/run_prepared_route.py), and
[`src/adaptive_pursuit_controller.py`](src/adaptive_pursuit_controller.py).

The strongest reference candidate is `data/manual_routes/smoke_run01_c`, but
it is not field-proven. Read the field-trial note before using this branch with
`--send-control`.

## Outdoor Track

The outdoor track is a mission-checkpoint runtime. The current outdoor system is
built around:

- checkpoints from the SDK,
- optional OSM-expanded intermediate waypoints,
- LogoNav as the local motion policy,
- traversability, IMU, GPS, vision, and route-guard safety layers.

Main entrypoint:

- [`live_outdoor_runtime.py`](live_outdoor_runtime.py)

Core outdoor modules:

- [`src/outdoor_logonav_controller.py`](src/outdoor_logonav_controller.py)
- [`src/outdoor_gps_controller.py`](src/outdoor_gps_controller.py)
- [`src/osm_router.py`](src/osm_router.py)
- [`src/outdoor_traversability.py`](src/outdoor_traversability.py)
- [`src/semantic_risk_estimator.py`](src/semantic_risk_estimator.py)
- [`src/imu_safety.py`](src/imu_safety.py)
- [`src/vision_safety_monitor.py`](src/vision_safety_monitor.py)

## Setup

### 1. Enter the workspace

```bash
cd ERC-3-earthrover-challenge
```

### 2. Create the local SDK config

```bash
cp earth-rovers-sdk/.env.sample earth-rovers-sdk/.env
```

Then edit your local `.env` using [`earth-rovers-sdk/.env.sample`](earth-rovers-sdk/.env.sample) as the template.

Typical fields include:
- `SDK_API_TOKEN`
- `BOT_SLUG`
- `MISSION_SLUG`
- browser path and local SDK settings

The real `.env` file is intentionally ignored by git.

### 3. Put model weights in the expected local directories

- MBRA / LogoNav weights:
  `mbra_repo/deployment/model_weights/`
- project-level checkpoints:
  `models/`
- Depth Anything V2 checkpoints:
  `third_party/Depth-Anything-V2/checkpoints/`

### 4. Verify the workspace

```bash
python3 verify_workspace.py
```

## Typical Commands

Indoor competition-style run:

```bash
python live_indoor_runtime.py \
  --checkpoint-steps 45 480 761 821 1094 1208 1345 1430 1544 1638 1764 \
  --auto-advance-checkpoints --send-control --controller mbra --depth-safety
```

Outdoor mission run:

```bash
python live_outdoor_runtime.py --mission --send-control --controller logonav --osm-route
```

Outdoor marathon run:

```bash
python live_outdoor_runtime.py \
  --mission --send-control --controller logonav --osm-route --ultra-marathon
```

More operational details live in [guide.md](guide.md).

## Practical Notes

- Keep `.h5` recordings, weights, and generated debug media out of git.
- Treat active runtime code and the vault's source-of-truth notes as the
  current truth layer.
- Treat older planning notes as historical context, not as the final system.
- The indoor and outdoor stacks share some modules, but they are not the same
  navigation problem and should not be forced into one controller story.

## What To Read First

If you want to understand the project quickly:
- start with this README
- inspect [`live_indoor_runtime.py`](live_indoor_runtime.py) for indoor behavior
- inspect [`live_outdoor_runtime.py`](live_outdoor_runtime.py) for outdoor behavior
- inspect [`src/mbra_controller.py`](src/mbra_controller.py) and [`src/outdoor_logonav_controller.py`](src/outdoor_logonav_controller.py) for the controller wrappers

If you want the engineering story behind the current code:
- go to [`docs/`](docs/)

## Documentation

Project reports and longer technical notes live in [`docs/`](docs/).

Important deeper writeups include:
- indoor story and runtime evolution
- outdoor runtime explanation
- marathon story
- semantic segmentation review
- perception and traversability review

The reports are useful if you want the engineering history, failure analysis, and design reasoning behind the current code.

## Working Rules

- Keep secrets out of git.
- Keep recorded `.h5` files and model checkpoints out of git.
- Keep generated debug images out of git.
- Use relative paths inside this workspace.
- Treat this repository as a standalone project workspace.

## Practical Notes

- The repo contains source code and wrappers, not the full large-data environment used during development.
- Some scripts assume local recorded runs or checkpoints exist outside version control.
- The runtime code is the source of truth for current behavior; the reports explain how that behavior evolved.
- The README is meant to help someone orient quickly; the deeper reports are where the design history and failure analysis live.
