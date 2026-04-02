# ERC-3 EarthRover Challenge

This repository contains the working indoor and outdoor autonomy stack used for the NYU EarthRover Challenge effort. It is a self-contained workspace for running the rover, testing controller changes, hardening the runtime, and evaluating safety layers without depending on files outside this repo.

The project ended up splitting naturally into two different problems:
- indoor corridor navigation with strong map structure and visual localization
- outdoor mission navigation with live checkpoints, routed waypoints, and stricter safety handling

## What This Repository Is For

This repo was used to:
- run the indoor checkpoint pipeline with corridor localization, graph progression, and MBRA
- run the outdoor mission pipeline with LogoNav, mission checkpoints, OSM-expanded routes, and runtime safety layers
- test safety modules such as depth-based traversability, IMU protection, and semantic risk estimation
- keep the code, SDK bridge, wrappers, scripts, and supporting docs in one place

It is not intended to store large recorded datasets, model checkpoints, or generated debug media in git.

## Main Components

### Indoor stack

- Visual place recognition and temporal stabilization:
  [`src/corridor_localizer.py`](src/corridor_localizer.py),
  [`src/temporal_localization.py`](src/temporal_localization.py)
- Corridor graph planning and checkpoint progression:
  [`src/graph_planner.py`](src/graph_planner.py)
- MBRA controller wrapper:
  [`src/mbra_controller.py`](src/mbra_controller.py)
- Indoor runtime entrypoints:
  [`live_indoor_runtime.py`](live_indoor_runtime.py),
  [`live_indoor_runtime_mbra.py`](live_indoor_runtime_mbra.py),
  [`live_indoor_runtime_recovery.py`](live_indoor_runtime_recovery.py)

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

## Models and Methods Used

The repository uses or wraps the following main ideas and external components:

- [CosPlace](https://github.com/gmberton/CosPlace) style visual place recognition for indoor localization
- [MBRA / Model-Based Re-Annotation](https://model-base-reannotation.github.io/) as the short-horizon indoor controller reference
- [LogoNav](https://openreview.net/forum?id=9DyLaIHqrD) as the outdoor learned controller base
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) for monocular depth estimation
- [SegFormer](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512) for semantic segmentation experiments and runtime semantic scoring
- [OpenStreetMap](https://www.openstreetmap.org/) route structure for outdoor waypoint expansion

In this repo, those pieces are not used as a single monolithic system. They are wrapped inside task-specific runtimes:
- indoor: localization + graph progression + MBRA
- outdoor: mission checkpoints + routed waypoints + LogoNav + safety overrides

## Repository Layout

- [`src/`](src/): shared controllers, localization, safety, and runtime modules
- [`scripts/`](scripts/): diagnostics, calibration, probes, and preflight utilities
- [`earth-rovers-sdk/`](earth-rovers-sdk/): browser/FastAPI bridge for the rover
- [`mbra_repo/`](mbra_repo/): copied MBRA / LogoNav workspace used by the project
- [`third_party/Depth-Anything-V2/`](third_party/Depth-Anything-V2/): vendored depth dependency
- [`models/`](models/): local model checkpoint directory
- [`docs/`](docs/): technical reports, notes, and project documentation

## Current Project Direction

### Indoor

The indoor system is built around:
- corridor localization from recorded corridor imagery
- temporal stabilization of the pose estimate
- exact checkpoint-step planning on the corridor graph
- MBRA as a short-horizon local controller

The important engineering decision was to treat MBRA as a local controller on top of known corridor structure, not as a global planner.

### Outdoor

The outdoor system is built around:
- mission checkpoints from the rover SDK
- OSM-expanded intermediate waypoints
- LogoNav as the local motion policy
- runtime layers for rerouting, recovery, traversability, IMU safety, and semantic checks

The important engineering decision was to keep the existing outdoor controller base and harden the runtime around it, instead of replacing the entire stack.

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

## Running the Project

Indoor and outdoor have separate entrypoints.

Typical indoor runtime:

```bash
python3 live_indoor_runtime.py
```

Typical outdoor mission runtime:

```bash
python3 live_outdoor_runtime.py --mission --send-control --controller logonav --osm-route
```

The exact flags depend on the test mode, safety settings, and whether the run is indoor, outdoor, or marathon-oriented.

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
