# ERC-3 EarthRover Challenge

This repository contains the rover runtime, controller wrappers, safety modules, and supporting scripts for the NYU EarthRover Challenge.

The goal of the workspace is simple:
- run the indoor stack cleanly
- run the outdoor stack safely
- keep the project self-contained
- keep large data and model files out of git

## Repository Layout

- `earth-rovers-sdk/`: browser and FastAPI bridge for the rover
- `src/`: shared runtime modules, controllers, localization, and safety code
- `scripts/`: calibration, diagnostics, and evaluation helpers
- `mbra_repo/`: local copy of the MBRA / LogoNav research workspace
- `third_party/Depth-Anything-V2/`: vendored depth-estimation dependency
- `models/`: local model checkpoints and project-specific weights
- `docs/`: written reports and technical notes

## Current System

- Indoor: MBRA with corridor localization and checkpoint-step progression
- Outdoor: LogoNav with OSM-expanded waypoints and runtime safety layers
- Marathon: outdoor runtime with stricter safety and recovery handling

## Setup

1. Clone the repo and enter the workspace.

```bash
cd ERC-3-earthrover-challenge
```

2. Create the local SDK config.

```bash
cp earth-rovers-sdk/.env.sample earth-rovers-sdk/.env
```

Edit `earth-rovers-sdk/.env` with your local values.
Do not commit that file.

3. Put model weights in the expected local directories.

- MBRA / LogoNav weights: `mbra_repo/deployment/model_weights/`
- Project checkpoints: `models/`
- Depth Anything V2 checkpoints: `third_party/Depth-Anything-V2/checkpoints/`

4. Verify the workspace.

```bash
python3 verify_workspace.py
```

## Working Rules

- Keep secrets out of git.
- Keep large checkpoints out of git.
- Treat this repo as standalone.
- Use relative paths inside the workspace.
- Do not depend on files outside this directory unless they are explicitly vendored here.

## Notes

- The main engineering work in this repo is split between indoor and outdoor autonomy.
- The indoor stack is centered on localization, graph progression, and MBRA.
- The outdoor stack is centered on LogoNav, mission checkpoints, route expansion, and safety layers.
- The `.tex` files in `docs/` are technical reports and story-style writeups.
- The Markdown files are for quick reference and operator notes.

