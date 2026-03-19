# ERC-3 EarthRover Challenge

Standalone indoor-navigation workspace for the NYU EarthRover Challenge effort.

This directory is meant to be pushed, cloned, and worked on as its own project. It should not depend on files outside this directory.

## What Is In Here

- `docs/`: indoor competition notes, strategy, and team planning documents.
- `earth-rovers-sdk/`: FrodoBots / EarthRover browser + FastAPI bridge.
- `mbra_repo/`: MBRA / LogoNav research and deployment code used as the local-controller reference.
- `src/`: shared runtime modules copied for this project, including rover interface and safety helpers.
- `DBR/`: DBR-related material plus vendored `Depth-Anything-V2` dependency for optional depth safety.
- `models/`: local checkpoint location for project-specific model files.

## Local Setup

### 1. Clone the repo and enter this directory

```bash
cd ERC-3-earthrover-challenge
```

### 2. Create your local SDK config

Do not commit a real `.env` file. Each teammate must create their own local copy:

```bash
cp earth-rovers-sdk/.env.sample earth-rovers-sdk/.env
```

Then edit `earth-rovers-sdk/.env` with your actual SDK values:

- `SDK_API_TOKEN`
- `BOT_SLUG`
- `CHROME_EXECUTABLE_PATH`
- `MAP_ZOOM_LEVEL`
- `MISSION_SLUG`

The committed file is only the template: `earth-rovers-sdk/.env.sample`

The local file is ignored by git: `earth-rovers-sdk/.env`

### 3. Place model weights locally

This repo contains code and weight directories, but not the actual large checkpoints.

Expected locations:

- MBRA / LogoNav deployment weights:
  - `mbra_repo/deployment/model_weights/`
- Project-level checkpoints:
  - `models/`
- Optional depth checkpoints for safety:
  - `DBR/thirdparty/Depth-Anything-V2/checkpoints/`

### 4. Verify the workspace

Run:

```bash
python3 verify_workspace.py
```

The script checks:

- required folders and files exist
- the local SDK `.env` exists and is not still using placeholder values
- runtime weight directories exist and contain checkpoints
- no hardcoded parent-repo paths remain in important project files

If the script exits nonzero, the setup is not ready yet.

## Path Discipline

This workspace should be treated as a standalone project.

- New code should use paths relative to this directory.
- Do not add imports or runtime dependencies that reach back into the parent repo.
- Keep secrets out of git.
- Keep large checkpoints local or in approved artifact storage, not committed directly.

## Current Intended Stack

- Global navigation: topological graph + visual localization
- Local control: MBRA as a short-horizon controller only
- Safety: conservative override layer, optionally with depth and pedestrian checks
- Goal completion: separate image-based checkpoint verification

## Important Notes

- `mbra_repo/` here is a copied workspace version, not the original root copy.
- The local `.env` is intentionally not committed.
- Training configs inside `mbra_repo/train/config/` still contain dataset placeholders; that is expected.
- `verify_workspace.py` is the first check teammates should run after cloning.
# ERC-3-earthrover-challenge
