# ERC Indoor Context

## Project Goal

Build a reliable indoor navigation system for the EarthRover Challenge in a
known, repeated corridor environment at NYU.

This is **not** a generic unseen-building navigation problem.

## Current Baseline

The current baseline is:

1. `baseline.py`
   - CosPlace descriptor extraction
   - image retrieval
   - place graph construction
   - navigation/action graph construction
   - optional SuperGlue verification at query time

2. `src/temporal_localization.py`
   - temporal smoothing over retrieval results
   - continuity-aware node selection
   - optional heading-aware scoring
   - ambiguity handling to avoid unstable jumps

3. `src/corridor_localizer.py`
   - runtime-facing localization API
   - loads the built database once
   - accepts a frame or image path
   - returns node, step, confidence, and ranked candidates

4. `src/graph_planner.py`
   - runtime graph planner
   - accepts the localizer result
   - computes path to a target node / step / image
   - returns a nearby subgoal for the controller

5. `src/navigation_runtime.py`
   - controller-facing coordinator
   - combines localization and planning into one call
   - returns current node, target, subgoal, and optional subgoal image

6. `src/local_controller.py`
   - first simple local-controller baseline
   - heading-aware control from current state to nearby graph subgoal
   - outputs `linear` and `angular` commands

7. `live_indoor_runtime.py`
   - conservative live loop runner
   - ties SDK input/output to localization, planning, and control
   - dry-run by default, send-control only when explicitly enabled

8. `src/earthrover_interface.py`
   - live robot interface for front camera, telemetry, orientation, and IMU data

9. `src/depth_estimator.py` + `src/depth_safety.py`
   - optional safety layer

10. `mbra_repo/`
   - research reference for MBRA / LogoNav
   - not yet the deployed indoor controller

## Important Conceptual Split

Do not mix these up:

- `baseline.py` = perception + localization + graph-planning backbone
- `mbra_repo` = learned navigation/control research code
- `earth-rovers-sdk` = robot I/O path

MBRA is **not automatically** the deployed online controller in this repo.

## Sensor Stance

The project is:

- `vision-primary`
- with `IMU / heading support`

Meaning:

- camera frames are the main localization signal
- orientation / IMU should support temporal filtering, heading consistency,
  recovery, and controller smoothing
- GPS is not part of the indoor backbone
- RPM / encoder data should be used only if live data proves they are reliable

## Current Data Assets

### Raw recording

- `data/corrider.h5`

This contains:

- front frames
- controls
- telemetry
- accelerometer
- gyroscope
- magnetometer
- RPMs

### Extracted dataset

- `data/corrider_extracted/front_images/`
- `data/corrider_extracted/metadata/data_info.json`
- `data/corrider_extracted/metadata/*.csv`
- `data/corrider_extracted/metadata/summary.json`

This was created with:

- `tools/extract_h5_dataset.py`

### Built database

- `data/corrider_db/`

Contains:

- `descriptors.npz`
- `config.json`
- `place_graph.json`
- `navigation_graph.json`

## What Has Already Been Verified

1. The `.h5` file is useful and contains real front frames, controls,
   telemetry, IMU data, and mostly nonzero RPM samples.
2. The extractor works and aligns frame metadata using relative time.
3. The baseline DB build works on the extracted corridor images.
4. Query-time retrieval works on this corridor.
5. Held-out retrieval against a subsampled DB lands in the correct local
   neighborhood, but corridor aliasing still exists in ambiguous regions.
6. Temporal localization has been evaluated on a held-out step-5 corridor DB:
   - exact step match rate: `95.7%`
   - near match rate: `97.9%`
   - moving-frame exact match rate: `99.1%`
   - moving-frame near match rate: `100%`
   - results file: `data/corrider_db_step5/temporal_eval.json`
   - most remaining large errors are concentrated in one ambiguous window around
     corridor steps `370--410`
   - inspection shows that window is largely a stationary / repeated-frame
     segment, so it should not be treated as a normal moving localization
     failure
7. Runtime localizer and planner wrappers now exist:
   - `src/corridor_localizer.py`
   - `src/graph_planner.py`
   - `src/navigation_runtime.py`
   - `src/local_controller.py`
    - smoke test succeeded for `localize -> plan -> subgoal -> local command`
8. A live runtime runner now exists:
   - `live_indoor_runtime.py`
   - CLI verified
   - defaults to dry-run for safety


## How To Run After EarthRover Is Open

Once the EarthRover SDK server is up and serving camera/data endpoints on
`http://localhost:8000`, the indoor runtime can be launched directly.

Recommended first pass:

1. Start with dry-run and the simple controller:
   ```bash
   python3 live_indoor_runtime.py --target-step 120 --controller simple
   ```
2. If checkpoint execution is preferred instead of one final target:
   ```bash
   python3 live_indoor_runtime.py --checkpoint-steps 120 180 240 --controller simple
   ```
3. Only send real commands after dry-run behavior looks correct:
   ```bash
   python3 live_indoor_runtime.py --target-step 120 --controller simple --send-control
   ```

Controller options:

- `--controller simple`
  - current hand-written baseline controller
  - lowest dependency risk
- `--controller mbra`
  - loads `mbra_repo_1/train/config/MBRA.yaml`
  - loads `mbra_repo_1/deployment/model_weights/mbra.pth`
- `--controller logonav`
  - loads `mbra_repo_1/train/config/LogoNav.yaml`
  - loads `mbra_repo_1/deployment/model_weights/logonav.pth`

MBRA / LogoNav runtime examples:

```bash
python3 live_indoor_runtime.py --target-step 120 --controller mbra
python3 live_indoor_runtime.py --target-step 120 --controller logonav
```

Useful overrides:

- `--mbra-config /path/to/config.yaml`
- `--mbra-checkpoint /path/to/checkpoint.pth`
- `--mbra-device cpu`
- `--mbra-device cuda:0`

What the runtime does each loop:

1. Reads the current front camera frame from the EarthRover SDK.
2. Reads heading/orientation from the SDK if available.
3. Runs corridor localization.
4. Plans a graph path and picks a nearby subgoal.
5. Passes `controller_input` into the chosen local controller.
6. Prints or sends `(linear, angular)` commands.

Important note on the MBRA controller currently wired in this repo:

- it is integrated as a local-controller candidate, not as a replacement for
  localization or graph planning
- it uses the current graph subgoal to build a relative goal-pose input for the
  MBRA/LogoNav model
- this is appropriate for dry-run and controlled testing, but it is still an
  engineering adaptation of the original deployment code rather than a fully
  validated final controller

Recommended validation order:

1. `--controller simple` with dry-run
2. `--controller mbra` with dry-run
3. `--controller logonav` with dry-run
4. only then retry with `--send-control`

## Known Issues Already Fixed

1. `baseline.py` had a JSON serialization bug when writing NumPy scalar types.
   This is fixed.
2. `baseline.py` action-graph construction broke for subsampled databases
   built with `--step > 1`. This is fixed.
3. The H5 extractor initially aligned streams by raw timestamps, which was
   wrong because of a clock offset. It now aligns by relative time.

## Main Open Technical Questions

1. What is the actual online local controller for moving between nearby graph
   nodes after the simple baseline?
2. How should IMU / heading support be integrated into runtime localization and
   local control?
3. What recovery logic should run when localization confidence drops?

## Immediate Next Steps

1. Keep the current localizer + planner stack as the perception/planning baseline.
2. Treat the `370--410` region as a motion-aware edge case, not a generic
   retrieval failure.
3. Add geometric verification only to ambiguous moving cases if needed.
4. Decide the first online controller candidate.
5. Connect:
   - frame input
   - temporal localization
   - graph path / subgoal selection
   - local controller
   - safety
6. Run `live_indoor_runtime.py` in dry-run mode against the SDK before sending
   any real commands.

## Files That Matter Most Right Now

- `baseline.py`
- `src/temporal_localization.py`
- `src/corridor_localizer.py`
- `src/graph_planner.py`
- `src/navigation_runtime.py`
- `src/local_controller.py`
- `live_indoor_runtime.py`
- `src/earthrover_interface.py`
- `tools/extract_h5_dataset.py`
- `tools/evaluate_temporal_localization.py`
- `data/corrider_extracted/metadata/data_info.json`
- `data/corrider_db/`

## Short Version

We already have a working corridor-specific perception/planning baseline.
The project is no longer about inventing an architecture from scratch.

The main work now is turning:

- corridor database
- stabilized localization
- graph progression
- local control
- safety

into one reliable runtime loop.

## Current Engineering Verdict

Yes, the project is currently moving in the right direction.

Why:

- we are using the known-corridor assumption properly
- we are not overcomplicating the backbone with unnecessary SLAM
- we now have a working runtime perception/planning path:
  `corridor_localizer -> graph_planner -> navigation_runtime -> local_controller`
- the remaining major unknown is whether MBRA should replace or augment the
  simple local controller, not whether the graph-localization backbone is sound
