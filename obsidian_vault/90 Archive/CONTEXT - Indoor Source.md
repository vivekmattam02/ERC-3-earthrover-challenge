# ERC Indoor / No-GPS Route-Repeat Context

## Project Goal

Build a reliable **no-GPS, repeated-route navigation system** for the
EarthRover platform in rough indoor / off-road-style terrain using a
teach-and-repeat workflow.

This is **not** a generic unseen-world exploration problem and it is **not**
the original GPS outdoor mission setup anymore.

## Active Direction (2026-04-24)

The current active branch of work is:

- `no GPS`
- `visual teach-and-repeat`
- `differential-drive rover`
- `rough / uneven terrain`
- `manually taught route, then autonomous repeat`

Operationally, the repo is no longer centered on the original corridor-only
baseline. The current source-of-truth problem is:

1. manually record a good reference traversal
2. convert it into a visual route package
3. run the live route follower with rough-terrain control
4. iterate on control + post-processing until repeat runs are stable

Important reality check:

- the FrodoBots mission session currently returns placeholder GPS
  (`latitude=1000`, `longitude=1000`)
- because of that, `live_outdoor_runtime.py` is **not** the active runtime for
  this work
- the active runtime is `live_indoor_runtime.py` repurposed as a
  no-GPS route-repeat runner

## Current Active Pipeline

The current active no-GPS pipeline is:

1. `scripts/record_sdk_session.py`
   - records manual rover runs from the local SDK
   - saves front camera, telemetry, IMU, magnetometer, RPMs
   - now auto-normalizes output to `.h5`

2. `tools/extract_h5_dataset.py`
   - converts a recorded H5 bag into route images + metadata
   - supports motion-window auto-trimming
   - trims using relative session time, not mismatched raw clocks

3. `baseline.py`
   - builds CosPlace descriptor DB + place graph + navigation graph
   - now preserves sequential navigation edges even when manual bags do not
     contain control labels

4. `scripts/prepare_manual_route.py`
   - one-command route preparation:
     `bag -> extracted dataset -> descriptor DB -> graph -> route_info.json`

5. `scripts/visualize_manual_route.py`
   - creates contact sheets for quick route quality inspection

6. `scripts/run_prepared_route.py`
   - short launcher for prepared routes
   - avoids long fragile CLI strings

7. `live_indoor_runtime.py`
   - current live no-GPS route-repeat runner
   - rough-terrain mode
   - startup relocalization probe
   - no-progress relocalization scan/probe
   - tilt-aware speed reduction

8. `src/sensor_state.py`
   - now computes filtered `roll`, `pitch`, and `tilt`

9. `src/local_controller.py`
   - still the active controller family
   - but now uses continuous heading correction while driving, not just
     turn-in-place alignment

## Current Recorded Bags

Canonical manual recordings are currently:

- `recordings/manual_flag_collection/2026-04-22/run_01_1521.h5`
- `recordings/manual_flag_collection/2026-04-22/run_02_1537.h5`
- `recordings/manual_flag_collection/2026-04-22/run_03_1551.h5`

Bag quality verdict:

- `run_01_1521.h5`
  - strongest teach bag so far
  - best current reference route
- `run_02_1537.h5`
  - rejected as primary route
  - contains a dark / near-black segment early and a badly tilted tail
- `run_03_1551.h5`
  - usable but weaker than run 1
  - active motion mostly in the first ~290 s, poor tail later

There is also a later malformed recording artifact:

- `recordings/manual_flag_collection/2026-04-2`
- `recordings/manual_flag_collection/2026-04-2.summary.json`

This is a valid saved recording with a bad filename and should be cleaned up
during post-processing rather than treated as a canonical route by default.

## Current Prepared Route Packages

Prepared route packages currently exist at:

- `data/manual_routes/smoke_run01_c`
- `data/manual_routes/smoke_run02_c`
- `data/manual_routes/smoke_run03_c`

Current recommendation:

- use `smoke_run01_c` as the active route package
- do **not** use `smoke_run02_c` as the primary route
- keep `smoke_run03_c` only as a secondary comparison route

Recommended live command:

```bash
python scripts/run_prepared_route.py \
  --route-dir data/manual_routes/smoke_run01_c \
  --sdk-url http://127.0.0.1:8000 \
  --rough-terrain \
  --send-control
```

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
   - current simple local-controller baseline
   - originally heading-gated and corridor-oriented
   - now extended for continuous heading correction during forward drive
   - still heuristic, not yet the final terrain-capable controller

7. `live_indoor_runtime.py`
   - conservative live loop runner
   - ties SDK input/output to localization, planning, and control
   - dry-run by default, send-control only when explicitly enabled

8. `src/earthrover_interface.py`
   - live robot interface for front camera, telemetry, orientation, and IMU data

9. `src/sensor_state.py`
   - lightweight motion-state filter
   - smooths heading from orientation
   - estimates turn rate from gyro z
   - summarizes RPM motion hints
   - now also estimates `roll`, `pitch`, and `tilt`
   - provides a motion prior for localization/control

10. `src/depth_estimator.py` + `src/depth_safety.py`
   - optional safety layer

11. `mbra_repo/`
   - research reference for MBRA / LogoNav
   - not yet the deployed indoor controller
12. `src/mbra_controller.py`
   - optional MBRA local-controller wrapper
   - intended only for short-horizon subgoal following
   - plugs into the same controller interface as the simple controller

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
- with a lightweight `motion-prior` layer

Meaning:

- camera frames are the main localization signal
- orientation / IMU should support temporal filtering, heading consistency,
  recovery, and controller smoothing
- GPS is not part of the indoor backbone
- RPM / encoder data should be used only if live data proves they are reliable

Current implementation:

- filtered heading is passed into localization and control
- gyro z is used as a filtered turn-rate hint for the controller
- RPM mean is used as a weak motion hint
- this is intentionally a simple filter, not a full EKF backbone

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

### No-GPS manual-route assets

- `recordings/manual_flag_collection/2026-04-22/`
- `data/manual_routes/smoke_run01_c/`
- `data/manual_routes/smoke_run02_c/`
- `data/manual_routes/smoke_run03_c/`

These are now the most important runtime assets for active work.

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
9. A lightweight motion filter now exists:
   - `src/sensor_state.py`
   - smooths heading from SDK orientation
   - estimates turn rate from gyro z
   - feeds a motion prior into localization/control

## Current Truthful Status

What is actually solid right now:

- corridor visual localization
- temporal stabilization of localization
- graph path planning and nearby subgoal selection
- basic runtime wiring from SDK input to localize -> plan -> control
- better debug output in the live runtime and controller
- manual bag recording -> route extraction -> route package build
- contact-sheet visualization for route quality checking
- no-GPS route-repeat runtime launch path

What is only partial right now:

- local control
  - improved compared to the first version
  - now includes rough-terrain steering, stall recovery, tilt slowdown, and
    relocalization search
  - still heuristic and not proven reliable enough for long live runs
- recovery behavior
  - low-confidence stop exists
  - no-path stop exists
  - stale motion-state stop exists
  - there is now a basic startup probe + relocalization scan/probe sequence
  - there is still no mature final recover / relocalize / resume state machine
- safety
  - optional depth-safety code exists in the repo
  - it is not yet fully integrated into the live runtime path
- IMU usage
  - filtered heading, gyro-z turn-rate, and RPM mean are now used
  - this is still a lightweight motion-prior layer, not full sensor fusion

What is still unproven:

- MBRA as a practical replacement for the simple controller on the real corridor (vel_past bug is fixed, needs live validation)
- full safety-aware autonomy on the real robot
- long-run stability (100+ steps) without human intervention
- repeated no-GPS route-follow on rough terrain without human intervention
- whether a pursuit-style controller will outperform the current simple controller

What has been proven in live tests:

- simple controller successfully navigated from step 1148 to 1189 (target reached)
- CosPlace localization + temporal filtering works reliably in the corridor
- graph planning and subgoal progression works correctly
- jump rejection and confidence-gated target detection prevent false positives
- GPU inference works (CosPlace 4.8ms, MBRA 14ms on CUDA)

What changed most recently:

- `scripts/record_sdk_session.py`
  - added manual no-GPS session recording
  - now auto-normalizes output names to `.h5`
- `tools/extract_h5_dataset.py`
  - added motion-window auto-trim
  - fixed time-base mismatch by trimming in relative session time
- `baseline.py`
  - preserves sequential navigation edges for manual bags with no control labels
- `scripts/prepare_manual_route.py`
  - new one-command bag-to-route builder
- `scripts/visualize_manual_route.py`
  - new contact-sheet visualizer for route quality
- `scripts/run_prepared_route.py`
  - new short launcher for prepared route packages
- `src/graph_planner.py`
  - loader now accepts either `links` or `edges` node-link JSON keys
- `src/sensor_state.py`
  - added tilt estimation
- `src/local_controller.py`
  - continuous heading correction during forward drive
- `live_indoor_runtime.py`
  - rough-terrain tuning
  - startup relocalization probe
  - no-progress relocalization search
  - tilt-aware slowdown

- `src/mbra_controller.py`
  - fully rewritten based on deep analysis of MBRA architecture and training code
  - vel_past now uses fixed constants (linear=0.5, angular=0.0) per authors' reference
  - removed EMA blending, velocity feedback, deadzone hacks
  - see `docs/our_mbra_discoveries.tex` for full technical analysis
- `live_indoor_runtime.py`
  - controller-dependent defaults (tick rate, subgoal hops)
  - MBRA: 3Hz / 4 hops; simple: 2Hz / 15 hops


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
  - now the active no-GPS route-repeat controller
  - still heuristic, but currently the only practical live option
  - default tick rate: 2Hz, default subgoal hops: 15

## Immediate Next Work: Post-Processing

The next real work after recording is post-processing, not random re-testing.

Priority order:

1. cleanly rename and organize all new bags
2. generate summaries and contact sheets for each candidate teach bag
3. reject bags with:
   - dark / black segments
   - extreme tilt
   - long dead time
   - bad route coverage
4. create one canonical teach bag for the active route
5. rebuild the route package from that canonical bag
6. compare autonomous repeat behavior against the current `smoke_run01_c`

The current best candidate remains:

- `run_01_1521.h5` / `smoke_run01_c`
- `--controller mbra`
  - loads `mbra_repo/train/config/MBRA.yaml`
  - loads `mbra_repo/deployment/model_weights/mbra.pth`
  - image-goal conditioned learned controller with visual steering
  - default tick rate: 3Hz, default subgoal hops: 4

MBRA runtime example:

```bash
python3 live_indoor_runtime.py --target-step 1189 --controller mbra --send-control
```

Useful overrides:

- `--mbra-weights /path/to/mbra.pth`
- `--tick-hz 4` (increase frame rate for better MBRA context)
- `--max-subgoal-hops 5` (adjust goal distance)

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
- it uses the current camera frame history + nearby subgoal image from the
  corridor database as the goal input
- vel_past uses fixed constants per the MBRA authors' deployment reference
  (NOT feedback from model predictions — see `docs/our_mbra_discoveries.tex`)
- the model runs on GPU with ~14ms inference time
- pending live validation with the vel_past fix applied

Recommended validation order:

1. `--controller simple` with dry-run
2. `--controller mbra` with dry-run
3. only then retry with `--send-control`

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
   local control beyond the current lightweight filter?
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
6. Continue using the lightweight motion-prior layer instead of jumping to a
   full EKF unless real tests prove we need more.
7. Run `live_indoor_runtime.py` in dry-run mode against the SDK before sending
   any real commands.

## Files That Matter Most Right Now

- `baseline.py`
- `src/temporal_localization.py`
- `src/corridor_localizer.py`
- `src/graph_planner.py`
- `src/navigation_runtime.py`
- `src/local_controller.py`
- `src/mbra_controller.py`
- `src/sensor_state.py`
- `live_indoor_runtime.py`
- `src/earthrover_interface.py`
- `tools/extract_h5_dataset.py`
- `tools/evaluate_temporal_localization.py`
- `data/corrider_extracted/metadata/data_info.json`
- `data/corrider_db/`
- `docs/our_mbra_discoveries.tex` — detailed MBRA architecture analysis and deployment findings

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

## Literature Sanity Check

The current high-level direction is aligned with how repeated-route visual
navigation is commonly approached in the literature.

What matches known successful approaches:

- topological place-based localization instead of forcing monocular SLAM to be
  the whole backbone
- known-environment / repeated-route memory built from prior traversals
- temporal filtering on top of visual place recognition
- graph or hop-based planning from current place to a nearby subgoal

This is consistent with:

- PlaceNav-style visual place recognition + filtering for topological
  navigation
- visual teach-and-repeat style systems that reuse a taught route and image
  registration rather than rebuilding full metric geometry every time
- RoboHop-style topological planning by moving through local place/segment
  subgoals

So the main architectural choice is not a blunder.

The part that is currently weak compared to recommended practice is the
controller/runtime layer:

- the simple controller is still heuristic
- runtime recovery is still basic
- safety is not yet fully integrated
- the graph is still effectively one-way in many cases

Current conclusion:

- localization/planning direction: good
- controller/runtime direction: still needs serious improvement

## MBRA Integration Status

MBRA is now **fully functional** as an optional local controller.

The intended role is:

- 6-frame observation history + nearby subgoal image → local velocity command (linear, angular)

The MBRA path is exposed through:

- `src/mbra_controller.py`
- `live_indoor_runtime.py --controller mbra`

Current status:

- Weights downloaded: `mbra_repo/deployment/model_weights/mbra.pth` (377MB from HuggingFace)
- Dependencies installed in `erv` env: `efficientnet-pytorch`, `einops`
- GPU inference working: ~14ms per call on CUDA (TITAN Xp)
- Non-MBRA imports made lazy in `utils_logonav.py` to avoid pulling in unused deps
- vel_past uses fixed constants (linear=0.5, angular=0.0) matching the MBRA authors' deployment reference
- No EMA blending — model handles temporal smoothing via its 6-frame context
- Default tick rate: 3Hz, default subgoal hops: 4
- Full technical analysis recorded in `docs/our_mbra_discoveries.tex`

Key deployment detail (see `docs/our_mbra_discoveries.tex` for full analysis):
MBRA's vel_past input is a **contextual hint**, not odometry feedback.
Feeding the model's own predictions back creates a self-reinforcing feedback
loop. The authors' own reference code uses fixed constants.

## Session Log

### 2026-03-23: Full docs pass + shared context workflow

- Read all docs `.tex` files now, including:
  - `docs/march19.tex`
  - `docs/discoveries.tex`
  - `docs/indoor_navigation_strategy.tex`
  - `docs/known_corridor_runtime_plan.tex`
  - `docs/laserfocus.tex`
  - `docs/nyu_indoor_track.tex`
  - `docs/our_approach.tex`
  - `docs/stack_breakdown.tex`
  - `docs/team_start_here.tex`
- The full crux is clear and consistent across the strongest/latest notes:
  - known-corridor visual localization + temporal filtering + graph planning is the baseline backbone
  - local control / runtime recovery is the main bottleneck
  - MBRA must be treated only as an optional short-horizon controller candidate, not the backbone
- Important doc note:
  - some older strategy docs still talk about MBRA more centrally than the corrected handoff/context files
  - the corrected interpretation in `codex_handoff_prompt.md`, `chat_session_reconstructed.md`, `chat_handoff.md`, `CONTEXT.md`, and `guide.md` should be treated as the current source of truth
- Concrete code fix completed in the current session:
  - `src/sensor_state.py` now detects stale motion state from non-advancing SDK timestamps instead of a self-overwritten wall-clock check
  - `src/local_controller.py` now stops on `motion_state_stale`
  - `live_indoor_runtime.py` now exposes `motion_state_stale` in the loop payload
- Shared workflow rule from the user:
  - `CONTEXT.md` is a living shared note between user and assistant
  - update it continuously as work progresses, regardless of task type
- Current shared understanding confirmed:
  - the project direction is now clear and stable
  - we are not re-arguing the backbone
  - current work is focused on making local control, recovery, and runtime behavior reliable
- Current next-step recommendation:
  - the next sensible action is real runtime validation of the simple controller path, not more architecture redesign
  - first verify SDK + manual teleop + dry-run simple controller on short targets
  - then inspect whether local execution still fails due to no-progress / oscillation, which would justify the next recovery patch
- Environment clarification:
  - activate a Python environment before running the repo
  - for the simple baseline path, use any env that has the SDK and baseline deps installed
  - for MBRA, the `mbra` conda env is the intended starting point, but it still also needs SDK/runtime deps if they are not already present
- Environment note from user discussion:
  - the active working conda environment may be `erv`, not `mbra`
  - for current simple/runtime work, use the environment that actually has the needed packages and can connect successfully
  - do not switch to `mbra` unless we are specifically validating MBRA deps/weights or it is confirmed to be the fully working env
- Environment decision guidance:
  - MBRA environment activation is not required for the current simple-controller baseline path
  - activate the MBRA-specific env only when we are intentionally testing `--controller mbra` or MBRA-specific dependencies
  - current priority remains making the simple runtime/control path work first in the environment that already runs SDK and runtime code
- Teleop usability fix completed:
  - `earth-rovers-sdk/examples/simple_control.py` now defaults to live single-key terminal control
  - old line-based behavior is still available with `--line-mode`
  - this was changed because the previous script required full line input and was easy to misuse during real teleop testing

- User pointed out `earth-rovers-sdk/examples/keyboard_control.py` is the correct teleop example.
- Inspection confirms it uses the `/control` endpoint and initializes `/sdk`, unlike the direct-RTM `simple_control.py` path.
- Current debugging focus shifted to validating the `/control` server path and any browser-session dependency behind it.

- Confirmed by code inspection that `earth-rovers-sdk/examples/keyboard_control.py` is the intended manual teleop example.
- `keyboard_control.py` drives the browser-backed `/control` path and initializes `/sdk`; `simple_control.py` uses the older `/control-legacy` direct RTM path.
- Manual debugging should use `keyboard_control.py`; autonomous runtime still uses `/api/set_control`.

- Manual teleop via `earth-rovers-sdk/examples/keyboard_control.py` is now confirmed working.
- This clears the SDK/browser control path as the immediate blocker.
- Next step is to run the autonomous stack on the `simple` controller path, first in dry-run and then with `--send-control`.

- User asked whether the robot needs to be physically placed in the corridor for runtime testing.
- Answer: yes for the current baseline, because localization assumes the robot is operating inside the known corridor captured in the database and starting from a visually compatible pose.

- First autonomous runtime attempt failed immediately with `ModuleNotFoundError: No module named 'torch'` from `src/corridor_localizer.py`.
- This is an environment issue in the active `erv` conda env, not a control/runtime logic issue.
- Even the `simple` controller path currently depends on the localization stack, which imports PyTorch.

- After installing `torch`, the next autonomous runtime attempt failed with `ModuleNotFoundError: No module named 'sklearn'` from `baseline.py`.
- This confirms the current blocker remains incomplete Python dependencies in the active `erv` env.
- Immediate priority is to finish installing baseline/runtime dependencies before further runtime testing.

- After installing the missing deps, `live_indoor_runtime.py` progressed far enough to download CosPlace weights and initialize the model.
- The next failure was `SDK server returned status 503` / `Failed to connect to SDK`, which indicates the problem has moved from Python environment setup to SDK telemetry availability.
- Current suspicion: the runtime is calling the SDK `/data` endpoint before browser-backed telemetry is active.

- User confirmed manual SDK control is working while `live_indoor_runtime.py` still gets SDK `/data` status 503.
- This narrows the issue to telemetry availability rather than command delivery.
- Current check: verify whether the SDK frontend is actually relaying RTM telemetry into `/api/update_data`, since `/data` depends on that cache.

- Patched `earth-rovers-sdk/main.py` so `/data` and `/v2/front` fall back to `browser_service` when the lightweight JS cache is empty.
- This addresses the concrete mismatch where manual control worked through the browser session but runtime telemetry failed with SDK 503 due to empty cache-only endpoints.
- Next step is to restart the SDK server and retry the simple runtime dry-run.

- After fixing SDK 503, the runtime connected successfully and began localization.
- The next failure is `torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device` during CosPlace inference.
- Root cause: the current PyTorch build auto-selected CUDA, but the installed build does not support the machine's TITAN Xp (`sm_61`).
- Immediate fix direction: force the corridor localizer onto CPU by default or gracefully fall back from CUDA to CPU.

- Patched `src/corridor_localizer.py` to avoid incompatible CUDA inference on this machine.
- The localizer now prefers CPU when the detected GPU capability is below the supported range for the installed PyTorch build, and it also retries on CPU if CUDA inference still fails at runtime.
- Next step is to rerun `live_indoor_runtime.py` without changing the SDK server.

- First successful dry-run reached localization and planning without SDK or CUDA crashes.
- Localization placed the robot at step `1007` with confidence `0.474`.
- The runtime then stopped with `runtime_no_path_stop` and `path_error=no_path:1007->120`, indicating a planning/graph-direction mismatch rather than a control failure.
- Immediate next check: determine whether the navigation graph is forward-only and the target step should instead be chosen ahead of the current localized step.

- Dry-run with a forward target (`1030`) now produces a valid plan: current step `1007`, subgoal `1010`, and initial `align_heading` commands.
- The next blocker is that the controller switches to `motion_state_stale_stop` after a few ticks, even in dry-run.
- Current investigation: determine whether motion-state freshness is being enforced too aggressively or based on telemetry fields that do not update in this SDK path.

- Patched `live_indoor_runtime.py` so dry-run no longer enforces `motion_state_stale_stop`.
- Real command sending still keeps the stale-motion safety stop.
- This allows dry-run to reveal planner/controller behavior without being masked by telemetry freshness limits.

- User observed that dry-run keeps printing repeated `align_heading` commands and feels like it is not working.
- Important clarification: in dry-run, `sent=False`, so the robot never rotates and the observed heading never changes; repeated `align_heading` output is therefore expected, not a new controller bug.
- Next step is to test a very short real-control run with `--send-control` on a nearby forward target, while keeping safety enabled.

- User reported that manual `keyboard_control.py` stopped working again after the recent runtime tests.
- Likely immediate cause is a stuck or stale SDK browser session rather than a new localization/planning bug.
- Current action is to reset the SDK server/browser session cleanly before further diagnosis.

- User asked whether the autonomous failure is simply because the overall system is not working properly.
- Current answer: no, not exactly. Manual control and core runtime pieces now work, so the remaining issue is narrower: autonomous execution/controller behavior under real feedback.

- Root cause of `send_control` not reaching robot was found: `earthrover_interface.py` was posting to `/api/set_control` which only queues commands in `pending_control`; nobody polls `/api/get_control` to forward them to Agora RTM.
- Fix: changed `earthrover_interface.py` to post to `/control` instead, which calls `browser_service.send_message()` directly — the same path confirmed working by `keyboard_control.py`.

- First real live run (`--send-control`) showed the robot IS physically moving: localization tracked position changes (cur went 1577→1579→20→214→1508).
- Main problem: robot stuck in `align_heading` mode indefinitely, spinning in circles without ever entering `drive_to_subgoal`.
- Root cause: `subgoal_orientation` from data_info.json is the compass heading from recording time, not a reliable drive-direction. The filtered heading error never resolves below `align_exit_threshold_deg=12°`, causing infinite spin.
- After spinning, visual localizer gets confused by facing random corridor directions → localization jumps (e.g., to step 20, 214, 1508).
- Fix applied to `src/local_controller.py`:
  - `align_enter_threshold_deg` raised from 32° to 65° (only enter align mode for large heading errors)
  - `align_exit_threshold_deg` raised from 12° to 20°
  - Added `max_align_ticks=8`: after 8 consecutive align ticks (~4s at 2Hz), force exit align mode and drive forward
  - Added `_align_ticks` counter to track this

- Also fixed: `stale_timeout_s` in `sensor_state.py` raised from 1.0s to 5.0s to stop the intermittent `motion_state_stale_stop` that was cutting commands every ~3 ticks.

- Second live run: robot now in `drive_to_subgoal` the whole time (spinning fix worked), but angular kept growing from 0.031→0.238 over ~20 ticks, causing robot to spiral.
- Root cause: `drive_heading_gain=0.007` was multiplying a heading error derived from unreliable indoor compass data in data_info.json, causing increasing angular correction and spiral motion.
- After spiraling, robot faces a different direction → localization jumps to step 298 (from 1535).
- Fix: `drive_heading_gain` set to 0.0 and heading-based linear scaling removed from drive_to_subgoal mode.
- Robot now drives straight forward with angular=0 during `drive_to_subgoal`. Align mode still handles large heading changes (>65°, capped at 8 ticks).
- Current controller behavior: just drive straight; let localization advance the step.

- Third live test: robot sent commands (sent=True) but linear velocity decayed to 0.026 via stacking multipliers (gyro noise → high_turn_rate_linear_scale × held_previous × rpm_slow), below the robot's physical friction threshold. Robot physically stopped, view didn't change, localizer stuck at same step → held_previous stayed True → further speed reduction. Classic death spiral.
- Also found: after align_heading exits, `_previous_angular` retained the last align angular value. Rate limiter bled it into drive_to_subgoal (e.g., angular=-0.140 instead of 0.000), causing unwanted turning after align.
- Fixes applied to `src/local_controller.py`:
  - `min_linear` raised from 0.06 to 0.10 (ensure commands exceed friction threshold)
  - Hard floor enforced: `linear = max(min_linear, ...)` in both drive_to_subgoal and no_heading_forward_crawl paths. Prevents multiplicative scaling from dropping below usable speed.
  - `_previous_angular` reset to 0.0 on align→drive transition (alongside `_align_ticks=0`). Prevents angular bleed from align mode into drive mode.
  - Removed `self._rate_limit_angular(angular)` call from drive_to_subgoal path (angular is always 0.0 in drive mode, rate limiter was only introducing bleed artifacts).

- Fourth live test: **ROBOT SUCCESSFULLY NAVIGATED** from step 1089 to 1098 (and past to 1099). Steps advanced: 1089→1092→1093→1097→1099. Confidence stayed ~0.6-0.7. Linear held at 0.100-0.160 (min_linear floor working).
- Problem 1: align_heading still fired (iterations 17-50) due to unreliable indoor compass in subgoal_orientation from data_info.json. Robot oscillated between align and drive modes. Despite this, it still advanced steps — localizer is resilient.
- Problem 2: when cur=1099 overshot target=1098, graph returned no_path (forward-only). Robot sat stopped forever with runtime_no_path_stop. No graceful exit.
- Fix 1: nulled `subgoal_orientation` in live_indoor_runtime.py (`controller_input["subgoal_orientation"] = None`). Controller takes `no_heading_forward_crawl` path → drives straight forward, no compass-based alignment. Align mode can't trigger without orientation data.
- Fix 2: added target-reached detection in runtime loop — when `cur >= target`, print success message, stop robot, and break out of loop. No more infinite runtime_no_path_stop.

- Fifth live test (target 1187): Robot navigated 1152→1168→1170→1173→1177 then localizer hallucinated to 1276 at conf=0.160. TARGET REACHED fired falsely because 1276 >= 1187 despite garbage confidence.
- Root causes: (1) no confidence gate on target-reached, (2) no localization jump rejection, (3) no proximity slowdown near target, (4) base speed too conservative at 0.100-0.120.
- MBRA assessment: `mbra_repo/deployment/model_weights/` is **EMPTY**. No .pth files. MBRA code wrapper exists but is non-functional without trained weights. Not viable right now.
- Fixes applied to live_indoor_runtime.py:
  - **Jump rejection**: if cur jumps >30 steps in one tick at conf < 0.35, reject the localization, revert to previous trusted step, and stop. Prevents hallucinated jumps from triggering false target-reached.
  - **Target-reached requires confidence**: must have conf >= 0.40 for 3 consecutive ticks before declaring target reached. Single-tick low-conf hallucinations can't trigger it.
  - **Proximity slowdown**: within 15 steps of target, scale linear speed down to 40% minimum. Prevents overshooting the target.
  - **Subgoal hops increased**: default max_subgoal_hops from 3 to 15. Gives controller more forward visibility for speed calculations.
- Fixes applied to local_controller.py:
  - Speed cap in no_heading_forward_crawl raised from max_linear×0.5 (=0.12) to max_linear×0.75 (=0.18). Allows faster driving while still being conservative without heading data.

- **MBRA local controller is now FUNCTIONAL.** This is the image-goal conditioned learned controller described in the docs.
  - Downloaded weights (377MB) from HuggingFace NHirose/MBRA_project_models → `mbra_repo/deployment/model_weights/mbra.pth`
  - Installed missing deps in erv env: `efficientnet-pytorch`, `einops`
  - Made non-MBRA model imports lazy in `utils_logonav.py` (GNM, ViNT, NoMaD, LeLaN, clip) to avoid pulling in heavy unused deps
  - Added CUDA sm_61 fallback in `mbra_controller.py` (same GPU incompatibility as CosPlace)
  - MBRA takes (current_frame_sequence, subgoal_image) → outputs (linear_vel, angular_vel) with steering. This replaces the blind "drive straight" heuristic.
  - Use: `python3 live_indoor_runtime.py --controller mbra --target-step N --send-control`

- **GPU now works!** Root cause: PyTorch 2.10.0+cu128 dropped sm_61 (Pascal/TITAN Xp). Downgraded to PyTorch 2.3.1+cu121 which includes sm_50/sm_60 arch support.
  - MBRA on GPU: **14ms (71 Hz)** — was 46ms on CPU
  - CosPlace localizer on GPU: **4.8ms (210 Hz)** — was ~50ms on CPU
  - Both models now on CUDA. Full loop (localize + plan + MBRA infer) well under 50ms.
  - Fixed GPU capability checks in both corridor_localizer.py and mbra_controller.py to use `torch.cuda.get_arch_list()` instead of hardcoded sm_70 threshold.

- **Stale data/frame problem fixed via JS push model.**
  - Root cause: `/data` had 2s cache, `/v2/front` had 5s cache. Localizer saw the same frame for ~10 ticks. The `/api/update_data` and `/api/update_frame` endpoints existed in main.py but nothing called them.
  - Fix 1 (`basicRtm.js`): Added `fetch("/api/update_data", ...)` inside the `MessageFromPeer` handler. Every RTM sensor message is immediately pushed to the backend cache. Zero latency vs the old pyppeteer eval polling.
  - Fix 2 (`basicVideoCall.js`): Added `_pushFrameLoop()` — captures front camera frame every 300ms via `getLastBase64Frame(1000)` and POSTs to `/api/update_frame`. Starts 3s after page load. Localizer now gets ~3.3 fps fresh frames instead of reusing a 5s-stale cached one.
  - Fix 3 (`main.py`): Reduced cache staleness thresholds — `/data` from 2.0s to 0.5s, `/v2/front` from 5.0s to 1.0s. If JS push stops for any reason, pyppeteer fallback still works at these tighter timeouts.
  - Net effect: localizer sees a new frame every ~300ms instead of every ~5s. Sensor data refreshes every RTM message instead of every 2s. Full pipeline latency dropped from seconds to sub-second.

- **MBRA `mbra_missing_images` bug fixed.**
  - Symptom: MBRA controller returned `mbra_missing_images` every tick, robot never moved. Confidence was fine (~0.6), localization and planning worked, but MBRA got `subgoal_image_rgb=None`.
  - Root cause: Graph node paths and descriptor archive paths are hardcoded to `/home/vivek/Desktop/rover/...` (original author's machine). On this machine (`/home/lunar/`), the paths don't exist, so `_load_subgoal_image()` returned None. The actual images DO exist at `data/corrider_extracted/front_images/` under the local repo.
  - Fix (`src/navigation_runtime.py`): `_load_subgoal_image()` now detects when the hardcoded path doesn't exist, extracts the relative `data/...` portion, and resolves it against the actual repo root. Subgoal images now load correctly.

- **MBRA persistent-turning bug fixed — vel_past feedback loop.**
  - Symptom: MBRA output angular ~0.15-0.18 every tick, robot veered into walls. cur=1152 never changed for 65+ ticks.
  - Root cause investigation: Read the full MBRA architecture (ExAug_dist_delay in exaug.py), training code (train_utils.py), and reference deployment (LogoNav_frodobot.py).
  - MBRA model architecture: EfficientNet-B0 encoder + Transformer decoder. Takes 6 observation images (context_size=5 + current), 1 goal image (concatenated with last observation → 6ch EfficientNet), robot_size, delay, and vel_past (6 linear + 6 angular = 12 tokens). Outputs 8-step velocity trajectory.
  - **The bug**: vel_past is 12 of 21 transformer tokens — more than half the sequence. We were feeding MBRA's own predicted angular velocities back as vel_past. Once the model output a small angular bias (natural for outdoor-trained model in an indoor corridor), that got fed back → amplified → locked into persistent turning. During TRAINING, vel_past starts as random values and is self-correcting via loss gradients. During DEPLOYMENT, there's no loss to correct the feedback.
  - **The fix**: The MBRA authors' own deployment reference (train_utils.py line 1815-1817) uses FIXED vel_past: `linear=0.5, angular=0.0`. This tells the model "robot is going straight at moderate speed" and lets visual context drive decisions. Rewrote `mbra_controller.py` to use pre-built fixed vel_past tensor.
  - Also removed: EMA blending (model handles temporal smoothing via its 6-frame context), angular deadzone hack, high_angular_linear_scale hack. These were band-aids that didn't address the root cause.
  - Also: default tick rate for MBRA set to 3Hz (matching reference deployment), subgoal hops default to 4.

- **MBRA corridor-stall bug fixed — near-zero linear velocity deadlock.**
  - Symptom: MBRA navigates well for stretches (1266→1288 in 27 ticks), then stalls at a step for 40-100+ ticks. Linear drops to 0.001-0.005, angular stays ~0.07-0.09. Robot spins in place.
  - Root cause: corridor aliasing. When current view and subgoal image (4 hops ahead) look nearly identical, MBRA interprets "I see the goal" as "I'm at the goal" → outputs near-zero linear. Robot stops → 6 context frames become identical → model keeps outputting near-zero linear → deadlock.
  - Fix 1 (`mbra_controller.py`): Added `min_linear=0.10` floor, same principle as the simple controller. `linear_cmd = max(min_linear, linear_cmd)`. Robot physically cannot stall above the friction threshold.
  - Fix 2 (`live_indoor_runtime.py`): No-progress detection. If cur_step doesn't change for 20 ticks, reset MBRA's observation context (`controller.reset()`). Fresh context breaks the stale-frame deadlock. Resets repeat every 20 ticks if still stuck.

- **Localization jump rejection hardened — scaled confidence + state revert.**
  - Old bug: jump rejection used flat conf < 0.35 threshold. A jump of 800 steps only needed 0.498 confidence to be accepted (slipped through after 2-3 rejection ticks because localizer's internal state already shifted to wrong position).
  - Fix 1 (`src/temporal_localization.py`): Added `save_state()` / `revert_state()`. Before each update the state is snapshotted. After rejection, it's restored — so the continuity cost for the wrong position remains high on the next tick.
  - Fix 2 (`src/corridor_localizer.py`): Calls `save_state()` before every `temporal_localizer.update()`. Exposes `revert_last_update()`.
  - Fix 3 (`src/navigation_runtime.py`): Exposes `revert_localization()`.
  - Fix 4 (`live_indoor_runtime.py`): Jump rejection threshold now scales with jump magnitude: `required_conf = min(0.85, 0.35 + 0.005 * jump_mag)`. Jump of 30 steps needs 0.50 confidence; 100+ steps needs 0.85. When rejected, calls `runtime.revert_localization()` to undo localizer state.

- **Robot randomly stopping mid-run — control path fragility + fallback added.**
  - Root cause: `/control` endpoint depends on a live browser/pyppeteer session. If Agora RTM disconnects, token expires, or browser page reloads, all commands fail silently. Robot stops.
  - Fix (`src/earthrover_interface.py`): `send_control()` now tries `/control` first (1s timeout). If it fails, automatically falls back to `/control-legacy` (direct Agora REST API, no browser dependency).
  - Fix (`earth-rovers-sdk/examples/keyboard_control.py`): Same fallback added to `send_command()`.
  - Note: Both `/control` and `/control-legacy` are officially supported — `simple_control.py` in the SDK's own examples already used `/control-legacy`.
  - Practical note: if robot stops and keyboard control also fails, restart SDK + reopen `/sdk` in browser.

- **MBRA wall-crash prevention — angular saturation limit, recovery backup, RPM stall detection.**
  - Observed: MBRA commanded sustained max angular (±0.340) for 20+ ticks → robot spun into wall. After wall hit, camera saw wall texture → garbage localization → `jump_rejected_stop` for 37+ ticks → stuck forever.
  - Fix 1 (`live_indoor_runtime.py`): Angular saturation counter. If angular > 85% of max_angular for 6 consecutive ticks, force `angular=0` for 4 ticks (straight ahead). Prevents wall-spinning.
  - Fix 2 (`live_indoor_runtime.py`): Jump rejection recovery. If `jump_rejected` fires for 12+ consecutive ticks, robot backs up at -0.12 m/s for 5 ticks, then resets localizer + controller for fresh start.
  - Fix 3 (`live_indoor_runtime.py`): RPM stall detection. If commanding forward (linear > 0.05) but `rpm_mean` < 1.5 for 8+ ticks, robot is physically stuck → same backup recovery triggered.

- **Depth Anything V2 obstacle avoidance wired up (optional `--depth-safety` flag).**
  - Infrastructure: `src/depth_estimator.py` + `src/depth_safety.py` already existed in the repo (written by Vivek) but were not connected to the runtime.
  - Checkpoint: `depth_anything_v2_metric_hypersim_vits.pth` — Indoor-metric Small model. Place in `third_party/Depth-Anything-V2/checkpoints/`.
  - Fix (`src/depth_estimator.py`): `_find_checkpoint()` updated to recognize actual HuggingFace filename `depth_anything_v2_metric_hypersim_{enc}.pth`.
  - Integration (`live_indoor_runtime.py`): Added `--depth-safety` flag. When enabled, runs depth inference every 2 ticks at 160×120. Computes forward clearance (center ±22.5° arc). If < `--depth-stop-m` (default 0.4m): stop + trigger backup recovery. If < `--depth-slow-m` (default 0.8m): scale down linear proportionally. Depth inference failures are non-fatal.
  - To use: `python3 live_indoor_runtime.py --controller mbra --send-control --depth-safety`

- **docs/our_mbra_discoveries.tex updated with corridor-stall and IMU findings.**
  - Added new subsection "Corridor Aliasing Stall (Near-Zero Linear Deadlock)" under Symptoms.
  - Added explantion of why IMU/side cameras are not the right fix (MBRA architecture accepts only front RGB + goal image).
  - Added Bug #6 (corridor aliasing stall) to Summary of Bugs.
  - Added Lessons #6 and #7 about min velocity floors and stale context self-reinforcement.

- User requested a deeper read-only pass over the current codebase, especially the SDK bridge, live runtime, simple-controller path, and MBRA path.
- A documentation artifact was created in `docs/current_codebase_deep_read.tex` to capture the current understanding without changing runtime logic.

- User asked for a review of the current `.gitignore` and whether more heavy or personal files should be excluded.
- Current action is read-only inspection of `.gitignore` and worktree state before proposing ignore additions.

- `.gitignore` was updated to ignore the current local handoff/context markdown files and the extra `nyu-earthrover-main/` clone.
- Existing broad ignore rules for `*.tex` and `docs/` already cover newly created LaTeX docs.

- User asked how to push the repo to GitHub from the current machine, given the current git config and remote.
- Important clarification: git commit identity (`user.name`, `user.email`) and the GitHub account used for SSH push access are related but not the same thing.

- User hit GitHub push failure: remote repo is correct, but SSH authentication is currently happening as `harsh-sutariya`, not `vivekmattam02`.
- Important clarification: commit identity (`git config user.*`) does not control which GitHub account SSH uses for push access.
- The fix is to point this repo at the correct SSH identity or switch the remote to HTTPS with the user's own credentials.

- User loaded the personal SSH key `~/.ssh/id_ed25519_vivek` successfully.
- `ssh -T git@github-vivek` no longer points to the wrong account issue; it now fails at DNS resolution for `github.com`, which suggests the SSH alias is being read but network name resolution is temporarily failing.

- User shifted focus to brainstorming the outdoor track.
- Goal for this prompt: develop several concrete strategy options for unknown outdoor navigation, likely using GPS where helpful, before reviewing teammate outdoor-only MBRA code.
- Current action: gather repo references to the outdoor track and combine them with primary-source method ideas before proposing 5 usable solution directions.

- User shared that the teammate's working outdoor GPS+MBRA file is `mbra_gps.py`.
- Current action is a read-only review of that file to place it correctly within the outdoor strategy space and compare it with the brainstormed options.

## 2026-03-24 Outdoor Planning Note

- Read `mbra_gps.py` carefully to understand the teammate's outdoor working baseline.
- `mbra_gps.py` is a GPS-conditioned LogoNav/MBRA-style controller, not just a reactive visual policy.
- It uses `/v2/front` plus `/data`, converts GPS to UTM, transforms the goal into the robot-local frame, and feeds a relative GPS/heading goal token into the learned policy.
- Outdoor planning direction now centers on five candidate strategies: classical GPS baseline, GPS+VFH/local avoidance, current MBRA-GPS baseline, GPS+OSM routing, and a stronger learned/hybrid model path.
- Indoor repo pieces reusable outdoors: SDK bridge, robot I/O, logging, runtime/recovery skeleton, optional depth safety.
- Indoor-specific corridor localization/graph planning should not be reused directly for outdoor GPS missions.

## 2026-03-24 Outdoor Top-3 Direction

- Narrowed outdoor brainstorming to the three most reliable strategies rather than five.
- Best reliability stack identified as: (1) current GPS-conditioned LogoNav baseline with safety/recovery, (2) classical GPS plus VFH/depth fallback, and (3) OSM-routed GPS waypoints executed by one of the first two local controllers.
- The working `mbra_gps.py` / LogoNav path should be treated as a serious primary solution, not an optional experiment.
- The classical GPS+VFH stack is the main non-ML fallback for robustness.
- OSM routing is the strongest global-planning upgrade for harder urban missions with turns and crossings.

## 2026-03-24 Outdoor Build Order

- Teammate is expected to handle Phase 1 fixes for `mbra_gps.py` / `outdoor_logonav.py`.
- Our focus shifts to the independent outdoor components that can be built in parallel: `src/outdoor_gps_controller.py`, `live_outdoor_runtime.py`, and `src/osm_router.py`.
- Recommended implementation order: (1) classical GPS+VFH controller, (2) outdoor runtime wrapper, (3) OSM routing layer.
- Key external inputs still needed later: sample outdoor data format, exact mission/checkpoint payload format, and whether OSM internet queries are allowed in the target deployment setting.

---

# ERC Outdoor Context

## Current Focus (2026-03-25)

All work is now on **outdoor GPS checkpoint navigation** on EarthRover Zero.
Indoor work is on hold (CosPlace VPR localizer + corridor graph is stable).

## Outdoor System Architecture

```
SDK server (hypercorn) → live_outdoor_runtime.py
  ├── LogoNav controller (default) — 6 frames + GPS + heading → linear/angular
  ├── GPS controller (fallback)   — classical bearing-error → linear/angular
  ├── Depth Anything V2           — metric depth for soft traversability bias
  ├── Outdoor traversability      — 15% angular nudge toward clearer space
  └── Stuck detection & recovery  — frozen-tick aware, turn-based recovery
```

## Competition Run Commands

### Terminal 1 — SDK server
```bash
sudo fuser -k 8000/tcp 2>/dev/null
cd ~/Desktop/rover/ERC-3-earthrover-challenge/earth-rovers-sdk
conda activate erv
hypercorn main:app --reload
```

### Terminal 2 — start mission (must run BEFORE controller)
```bash
curl -X POST http://127.0.0.1:8000/start-mission
```

### Terminal 3 — run controller

**Option A — base (same as previously working field test):**
```bash
cd ~/Desktop/rover/ERC-3-earthrover-challenge
conda activate erv
python live_outdoor_runtime.py --mission --send-control
```

**Option B — with soft traversability bias (low-risk addition):**
```bash
python live_outdoor_runtime.py --mission --send-control --traversability
```

**Difference:** Option B adds a 15% angular nudge away from broad vegetation/walls when depth detects a forward blockage. It never stops or slows the robot.

## What Is Working

1. **LogoNav (IL_gps architecture)** — default outdoor controller
   - Vision-based GPS-conditioned learned navigation
   - 6 camera frames + GPS + orientation → linear/angular (max 0.3/0.3)
   - Weights: `mbra_repo/deployment/model_weights/logonav.pth` (295MB)

2. **GPS waypoint following** — reached CP=3/9 in earlier field run
   - Ordered checkpoint list from SDK `/start-mission`
   - OSM pedestrian routing optional (`--osm-route`)
   - Auto-reports `/checkpoint-reached` when within `--goal-radius-m` (default 8m)

3. **Stuck detection and recovery**
   - Frozen telemetry skip: only counts fresh GPS ticks for displacement
   - Stuck window: 15 fresh ticks, must displace >0.30m and make progress toward goal
   - Recovery: reverse 5 ticks → turn 8 ticks (direction based on bearing error)
   - Fast wall-hit: 3 consecutive ticks with linear≥0.15 and <0.10m displacement
   - LogoNav stuck: turn-only recovery (no reverse), relaxed progress threshold

4. **Depth Anything V2 with auto max_depth**
   - Auto-detects from checkpoint: vkitti→80.0, hypersim→20.0
   - Was broken at max_depth=5.0 (everything read 0.3-0.8m) — now fixed

5. **Soft traversability bias** (with `--traversability` flag)
   - Module: `src/outdoor_traversability.py`
   - Middle-band depth crop (rows 15%-60%)
   - 10th percentile per angular bin (16 bins, 90° FOV)
   - 4-frame temporal min-pool (~1.3s obstacle memory)
   - Only applies 15% angular blend when `forward_blocked` is true
   - Never stops, never slows — depth model too weak for hard safety

6. **Debug output**
   ```
   [0102] CP=2/9 wp=3/8 dist=54.1m bear=-35° hdg=125.0°(Δ+7°) fwd=5.40m TRV mode=DRIVING lin=+0.250 ang=-0.600 SENT
   ```

## EarthRover Zero Hardware Constraints

- **Camera:** RGBA 576×1024, front-mounted low (~30cm height), wide-angle
- **Compass:** jumps 50-150°/tick from motor magnetic interference. Raw compass used (EMA causes spirals)
- **GPS:** updates at ~1.5Hz instead of 3Hz. FRZ=1 on every other tick
- **Robot toppled** in one run — likely stuck recovery + slope terrain

## Key Tuning Parameters

| Parameter | Value | Why |
|---|---|---|
| controller | logonav | Visual navigation, default |
| nominal_linear | 0.33 | Tuned for EarthRover Zero |
| max_angular | 0.45 | Reduced from 1.0 to prevent snaking |
| angular_gain | 0.4 | Reduced from 1.2 to prevent snaking |
| in_place_turn_threshold | 90° | Lower values cause TURNING/DRIVING flicker with noisy compass |
| stuck_window_ticks | 15 | Counts only fresh (non-frozen) telemetry |
| stuck_min_displacement | 0.30m | Raised from 0.15m to reduce false positives |
| goal_radius_m | 8.0 | Checkpoint reach distance |
| depth max_depth | auto (80 for vkitti) | Was hardcoded 5.0, caused 100% false stops |
| TRAV_BIAS_WEIGHT | 0.15 | 15% angular blend when blocked |

## Depth Model Findings (2026-03-25)

### The max_depth bug (root cause of all depth issues)
- `DepthEstimator` was initialized with `max_depth=5.0`
- Depth Anything V2: `depth = max_depth * sigmoid(head(features))`
- vkitti checkpoint trained with max_depth=80 → at max_depth=5, everything reads 0.3-0.8m
- **Fix:** `_infer_max_depth()` auto-detects from checkpoint filename

### Calibration results (200 outdoor frames, max_depth=80)
```
Forward clearance: min=3.49m, p10=4.63m, median=5.80m, p90=11.35m
Trigger rates: STOP(<0.6m)=0%, SLOW(<1.2m)=0%, BLOCKED(<1.5m)=0%
```

### Parameter sweep findings
- Resolution doesn't matter: 120×160 and 240×320 give identical results
- Percentile doesn't matter much: p1 vs p25 shifts ~10-15%, same ordering
- **Thin objects invisible:** person+dog reads 25-30m (should be ~5-10m actual)
- **Broad vegetation weakly detectable:** tree/bushes at 4-5m vs open at 5-7m (narrow 1m gap)
- **Conclusion:** Metric depth usable ONLY as soft directional bias. NOT usable for hard safety or thin obstacle detection.

## What Was Tried and Reverted

| Change | Result | Action |
|---|---|---|
| Compass EMA alpha=0.35/0.20 | Lag → outward spirals | Reverted to raw compass |
| Speed 0.35/0.40 | Noisy compass + high speed → worse snaking | Settled at 0.33/0.38 |
| angular_gain=0.7/1.2 | Robot snakes on heavy platform | Reduced to 0.4 |
| max_angular=0.65/1.0 | Same snaking | Reduced to 0.45 |
| in_place_turn_threshold=45°/70° | Compass noise → TURNING/DRIVING flicker | Reverted to 90° |
| Depth hard stop/slow | max_depth=5 → 100% false stop; max_depth=80 → thin objects invisible | Demoted to soft 15% angular bias |
| LogoNav weights: mbra.pth | Wrong architecture (ExAug_dist_delay vs IL_gps) — crash | Fixed: downloaded logonav.pth (295MB) |

## Outdoor Key Files

| File | Purpose |
|---|---|
| `live_outdoor_runtime.py` | Main outdoor runtime loop |
| `src/outdoor_gps_controller.py` | GPS waypoint follower |
| `src/outdoor_logonav_controller.py` | LogoNav visual controller wrapper |
| `src/outdoor_traversability.py` | Soft traversability bias |
| `src/depth_estimator.py` | Depth Anything V2 (auto max_depth) |
| `src/earthrover_interface.py` | SDK interface |
| `src/osm_router.py` | OSM pedestrian routing |
| `scripts/calibrate_traversability.py` | Offline calibration tool |
| `scripts/sweep_trav.py` | Parameter sweep tool |
| `mbra_repo/deployment/model_weights/logonav.pth` | LogoNav weights (IL_gps, 295MB) |
| `third_party/Depth-Anything-V2/checkpoints/depth_anything_v2_metric_vkitti_vits.pth` | Outdoor depth model |
| `test_outdoor/test_outdoor_*.h5` | Recorded outdoor frames (~2000 frames each) |

## Outdoor Agreed Next Steps (2026-03-25)

1. **Semantic segmentation — DONE** — SegFormer-B0 on ADE20K, `src/semantic_risk_estimator.py`, `--semantics` flag
2. **Field test when available (outdoor on hold due to cars on course):**
   - Without `--traversability` first (confirm base unchanged)
   - Then with `--traversability` (confirm soft bias doesn't hurt)
   - Then optionally with `--semantics`

---

# Indoor Session Log (2026-03-25)

## Indoor Fixes Applied (6 total)

1. **max_depth hardcode removed** — was `max_depth=5.0` in `live_indoor_runtime.py` (same bug as outdoor). Now auto-detects from checkpoint.
2. **Dead code removed** — duplicate `motion_state_stale` check was unreachable in `local_controller.py`.
3. **Gyro-based steering correction** — THE MAIN FIX. Controller was driving with `angular=0.0` (no course correction). Now uses gyro Z to counteract unwanted rotation. `gyro_drive_correction_gain=0.008`, deadband 3.0 dps, capped at 50% max_angular. In `src/local_controller.py`.
4. **min_linear raised** — from 0.10 to 0.12 in `src/local_controller.py`. Prevents velocity death spiral from stacking multipliers.
5. **EMA command smoothing** — alpha=0.7 linear, 0.6 angular in `live_indoor_runtime.py`. Recovery bypasses smoothing.
6. **Telemetry freeze detection** — FRZ counter in `live_indoor_runtime.py`, warns when sensor timestamps stall.

## Indoor Codex Additions (same session)

- `--checkpoint-images` flag: target by image filename directly
- `--target-image-name` flag: single image target
- `--checkpoint-steps` flag: target by pre-localized graph steps
- `--auto-advance-checkpoints` flag: auto-advance through checkpoint list

## Competition Advantage

- Test is in the **same corridor** as the recorded dataset
- User has **11 checkpoint images** before competition
- Pre-localization tool ready: `scripts/prelocalize_checkpoints.py`
- Can run checkpoint images through CosPlace to get exact graph steps

## Checkpoint Steps (from corridor DB frames)

```
CP1=45  CP2=480  CP3=761  CP4=821  CP5=1094  CP6=1208
CP7=1345  CP8=1430  CP9=1544  CP10=1638  CP11=1764
```

These are frame indices from the corridor database (0-1864). All verified present in the navigation graph.

## Additional Fixes (session continued)

7. **Last-checkpoint crash fix** — After reaching checkpoint 11, auto-advance moved index past end → next tick `plan_to_active_checkpoint` raised ValueError. Fixed: main loop now checks `checkpoint_reached + next_active_checkpoint is None` and breaks cleanly with "ALL CHECKPOINTS COMPLETED".
8. **Checkpoint reach tolerance** — `checkpoint_reached` was exact node match only. If localizer returned node 44 instead of 45, checkpoint wouldn't trigger. Added `checkpoint_reach_tolerance=3` (±3 steps) in `graph_planner.py`.

## Indoor Competition Command

```bash
python live_indoor_runtime.py --checkpoint-steps 45 480 761 821 1094 1208 1345 1430 1544 1638 1764 --auto-advance-checkpoints --send-control
```

## Indoor Pending

- **Live indoor test**: All fixes need live validation — especially gyro correction gain tuning
- Prelocalization not needed — checkpoints are exact DB frame indices

## .env Config
- `BOT_SLUG=joint-frog-grasp`
- `MISSION_SLUG=leg-1`
