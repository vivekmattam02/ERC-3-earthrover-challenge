# Codex Handoff Prompt

Copy everything below into the other laptop's Codex session.

---

We are continuing work in:

`/home/lunar/Desktop/rover/ERC-3-earthrover-challenge`

Read this carefully and **keep it fixed in memory for the rest of the session**.
Do not casually re-argue or restart the architecture unless you find a
concrete technical flaw in the code or evidence.

## 1. Read order: do this first

Before editing anything, read these files in this exact order:

1. `chat_session_reconstructed.md`
2. `chat_handoff.md`
3. `CONTEXT.md`
4. `guide.md`
5. `docs/march19.tex`
6. `docs/discoveries.tex`

Then inspect these code files:

1. `baseline.py`
2. `tools/extract_h5_dataset.py`
3. `tools/evaluate_temporal_localization.py`
4. `src/temporal_localization.py`
5. `src/corridor_localizer.py`
6. `src/graph_planner.py`
7. `src/navigation_runtime.py`
8. `src/local_controller.py`
9. `src/sensor_state.py`
10. `src/mbra_controller.py`
11. `live_indoor_runtime.py`
12. `earth-rovers-sdk/examples/simple_control.py`

Do not skip this read order.

## 2. The project in one sentence

This is a **known-corridor indoor navigation system** for the EarthRover
Challenge, built around:

- visual localization against a previously recorded corridor
- temporal stabilization of that localization
- graph planning over remembered corridor places
- nearby subgoal selection
- a short-horizon local controller
- safety and recovery around that controller

This is **not** a generic unknown-building navigation system.

## 3. Current truths you should treat as fixed unless you find concrete evidence otherwise

### 3.1 What is strong

- localization is the strongest validated part
- temporal localization is strong
- live localization on the real corridor looked good
- graph planning / nearby subgoal selection is broadly correct for the baseline

### 3.2 What is weak

- local controller quality
- runtime recovery logic
- full safety integration
- MBRA runtime readiness / validation

### 3.3 What is the current bottleneck

The main current engineering problem is:

**short-horizon execution**

not:

- localization architecture
- graph-planning architecture

## 4. Absolutely critical architectural interpretation

Do not confuse these layers.

### 4.1 Backbone

The main backbone is:

- `baseline.py`
- `src/temporal_localization.py`
- `src/corridor_localizer.py`
- `src/graph_planner.py`
- `src/navigation_runtime.py`

This backbone is based on:

- corridor image memory
- visual place recognition
- temporal filtering
- graph path planning

### 4.2 Controller

The controller layer is separate.

Current baseline controller:

- `src/local_controller.py`

Optional experimental controller:

- `src/mbra_controller.py`

### 4.3 Robot I/O

Robot and teleop path:

- `live_indoor_runtime.py`
- `earth-rovers-sdk/examples/simple_control.py`

## 5. MBRA: remember this clearly

For this project, MBRA is **not**:

- the localization system
- the graph planner
- the checkpoint manager
- the whole autonomy stack

For this project, MBRA is only a possible:

- **short-horizon local controller**

Meaning:

- input: recent observation frames + nearby subgoal image
- output: local motion command

Current MBRA status:

- architecturally wired in
- not fully validated
- still needs proper env/deps/weights
- should not be spoken about as if it is already solved

Do not accidentally promote MBRA back into ``the whole system.''

## 6. What has already been done

These are not ideas. These are already completed changes.

### 6.1 Baseline cleanup

- `baseline.py` was cleaned up from hardcoded, personal, notebook-style code
- it was turned into a usable shared baseline CLI
- it now builds:
  - descriptor database
  - place graph
  - navigation graph

### 6.2 Dataset extraction

- `tools/extract_h5_dataset.py` was created
- it converts `data/corrider.h5` into:
  - extracted corridor images
  - metadata JSON
  - metadata CSVs
- important detail:
  - stream alignment was handled by relative time carefully

### 6.3 Localization and evaluation

- query-time retrieval was tested
- temporal localization was added
- temporal localization was evaluated
- held-out results were strong
- live corridor localization looked good

### 6.4 Runtime modules

These were created to make the system modular:

- `src/corridor_localizer.py`
- `src/graph_planner.py`
- `src/navigation_runtime.py`

### 6.5 Controller baseline

- `src/local_controller.py` was created
- later improved with:
  - align-heading mode
  - hysteresis
  - no-progress handling
  - debug fields
  - stale motion-state stop

### 6.6 IMU / motion-prior support

- `src/sensor_state.py` was added
- it provides:
  - filtered heading
  - gyro-z turn-rate hint
  - RPM mean hint
- it is a lightweight motion-prior layer
- it is not a full EKF backbone

### 6.7 Live runtime

- `live_indoor_runtime.py` was created
- it ties:
  - SDK input
  - localization
  - planning
  - controller
  - optional command sending

### 6.8 Teleop

- `earth-rovers-sdk/examples/simple_control.py` exists as the simple Python teleop path

### 6.9 Literature note

- `docs/discoveries.tex` exists
- it already records that the overall architecture is broadly supported by the literature
- the controller/runtime layer is the weak point

## 7. What not to do

Do **not** do these things casually:

- do not throw away the localization + graph backbone
- do not propose MBRA as the whole stack
- do not act like the current heuristic controller is already good enough
- do not ignore the local context files and then re-derive everything from scratch
- do not assume the current repo is a clean final competition system

## 8. How each local context file is being used

### `chat_session_reconstructed.md`

- closest thing to a turn-by-turn session memory
- use when you want the actual flow of questions, answers, and corrections

### `chat_handoff.md`

- structured project handoff
- use when you want the organized project story

### `CONTEXT.md`

- running local truth file
- use for:
  - current state
  - verified vs partial
  - what has already been done

### `guide.md`

- practical runbook
- use for commands and operational steps

### `docs/march19.tex`

- broad internal explanation
- use when you want the bigger picture explained carefully

### `docs/discoveries.tex`

- literature sanity check and reference inventory
- use when you want to know whether the architecture itself is sensible

## 9. What the user has been trying to do

The user has been trying to:

- make the repo a clean shared indoor baseline
- validate localization on real corridor data
- avoid architecturally stupid decisions
- understand where MBRA really fits
- improve controller reliability
- keep local notes that explain the project very clearly

The user values:

- very clear direct explanations
- practical next steps
- not being hand-waved
- not restarting from zero when something already works

## 10. If you must summarize the whole project in four lines

- The project uses remembered corridor images to localize the robot.
- A graph planner turns that location into a nearby subgoal.
- A local controller tries to move the robot toward that subgoal.
- Localization is strong; control/recovery is the main weak point.

## 11. What you should do first in your first reply

Your first reply should:

1. confirm you read the listed context files
2. summarize the current stack
3. separate what is already verified from what is still partial
4. state clearly that the current main bottleneck is local control / runtime recovery
5. only then continue with code work

## 12. Final reminder: keep this fixed in memory

Remember these three facts clearly:

1. localization is the strongest validated part
2. planning is broadly acceptable as a baseline
3. control and recovery are the main unfinished parts

And remember this one architectural rule clearly:

**MBRA is only a candidate short-horizon controller here. It is not the backbone.**

---

Begin by reading the files listed above and then continue from that exact state.
