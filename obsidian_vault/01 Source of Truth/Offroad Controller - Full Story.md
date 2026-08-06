# Offroad Controller - Full Story

> **Purpose:** Full detailed explanation of the current no-GPS off-road controller branch: what controller we started with, what we changed, what failure modes remain, and what the final controller should become.
> **Read when:** I need to understand the current control problem in a technically serious way.
> **Authority:** Current controller-story source of truth for the off-road branch. Trust active code for exact parameter values.

## Why This Note Exists

The current off-road branch is not blocked only by localization.

It is also blocked by control quality.

The rover can:

- record bags
- build route packages
- localize against a taught route
- start moving

But that is not enough.

The real remaining question is:

> what kind of controller should a no-GPS, rough-terrain, differential-drive EarthRover actually use?

This note exists to answer that from the current repo state.

## What Controller We Started With

The current baseline controller lives in:

- `src/local_controller.py`

It is still called:

- `SimpleLocalController`

That name matters. It is a hint, not an insult.

This controller began as a pragmatic local controller for graph subgoals, not as a final off-road controller.

Its original structure was basically:

- use confidence gating
- compare current heading to subgoal heading
- if heading error is large, align first
- otherwise drive forward toward the next subgoal

That is a reasonable starter controller for a stable corridor-like setting.

It is not enough by itself for rough terrain.

## Why The Original Controller Was Not Enough

The original failure was not just “the robot moved badly.”

The deeper issue was that the controller was too close to an:

- align first
- then drive

style of logic.

That is brittle for a differential-drive rover on rough terrain because:

- rocks and bumps perturb heading continuously
- the camera pose changes under body roll and pitch
- the robot should keep making forward progress while correcting
- repeated stop-align-stop-align behavior turns into stalls and oscillation

So the main controller lesson was:

> the robot cannot behave like a static tripod that pauses for every heading disagreement.

## What The Current Controller Now Does

The current `SimpleLocalController` is no longer pure align-or-drive.

The main meaningful improvements already in code are:

### 1. Continuous steering during forward drive

During `drive_to_subgoal`, the controller now:

- computes angular correction from heading error
- keeps steering while moving
- damps steering with gyro turn-rate information
- slows linear speed when heading error is large

This is important because it makes the controller more like a moving tracker and less like a brittle two-mode switch.

### 2. Confidence-aware speed scaling

The controller already reduces forward speed when:

- localization confidence is weaker
- the heading error is larger
- the rover is already turning quickly
- progress is stale

That is the right general direction.

### 3. Motion/RPM awareness

The controller consumes:

- filtered RPM mean
- filtered heading rate
- filtered heading

So it is already using a lightweight motion prior rather than pretending the camera is the only signal.

## What The Runtime Adds On Top Of The Controller

The controller is not acting alone.

`live_indoor_runtime.py` adds a lot of rough-terrain behavior around it:

- angular saturation override
- jump-reject recovery
- RPM stall detection
- tilt-aware speed reduction
- startup relocalization probing
- no-progress relocalization search

So the real behavior is already a controller plus a runtime state machine.

That distinction matters.

If someone asks “what controller are we using?”, the current answer is:

- rough-terrain prepared-route runs select `AdaptivePursuitController`;
- the older `SimpleLocalController` remains a baseline and reference point;
- `live_indoor_runtime.py` adds an important recovery state machine around
  either controller.

## What The Sensor Layer Contributes

`src/sensor_state.py` now provides:

- filtered heading
- filtered heading rate
- filtered RPM mean
- filtered roll
- filtered pitch
- filtered tilt

This is intentionally not a full EKF.

It is a lightweight motion-state prior.

That is a good choice for now because it gives the controller the signals it needs without pretending we have a full state-estimation backbone.

## What Rough-Terrain Mode Actually Means

In `live_indoor_runtime.py`, rough-terrain mode changes the control profile.

It tightens the controller and runtime in a conservative direction:

- lower max linear speed
- lower min linear speed
- lower max angular speed
- tighter align thresholds
- lower confidence speed scaling
- more conservative stall thresholds
- stronger relocalization/search behavior
- more aggressive tilt slowdown

This is not enough by itself to make the rover terrain-capable. It is a
runtime profile around the controller and must be judged by field progress,
not by the presence of tuning flags.

## What The Real Failure Modes Still Are

Even with the current patches, the remaining failure modes are clear.

### 1. Over-dependence on heading heuristics

The controller still reasons too much in terms of heading error and not enough in terms of continuous geometric path tracking.

That means it can still feel too discrete when the route geometry becomes messy.

### 2. Recovery is still heuristic

The relocalization scan/probe behavior is a real improvement, but it is still heuristic:

- scan
- forward probe
- localizer reset

Earlier reverse-heavy recovery was a poor fit for rocks, so the rough-terrain
path is now forward-biased. That is a design correction, not proof that the
recovery is solved.

That is much better than doing nothing, but it is not yet a fully principled terrain-local controller.

### 3. Terrain reaction is still indirect

Tilt-aware slowdown helps, but it is only one signal.

The controller still does not explicitly optimize curvature or traction for the body mechanics of a differential-drive rover on uneven ground.

### 4. Route-follow quality still depends heavily on the teach route

A bad route package can still make a decent controller look bad.

That is why control and post-processing have to be developed together.

## What The Correct Controller Family Should Be

The right controller family for this rover is:

- visual teach-and-repeat localization
- graph/subgoal progression
- adaptive pursuit or curvature tracking
- terrain-aware speed scheduling
- local reactive recovery

Not:

- pure align-turn-in-place logic
- not generic MPC as the first move
- not end-to-end learned control as the immediate answer

The practical reason is simple:

- the rover is differential-drive
- the terrain is uneven
- the route is taught
- startup location is approximately known
- global exploration is not the main problem

So the right controller is one that:

- continuously tracks a route direction
- keeps moving while correcting
- slows intelligently for terrain and confidence
- only resorts to turn-in-place when absolutely necessary

## What The Controller Should Become

The repo now contains `AdaptivePursuitController`; it is the right *family* to
evaluate, not a validated final answer.

Its outputs should still be:

- `linear`
- `angular`

But its logic should be more structured:

### Linear speed should depend on:

- localization confidence
- tilt
- heading error
- turn rate
- RPM/slip proxy
- progress/stall state

### Angular speed should depend on:

- route heading / lookahead direction
- heading error
- current turn rate
- local recovery mode

### Controller states should be explicit:

- `BOOTSTRAP_SEARCH`
- `TRACK`
- `STALL_RECOVERY`
- `LOST_RELOCALIZE`

That is already where the runtime is heading. The required next step is not
another abstract controller redesign; it is evidence that those states produce
stable progression on the real route.

## What Now Exists In Code

That controller is no longer just a proposal.

The repo now has:

- `src/adaptive_pursuit_controller.py`

And the rough-terrain prepared-route launcher now defaults to it when using:

- `python scripts/run_prepared_route.py --route-dir ... --rough-terrain --send-control`

So the current state is:

- the old `SimpleLocalController` still exists
- MBRA still exists as a separate local-controller branch
- the new adaptive pursuit controller exists as the intended no-GPS rough-terrain controller family

That does **not** mean the controller problem is finished.
It means the code is now on the right controller branch instead of only discussing it.

## What We Should Not Say Yet

We should not claim:

- the current controller is already the final off-road controller
- rough-terrain mode alone solved the control problem
- relocalization probes mean the system is now fully robust
- post-processing can compensate for a weak controller indefinitely

That would be overstating the current state.

## What The Field Tests Actually Proved

The current controller branch is not fake progress, but the field evidence is
narrower than it first appeared:

- the rover can move under no-GPS repeat commands;
- tilt, traction, and camera pose are real constraints;
- heading/route-orientation signals were not reliable enough to trust;
- repeated scan/probe cycles did not establish sustained step progression.

So continuous control, terrain scheduling, and active relocalization remain
the correct direction, while the actual success criterion remains unproven.
See [[01 Source of Truth/No-GPS Field Trial - Findings]].

## The Short Version

The current controller is:

- an implemented adaptive-pursuit experiment with a runtime recovery layer;
- not field-validated as a reliable route follower;
- still subject to localization/reference-route failure, not only control tuning.

The final controller should be:

- adaptive pursuit / curvature tracking
- terrain-aware
- confidence-aware
- recovery-aware

That is the correct control direction for this branch.

## Read Next

- [[01 Source of Truth/Offroad Track - Full Story]]
- [[00 Home/Current No-GPS - Read This First]]
- [[03 Personal Notes/Current Truth]]
- [[03 Personal Notes/Architecture in My Words]]
