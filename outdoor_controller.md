# Outdoor Controller Plan

This file is the living working document for the **EarthRover outdoor competition track**.

Use this file instead of `CONTEXT.md` for outdoor-planning updates, implementation notes, strategy decisions, and competition-specific reasoning.

## Purpose

The goal of this document is to keep one clean, evolving record of:

- what the outdoor task actually is
- what the official constraints imply technically
- what we have already decided
- what the best outdoor strategy options are
- what code already exists
- what should be built next
- what still needs clarification from the team

This is intentionally different from the indoor `CONTEXT.md`, because the outdoor problem is fundamentally different.

## Outdoor Task Understanding

The outdoor EarthRover task is **GPS-checkpoint navigation in unknown outdoor environments**.

This is not the same as the indoor corridor problem.

### What the official task implies

Based on the EarthRover Challenge description:

- missions consist of a sequence of GPS-defined checkpoints
- environments can be seen or unseen
- outdoor routes may include sidewalks, paths, roads, parks, turns, crossings, people, bikes, cars, and clutter
- the robot has front/back cameras, GPS, IMU, and remote/cloud compute access
- the robot is evaluated on reaching checkpoints and finishing missions under real-world uncertainty
- a checkpoint is considered reached when the robot gets sufficiently close in GPS space

### Core decomposition of the outdoor problem

Every outdoor solution has to solve three subproblems:

1. `Where should I go globally?`
   - next GPS checkpoint
   - or an intermediate GPS waypoint from a router

2. `What short-horizon motion should I execute right now?`
   - learned local controller
   - or classical bearing-following controller

3. `How do I avoid unsafe or deadlocked behavior?`
   - obstacle avoidance
   - stop / backup / retry logic
   - stuck detection
   - graceful control fallback

That decomposition is the correct outdoor backbone.

## High-Level Contrast With Indoor

### Indoor backbone

Indoor is:

- vision localization in a known corridor
- temporal stabilization
- graph planning over a prebuilt route memory
- short-horizon controller to follow graph subgoals

### Outdoor backbone

Outdoor should be:

- GPS checkpoint mission manager
- optional route planner for intermediate GPS waypoints
- short-horizon local controller
- safety / recovery wrapper

### Important consequence

The indoor corridor localizer and graph planner should **not** be reused as the outdoor navigation backbone.

The parts of the repo that *are* reusable outdoors are:

- EarthRover SDK bridge
- front camera and telemetry interface
- live runtime skeleton
- control sending and graceful stopping
- optional depth estimation / safety hooks
- debug logging and dry-run patterns

## What Existing Outdoor Code Already Proves

The key working outdoor file we examined is:

- `mbra_gps.py`

### What `mbra_gps.py` actually is

This file is not best thought of as the indoor MBRA controller reused outdoors.

It is better understood as a **GPS-conditioned LogoNav / ViNT-like local navigation policy**.

### What it does

At a high level, `mbra_gps.py`:

- reads the latest front image from `/v2/front`
- reads GPS + orientation telemetry from `/data`
- converts current GPS to UTM
- converts the next goal GPS to UTM
- transforms the goal into the robot-local coordinate frame
- constructs a relative goal token
- feeds image history + goal token into a learned policy
- gets predicted waypoints from the model
- selects a predicted waypoint
- converts that waypoint into linear/angular commands
- sends those commands continuously at about 3 Hz

### Why this matters

This means the teammate's outdoor code is already a **serious learned outdoor baseline**.

It is not just:

- teleoperation
- or GPS bearing pasted on top of a random model
- or indoor MBRA misapplied outdoors

It is a legitimate GPS-goal-conditioned short-horizon controller.

That matters because it means we already have one credible outdoor strategy in hand.

## Current Best 3 Outdoor Strategies

We considered many possible approaches. The cleanest and most reliable shortlist is now the following three.

### Strategy 1: LogoNav + Safety + Recovery

This should be treated as the **primary learned outdoor solution**.

#### Core idea

Use the current GPS-conditioned LogoNav-style controller as the local motion policy, but wrap it with:

- obstacle veto / slowdown
- stuck detection
- backup / recovery
- robust command sending

#### Why it is strong

- already aligned with the EarthRover sensor stack
- already compatible with GPS-defined checkpoints
- already proven to run in the repo in some form
- likely stronger than pure classical control in visually structured outdoor scenes
- can naturally exploit camera semantics without explicit hand-coded rules

#### What it still lacks

The raw teammate file currently lacks several deployment-critical layers:

- obstacle avoidance / depth veto
- stuck detection
- graceful control-send failure handling
- fallback behavior when telemetry is temporarily bad
- competition-hardened runtime wrapper

#### Correct interpretation

This is not the whole outdoor architecture by itself.

The correct architecture is:

`mission manager -> next GPS waypoint -> LogoNav local policy -> safety/recovery wrapper`

### Strategy 2: Classical GPS + VFH / Depth Avoidance

This should be treated as the **primary non-ML fallback**.

#### Core idea

Use:

- current GPS and next goal GPS
- current heading / orientation
- depth or free-space based local avoidance

The global objective remains simple: move toward the next GPS goal.

The local objective becomes: choose a safe steering direction that is as close as possible to the GPS bearing.

#### Why it is strong

- deterministic
- easy to debug
- independent of learned policy failures
- does not depend on LogoNav weights or ML inference reliability
- useful as a competition-day backup when the learned controller misbehaves

#### What it will not solve

By itself it does not understand:

- long-range route structure
- sidewalks vs buildings
- semantic crossing behavior
- complex urban topology

So it is best viewed as a robust local follower, not a full-city planner.

### Strategy 3: OSM Routing + One Local Controller

This should be treated as the **global-planning upgrade**.

#### Core idea

Instead of telling the robot to drive directly from current GPS to final checkpoint GPS, use OpenStreetMap to generate a sequence of intermediate pedestrian-appropriate GPS waypoints.

Then let either:

- Strategy 1 follow those waypoints
- or Strategy 2 follow those waypoints

#### Why it is strong

This solves one of the biggest practical outdoor failures:

- straight-line GPS targets are often geometrically wrong in cities
- a direct bearing may point through a fence, building, grass patch, or unsafe crossing

OSM routing gives the system a more realistic global path:

- along sidewalks
- along paths
- around blocks
- through legal pedestrian routes

#### Why it is not standalone

OSM routing does not drive the robot.

It only provides intermediate goals.

You still need a local controller and safety stack.

## Recommended Priority Order

If we care about reliability over novelty, the order should be:

1. Strategy 1: `LogoNav + Safety + Recovery`
2. Strategy 2: `GPS + VFH / Depth`
3. Strategy 3: `OSM routing + one of the above`

This ranking is based on practical competition value, not paper novelty.

### Why this order is correct

- Strategy 1 has the highest ceiling and already has working code roots
- Strategy 2 is the most robust fallback and gives operational insurance
- Strategy 3 is the strongest global improvement once local control exists

## What Should Not Be Prioritized Right Now

The following ideas are interesting but should not be the immediate focus.

### 1. Full new foundation-model replacement

Building a totally new ViNT / NoMaD / GNM stack from scratch is too expensive unless the current LogoNav line completely fails.

### 2. VLM semantic controller layer

Adding CLIP / captioning / semantic hints may be useful later, but it is not the most reliable first path.

It adds complexity and uncertainty before the simpler stacks are battle-tested.

### 3. Reusing the indoor corridor localization backbone outdoors

This is conceptually wrong for the outdoor mission format.

Outdoor is not a repeated visual memory corridor task.

## Concrete Build Plan

The current team split should be:

### Teammate-owned work

The teammate is expected to address the Phase 1 stabilization of the current LogoNav-style code.

That includes things like:

- robust send-control behavior
- removing crash-on-send failure behavior
- cleanup of obvious bugs in the current `mbra_gps.py` path
- turning the current file into a cleaner production-ready outdoor controller

### Our parallel work

The pieces that can be built independently right now are:

1. `src/outdoor_gps_controller.py`
2. `live_outdoor_runtime.py`
3. `src/osm_router.py`

### Recommended implementation order

#### Step 1: `src/outdoor_gps_controller.py`

Build the classical controller first.

It should:

- take current GPS / UTM
- take next target GPS / UTM
- take current heading
- optionally take depth / free-space observation
- output `linear` and `angular`

This gives a strong non-ML baseline immediately.

#### Step 2: `live_outdoor_runtime.py`

Build the outdoor runtime wrapper next.

It should handle:

- dry-run mode
- send-control mode
- front image fetch
- telemetry fetch
- checkpoint progression
- controller selection
- graceful stop behavior
- logging for debugging

This should be the outdoor analog of the indoor live runtime, but with GPS mission structure instead of corridor-step structure.

#### Step 3: `src/osm_router.py`

Build the route-planning layer after the runtime and controller exist.

It should:

- accept start GPS and goal GPS
- query or load pedestrian-relevant map structure
- generate intermediate route waypoints
- feed those waypoints into the runtime

This is the right order because routing is only valuable once the system already knows how to follow GPS waypoints robustly.

## Proposed File Structure

A clean outdoor structure could look like this:

```text
outdoor_controller.md              # this living planning document
outdoor_logonav.py                 # cleaned learned outdoor local controller
live_outdoor_runtime.py            # outdoor runtime / mission runner
src/
  outdoor_gps_controller.py        # classical GPS + VFH controller
  osm_router.py                    # OSM routing and intermediate waypoint logic
  depth_estimator.py               # reused safety component
  earthrover_interface.py          # reused robot interface
```

The original `mbra_gps.py` can remain as a reference implementation until the cleaned version is ready.

## Current Assumptions About Competition Deployment

These assumptions should be revisited if the organizers or teammate provide new details.

### Assumption 1

We will receive either:

- the next GPS checkpoint
- or the full list of mission checkpoints

at runtime.

### Assumption 2

The SDK `/data` endpoint will expose enough telemetry for:

- latitude
- longitude
- heading / orientation
- maybe wheel / RPM hints

### Assumption 3

The `/v2/front` image endpoint is sufficiently stable for local control.

### Assumption 4

Remote/cloud compute is allowed, so CPU/GPU inference outside the robot is acceptable.

### Assumption 5

OSM query access may be available, but if internet/query reliability is uncertain we may need route caching or prefetch.

## What Inputs We Still Need From The Team

To build the outdoor system correctly, we still need a few concrete things.

### 1. Outdoor mission payload format

We need to know whether the competition runtime gives:

- one next checkpoint at a time
- or the full list of mission checkpoints
- or extra route / neighborhood map context

### 2. Real outdoor `/data` sample

We need one real outdoor telemetry sample so we can verify:

- field names
- coordinate formats
- heading conventions
- timestamp behavior
- RPM availability

### 3. Outdoor data sample

If the team collected outdoor data, we need:

- sample images
- GPS logs
- metadata shape
- folder structure

That will help for:

- controller testing
- future tuning
- possible model fine-tuning

### 4. OSM deployment assumption

We need to know whether OSM can be queried online during missions or whether routing must be cached/precomputed.

### 5. Teammate's stabilized LogoNav code

Once the teammate finishes the Phase 1 fixes, we need the updated file so the learned outdoor controller can be integrated cleanly with the rest of the runtime.

## Round-Level Strategy Thinking

A likely practical competition usage pattern is:

### Round 1

Use the simplest reliable learned stack:

- LogoNav local controller
- plus safety / recovery

The purpose is to calibrate what breaks under real mission conditions.

### Round 2

Continue with the learned stack if it behaves well.

If it is unstable, use the classical GPS + VFH fallback.

### Rounds 3 to 5

Prefer OSM-routed waypoint execution for more complex urban geometry, using whichever local controller has proven more reliable in prior rounds.

The real competition winner will likely not be the most elegant model, but the stack that finishes missions consistently.

## Reliability-Centered Philosophy

This outdoor track should be built around a simple principle:

> A slightly less intelligent system that keeps moving safely and recovers from problems is more valuable than a more impressive model that stalls once and scores zero.

This is especially important because EarthRover scoring is mission-completion oriented.

That means:

- safety layers matter
- graceful degradation matters
- fallback controllers matter
- stuck recovery matters
- clean runtime behavior matters

## Current Final Recommendation

At this point, the recommended outdoor plan is:

### Primary solution

- `LogoNav + Safety + Recovery`

### Fallback solution

- `GPS + VFH / Depth`

### Harder-mission upgrade

- `OSM routing + whichever local controller proves best`

This is the current best balance of:

- reliability
- development speed
- compatibility with the current repo
- competition realism

## Update Policy For This File

Going forward, use this file as the living outdoor log.

It should be updated whenever we do any of the following:

- refine the outdoor strategy
- inspect outdoor code
- add new outdoor modules
- learn something from a teammate's code
- clarify the official task constraints
- change our ranking of the top strategies
- discover a new blocker or requirement

## Last Updated Summary

As of now:

- the outdoor task is understood as GPS-checkpoint navigation in unknown urban environments
- the teammate's `mbra_gps.py` has been understood as a GPS-conditioned LogoNav-style learned local controller
- the best three outdoor strategies have been narrowed to:
  1. LogoNav + safety + recovery
  2. GPS + VFH / depth fallback
  3. OSM routing + one local controller
- the correct next independent work is:
  1. `src/outdoor_gps_controller.py`
  2. `live_outdoor_runtime.py`
  3. `src/osm_router.py`
- the teammate is expected to handle the Phase 1 stabilization of the learned controller path

## 2026-03-24 Implementation Update

### Built: `src/outdoor_gps_controller.py`

The first independent outdoor module now exists.

What it currently does:

- computes GPS bearing error in UTM space
- outputs a classical proportional heading-following command by default
- supports goal-reached stopping within a configurable radius
- optionally accepts polar clearance vectors and bin centers for VFH steering
- optionally accepts a depth estimator + depth map and computes polar clearance internally
- stops when forward clearance is too low
- slows when forward clearance is marginal
- returns structured debug information for runtime logging

This is intended to be the non-ML fallback controller and the base controller for the future outdoor runtime.

What still needs to be built around it:

- `live_outdoor_runtime.py`
- integration with `DepthEstimator` in a live loop
- `src/osm_router.py`
- controller selection / switching logic in the outdoor runtime

### Built: `live_outdoor_runtime.py`

The outdoor runtime wrapper now exists.

What it currently does:

- accepts a single goal, flattened checkpoint pairs, or a checkpoint JSON file
- converts mission checkpoints into UTM once at startup
- uses the same dry-run vs send-control pattern as the indoor runtime
- connects through the existing `EarthRoverInterface`
- fetches telemetry and optional front-camera frames each loop
- converts SDK compass heading into a math-angle convention for control
- runs the new `OutdoorGPSController` each tick
- optionally runs depth estimation every N ticks and reuses cached polar clearance
- tracks active checkpoint index and advances when the checkpoint is confirmed reached
- logs checkpoint progress, including partial-completion progress fraction
- stops the robot cleanly on final checkpoint completion or interrupt

Why this matters:

The outdoor challenge awards partial points by checkpoints, so mission/checkpoint progression is now represented explicitly in code instead of being left as an abstract planning idea.

What is next:

- `src/osm_router.py`
- optional controller selection between GPS fallback and the teammate's learned outdoor controller
- future stuck/recovery wrapper once real outdoor tests expose the failure modes

### Built: `src/osm_router.py`

The outdoor routing module now exists.

What it currently does:

- computes a bounding box around start and goal GPS coordinates
- builds an Overpass query for pedestrian-relevant highway classes
- parses Overpass JSON directly instead of depending on a new OSM client library
- constructs a lightweight directed graph from OSM nodes and ways
- applies higher traversal cost to road-like classes and lower cost to footway/path classes
- snaps the start and goal to the nearest graph nodes within a configurable tolerance
- runs A* over the graph
- densifies long route segments into shorter waypoint hops
- thins redundant waypoints to keep the route usable by the runtime/controller
- returns an `OSMRouteResult` with waypoints, node path, total distance, and debug metadata

What has been verified so far:

- the module compiles
- a small offline smoke test with a synthetic Overpass payload succeeded

What is still not verified yet:

- real Overpass API query behavior from this environment
- route quality on actual competition locations
- integration into `live_outdoor_runtime.py`

This means the next real step after this is not more brainstorming. It is runtime integration: adding an OSM-routed checkpoint expansion path into `live_outdoor_runtime.py`.

### Integrated: OSM Routing Into `live_outdoor_runtime.py`

The outdoor runtime now supports optional OSM route expansion at startup.

What changed:

- `src/osm_router.py` now uses the configured Overpass timeout in the query string
- `src/osm_router.py` now retries requests and falls back to a straight-line route instead of crashing when routing fails
- `live_outdoor_runtime.py` now supports `--osm-route`
- when enabled, the runtime expands each mission leg into intermediate pedestrian waypoints before the loop starts
- the runtime now distinguishes between:
  - internal navigation targets
  - real mission checkpoints

This distinction is important because the competition awards partial points only for real checkpoints, not for OSM-generated intermediate waypoints.

Current verification status:

- `osm_router.py` compiles
- `live_outdoor_runtime.py` compiles
- the OSM fallback path was smoke-tested and returns a straight-line route when Overpass is unavailable

What remains next:

- integrate the teammate's cleaned learned outdoor controller into the same runtime skeleton
- decide whether the default competition path should be:
  - classical GPS fallback
  - learned LogoNav path
  - or runtime-switchable between both
- add real outdoor stuck/recovery behavior once field testing starts

### Read: `new_mbra_gps.py`

The teammate's updated learned outdoor controller makes three meaningful improvements over the older `mbra_gps.py`:

- control sending is now robust to `/control` failures and falls back to `/control-legacy`
- control-send failures no longer crash the whole program; they are logged instead
- command smoothing / rate limiting was added to reduce abrupt linear/angular jumps

Additional cleanup in the new file:

- removed the per-tick `front_frame.png` disk write
- fixed the old uninitialized-command path by initializing linear/angular to zero at the start of `policy_calc()`
- resets previous smoothed commands to zero on sensor / GPS / image failure and on final goal completion
- changes angular command conversion from `atan(dy/dx)` to `atan2(dy, dx)` for safer quadrant handling
- adds turn-based linear slowdown before final command limiting

The new file is therefore a real stabilization of the teammate's learned controller path, but it is still a standalone script rather than being integrated into the shared `live_outdoor_runtime.py` runtime yet.

### Integrated: Learned Controller Into Shared Outdoor Runtime

The teammate's stabilized learned controller path is now wrapped and integrated into the shared outdoor runtime.

New file:

- `src/outdoor_logonav_controller.py`

What this wrapper does:

- loads LogoNav model/config from `mbra_repo`
- preserves the original heading convention from the teammate's standalone script
- accepts raw SDK `orientation_deg`
- maintains image context queue internally
- exposes `update_goal(goal_utm, goal_compass_rad)`
- exposes `compute_command(frame_rgb, current_utm, orientation_deg)`
- returns the same structured control-command style used by the shared runtime

Runtime integration:

- `live_outdoor_runtime.py` now supports `--controller gps` and `--controller logonav`
- GPS mode still uses the math-angle conversion path
- LogoNav mode intentionally uses raw `orientation_deg` and lets the wrapper compute `cur_compass` internally
- OSM routing and mission checkpoint accounting remain in the runtime, not inside the controller
- target items now carry `goal_compass_rad` so the learned controller can receive a segment-level goal heading

This means the shared outdoor runtime now supports:

- classical fallback controller
- learned LogoNav controller
- OSM-expanded waypoint execution
- mission-level checkpoint scoring in one place

### Deferred: Offline Replay Evaluation After Competition

A more thorough offline replay/evaluation harness has been discussed, but it is intentionally deferred rather than being treated as an immediate build priority.

What that deferred work would be:

- load recorded outdoor HDF5 runs
- replay front frames plus GPS/orientation through the outdoor controllers
- define artificial short/medium GPS waypoint tasks inside recorded trajectories
- measure command sanity, heading alignment, forward progress, spin/stall behavior, and safety-trigger frequency
- use the results to tune recovery thresholds and controller robustness in a more systematic way

Why it is deferred for now:

- the current rough outdoor dataset is off-road / trail-like and does not match the competition's urban sidewalk domain closely enough to justify spending competition time on a full replay-analysis workflow
- there is a nontrivial timestamp-alignment issue between some frame streams and telemetry streams, which would need careful treatment before replay conclusions could be trusted
- one file has partially corrupted controls, which reduces the value of building a control-centric replay pipeline immediately
- the near-term engineering priority is still live competition robustness, not post-hoc analysis infrastructure

How the current rough dataset should be used instead:

- pipeline verification only
- rough stress-testing intuition for vibration, glare, rough ground, and dynamic obstacles
- sanity-checking that the runtime/controller stack does not crash on real robot data

Recommended immediate priorities instead of replay-harness work:

- GPS-based stuck detection and recovery in the shared outdoor runtime
- one real urban outdoor field test as soon as possible
- cautious validation of depth/safety behavior in real competition-like scenes

When to revisit this deferred work:

- after the competition, or earlier only if there is unexpected schedule slack
- once more representative urban outdoor data is available
- once timestamp synchronization expectations are understood and documented more clearly

