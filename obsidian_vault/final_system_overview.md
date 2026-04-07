# Final System Overview - The One Architecture Document for ERC-3 EarthRover Challenge

> Source: `final_system_overview.tex`
> Master Note: [[erc3_full_documentation]]

# Project Overview

This repository contains the final working architecture for the ERC-3 EarthRover
Challenge project. In practice the project split into two quite different systems:

- an **indoor known-corridor system**, where repeated corridor structure
and pre-recorded data could be exploited heavily, and
- an **outdoor mission runtime**, where the rover had to follow SDK
mission checkpoints in a live environment while staying inside a stricter safety
envelope.

Those two tracks share code and hardware, but they are not just two parameter
settings of the same algorithm. The indoor stack is built around visual place
recognition and graph progression. The outdoor stack is built around mission
checkpoints, routed waypoints, and controller hardening.

# The Story in One Page

The project started from a simpler indoor question: can we turn a known corridor
recording into a usable navigation memory and drive to competition checkpoints
reliably? That immediately pushed the system toward a practical stack:

- CosPlace-style visual retrieval for localization,
- temporal smoothing to reduce flicker,
- graph planning over the corridor memory,
- and a short-horizon controller for immediate motion.

The outdoor track changed the project. GPS checkpoints, sidewalks, route structure,
and safety constraints made it obvious that the corridor backbone could not simply
be reused outdoors. That led to the split:

- **Indoor:** corridor memory + temporal localization + graph progression +
MBRA as local controller.
- **Outdoor:** mission checkpoints + optional OSM routing + LogoNav as
local controller + runtime safety layers.

# Full Architecture Diagram

```
                         +----------------------+
                         |  EarthRover SDK/API  |
                         |  camera / telemetry  |
                         +----------+-----------+
                                    |
                    +---------------+---------------+
                    |                               |
                    v                               v
        +------------------------+      +--------------------------+
        | Indoor corridor stack  |      | Outdoor mission stack    |
        +------------------------+      +--------------------------+
        | corridor_localizer.py  |      | live_outdoor_runtime.py  |
        | temporal_localization  |      | earthrover_interface.py  |
        | graph_planner.py       |      | osm_router.py            |
        | navigation_runtime.py  |      | outdoor_logonav_control  |
        | mbra_controller.py     |      | outdoor_gps_controller   |
        +-----------+------------+      +------------+-------------+
                    |                                |
                    v                                v
         checkpoint-step subgoals         mission checkpoints / route waypoints
                    |                                |
                    v                                v
           local controller output         controller output + safety envelope
                    |                                |
                    +---------------+----------------+
                                    |
                                    v
                           rover control commands
```

# Indoor Track

The indoor pipeline is assembled in `live_indoor_runtime.py`. The most
important modules are:

```text
\toprule
\textbf{File} & \textbf{Main symbol} & \textbf{Role} \\
\midrule
\texttt{src/corridor\_localizer.py} & \texttt{CorridorLocalizer} & CosPlace-based retrieval over the corridor descriptor database; turns a live image into a candidate corridor step and confidence. \\
\texttt{src/temporal\_localization.py} & \texttt{TemporalLocalizer}, \texttt{TemporalLocalizerConfig} & Adds continuity penalties and ambiguity handling so one bad frame does not rewrite the pose. \\
\texttt{src/graph\_planner.py} & \texttt{GraphPlanner}, \texttt{GraphPlannerConfig} & Converts current step and active checkpoint into a path and nearby subgoal step. \\
\texttt{src/navigation\_runtime.py} & \texttt{NavigationRuntime} & Orchestrates localizer + planner and returns controller input bundles. \\
\texttt{src/mbra\_controller.py} & \texttt{MBRAController} & Short-horizon goal-image controller for indoor motion. \\
\texttt{src/local\_controller.py} & \texttt{SimpleLocalController} & Backup heading-based controller when MBRA is not used. \\
\bottomrule
```

In current code, `TemporalLocalizerConfig` defaults include
`top_k=10`, `max_step_jump=20`, `jump_penalty=0.05`,
`backward_penalty=0.15`, `heading_penalty=0.002`,
`ambiguity_margin=0.05`, and `hold_on_ambiguity=True`.

For MBRA-based indoor competition mode, `live_indoor_runtime.py` sets:

- `max_subgoal_hops = 8`
- `tick_hz = 3.0`
- `depth_stop_m = 0.25`
- `depth_slow_m = 0.60`

The indoor tick loop is conceptually:

```
frame -> corridor localizer -> temporal stabilization -> graph planner
      -> choose active checkpoint / subgoal image
      -> MBRA or simple controller
      -> optional depth veto
      -> send command or hold
```

# Outdoor Track

The outdoor pipeline is concentrated in `live_outdoor_runtime.py`. It pulls
together:

- mission checkpoints from the SDK via `src/earthrover_interface.py`,
- optional OSM route expansion via `src/osm_router.py`,
- local motion from either `src/outdoor_logonav_controller.py` or
`src/outdoor_gps_controller.py`,
- and a thick runtime safety envelope around the chosen controller.

Key outdoor runtime defaults visible in the code include:

- `--controller logonav`
- `--goal-radius-m 8.0`
- `--intermediate-goal-radius-m 3.0`
- `--checkpoint-confirm-ticks 1`
- `--stuck-window-ticks 15`
- `--route-corridor-stop-m 10.0`

# Safety Architecture

```text
\toprule
\textbf{Authority level} & \textbf{Typical layers} & \textbf{Meaning} \\
\midrule
Hard stop & IMU safety, health gate, explicit operator intervention, severe GPS / vision faults & Motion must stop immediately. \\
Hold / halt & nav-ready gate, camera watchdog, mission setup wait, hard-stop cooldown & The runtime pauses until preconditions return. \\
Constraint & traversability slowdown/stop, speed caps, route corridor guard, recovery attempt limits & Controller output is clipped or replaced. \\
Soft bias & semantic angular bias, caution scaling, informational diagnostics & Motion is nudged rather than vetoed. \\
Info only & richer logs, waypoint / checkpoint counters, visibility diagnostics & Helps the operator understand behavior. \\
\bottomrule
```

# Composite Safety Modes

## `--ultra-marathon`

Turns on IMU safety, health gate, leg pause, no reverse, route corridor guard,
nav-ready gating, and more conservative recovery behavior. It also caps
`max_linear` and `logonav_max_linear` at `0.24`, tightens
angular caps to about `0.32`, widens the stuck window, and enables
behind-waypoint pruning.

## `--night-safe`

Turns on lamp, vision safety, GPS safety, IMU safety, route corridor guard,
nav-ready gating, and traversability. It caps `logonav_max_linear` at
`0.22`, `logonav_max_angular` at `0.28`, and makes
traversability thresholds more conservative.

## `--sidewalk-strict`

Forces OSM routing, disables straight-line fallback, and turns on route corridor
guard, GPS safety, traversability, semantics, semantic hard-stop, semantic yield,
and semantic sidewalk-stop.

# Design Decisions

- **MBRA indoors**: used as a local controller on top of corridor
localization and graph progression, not as a planner.
- **LogoNav outdoors**: kept as the main learned outdoor controller while
runtime reliability was hardened around it.
- **CosPlace over heavier matching**: simpler and sufficient for the
known-corridor problem.
- **Depth and semantics as support**: useful for caution and vetoes, but
not reliable enough to become the whole intelligence layer.

# Operations

## Indoor

```
python live_indoor_runtime.py \
  --checkpoint-steps 45 480 761 821 1094 1208 1345 1430 1544 1638 1764 \
  --auto-advance-checkpoints --send-control --controller mbra --depth-safety
```

## Outdoor

```
python live_outdoor_runtime.py --mission --send-control --controller logonav --osm-route
```

## Outdoor marathon

```
python live_outdoor_runtime.py \
  --mission --send-control --controller logonav --osm-route --ultra-marathon
```

# Troubleshooting

```text
\toprule
\textbf{Symptom} & \textbf{Likely cause} & \textbf{What to check first} \\
\midrule
Indoor says \texttt{no path} forever & Robot already moved past target in forward-only graph & checkpoint-step list, confidence, skip-past logic \\
Indoor spins or stalls repeatedly & stale controller context or poor recovery mode & MBRA mode, no-progress handling, depth veto logs \\
Outdoor reroutes too often & route corridor threshold too aggressive or waypoint handoff too loose & route corridor flags, intermediate waypoint radius, prune settings \\
Outdoor spins near a target & alignment logic applied to wrong target regime & whether target is mission checkpoint or routed waypoint \\
Night run keeps stopping & image quality gate or conservative traversability & vision safety logs, lamp state, clearance thresholds \\
\bottomrule
```

# Current Status

- **Working strongly:** indoor localization backbone, checkpoint-step
planning, indoor MBRA deployment path, outdoor mission runtime skeleton, outdoor
LogoNav path, multiple outdoor safety layers.
- **Partial:** outdoor reliability under repeated live transitions,
semantics as a robust field signal, depth/traversability calibration across all
terrain.
- **Still weak:** physical stability and target-transition stability in
long outdoor runs.
