# Current Codebase Deep Read - [0.3em] Read-only summary of the current SDK, runtime, and MBRA paths

> Source: `docs/current_codebase_deep_read.tex`
> Master Note: [[erc3_full_documentation]]

# Purpose

This note is a **read-only understanding document** for the current state
of the repository after many iterative changes.

The goal is not to propose a redesign. The goal is to capture what the code
**actually does now**, especially in the following areas:

- the SDK bridge,
- the live runtime,
- the current simple-controller path,
- the localization / temporal state path,
- the current MBRA deployment wrapper,
- the optional depth-safety path.

# High-Level Bottom Line

\begin{mdframed}
The repo is no longer just "baseline localization + graph planner + a small
controller." It is now a more integrated system with:

- a **hybrid SDK bridge** that supports both browser-relay and direct RTM fallback,
- a much heavier **live runtime wrapper** with recovery and guardrails,
- a **simple controller** that has become more forward-biased and less heading-driven,
- a **temporal localizer** that can now be reverted after jump rejection,
- a more faithful **MBRA inference wrapper**,
- optional **depth veto / slowdown** support.

\end{mdframed}

# SDK Bridge: What It Does Now

## Core idea

The SDK side is now built as a **hybrid bridge**.

It no longer depends on only one path for command or telemetry flow.

## Control paths

There are now two meaningful control paths:

1. `/control`
1. `/control-legacy`

\textbf{`/control`} uses the browser-backed path:
- the backend initializes a browser session,
- opens `/sdk`,
- joins the SDK page,
- sends control through the JavaScript `window.sendMessage(...)` hook.

\textbf{`/control-legacy`} uses direct RTM send without requiring the
browser relay.

This means manual control and autonomous control are now more resilient to
browser-session failures than before.

## Telemetry paths

Telemetry is also hybrid now.

The frontend JavaScript actively pushes data back into the backend:

- `basicRtm.js` listens to incoming RTM peer messages and posts them to `/api/update_data`.
- `basicVideoCall.js` periodically captures the front camera frame and posts it to `/api/update_frame`.

Then the backend exposes:

- `/data`
- `/v2/front`

Those endpoints now:

- prefer the lightweight cache populated by the JS push path,
- but fall back to `browser_service` if the cache is empty.

## Practical consequence

This is a major structural improvement over a pure pyppeteer-pull design.

The current SDK bridge is trying to achieve three things at once:

- low-latency data access through pushed caches,
- survivability when the lightweight cache is empty,
- a legacy fallback when the browser-backed command path is not healthy.

## Manual teleop

The current intended manual teleop is `examples/keyboard_control.py`.

Its behavior is:

- initialize `/sdk`,
- capture live W/A/S/D key state with `pynput`,
- smooth target velocities,
- send commands through `/control`,
- fall back to `/control-legacy` if needed.

`examples/simple_control.py` was also upgraded, but it is more of a
minimal terminal teleop helper than the main curses-based teleop path.

# Live Runtime: What It Does Now

## Original role vs current role

Originally, the live runtime was conceptually a thin loop:

\fbox{\parbox{0.92\textwidth}{
camera $\rightarrow$ localize $\rightarrow$ plan $\rightarrow$ controller
$\rightarrow$ send command
}}

That is no longer the full story.

The current `live_indoor_runtime.py` contains a large amount of
**runtime policy** and **recovery logic** around the controller.

## What the live loop does now

The loop still performs the same main phases:

1. get camera frame from the SDK,
1. get telemetry from the SDK,
1. run the motion-state filter,
1. run localization,
1. run graph planning,
1. assemble controller input,
1. ask the selected controller for a command,
1. post-process that command through runtime safety / recovery logic,
1. optionally send the command.

But after controller inference, the runtime now adds several additional layers.

## Runtime layers added on top of the controller

1. **Controller-specific defaults**:
1. MBRA defaults to 3 Hz and 4 subgoal hops.
1. Simple defaults to 2 Hz and 15 subgoal hops.

1. **Target reached confirmation**:
1. target is not declared reached from one frame alone,
1. it requires confidence and repeated confirmation ticks.

1. **Localization jump rejection**:
1. large step jumps require higher confidence,
1. rejected jumps trigger localizer state revert.

1. **Proximity slowdown**:
1. forward speed is reduced near the target.

1. **Recovery backup mode**:
1. a reverse override can temporarily replace normal control,
1. after recovery, runtime and controller state can be reset.

1. **Angular saturation override**:
1. repeated near-max angular output triggers a forced straight override.

1. **Optional monocular depth veto**:
1. slows or stops forward motion if forward clearance looks small.

1. **RPM stall detection**:
1. low RPM during commanded forward motion can trigger backup recovery.

1. **No-progress reset**:
1. if the localized step does not change for long enough, the controller can be reset.

## Most important conceptual consequence

The controller file is no longer the whole control story.

The current on-robot behavior is a combination of:

- the selected controller,
- the sensor-state filter,
- localization confidence and jump rejection,
- runtime overrides,
- recovery logic,
- optional depth safety.

So when the robot behaves a certain way, it may not be coming from the
controller alone.

# Current Simple Controller: What It Really Is

## What changed in spirit

The current simple controller is much less like a continuous heading controller
than the original baseline intuition suggested.

It is now closer to:

\fbox{\parbox{0.92\textwidth}{
brief alignment when needed $\rightarrow$ then mostly forward crawl
with outer runtime logic handling many failures
}}

## Important current settings

Key changes in the current simple controller:

- `min_linear` increased to 0.10,
- `drive_heading_gain` is now 0.0,
- alignment thresholds are much wider,
- alignment duration is capped by `max_align_ticks`,
- no-heading mode is more aggressive and always maintains forward crawl.

## Very important runtime interaction

The runtime now explicitly sets:

`controller_input["subgoal_orientation"] = None`

That means the simple controller usually does **not** operate with real
subgoal heading information during the live loop.

In practice, this pushes it into the `no_heading_forward_crawl` style
of behavior much more often.

## Implication

The current simple path is best described as:

- localization says where we are,
- planning says which nearby graph step is ahead,
- the controller tries to move forward conservatively,
- the runtime wrapper provides many of the recovery behaviors.

This is not a rich trajectory-following controller at the moment.

# Motion-State Filter

## Role

`sensor_state.py` is still a lightweight motion-prior layer, not a full
sensor-fusion backbone.

## Current signals used

It currently derives:

- filtered heading from orientation,
- filtered heading-rate from gyro Z,
- filtered RPM mean from the latest RPM packet,
- telemetry freshness / stale status.

## Important change

The stale logic no longer depends on "did update() get called just now?"

It now tries to use the sensor timestamp to decide whether data is actually
fresh. That is a more correct design.

The timeout is also now more tolerant at 5 seconds.

## Practical effect

This filter is not driving localization directly in a heavy fusion sense.
Instead, it provides:

- heading hints,
- turn-rate damping hints,
- RPM-based motion hints,
- a freshness stop signal.

# Localization and Temporal State

## Corridor localizer

The core localization backbone remains the same:

- preprocess frame,
- run CosPlace descriptor inference,
- retrieve nearest database matches,
- score candidates through the temporal localizer.

## New practical changes

The corridor localizer now adds two deployment-focused capabilities:

- explicit device selection / CPU fallback for CUDA incompatibility,
- the ability to revert the last temporal update.

## Temporal localizer change

The temporal localizer now supports:

- `save_state()`
- `revert_state()`

This is specifically there so runtime jump rejection can say:

> "That large localization jump was not trusted, so roll back the temporal
> state and keep the previous estimate."

## Meaning

This is an important conceptual change.

Before, temporal localization was just a forward accumulator.
Now it is part of a runtime closed loop where:

- localization proposes a step,
- runtime judges whether that jump is believable,
- runtime can reject it and restore the previous temporal state.

# Navigation Runtime Handoff

`navigation_runtime.py` is still a fairly clean handoff layer.

Its main job is:

1. localize,
1. plan path to target,
1. load the subgoal image if available,
1. package controller input.

## Important portability fix

It now repairs absolute image paths from graph artifacts when they were created
on another machine.

This is practical and important because many graph databases were generated on a
different system and carried over.

# Current MBRA Path

## Main conceptual stance

The code now treats MBRA as a **short-horizon local controller only**.

That matches the intended project architecture and also matches the strongest
handoff documents.

## What the current wrapper does

The MBRA wrapper now:

- loads model config and weights,
- maintains only the observation image history,
- uses a fixed `vel_past` tensor,
- uses timestep `[0,0]` as the immediate action,
- clamps output to the repo's safe command range,
- enforces a minimum forward command to avoid deadlock.

## Why this matters

Earlier deployment-style hacks such as:

- feeding back previous predicted commands into `vel_past`,
- EMA blending over outputs,
- keeping controller-side motion history,

have been removed.

The current wrapper is therefore much closer to:

> "run the MBRA model the way the original design intended, then only add small
> deployment clamps and confidence gating."

## MBRA utility loading

`mbra_repo/deployment/utils_logonav.py` was changed to lazy-load heavy
model families.

This is not a navigation-policy change. It is a deployment / dependency change.

Its practical value is:

- MBRA and IL paths can load with fewer unnecessary imports,
- unrelated heavyweight stacks do not need to be imported unless actually used.

# Depth Path

`depth_estimator.py` remains an optional helper around Depth Anything V2.

The most important practical change is not the inference logic itself, but the
checkpoint search:

- it now searches the actual HuggingFace-style metric indoor filenames,
- it also preserves some legacy filename variants.

In the live runtime, depth currently acts only as:

- a forward-clearance slowdown layer,
- a forward-clearance stop layer,
- a trigger for recovery backup.

It is not part of localization or planning.

# Most Important Practical Takeaways

1. **The SDK path is much stronger than before.**
Command and telemetry now both have redundancy.

1. **The live runtime is now a major source of behavior.**
Reading only the controller file is no longer enough to understand the robot.

1. **The simple controller is currently a crawl-heavy baseline.**
It is not doing rich heading-based local control in the live loop.

1. **Localization remains the strongest validated subsystem.**
The newer changes mainly make it more robust to deployment issues and jump rejection.

1. **MBRA has been made more faithful to the original reference behavior.**
That makes live evaluation more meaningful than before.

1. **Many changes were practical deployment fixes, not architectural replacements.**
The codebase has been hardened in-place rather than rebuilt from scratch.

# Current Interpretation of the Stack

\begin{mdframed}
**Best current reading of the codebase:**

- visual localization + temporal filtering + graph planning remain the backbone,
- the SDK bridge is now materially better than it was before,
- the runtime wrapper has become much more responsible for recovery and command shaping,
- the simple controller is conservative and forward-biased,
- MBRA is now implemented more faithfully as an optional short-horizon controller,
- the main uncertainty is not plumbing anymore, but real autonomous execution behavior.

\end{mdframed}

# Related Documents

- [[erc3_full_documentation]] --- single master guide for the complete project story and current architecture.
- [[live_indoor_runtime_story]] --- indoor evolution, MBRA integration, and checkpoint-step runtime behavior.
- [[live_outdoor_ultra_marathon_story]] --- outdoor and marathon runtime evolution with safety-layer reasoning.
- [[outdoor_perception_review]] --- depth/semantic perception findings and their runtime implications.
