# ERC-3 EarthRover Challenge - Definitive Full Documentation

> Source: `erc3_full_documentation.tex`

> [!abstract] Note Header
> **Purpose:** Canonical master note for the whole project: story, architecture, decisions, outcomes, and current status.
> **Read when:** I need the full project picture or need to trace a claim back to the main documentation.
> **Authority:** Highest documentation-level note in the vault. If it conflicts with active runtime behavior, trust current code first.

> [!important] Current Branch Update (2026-07-16)
> This converted master document explains the completed indoor, GPS-outdoor, and
> marathon story. A later no-GPS rough-terrain teach-and-repeat branch is now
> active. Its data and command pipeline work, but field testing has not yet
> proved reliable autonomous repeat. Do not infer that it is competition-ready
> from the historical systems described below. Read [[00 Home/Current No-GPS - Read This First]] and [[01 Source of Truth/No-GPS Field Trial - Findings]] for the current evidence.

# What This Document Is

This is the single definitive documentation file for the ERC-3 EarthRover
Challenge project.

The point of this file is simple: if someone asks for one document that explains
what this project was, what code was used, what the main architecture became, what
worked, what failed, and what still remains open, this should be the file they
read.

This document deliberately combines:

- the project story,
- the current architecture,
- the important code structure,
- the reasoning behind the main design decisions,
- the competition results,
- and the real failure modes.

It is not a short README and it is not a polished paper. It is meant to be the
technical truth layer in narrative form.

> [!note] How to use this document
> Read this once from top to bottom. After that, use it by question. If you want
> the project story, read the phase and design-decision sections. If you want to
> run the system, jump to operations. If you want to defend it in an interview,
> read the interview section. If you want the math slowly and from intuition,
> read the companion file `erc3_mathematics_guide.txt`.

# Documentation Control and Revision Log

## Canonical source and companion docs

- **Canonical master file**: [[erc3_full_documentation]]
- **Companion narrative/history docs**: all files listed in
`Related Documents` at the end of this master file
- **Code authority rule**: if any doc conflicts with runtime behavior,
trust active code paths in `live_indoor_runtime.py` and
`live_outdoor_runtime.py`
- **Current field-evidence rule:** for no-GPS rough-terrain behavior, trust
  [[01 Source of Truth/No-GPS Field Trial - Findings]] over historical
  deployment language in this converted narrative

## Verification tooling used for this documentation

- `scripts/docs/lint_tex_docs.py`
- `scripts/docs/check_links.py`
- `scripts/docs/check_command_references.py`

## Revision log (most recent first)

```text
\toprule
\textbf{Date} & \textbf{Change summary} \\
\midrule
2026-04-06 & Added canonical runbook section, command conflict table, and evidence appendix with representative field log signatures. \\
2026-04-06 & Added C4 architecture views, complete CLI inventory, and expanded code-anchor sections for commonly disputed runtime behavior. \\
2026-04-06 & Added this documentation-control section to clarify source-of-truth, tooling, and update policy. \\
\bottomrule
```

## Reader Paths

If you do not want to read all 3000+ lines in one shot, use one of these paths:

- **5-minute overview path**: Project Summary $\rightarrow$ Competition
Results $\rightarrow$ Current Final Mental Model $\rightarrow$ Current Status.
- **Operator run path**: Practical Commands $\rightarrow$ Preflight
Sequence $\rightarrow$ Operations $\rightarrow$ Troubleshooting.
- **Debug path**: Indoor Tick-by-Tick $\rightarrow$ Outdoor Tick-by-Tick
$\rightarrow$ Code Anchors $\rightarrow$ Layer-by-Layer Debug Playbooks.
- **Interview path**: Main Design Decisions $\rightarrow$ Algorithms and
Math $\rightarrow$ Interview Mode $\rightarrow$ Harder Interview Questions.
- **Onboarding path**: Source of Truth section $\rightarrow$ Repo Map by
Question $\rightarrow$ One-Week Onboarding Plan $\rightarrow$ First-Hour section.

# Project Summary

The ERC-3 project turned into three related but structurally different systems:

- **Indoor known-corridor navigation**
- **Outdoor GPS mission navigation**
- **Outdoor marathon mode**

At the highest level, all three asked the rover to reach checkpoints. But that
surface similarity was misleading. The indoor and outdoor tracks became very
different engineering problems.

## Indoor in one sentence

Indoor was a **known-corridor checkpoint problem** where repeated structure
could be exploited heavily.

## Outdoor in one sentence

Outdoor was a **mission runtime problem** where controller quality, routing,
waypoint semantics, safety checks, and recovery logic all mattered together.

## Marathon in one sentence

Marathon was the outdoor stack placed under a stricter safety envelope and a much
harder stability requirement.

# Who Built It and Why That Matters

This repo did not emerge from one clean top-down design pass. It was shaped by:

- earlier code already present in the repository,
- teammate controller paths that were already partially alive,
- repeated debugging passes on the runtime,
- later documentation and architecture cleanup that made the system more
legible.

That matters because many of the most important decisions were not abstract design
preferences. They were forced by actual bad behavior in logs and field runs.

# The Story of the Project

## The original wrong question

At the very beginning, it is easy to ask the wrong question:

> How do we make the rover autonomously reach checkpoints?

That question is too vague. It hides the fact that indoor and outdoor are
different problems.

## The better indoor question

The indoor question became:

> How do we use the fact that the corridor is known and repeated so the rover can
> move from exact checkpoint step to exact checkpoint step?

## The better outdoor question

The outdoor question became:

> How do we keep a mission runtime stable and safe in a live environment where the
> meaning of the target changes over time?

## What failed early

Several bad assumptions had to die:

- that one controller name could explain the whole project,
- that more recovery logic automatically helps,
- that depth could instantly become a trusted hard-stop outdoor backbone,
- that semantics could immediately become the authoritative scene
understanding layer,
- that more caution automatically means more stability.

## What survived all the way through

What survived was:

- CosPlace-style indoor localization,
- temporal stabilization,
- graph progression over checkpoint steps,
- MBRA as indoor short-horizon control,
- LogoNav as outdoor short-horizon control,
- OSM routing as useful outdoor route structure,
- safety layers with limited, carefully scoped authority.

> [!note] The real project question
> By the end, the most honest project question was not "which controller did we
> use?" It was "how do we make the runtime remain coherent under uncertainty,
> especially during transitions?" That is the question that explains most of the
> later fixes.

# Competition Results

```text
\toprule
\textbf{Track} & \textbf{Goal} & \textbf{Achieved} & \textbf{Main Limitation} \\
\midrule
Indoor & 11 checkpoints & 8 reached & transition / ambiguity / recovery edge cases \\
Outdoor & full checkpoint mission & 1 full success, others partial & waypoint transition instability \\
Marathon & long endurance run & 1 checkpoint, then toppled & unstable post-transition turning + platform stability \\
\bottomrule
```

Those numbers matter, but they are not enough by themselves. The useful question is
what system produced them and what those results actually mean.

# Research Components Used

The project used several research ideas and model families. None of them were used
as a complete end-to-end magic solution. Each one had to be placed in the right
role inside the runtime.

## CosPlace

CosPlace was used for indoor visual place recognition. The basic role was:

- encode the current live corridor frame,
- compare it to stored corridor descriptors,
- recover the most likely corridor position.

This was the indoor localization backbone.

## MBRA

MBRA was used as the indoor learned local controller reference. The important point
is what it *was not*. MBRA was not used as:

- the indoor localizer,
- the indoor planner,
- or the whole indoor autonomy stack.

Instead, MBRA was used in the role it fit best: short-horizon control toward a
chosen nearby visual goal.

## LogoNav

LogoNav was used as the outdoor learned controller path. Like MBRA, it was not the
whole stack. It was the controller inside a larger mission runtime.

## Depth Anything V2

Depth Anything V2 provided monocular depth. It was useful for caution, obstacle
screening, and traversability-style reasoning, but it was not strong enough in this
project to be the only trusted outdoor safety mechanism.

## SegFormer

SegFormer was used for semantics. The main runtime use was semantic risk scoring and
steering bias, not full semantic autonomy. The labels were helpful, but the runtime
still needed strong safety logic above them.

## OpenStreetMap

OpenStreetMap was used to expand outdoor mission checkpoints into local route
structure. This became very important because a far GPS checkpoint is too coarse to
follow directly in a stable way.

# What The Final Project Split Became

The cleanest way to understand the project is to accept the split explicitly.

## Indoor

Indoor is:

- corridor localization,
- temporal stabilization,
- exact checkpoint-step graph progression,
- MBRA as local control,
- depth veto as safety.

## Outdoor

Outdoor is:

- SDK mission checkpoints,
- optional OSM-expanded intermediate waypoints,
- LogoNav as local control,
- runtime safety and recovery around it.

## Marathon

Marathon is:

- the outdoor stack,
- plus stricter safety, slower motion, stronger checks,
- and a much heavier requirement on stable target transitions.

> [!note] Architectural truth
> There is no single honest sentence like "we used MBRA" or "we used LogoNav"
> that explains this project. The real explanation is layered. Indoor uses
> localization + temporal stabilization + graph planning + MBRA. Outdoor uses
> mission semantics + route semantics + LogoNav + runtime safety.

# Hardware and Platform Reality

Even though most of this repository is software, the platform mattered constantly.

The project relied on:

- front camera imagery as the main live visual stream,
- GPS for outdoor waypoint and mission geometry,
- IMU data for tilt and angular-rate safety,
- telemetry through the EarthRover SDK,
- mission and control APIs through the SDK bridge.

The reason this belongs in the main document is that several software decisions
only make sense once the platform constraints are visible:

- monocular depth depends heavily on the camera geometry,
- indoor heading signals were noisy enough to hurt localization,
- repeated aggressive turning was not only a controller issue but a physical
stability issue,
- telemetry delays can imitate controller failure if not checked explicitly.

# Indoor System: The Full Story

## What Indoor Really Was

Indoor was not general building navigation. It was not SLAM. It was not GPS-free
autonomy in a random unseen structure. It was a repeated corridor problem with a
recorded dataset and a known topological route.

That single fact changed almost every good engineering decision downstream.

Because the corridor was known, we could do three things that made the final system
much more tractable:

- build a corridor memory offline,
- localize against that memory online,
- define checkpoints as exact corridor steps instead of vague goals.

## Indoor Data

The indoor stack started from a recorded H5 run of the corridor. That recording was
extracted into:

- front camera frames,
- metadata and telemetry,
- and later into a corridor database with descriptors and graph structure.

That dataset was not only for convenience. It was the foundation for both indoor
localization and indoor checkpoint definition.

## Indoor Code Structure

The key files in the indoor stack are:

```text
\toprule
\textbf{File} & \textbf{Role} \\
\midrule
\texttt{baseline.py} & Builds and queries the corridor descriptor database. \\
\texttt{src/corridor\_localizer.py} & Loads the corridor database and localizes a frame against it. \\
\texttt{src/temporal\_localization.py} & Smooths retrieval outputs over time using continuity penalties. \\
\texttt{src/graph\_planner.py} & Plans progress over the corridor graph and chooses a nearby subgoal. \\
\texttt{src/navigation\_runtime.py} & Orchestrates localization and planning. \\
\texttt{src/mbra\_controller.py} & Runs the MBRA learned policy as the local controller. \\
\texttt{src/local\_controller.py} & Provides the simple heading-based backup controller. \\
\texttt{live\_indoor\_runtime.py} & Main indoor runtime loop used in practice. \\
\bottomrule
```

## Indoor known-good defaults

One thing that older notes often blur is that the runtime uses different defaults
depending on controller choice. In the active indoor runtime:

```text
\toprule
\textbf{Quantity} & \textbf{MBRA mode} & \textbf{Simple mode} \\
\midrule
tick rate & 3.0 Hz & 2.0 Hz \\
subgoal hops & 8 & 15 \\
depth stop & 0.25 m & 0.40 m \\
depth slow & 0.60 m & 0.80 m \\
\bottomrule
```

This matters because it shows the runtime is not treating MBRA and the simple
controller as the same thing with different names. They are given different
operating envelopes.

## Indoor Runtime Flow

The indoor runtime can be understood as a very concrete loop:

```
camera frame
  -> corridor localizer
  -> temporal stabilization
  -> graph planner
  -> choose current checkpoint + nearby subgoal
  -> MBRA or simple controller
  -> optional depth safety
  -> send command
```

This loop matters because it shows the true division of labor. The local controller
was never asked to solve the entire indoor problem by itself.

## Why checkpoint-step mode was such a major simplification

Earlier indoor thinking mixed together:

- target images,
- target places,
- target checkpoints.

Checkpoint-step mode forced all three into one representation: a checkpoint was an
exact corridor step. That made the logs, the planner, and the runtime all speak
the same language.

## Indoor Localization

Indoor localization was based on CosPlace-style visual place recognition. A live
frame was embedded and matched against stored corridor embeddings. That produced a
candidate location in corridor-step space.

The important practical point is that the system did not trust one frame by itself.
That is why temporal stabilization exists.

## What Temporal Stabilization Means

Temporal stabilization means the system uses recent localization history to avoid
jumping to a new pose estimate just because one frame looked accidentally similar to
the wrong corridor region.

In the current code, the temporal localizer adds:

- jump penalties,
- backward penalties,
- heading penalties,
- and ambiguity hold behavior.

The defaults are:

- `top_k = 10`
- `max_step_jump = 20`
- `jump_penalty = 0.05`
- `backward_penalty = 0.15`
- `heading_penalty = 0.002`
- `ambiguity_margin = 0.05`
- `hold_on_ambiguity = True`

This is not heavy probabilistic filtering. It is a continuity-aware stabilizer. That
was enough for the corridor problem.

## Graph Progression Over Corridor Steps

Once localization produces a stable corridor step, the planner treats the indoor
route as a graph. Each competition checkpoint is an exact step. The planner then
selects a nearby subgoal some number of hops ahead.

This was one of the biggest conceptual wins in the project. Once the problem became
"move from exact step to exact step inside a known corridor graph," the rest of
the stack became easier to reason about.

## MBRA: What It Was Intended to Be

MBRA was intended as a short-horizon image-goal-conditioned local controller. The
idea from the paper direction is not "let MBRA solve the whole navigation
problem." The intended role is:

- something else tells you where you are and where you should head next,
- MBRA turns current image plus goal image into the immediate motion command.

That is exactly how it was most useful in this project.

## How MBRA Was Used Here

In this repo, MBRA became the indoor local controller on top of:

- corridor localization,
- temporal stabilization,
- graph progression,
- exact checkpoint-step mode.

So the division was:

- localization says where the rover is,
- the planner says what checkpoint/subgoal comes next,
- MBRA says how to move right now.

## MBRA Inference Details

The MBRA wrapper in `src/mbra_controller.py` keeps a short visual context,
uses fixed prior velocity inputs, and runs the model to produce a predicted
trajectory. The immediate command is taken from the first step of that output.

Important details used in this project:

- a 6-frame context was used,
- the model operated on resized images,
- fixed `vel_past` values were used rather than trying to feed noisy
live command history back into the model,
- the output is an immediate local command, not a global route.

## Why MBRA Was Made Sole Motion Authority

One of the main indoor discoveries was that MBRA behaved better when it was allowed
to remain the sole motion authority.

That means:

- no reverse-based recovery fighting the learned controller,
- no separate backup steering law trying to override it every few ticks,
- no competing runtime correction trying to reshape every turn.

The reason was practical: when the learned controller is already building motion
from a short visual context, piling another steering system on top of it can create
more oscillation rather than more safety.

## Why No Reverse for MBRA

Reverse behavior was deliberately avoided for MBRA-based indoor runs.

The important reason was not GPS or any external signal. The reason was visual
context. MBRA is fundamentally operating on a forward visual context. Reverse motion
changes that context in a way that made recovery behave worse in practice.

So the project chose:

- no reverse for MBRA,
- skip past impossible backward graph situations,
- reset stale context when needed.

## Forward-Only Graph Behavior

The corridor graph behaved like a forward-only graph in practice. That created a
very specific indoor bug: if the rover had already localized beyond a checkpoint,
the planner could legitimately return "no path" to that older target.

At first this looked like the planner being wrong. It was not wrong. The runtime
was asking the wrong question.

## Skip-Past-Checkpoint Logic

The fix was to add skip-past-checkpoint logic. If the rover was already beyond the
active checkpoint with enough confidence, the runtime advanced instead of stopping
forever.

This was one of the key runtime corrections that made indoor competition mode much
more usable.

## What Stale Context Reset Means

If MBRA stopped making progress for multiple ticks, the runtime stopped trusting the
old local context and forced a refresh from localization and planning. In simple
terms: do not keep giving the controller the same stale target if the robot is no
longer actually advancing toward it.

## Indoor Safety

Indoor safety was relatively simple compared to outdoor. The main additional layer
was monocular depth safety:

- hard stop if forward clearance is too small,
- slow down in a band above that threshold.

For MBRA mode in the main indoor runtime:

- `tick_hz = 3.0`
- `max_subgoal_hops = 8`
- `depth_stop_m = 0.25`
- `depth_slow_m = 0.60`

## Indoor Competition Run

The main indoor competition-style command was:

```
python live_indoor_runtime.py \
  --checkpoint-steps 45 480 761 821 1094 1208 1345 1430 1544 1638 1764 \
  --auto-advance-checkpoints --send-control --controller mbra --depth-safety
```

## Indoor Outcome

Indoor reached **8 out of 11 checkpoints**. This was the strongest completed
stack in the project.

What went right:

- localization was strong enough to structure the corridor,
- temporal smoothing prevented many one-frame mistakes,
- graph progression made the target semantics much clearer,
- MBRA worked well once treated as a local controller rather than the whole
navigation solution.

What still failed:

- some corridor regions remained visually ambiguous,
- checkpoint transitions still had edge cases,
- stale-context and no-progress conditions still needed careful handling,
- depth safety could occasionally be too conservative.

## What the indoor result proves and what it does not

Indoor reaching 8 of 11 checkpoints proves that the backbone was real. It proves:

- corridor memory plus graph progression was the right frame,
- MBRA was worth keeping as the indoor local controller,
- the runtime was strong enough to complete most of the track.

It does not prove:

- full immunity to corridor aliasing,
- perfect checkpoint transitions,
- perfect robustness to stale-context conditions.

# Outdoor System: The Full Story

## What Outdoor Really Was

Outdoor was not just the indoor stack with GPS. This was one of the most important
project realizations.

Outdoor is a live mission runtime problem. That means the system had to deal with:

- SDK mission state,
- checkpoint sequencing,
- GPS noise,
- intermediate waypoint semantics,
- route expansion,
- safety intervention,
- and unstable behavior under real-world uncertainty.

## Outdoor Code Structure

The main outdoor files are:

```text
\toprule
\textbf{File} & \textbf{Role} \\
\midrule
\texttt{live\_outdoor\_runtime.py} & Main outdoor mission runtime. \\
\texttt{src/earthrover\_interface.py} & SDK bridge for camera, telemetry, mission, and control. \\
\texttt{src/outdoor\_logonav\_controller.py} & Learned outdoor controller wrapper. \\
\texttt{src/outdoor\_gps\_controller.py} & Classical GPS controller backup. \\
\texttt{src/osm\_router.py} & OSM-based route expansion into intermediate waypoints. \\
\texttt{src/outdoor\_traversability.py} & Depth-based obstacle and steering support. \\
\texttt{src/semantic\_risk\_estimator.py} & Semantic caution, stop, and bias scoring. \\
\texttt{src/imu\_safety.py} & Anti-flip guard. \\
\texttt{src/vision\_safety\_monitor.py} & Night image quality monitor. \\
\bottomrule
```

## Outdoor runtime defaults that matter most

The outdoor runtime has 80+ CLI flags, but a smaller set controls the personality of
the run before any composite mode is enabled:

```text
\toprule
\textbf{Quantity} & \textbf{Default} & \textbf{Flag / Source} \\
\midrule
controller & \texttt{logonav} & \texttt{--controller} \\
GPS max linear & 0.38 m/s & \texttt{--max-linear} \\
GPS max angular & 0.45 rad/s & \texttt{--max-angular} \\
LogoNav max linear & 0.30 m/s & \texttt{--logonav-max-linear} \\
LogoNav max angular & 0.30 rad/s & \texttt{--logonav-max-angular} \\
goal radius & 8.0 m & \texttt{--goal-radius-m} \\
intermediate waypoint radius & 3.0 m & \texttt{--intermediate-goal-radius-m} \\
stuck window & 15 ticks & \texttt{--stuck-window-ticks} \\
route corridor stop & 10.0 m & \texttt{--route-corridor-stop-m} \\
semantic yield risk & 0.25 & \texttt{--semantic-yield-risk} \\
semantic stop risk & 0.55 & \texttt{--semantic-stop-risk} \\
max tilt & 40.0\textdegree & \texttt{--max-tilt-deg} \\
max pitch/roll rate & 150\textdegree/s & \texttt{--max-gyro-dps} \\
\bottomrule
```

> [!warning] Speed cap differences by mode
> These defaults are overridden by composite modes. Ultra-marathon caps all linear to
> 0.20\,m/s and angular to 0.25\,rad/s. Night-safe caps even tighter: linear 0.16\,m/s,
> angular 0.22\,rad/s. The difference is significant --- night-safe is 47% slower than
> default.

## What LogoNav Was Intended to Be

LogoNav is best understood as the learned local motion policy in a larger
navigation system. It is not the whole outdoor architecture. It converts the
current visual state and target information into motion commands.

## How LogoNav Was Used Here

In this project, LogoNav sat inside a larger runtime:

```
mission checkpoints
  -> optional OSM route expansion
  -> active local waypoint
  -> LogoNav controller
  -> safety / recovery / logging / checkpoint handling
```

So the outdoor intelligence was not "LogoNav alone." It was the combination of:

- mission semantics,
- waypoint semantics,
- controller output,
- safety overrides,
- and recovery logic.

## What OSM-Expanded Waypoints Mean

A single far GPS checkpoint is usually too coarse for stable local control. The OSM
router therefore expanded a checkpoint into a pedestrian-appropriate route with
intermediate local waypoints.

That made the motion problem easier, but it also created one of the central outdoor
challenges: the runtime now had to manage *two kinds of targets*:

- real mission checkpoints,
- temporary intermediate routed waypoints.

That distinction turned out to be crucial.

## Why target semantics became the main outdoor issue

Many outdoor failures make more sense once you stop thinking in terms of
"controller good / controller bad" and start thinking in terms of
"what did the runtime believe the current target meant?"

The rover kept getting in trouble when:

- an intermediate routed waypoint was treated too much like a real
checkpoint,
- the next routed target ended up behind the rover after a reroute,
- alignment logic reacted too strongly to the wrong target regime.

## Why Outdoor Already Had a Runnable Base

The repo already had a usable outdoor base before the final hardening pass. That
base included:

- mission-mode support,
- a LogoNav controller path,
- GPS and telemetry access,
- basic loop execution.

So the real project was not inventing outdoor autonomy from zero. It was making the
existing outdoor runtime less brittle.

## Outdoor Runtime Layers

The outdoor runtime gradually became a layered supervised navigation system. The
important layers were:

- mission startup and checkpoint fetching,
- mission resume and auto-claim logic,
- route construction,
- active waypoint selection,
- controller output,
- traversability and semantic intervention,
- IMU, GPS, and vision safety,
- stuck detection and recovery,
- route corridor guarding,
- richer runtime logging.

## Composite safety modes

The runtime contains three especially important composite modes.

### `--ultra-marathon`

This turns on:

- IMU safety,
- health gate,
- leg pause,
- no reverse,
- route corridor guard,
- nav-ready gate,
- behind-waypoint pruning,
- more conservative stuck behavior.

It also tightens motion caps and recovery:

```text
\toprule
\textbf{Flag} & \textbf{Cap} & \textbf{Line} \\
\midrule
\texttt{logonav\_max\_linear} & $\leq 0.20$ m/s & 526 \\
\texttt{logonav\_max\_angular} & $\leq 0.25$ rad/s & 527 \\
\texttt{max\_linear} & $\leq 0.20$ m/s & 528 \\
\texttt{max\_angular} & $\leq 0.25$ rad/s & 529 \\
\texttt{recovery\_turn\_angular} & $\leq 0.30$ rad/s & 530 \\
\texttt{recovery\_turn\_ticks} & $\leq 5$ & 531 \\
\texttt{stuck\_window\_ticks} & $\geq 25$ & 532 \\
\texttt{max\_recovery\_attempts} & 3 (if unset) & 534 \\
\texttt{camera\_watchdog\_ticks} & $\leq 10$ & 535 \\
\bottomrule
```

All line references are to `live_outdoor_runtime.py`.

### `--night-safe`

This enables:

- lamp on,
- vision safety,
- GPS safety,
- IMU safety,
- route corridor guard,
- nav-ready gate,
- traversability.

It also changes the operating profile:

```text
\toprule
\textbf{Flag} & \textbf{Cap} & \textbf{Line} \\
\midrule
\texttt{tick\_hz} & $\leq 2.0$ Hz & 546 \\
\texttt{depth\_every\_n} & 1 (every tick) & 547 \\
\texttt{logonav\_max\_linear} & $\leq 0.16$ m/s & 548 \\
\texttt{logonav\_max\_angular} & $\leq 0.22$ rad/s & 549 \\
\texttt{max\_linear} & $\leq 0.16$ m/s & 550 \\
\texttt{max\_angular} & $\leq 0.22$ rad/s & 551 \\
\texttt{route\_corridor\_stop\_m} & $\leq 5.0$ m & 552 \\
\texttt{camera\_watchdog\_ticks} & $\leq 6$ & 553 \\
\texttt{battery\_warn\_pct} & $\geq 35.0\%$ & 554 \\
\bottomrule
```

Night-safe also auto-enables `logonav_device="auto"` (GPU if available) since
slower tick rates give more inference budget per tick.

> [!note] `--sidewalk-strict` status
> In this active runtime, `--sidewalk-strict` *does exist* as a real flag
> (`live_outdoor_runtime.py`, parser line~402). It auto-enables a stricter
> bundle centered on pedestrian routing and sidewalk-preservation behavior, including
> `--osm-route`, `--osm-no-fallback`, `--route-corridor-guard`,
> `--gps-safety`, `--traversability`, `--semantics`,
> `--semantic-yield`, and `--semantic-sidewalk-stop`.

## Outdoor Traversability

Traversability used depth to measure whether the forward corridor looked blocked. It
was not treated as a full world model. It became a stronger local override.

The main role of traversability was:

- slow down when the forward corridor starts to look tight,
- stop if the forward corridor is clearly too blocked,
- optionally bias steering based on left/right clearance structure.

## Outdoor Semantics

Semantics used SegFormer-derived labels grouped into categories such as:

- drivable,
- neutral,
- caution,
- hazard,
- ignored.

This helped with caution and some stop/yield logic, but semantics stayed a support
layer because the label space and field conditions were not reliable enough to make
it the full outdoor intelligence layer.

## Complete Outdoor Safety Layer Table

This is the table many documents reference but none previously contained in full.
Every layer is verified against `live_outdoor_runtime.py` with line numbers.

```text
\toprule
\textbf{Layer} & \textbf{Source} & \textbf{CLI flag} & \textbf{Threshold} & \textbf{Confirm} & \textbf{Mara.} & \textbf{Night} \\
\midrule
GPS signal/jump & L1043 & \texttt{--gps-safety} & signal$<$4, jump$>$20\,m & 3 ticks & opt & yes \\
Battery stop & L1082 & \texttt{--battery-stop-pct} & user-set & 0 & opt & opt \\
Telemetry freeze & L1096 & always & 50 ticks / 2.0\,s & 0 & yes & yes \\
IMU anti-flip & L1111 & \texttt{--imu-safety} & tilt$>$40\textdegree, gyro$>$150\textdegree/s & 2 & yes & yes \\
Vision quality & L1128 & \texttt{--vision-safety} & bright$<$42, dark$>$65\% & 3 ticks & opt & yes \\
Nav-ready gate & L1142 & \texttt{--nav-ready-gate} & cam+GPS+telem & confirm & yes & yes \\
Route corridor & L1184 & \texttt{--route-corridor-guard} & drift$>$10\,m & 3 ticks & yes & yes \\
Semantic hard stop & L1238 & \texttt{--semantic-hard-stop} & risk$\geq$0.55 & 2 ticks & opt & opt \\
Semantic sidewalk & L1258 & \texttt{--semantic-sidewalk-stop} & road$>$55\%, sw$<$5\% & 2 ticks & opt & opt \\
Semantic yield & runtime & \texttt{--semantic-yield} & risk$\geq$0.25 & 0 & opt & opt \\
Depth stop/slow & runtime & \texttt{--depth-safety} & stop$<$0.4\,m, slow$<$0.8\,m & 0 & opt & opt \\
Traversability & runtime & \texttt{--traversability} & obstacle steering & 0 & opt & yes \\
Stuck detection & runtime & always & window 15 ticks & 0 & yes & yes \\
Camera watchdog & runtime & \texttt{--camera-watchdog-ticks} & 10 ticks & 0 & yes & yes \\
Leg pause & runtime & \texttt{--leg-pause} & operator & --- & yes & opt \\
No-reverse & runtime & \texttt{--no-reverse} & --- & --- & yes & opt \\
Max recovery & runtime & \texttt{--max-recovery-attempts} & 3 (marathon) & --- & yes & opt \\
\bottomrule
```

**Authority hierarchy**: Hard stop $>$ Halt for operator $>$ Hold (wait) $>$ Constraint (limit) $>$ Soft bias (nudge) $>$ Info (warn).

**"Mara."** = always on in `--ultra-marathon`. **"Night"** = always on in `--night-safe`. **"opt"** = available but must be enabled explicitly.

## Semantic Label Taxonomy

The semantic estimator groups ADE20K labels into five categories. This grouping is
the single most important semantic design decision in the project, and it was the
direct result of the offline probe showing that naive label grouping produced 83%
false-positive obstacle fraction on open off-road terrain.

From `src/semantic_risk_estimator.py`, lines 19--24:

```
DRIVABLE_LABELS = {"road", "earth", "path", "sidewalk", "dirt track"}
NEUTRAL_LABELS  = {"grass", "field"}
HAZARD_LABELS   = {"person", "animal", "pole", "wall", "fence"}
CAUTION_LABELS  = {"tree", "plant"}
IGNORE_LABELS   = {"sky"}
```

```text
\toprule
\textbf{Category} & \textbf{Labels} & \textbf{Role in scoring} \\
\midrule
Drivable & road, earth, path, sidewalk, dirt track & Positive weight (+1.0 in free score) \\
Neutral & grass, field & Weak positive (+0.30 in free score) \\
Hazard & person, animal, pole, wall, fence & Hard alerts if $>$ threshold \\
Caution & tree, plant & $-0.60$ in free score; compound vegetation block \\
Ignore & sky & Excluded from all scoring \\
\bottomrule
```

**Why grass is neutral, not obstacle**: The first offline probe treated grass as
an obstacle. On off-road terrain, that produced an 83% "obstacle" fraction on an
open baseline frame. Moving grass to neutral with 30% drivable weight was the single
most important fix for false positives.

**The compound vegetation threshold**: Vegetation blockage fires only when BOTH
conditions are true: drivable center fraction $< 0.10$ AND caution center fraction $>
0.60$. This prevents false positives on scenes where some earth or path labeling keeps
the drivable fraction above 10%.

## What the semantic estimator is actually computing

The semantic estimator is not just reading the most common label in the image. It
measures fractions inside carefully chosen regions:

- center corridor fractions,
- left-half fractions,
- right-half fractions.

Then it builds:

- a risk score,
- a vegetation-blocked condition,
- and a left-right steering bias.

## IMU Safety

The IMU safety layer existed to detect unsafe tilt and dangerous angular motion,
especially in the marathon setting. It is best thought of as a last-resort
protection layer. It can stop the rover once the motion becomes unsafe, but it does
not remove the need for stable upstream controller behavior.

## Vision Safety

Night-time or poor visual quality can quietly destroy controller behavior. The
vision safety monitor existed to gate the controller when the image looked too dark,
too washed out, or too low-texture to trust.

## Route Corridor Guard

The route corridor guard existed because a route is only useful if the rover is
still plausibly following it. If the rover deviated too far from the intended route
structure for too long, the runtime could stop, reroute, or hold.

## What Went Wrong Outdoors

Outdoor failures were less about one single bug and more about a family of target
transition problems.

The main failure themes were:

- the runtime could hand off routed waypoints too early,
- reroutes could create targets behind the rover,
- alignment behavior could become too aggressive,
- route corridor logic could fight forward progress,
- stopping and resuming after mission transitions could create unstable loops.

## Failure chronology in plain language

The most common outdoor failure pattern looked like this:

1. the rover makes real progress,
1. a waypoint transition happens,
1. the new target is interpreted badly,
1. the runtime starts aligning, rerouting, or oscillating,
1. smooth forward progress disappears.

## Important Outdoor Corrections

The outdoor runtime therefore accumulated a set of very specific corrections:

- distinguish mission checkpoints from routed intermediate waypoints,
- shrink and later make dynamic the routed waypoint handoff radius,
- prune behind-the-rover waypoints,
- reroute from live pose rather than stale route state,
- gate hard alignment more carefully,
- strengthen traversability as a local override.

## Outdoor Competition Outcome

The outdoor stack had:

- one run that reached all checkpoints,
- several other runs around fifty percent successful overall,
- three failed rounds.

This is not a cleanly solved system, but it is also not a fake system. One full run
means the stack was real. The incomplete reliability means the runtime still had
too many unstable transition cases.

## What the outdoor result proves and what it does not

It proves:

- the outdoor stack could complete a full run,
- the controller path was not fake,
- the runtime and safety layers were operationally meaningful.

It does not prove:

- repeatable robustness,
- solved waypoint-transition behavior,
- solved marathon-scale stability.

# Marathon System: The Full Story

## What Marathon Mode Was Trying to Do

Marathon mode was an attempt to push the outdoor system into a narrower, safer
operating envelope for a longer run. The point was not more autonomy freedom. The
point was safe completion.

## What Marathon Added

Marathon mode added or emphasized:

- IMU safety,
- health gating,
- leg pauses,
- no reverse,
- tighter speed caps,
- route corridor guard,
- stricter recovery behavior,
- preflight checking.

## Why That Still Was Not Enough

The marathon failure is important because it shows the limit of adding safety
layers without enough physical transition testing.

The rover reached one checkpoint. After that:

- the runtime did not settle the next target cleanly enough,
- the rover entered repeated turning behavior,
- persistent aggressive turning on real terrain created instability,
- the rover toppled.

So the failure was not well described as "software failed" or "hardware failed"
alone. It was a coupled runtime-to-platform stability failure.

## What We Learned From Marathon

The marathon result forced three lessons:

- safer logic does not automatically mean more stable logic,
- waypoint and checkpoint transitions are the most dangerous phase of the
current outdoor system,
- physical stability matters just as much as software behavior in long runs.

## What Should Change After Marathon

The next improvements are clear:

- slower transition behavior,
- wider and calmer turning behavior,
- fewer in-place or aggressive alignment situations,
- improved platform balance or ballast,
- more field-tested runtime behavior before long attempts.

# Main Design Decisions and Why They Were Chosen

## Why CosPlace

Because the indoor problem was corridor-specific and repeated. We did not need the
most elaborate matching system available. We needed a fast and good-enough visual
localizer that made the corridor legible.

## Why MBRA for Indoor

Because MBRA fit the role of short-horizon visual control once the corridor memory
and graph progression carried the larger navigation structure.

## Why Not Treat MBRA as Everything

Because that would have overloaded the learned controller. The project became much
better once localization, planning, and control were separated cleanly.

## Why LogoNav for Outdoor

Because the repo already had a working outdoor controller path, and the project’s
real outdoor problem was runtime robustness rather than lack of a motion policy.

## Why Not Pure GPS as Main Direction

Because pure GPS was useful as a fallback but not rich enough as the main outdoor
behavior layer. The rover still needed a better local motion policy.

## Why Depth Did Not Become the Main Backbone

Because depth was helpful but not trustworthy enough across all real field cases to
become the only hard truth layer. It became a strong local override instead.

## Why Semantics Stayed Secondary

Because semantics was useful for caution and guidance, but ADE20K-style labels and
real outdoor terrain did not line up cleanly enough to justify full authority.

## Why No Reverse for Learned Controllers

Because reverse recovery damaged the learned visual context and created worse
behavior rather than cleaner recovery.

## Why Sole Motion Authority Mattered

Because mixing a learned controller with competing backup steering logic often made
oscillation worse, not better.

# Current Final Mental Model

If the project has to be explained in one page, the clearest version is this.

## Indoor

Indoor is a corridor-memory problem:

- CosPlace localizes,
- temporal filtering stabilizes,
- graph planning chooses the next checkpoint/subgoal,
- MBRA executes short-horizon motion,
- depth veto protects against obvious close obstacles.

## Outdoor

Outdoor is a mission-runtime problem:

- the SDK defines checkpoints,
- OSM can define intermediate route structure,
- LogoNav produces local commands,
- runtime layers enforce safer behavior.

## Marathon

Marathon is the outdoor stack under stricter runtime control, where the biggest
remaining weakness is stable behavior during and after target transitions.

# What Still Remains Open

The biggest remaining open problems are:

- calmer and more reliable waypoint handoff outdoors,
- less spin-prone transition behavior,
- better physical stability for long outdoor runs,
- more field validation of depth and semantic intervention,
- cleaner recovery logic that does not fight learned control.

# Practical Commands

## Indoor competition-style run

```
python live_indoor_runtime.py \
  --checkpoint-steps 45 480 761 821 1094 1208 1345 1430 1544 1638 1764 \
  --auto-advance-checkpoints --send-control --controller mbra --depth-safety
```

## Indoor dry run

```
python live_indoor_runtime.py \
  --checkpoint-steps 45 480 761 821 1094 1208 1345 1430 1544 1638 1764 \
  --auto-advance-checkpoints --controller mbra --depth-safety
```

## Outdoor mission run

```
python live_outdoor_runtime.py --mission --send-control --controller logonav --osm-route
```

## Outdoor traversability run

```
python live_outdoor_runtime.py --mission --send-control --traversability
```

## Outdoor marathon run

```
python live_outdoor_runtime.py \
  --mission --send-control --controller logonav --osm-route --ultra-marathon
```

## Outdoor night-safe run

```
python live_outdoor_runtime.py \
  --mission --send-control --controller logonav --osm-route --night-safe
```

## Canonical Runbook (Use These First)

If you only keep four commands on hand, keep these:

```text
\toprule
\textbf{Mode} & \textbf{Canonical command} \\
\midrule
Indoor (competition) &
\texttt{python live\_indoor\_runtime.py --checkpoint-steps 45 480 761 821 1094 1208 1345 1430 1544 1638 1764 --auto-advance-checkpoints --send-control --controller mbra --depth-safety} \\
Outdoor (standard) &
\texttt{python live\_outdoor\_runtime.py --mission --send-control --controller logonav --osm-route} \\
Outdoor (marathon profile) &
\texttt{python live\_outdoor\_runtime.py --mission --send-control --controller logonav --osm-route --ultra-marathon} \\
Outdoor (night profile) &
\texttt{python live\_outdoor\_runtime.py --mission --send-control --controller logonav --osm-route --night-safe} \\
\bottomrule
```

## Do-Not-Combine Command Conflicts

```text
\toprule
\textbf{Conflict} & \textbf{Why to avoid it} \\
\midrule
\texttt{--depth-safety} + \texttt{--no-depth-safety} & Contradictory intent. Keep one policy per run and make it explicit in logs. \\
\texttt{--ultra-marathon} + manual high-speed overrides (\texttt{--max-linear}, \texttt{--nominal-linear}, \texttt{--logonav-max-linear}) & Marathon profile is meant to cap speed conservatively. Manual speed-up defeats the profile's purpose and makes runs harder to compare. \\
\texttt{--night-safe} + manual high-speed overrides (\texttt{--max-linear}, \texttt{--nominal-linear}, \texttt{--logonav-max-linear}) & Night-safe is intentionally slower for low-visibility stability; speed-up changes the safety envelope. \\
\texttt{--controller gps} + LogoNav-specific tuning-only sessions & GPS mode bypasses LogoNav policy behavior. If tuning LogoNav behavior, keep \texttt{--controller logonav}. \\
\texttt{--mission} omitted during field mission testing & Without \texttt{--mission}, you are not exercising real checkpoint progression logic from the SDK mission state machine. \\
\bottomrule
```

> [!warning] Operational consistency rule
> For field comparisons, do not change profile flags and low-level safety thresholds in the same run.
> Change one layer at a time so failures can be attributed to one decision.

# Verified Reproduction Checklist (Fast Path)

This checklist is the shortest reliable path to reproduce baseline behavior with
commands that are already reflected in this documentation and validated by parser
checks.

## Step 1: Indoor dry-run sanity check

```
python live_indoor_runtime.py \
  --checkpoint-steps 45 480 761 821 1094 1208 1345 1430 1544 1638 1764 \
  --auto-advance-checkpoints --controller mbra --depth-safety
```

Expected signatures:
- startup block beginning with `Live indoor runtime`
- periodic tick lines with step/progress context
- if edge cases occur: `skipping past checkpoint ...` or
`no-progress reset ...`

## Step 2: Outdoor mission baseline

```
python live_outdoor_runtime.py \
  --mission --send-control --controller logonav --osm-route
```

Expected signatures:
- `navigation_ready ticks=3`
- `DONE=... LEG=... WP=... dist=... mode=...`
- `intermediate waypoint reached at (...)`

## Step 3: Marathon profile behavior

```
python live_outdoor_runtime.py \
  --mission --send-control --controller logonav --osm-route --ultra-marathon
```

Expected signatures:
- conservative speed behavior relative to standard mission mode
- stricter intervention signatures when route or state becomes inconsistent
- in adverse cases: `route_corridor_stop ...` followed by reroute

> [!warning] Repro discipline
> Do not tune thresholds while reproducing baseline outcomes. First reproduce
> baseline behavior exactly. Then change one subsystem at a time.

# Preflight Sequence

Before a marathon or night run, the preflight validator
(`scripts/preflight_marathon.py`, 216 lines) performs checks in this order:

1. **Mission start** (if `--mission`): calls `/start-mission`,
validates checkpoint list
1. **SDK connection**: verifies the SDK bridge is reachable
1. **Telemetry availability**: confirms initial telemetry response
1. **Battery check**: fails if battery $<$ `--battery-min-pct`
(default 70%)
1. **GPS live**: confirms lat/lon are valid and non-zero
1. **Telemetry advancing**: samples multiple times to verify timestamps are
incrementing (not frozen)
1. **Front camera live**: waits up to `--camera-timeout-s` for a
valid frame
1. **GPS signal level**: fails if `gps_signal` $<$
`--gps-min-signal` (default 2)
1. **Night-time checks** (if `--night-safe`): vision safety
(brightness, dark fraction, glare, texture) and IMU rest-pose tilt $<$ 20°
1. **Model weights**: verifies LogoNav weights and config files exist on
disk
1. **Mission checkpoint status** (if `--mission`): confirms checkpoint
list and completion state
1. **OSM routing validation** (if `--osm-route`): expands one test
route, checks for fallback straight-line segments

Any check failure prints a clear diagnostic and exits before the rover moves.

# System Architecture Diagram

```
  INDOOR PIPELINE                         OUTDOOR PIPELINE
  ==============                          ================

  Camera Frame                            SDK Mission Checkpoints
       |                                        |
  CosPlace VPR                            OSM Route Expansion
  (corridor_localizer.py)                 (osm_router.py)
       |                                        |
  Temporal Stabilization                  Active Local Waypoint
  (temporal_localization.py)                    |
       |                                  LogoNav or GPS Controller
  Graph Planner                           (outdoor_logonav_controller.py
  (graph_planner.py)                       outdoor_gps_controller.py)
       |                                        |
  MBRA or Simple Controller         +-----+-----+-----+-----+-----+
  (mbra_controller.py               |     |     |     |     |     |
   local_controller.py)            IMU  Vision GPS  Route Depth Semantic
       |                           Safe  Safe  Safe Guard Trav  Risk
  Depth Safety Veto                (imu_  (vis_ (gps (route (trav (sem_
       |                          safety safety safe  corr  ersab risk
  SDK Motor Command               .py)  _mon   .py)  .py)  .py)  .py)
                                         .py)
                                        |
                                   Stuck Detection + Recovery
                                        |
                                   SDK Motor Command

  SHARED: earthrover_interface.py (SDK bridge)
          depth_estimator.py (Depth Anything V2)
          sensor_state.py (heading/gyro filtering)
```

## Subsystem Diagram: Indoor Localization Pipeline

```
front RGB frame
    |
    v
CosPlace descriptor embedding (corridor_localizer)
    |
    v
Top-k retrieval candidates (index, distance)
    |
    v
Temporal scoring
  score = distance + continuity penalties + optional heading penalty
    |
    +--> ambiguity hold? keep previous node
    |
    v
stable corridor step + confidence
    |
    v
graph planner + checkpoint semantics
```

## Subsystem Diagram: Outdoor Waypoint Lifecycle

```
SDK mission checkpoint
    |
    +--> direct mode: active target = checkpoint
    |
    +--> OSM mode:
         checkpoint -> routed polyline -> intermediate waypoints
             |
             v
         active waypoint selected
             |
             v
         dynamic intermediate radius handoff
             |
             +--> prune behind-waypoint cases
             |
             v
         reached intermediate? advance
             |
             v
         reached mission checkpoint? report + next leg
```

## Subsystem Diagram: Semantic Scoring Pipeline

```
front RGB frame
    |
    v
SegFormer inference -> per-pixel labels
    |
    v
ROI masks (center corridor, left half, right half)
    |
    v
label fractions:
  drivable, neutral, caution, person/animal/pole/wall
    |
    v
risk score + hard alerts + vegetation_blocked
    |
    v
left/right free-score -> angular bias
    |
    v
runtime policy:
  semantic_yield / semantic_hard_stop / semantic_sidewalk_stop
```

# C4 Architecture Views

## C4 Level 1 — System Context

```
                         +----------------------+
                         |    Mission Operator  |
                         |  (supervised runtime)|
                         +----------+-----------+
                                    |
                                    v
 +------------------+       +-------+--------+       +-------------------+
 | Indoor Corridor  |<----->| ERC-3 Rover SW |<----->| EarthRover SDK/API|
 | Environment      |       | (this repo)    |       | (telemetry/control)|
 +------------------+       +-------+--------+       +-------------------+
                                    |
                                    v
                         +----------+-----------+
                         | Outdoor Mission Area |
                         | (GPS/OSM pedestrian) |
                         +----------------------+
```

## C4 Level 2 — Container View

```
 +--------------------------- ERC-3 Rover Software ---------------------------+
 |                                                                           |
 |  Indoor Runtime Container         Outdoor Runtime Container               |
 |  (live_indoor_runtime.py)         (live_outdoor_runtime.py)              |
 |        |                                     |                            |
 |        v                                     v                            |
 |  Localization + Planner               Mission + Routing                   |
 |  (corridor_localizer,                (earthrover_interface,               |
 |   temporal_localization,              osm_router)                         |
 |   graph_planner)                             |                            |
 |        |                                     v                            |
 |        v                              Local Controller                    |
 |  MBRA/Simple Controller              (outdoor_logonav / gps)              |
 |        |                                     |                            |
 |        +------------> Shared Safety Layers <+                            |
 |                      (depth, traversability, semantics,                   |
 |                       imu, vision, gps, corridor guard)                   |
 +---------------------------------------------------------------------------+
```

## C4 Level 3 — Dynamic View (Outdoor Tick)

```
1) Read telemetry + frame from SDK
2) Resolve active mission checkpoint / routed waypoint
3) Controller proposes command (LogoNav or GPS)
4) Apply safety layers in order:
   route corridor -> semantics -> depth/traversability -> IMU -> vision/GPS
5) Apply recovery/align gating if needed
6) Send command or hold/stop
7) Evaluate reach conditions and transition target state
```

# Complete Outdoor CLI Flag Inventory (112 Flags)

This section is the exhaustive runtime flag inventory from
`live_outdoor_runtime.py` argument parsing, grouped by category and including
default values and source line anchors.

```
| Category | Flag | Type | Default | Description | Source |
|---|---|---|---|---|---|
| Control | `--max-linear` | value | `0.38` | Controller max linear speed. | L353 |
| Control | `--min-linear` | value | `0.08` | Controller min forward speed. | L354 |
| Control | `--nominal-linear` | value | `0.33` | Nominal cruise speed. | L355 |
| Control | `--max-angular` | value | `0.45` | Controller max angular speed. | L356 |
| Control | `--angular-gain` | value | `0.4` | Proportional gain for steering. | L357 |
| Control | `--in-place-turn-threshold-deg` | value | `90.0` | Stop forward motion and turn in place above this bearing error. | L358 |
| Control | `--in-place-turn-exit-deg` | value | `60.0` | Exit turn-in-place mode when bearing error drops below this (hysteresis). | L359 |
| Control | `--logonav-stuck-progress-epsilon-m` | value | `-0.05` | LogoNav-specific progress threshold for stuck recovery. Raise toward 0.0 to recover sooner from curb / wall traps. | L366 |
| Control | `--stuck-command-linear` | value | `0.18` | Forward command threshold for no-progress detection. | L367 |
| Control | `--logonav-min-effective-linear` | value | `0.10` | Minimum effective forward speed to maintain when LogoNav is trying to move and no safety layer is actively slowing it. | L368 |
| Control | `--logonav-align-turn-threshold-deg` | value | `35.0` | LogoNav: above this bearing error, prioritize turning toward the waypoint before driving forward. | L369 |
| Control | `--logonav-align-turn-exit-deg` | value | `18.0` | LogoNav: exit the turn-priority mode once bearing error drops below this. | L370 |
| Control | `--logonav-align-max-linear` | value | `0.04` | LogoNav: maximum forward speed while turn-priority mode is active. | L371 |
| Control | `--logonav-align-min-angular` | value | `0.18` | LogoNav: minimum angular command while turn-priority mode is active. | L372 |
| Control | `--logonav-align-distance-m` | value | `6.0` | LogoNav: only force turn-priority below this waypoint distance unless bearing error is very large. | L373 |
| Control | `--logonav-align-extreme-deg` | value | `60.0` | LogoNav: always allow turn-priority above this bearing error, even for farther waypoints. | L374 |
| Control | `--logonav-weights` | value | `REPO_ROOT / "mbra_repo" / "deployment" / "model_weights" / "logonav.pth"` | Path to LogoNav weights. | L410 |
| Control | `--logonav-config` | value | `REPO_ROOT / "mbra_repo" / "train" / "config" / "LogoNav.yaml"` | Path to LogoNav config. | L411 |
| Control | `--logonav-device` | value | `"cpu"` | Device for LogoNav inference. | L412 |
| Control | `--logonav-max-linear` | value | `0.30` | LogoNav max linear speed cap. | L413 |
| Control | `--logonav-max-angular` | value | `0.30` | LogoNav max angular speed cap. | L414 |
| Depth/Traversability | `--depth-safety` | bool | `-` | Enable depth-based obstacle avoidance (disabled by default - camera geometry causes false stops on this platform). | L382 |
| Depth/Traversability | `--depth-model-size` | value | `"small"` | Depth model size. | L384 |
| Depth/Traversability | `--depth-every-n` | value | `2` | Run depth inference every N ticks. | L385 |
| Depth/Traversability | `--depth-slow-m` | value | `0.8` | Slow down below this forward clearance. | L386 |
| Depth/Traversability | `--depth-stop-m` | value | `0.4` | Stop below this forward clearance. | L387 |
| Depth/Traversability | `--traversability` | bool | `-` | Enable middle-band traversability layer (obstacle detection at tree-trunk / wall height). | L388 |
| Depth/Traversability | `--trav-obstacle-m` | value | `1.5` | Bins below this clearance are treated as blocked by the traversability layer. | L389 |
| Depth/Traversability | `--trav-stop-m` | value | `0.60` | Traversability: stop below this forward clearance. | L390 |
| Depth/Traversability | `--trav-slow-m` | value | `1.20` | Traversability: slow below this forward clearance. | L391 |
| Depth/Traversability | `--trav-memory-frames` | value | `4` | Traversability obstacle memory: use min-pool over last N depth frames. | L392 |
| Depth/Traversability | `--vfh-bins` | value | `16` | Number of polar clearance bins. | L398 |
| Depth/Traversability | `--vfh-fov-deg` | value | `90.0` | Horizontal FOV for VFH clearance bins. | L399 |
| Depth/Traversability | `--vfh-blocked-distance-m` | value | `0.8` | Bins below this clearance are treated as blocked. | L400 |
| GPS Safety | `--gps-safety` | bool | `-` | Enable GPS signal/fix safety gating and implausible-jump stops. | L447 |
| GPS Safety | `--gps-min-signal` | value | `2` | Minimum GPS signal level required before continuing. | L448 |
| GPS Safety | `--cell-min-signal` | value | `1` | Minimum cellular signal level required before continuing. | L449 |
| GPS Safety | `--gps-safety-confirm-ticks` | value | `3` | Require this many consecutive bad GPS signal/fix ticks before stopping. | L450 |
| GPS Safety | `--gps-jump-stop-m` | value | `12.0` | Stop if raw GPS position jumps by more than this many meters in a single telemetry step. | L451 |
| IMU Safety | `--imu-safety` | bool | `-` | Enable IMU-based anti-flip monitoring. | L417 |
| IMU Safety | `--max-tilt-deg` | value | `40.0` | IMU: emergency stop if tilt exceeds this angle. | L418 |
| IMU Safety | `--max-gyro-dps` | value | `150.0` | IMU: emergency stop if roll/pitch rate exceeds this (deg/s). | L419 |
| Loop/Runtime | `--tick-hz` | value | `3.0` | Loop frequency in Hz. | L344 |
| Loop/Runtime | `--max-steps` | value | `None` | Optional max loop iterations. | L345 |
| Loop/Runtime | `--send-control` | bool | `-` | Actually send commands to the robot. | L348 |
| Loop/Runtime | `--print-json` | bool | `-` | Print each loop state as JSON. | L349 |
| Loop/Runtime | `--heading-fusion` | bool | `-` | Blend compass heading with windowed course-over-ground after stable motion is detected. | L360 |
| Mission/Targets | `--goal-lat` | value | `None` | Single goal latitude. | L318 |
| Mission/Targets | `--goal-lon` | value | `None` | Single goal longitude. | L319 |
| Mission/Targets | `--goal-radius-m` | value | `8.0` | Distance threshold for checkpoint reach. | L350 |
| Mission/Targets | `--intermediate-goal-radius-m` | value | `3.0` | Maximum distance threshold for routed intermediate waypoint updates. The runtime further tightens this dynamically from actual waypoint spacing so it does not skip ahead too early. | L351 |
| Mission/Targets | `--checkpoint-confirm-ticks` | value | `1` | How many consecutive ticks inside radius are required. | L352 |
| Ops/Safety Profiles | `--ultra-marathon` | bool | `-` | Enable all marathon safety features: IMU anti-flip, conservative recovery, speed caps, health gate, leg pauses. | L416 |
| Ops/Safety Profiles | `--health-gate` | bool | `-` | Run health checks before starting navigation and between legs. | L420 |
| Ops/Safety Profiles | `--leg-pause` | bool | `-` | Pause for operator confirmation after each mission checkpoint. | L421 |
| Ops/Safety Profiles | `--camera-watchdog-ticks` | value | `10` | Stop after this many consecutive ticks without a camera frame. | L422 |
| Ops/Safety Profiles | `--battery-warn-pct` | value | `30.0` | Warn when battery falls below this percentage during the run. | L428 |
| Ops/Safety Profiles | `--battery-stop-pct` | value | `None` | Emergency stop if battery falls to or below this percentage. | L429 |
| Ops/Safety Profiles | `--night-safe` | bool | `-` | Enable a night-time outdoor safety profile: lamp on, tighter speed caps, vision safety gate, and semantic pedestrian stops. | L430 |
| Ops/Safety Profiles | `--lamp-on` | bool | `-` | Keep the rover lamp on while this runtime is active. | L431 |
| Ops/Safety Profiles | `--nav-ready-gate` | bool | `-` | Require several consecutive healthy ticks before allowing motion after startup, hard stops, or leg transitions. | L452 |
| Ops/Safety Profiles | `--nav-ready-confirm-ticks` | value | `3` | Navigation readiness gate: consecutive healthy ticks required before motion is allowed. | L453 |
| Ops/Safety Profiles | `--report-interventions` | bool | `-` | Report hard safety stops to the SDK interventions endpoints. | L454 |
| Ops/Safety Profiles | `--operator-confirm-hard-stop` | bool | `-` | On hard safety stops, wait for operator ENTER before resuming and optionally report intervention end. | L455 |
| Other | `--telemetry-freeze-ticks` | value | `4` | Count telemetry as potentially frozen after this many repeated timestamps. | L361 |
| Other | `--telemetry-freeze-timeout-s` | value | `4.0` | Actually stop only if telemetry has been frozen for at least this many wall-clock seconds. | L362 |
| Recovery | `--stuck-window-ticks` | value | `15` | Window size for no-progress detection (counts only fresh telemetry ticks). | L363 |
| Recovery | `--stuck-min-displacement-m` | value | `0.30` | Minimum displacement expected over the stuck window when commanding forward motion. | L364 |
| Recovery | `--stuck-progress-epsilon-m` | value | `0.20` | Minimum reduction in distance-to-goal expected over the stuck window when commanding forward motion. | L365 |
| Recovery | `--recovery-reverse-linear` | value | `-0.20` | Reverse command during simple stuck recovery. | L378 |
| Recovery | `--recovery-turn-angular` | value | `0.85` | Turn command during simple stuck recovery. | L379 |
| Recovery | `--recovery-reverse-ticks` | value | `5` | How many ticks to reverse during stuck recovery. | L380 |
| Recovery | `--recovery-turn-ticks` | value | `8` | How many ticks to turn during stuck recovery. | L381 |
| Recovery | `--no-reverse` | bool | `-` | Disable all reverse maneuvers in stuck recovery. | L423 |
| Recovery | `--max-recovery-attempts` | value | `0` | Halt for operator after this many stuck recoveries per waypoint (0=unlimited). | L424 |
| Routing/Corridor | `--osm-prune-behind-waypoints` | bool | `-` | Drop initial routed waypoints that are very close and behind the rover after OSM re-routing. | L375 |
| Routing/Corridor | `--osm-prune-behind-distance-m` | value | `18.0` | Maximum distance for pruning a behind-the-rover initial waypoint after OSM routing or waypoint handoff. | L376 |
| Routing/Corridor | `--osm-prune-behind-bearing-deg` | value | `100.0` | Prune initial routed waypoints if they lie farther than this angle behind the rover heading. | L377 |
| Routing/Corridor | `--osm-route` | bool | `-` | Expand mission legs into pedestrian waypoints using OSM at startup. | L401 |
| Routing/Corridor | `--sidewalk-strict` | bool | `-` | Enforce strict sidewalk-first outdoor behavior: pedestrian-only OSM routing, no straight-line fallback, tighter route corridor, and stronger semantic sidewalk checks. | L402 |
| Routing/Corridor | `--osm-no-fallback` | bool | `-` | Abort startup if any OSM leg falls back to a straight-line route. | L403 |
| Routing/Corridor | `--osm-buffer-m` | value | `300.0` | Bounding-box padding for OSM queries. | L404 |
| Routing/Corridor | `--osm-query-timeout-s` | value | `25` | Overpass server-side timeout in seconds. | L405 |
| Routing/Corridor | `--osm-request-retries` | value | `2` | Number of Overpass request attempts before fallback. | L406 |
| Routing/Corridor | `--osm-max-segment-m` | value | `20.0` | Max gap between routed waypoints after densification. | L407 |
| Routing/Corridor | `--osm-min-waypoint-spacing-m` | value | `8.0` | Minimum spacing between retained routed waypoints. | L408 |
| Routing/Corridor | `--osm-max-snap-distance-m` | value | `60.0` | Max allowed snap distance to the OSM graph. | L409 |
| Routing/Corridor | `--route-corridor-guard` | bool | `-` | Stop if GPS drifts too far away from the active routed path corridor. | L425 |
| Routing/Corridor | `--route-corridor-stop-m` | value | `10.0` | Maximum lateral deviation from the active routed corridor before stopping. | L426 |
| Routing/Corridor | `--route-corridor-confirm-ticks` | value | `3` | Require this many consecutive off-corridor ticks before stopping. | L427 |
| SDK/IO | `--sdk-url` | value | `"http://localhost:8000"` | EarthRover SDK base URL. | L346 |
| SDK/IO | `--sdk-timeout` | value | `10.0` | SDK request timeout in seconds. | L347 |
| Semantics | `--semantics` | bool | `-` | Enable semantic scene-understanding soft bias (people / vegetation corridor checks). | L393 |
| Semantics | `--semantic-model-profile` | value | `None` | Optional semantic model preset. cityscapes is stronger for sidewalk/road structure; mapillary is the heaviest street-scene option. | L394 |
| Semantics | `--semantics-model-id` | value | `"nvidia/segformer-b0-finetuned-ade-512-512"` | Semantic segmentation model id. Supports plug-and-play Hugging Face semantic/universal segmentation backends. | L395 |
| Semantics | `--semantics-device` | value | `"cpu"` | Device for semantic segmentation inference. | L396 |
| Semantics | `--semantics-every-n` | value | `3` | Run semantic inference every N ticks. | L397 |
| Semantics | `--semantic-hard-stop` | bool | `-` | Enable a hard stop when the semantic model sees a person or animal in the center corridor. | L438 |
| Semantics | `--semantic-yield` | bool | `-` | Slow aggressively and yield when semantic risk indicates people, animals, or ambiguous sidewalk occupancy ahead. | L439 |
| Semantics | `--semantic-yield-risk` | value | `0.25` | Semantic yield activation risk threshold. | L440 |
| Semantics | `--semantic-yield-max-linear` | value | `0.08` | Maximum linear speed while semantic yield mode is active. | L441 |
| Semantics | `--semantic-stop-risk` | value | `0.55` | Semantic hard stop risk threshold. | L442 |
| Semantics | `--semantic-stop-confirm-ticks` | value | `2` | Require this many consecutive semantic hazard ticks before stopping. | L443 |
| Semantics | `--semantic-sidewalk-stop` | bool | `-` | Stop when the center corridor looks road-dominant and not sidewalk-like for several ticks. | L444 |
| Semantics | `--semantic-road-dominance` | value | `0.55` | Semantic sidewalk stop: road fraction threshold in the center corridor. | L445 |
| Semantics | `--semantic-sidewalk-min` | value | `0.05` | Semantic sidewalk stop: minimum sidewalk+path fraction expected in the center corridor. | L446 |
| Vision Safety | `--vision-safety` | bool | `-` | Enable an image-quality safety gate for dark / glare / low-detail night frames. | L432 |
| Vision Safety | `--vision-min-brightness` | value | `42.0` | Vision safety: minimum mean grayscale brightness (0-255). | L433 |
| Vision Safety | `--vision-max-dark-fraction` | value | `0.65` | Vision safety: maximum fraction of dark pixels before visibility is considered unsafe. | L434 |
| Vision Safety | `--vision-max-glare-fraction` | value | `0.12` | Vision safety: maximum fraction of saturated bright pixels before glare is considered unsafe. | L435 |
| Vision Safety | `--vision-min-texture` | value | `8.0` | Vision safety: minimum gradient-texture score before a frame is considered too low-detail. | L436 |
| Vision Safety | `--vision-confirm-ticks` | value | `3` | Vision safety: require this many consecutive bad frames before stopping. | L437 |
```

**Inventory note:** This is exhaustive for parser-defined flags in
`live_outdoor_runtime.py` at the time of writing (112 flags excluding
suppressed legacy aliases).

# Key File Map

```text
\toprule
\textbf{File} & \textbf{Main purpose} \\
\midrule
\texttt{live\_indoor\_runtime.py} & main indoor runtime used in practice \\
\texttt{live\_indoor\_runtime\_mbra.py} & indoor MBRA-oriented variant \\
\texttt{live\_indoor\_runtime\_recovery.py} & indoor recovery-heavy variant \\
\texttt{live\_outdoor\_runtime.py} & main outdoor mission runtime \\
\texttt{src/corridor\_localizer.py} & indoor visual localization \\
\texttt{src/temporal\_localization.py} & indoor localization stabilization \\
\texttt{src/graph\_planner.py} & indoor graph planning \\
\texttt{src/navigation\_runtime.py} & indoor orchestration \\
\texttt{src/mbra\_controller.py} & indoor learned controller wrapper \\
\texttt{src/local\_controller.py} & indoor simple backup controller \\
\texttt{src/outdoor\_logonav\_controller.py} & outdoor learned controller wrapper \\
\texttt{src/outdoor\_gps\_controller.py} & outdoor GPS controller backup \\
\texttt{src/osm\_router.py} & outdoor route expansion \\
\texttt{src/outdoor\_traversability.py} & depth-based outdoor obstacle logic \\
\texttt{src/semantic\_risk\_estimator.py} & semantic risk scoring \\
\texttt{src/imu\_safety.py} & IMU anti-flip safety \\
\texttt{src/vision\_safety\_monitor.py} & image quality gating \\
\texttt{src/earthrover\_interface.py} & SDK interface layer \\
\texttt{baseline.py} & indoor descriptor / database build and query \\
\bottomrule
```

# How To Read This Project Without Getting Lost

One of the easiest ways to misunderstand this repository is to assume that every
file has the same status. They do not.

There are really four layers:

- **Active runtime code**: this is the true behavior layer.
- **Current narrative docs**: these explain the active behavior and the
reasoning behind it.
- **Historical planning docs**: these explain what people intended at an
earlier stage, but not always what finally shipped.
- **Experiment records**: these explain what was tested, what seemed
promising, and what was later demoted.

So the right reading order is not folder order. The right reading order is:

1. understand the current split between indoor and outdoor,
1. understand what the indoor runtime actually does,
1. understand what the outdoor runtime actually does,
1. then go back to the historical docs to understand why the final shape looks
the way it does.

# Timeline and Major Phases

## Phase 1: Indoor Corridor Framing

The project began by realizing that indoor navigation should not be treated as a
generic robotics problem. The corridor was known, the dataset could be recorded in
advance, and the competition checkpoints could be tied to exact corridor steps.

That phase established three durable truths:

- localization mattered more than a fancy local controller,
- the corridor graph was a useful abstraction,
- and exact step targets were cleaner than vague image goals.

## Phase 2: Indoor Controller Confusion and Cleanup

The indoor work then passed through a messy phase where the project had too many
runtime ideas at once: stronger recovery logic, multiple runtime variants, and too
much confusion about whether MBRA itself was supposed to be the whole system.

What survived from that phase was the clearer view:

- MBRA is the local controller,
- corridor localization and graph progression remain the backbone,
- no reverse and stale-context reset are better than trying to make MBRA
behave like a classical backup controller.

## Phase 3: Outdoor Runtime Reality

Outdoor started with a different gift: the repo already had a runnable outdoor
controller path. That meant the work was not "invent an outdoor controller from
zero." The work became "make the outdoor runtime stop doing stupid things in the
real world."

## Phase 4: Safety and Marathon Hardening

The marathon and night-safe work made the project more honest. At that point the
question was no longer "can the rover move." The question became "can the rover
remain physically and operationally sane over a long run."

# Version-by-Version Evolution (Chronological)

This section captures the practical evolution in the order the team actually lived
it. Dates are anchored to document timestamps and runtime change windows in this
repo, not invented from memory.

## V0 — Indoor baseline scaffold (early March)

- Corridor descriptors and graph artifacts existed.
- Runtime could localize and move, but behavior quality was inconsistent.
- The team still treated local-controller choice as the main indoor question.

## V1 — Indoor checkpoint-step reframing (mid March)

- Checkpoints were reframed as exact corridor steps.
- Localization, temporal continuity, and graph semantics became central.
- The problem shifted from "make turning smarter" to "make state semantics
unambiguous."

## V2 — MBRA-first indoor cleanup (late March)

- MBRA was made a short-horizon local controller, not a stack replacement.
- No-reverse behavior and stale-context reset were emphasized.
- Skip-past-checkpoint behavior was added for forward-only graph edge cases.
- Indoor reached 8/11 checkpoints in the competition-facing run profile.

## V3 — Outdoor runtime hardening pass (late March)

- Outdoor was treated as a mission-runtime problem, not only controller tuning.
- OSM route semantics, waypoint semantics, and corridor guard behavior were
tightened.
- Traversability and semantic layers remained support layers, not sole truth.

## V4 — Ultra-marathon and night-safe profiles (late March)

- Composite safety profiles were codified in runtime flags.
- IMU, vision, GPS, and route discipline were integrated into stricter modes.
- Field behavior exposed transition instability as the dominant remaining risk.

> [!note] Why this chronology matters
> Without chronology, the architecture looks arbitrary. With chronology, each layer
> is easier to defend: it was introduced to fix a specific class of observed failure,
> not to make the code look sophisticated.

# Hardware and Platform Reality

Even though this repository is mostly software, the platform mattered constantly.

## Sensors and Interfaces We Actually Relied On

The project relied on:

- front camera frames,
- GPS,
- IMU data,
- orientation estimates,
- battery and telemetry from the SDK,
- browser / server-side control through the EarthRover SDK bridge.

## Why The Platform Matters To The Software Story

Several software choices only make sense once the hardware constraints are visible:

- monocular depth can be very sensitive to camera geometry and scale,
- compass/orientation can be noisy indoors,
- aggressive turning is not just a controller issue if the rover has a high
center of gravity,
- telemetry freezes can look like controller failure if they are not detected
explicitly.

## Physical Mistakes That Became Software Lessons

The marathon result made this brutally clear: a runtime can be "cautious" and
still produce unstable turning behavior that becomes a mechanical tipping problem.
That is why platform stability is part of the architecture story, not an external
footnote.

# Software Architecture, But Told Like It Actually Feels

## Indoor Data Flow

The indoor data flow is easiest to understand as a layered question chain:

1. What corridor step does the current image look like?
1. Among the plausible steps, which one is consistent with recent history?
1. Given that step, which checkpoint are we trying to reach now?
1. Given that checkpoint, what nearby subgoal should the controller chase?
1. Given that subgoal, what command should MBRA output right now?
1. Is the forward depth safe enough to allow that command?

That layered structure is why the indoor system became explainable.

## Outdoor Data Flow

The outdoor flow asks a different chain of questions:

1. Has the SDK given us a valid mission and telemetry stream?
1. What mission checkpoint is currently active?
1. Should we route to that checkpoint directly or through OSM-expanded
waypoints?
1. What is the current active local target?
1. What command does LogoNav or the GPS controller want to send?
1. Do traversability, semantics, IMU, vision, GPS, or route-corridor logic
say that this command should be clipped, replaced, or blocked?

This is why the outdoor runtime is large. It is answering a lot of operational
questions, not just computing one control law.

## Why The Responsibilities Are Separated

The separation exists because each layer is answering a fundamentally different
question:

- localization answers *where am I?*
- planning answers *what should I be moving toward next?*
- control answers *how should I move in the next instant?*
- safety answers *should I even allow that motion?*
- recovery answers *what do I do if the main loop stops making sense?*

Whenever the code mixed these responsibilities too aggressively, the system got
harder to debug.

# Design Decision Diary

## Why Exact Checkpoint Steps Instead of Indoor Image Lists

The obvious alternative was to treat indoor checkpoints as arbitrary images. The
problem with that idea is that an image by itself does not tell the planner where it
belongs in the corridor graph. Exact step targets were cleaner because they aligned
localization, planning, and reporting around one common coordinate system.

## Why MBRA Instead of Keeping the Simple Controller as the Main Indoor Path

The simple controller remained useful as a backup and for understanding the runtime,
but it had a basic weakness: indoors it effectively became a forward-driving
controller with almost no real directional intelligence once subgoal orientation was
not trusted. MBRA, by contrast, could actually steer toward a visual target image.

## Why We Did Not Try To Make MBRA Solve Global Planning

Because that would have forced the wrong tool into the wrong job. The corridor graph
and place-recognition structure already solved the global part much better than a
raw learned controller could in this problem setting.

## Why Outdoor Stayed With LogoNav

The obvious temptation was to keep searching for a totally different outdoor
controller. The project instead accepted a more pragmatic truth: the existing
controller path was good enough to justify runtime hardening around it. That choice
saved time and exposed the real bottleneck: transition logic.

## Why OSM Routing Was Worth Keeping

The simpler alternative was to just chase the final GPS checkpoint directly. That is
easy to code, but it throws away route structure. OSM routing was worth keeping
because it gave the rover a more human-plausible local path, even though it also
introduced waypoint-transition complexity.

## Why Semantics and Depth Were Kept But Demoted

This is one of the most important design lessons in the project. Both depth and
semantics were useful enough to keep. Neither was trustworthy enough to rule the
whole runtime. So the correct design was not "delete them" and not "trust them
absolutely." The correct design was to use them as supporting safety layers with
limited authority.

## What this choice cost us

Every major design decision in this repo bought clarity at the price of something
else.

- Splitting indoor and outdoor gave honesty and better debugging, but it also
meant the repo stopped looking like one clean universal stack.
- Making MBRA the indoor local controller simplified responsibility, but it
forced the runtime to become explicit about localization and checkpoint semantics.
- Keeping LogoNav outdoors saved time and preserved a real runnable base, but
it forced the team to solve ugly runtime-transition problems instead of hiding
behind a new-controller fantasy.
- Demoting depth and semantics kept the stack honest, but it also meant the
final system still looks less glamorous than a fake "fully scene-aware" story.

## What these choices enabled

The same choices also enabled the strongest parts of the final project:

- a real indoor backbone that reached 8 of 11 checkpoints,
- an outdoor runtime that could actually complete a mission at least once,
- a project explanation that can survive technical questioning,
- and a codebase whose current failure modes are understandable rather than
mystical.

> [!note] Design lesson
> The final architecture is not elegant because it is small. It is defensible because
> each layer now has a reason to exist, and because the repo no longer pretends that
> one model solved every problem.

# Algorithms and Math, But In The Right Order

This section is not meant to replace the separate math teaching file. It is meant to
tie the math back to the runtime behavior.

## Visual Place Recognition

Plain English first:

- turn each image into a descriptor,
- compare that descriptor to stored corridor descriptors,
- choose the closest match.

In the code, temporal localization then adds extra cost terms:

\[
\text{score} =
\text{descriptor distance}
+ \text{continuity penalty}
+ \text{heading penalty}
\]

The continuity penalty itself is piecewise:

- no jump penalty if the step jump is small,
- extra cost if the jump is larger than `max_step_jump`,
- extra cost if the candidate goes backward.

That math lives directly in `src/temporal_localization.py`.

## Planner Logic

The planner is not doing continuous optimization. It is doing graph shortest-path
reasoning, then selecting a node some number of hops ahead as the local subgoal.

Plain English:

- find current node,
- find target checkpoint node,
- compute shortest path,
- pick a subgoal a few hops along that path.

## Semantic Risk

The semantic runtime is easier to understand if you think in \emph{fractions of the
image region} rather than in raw labels.

The code measures quantities like:

- person fraction in the center region,
- animal fraction in the center region,
- drivable fraction in the center region,
- caution fraction in the center region.

Then it builds a risk score using thresholds such as:

- person threshold \(= 0.002\)
- animal threshold \(= 0.002\)
- pole threshold \(= 0.002\)
- wall threshold \(= 0.010\)

The vegetation-blocked condition is compound:

\[
\text{drivable_center} < 0.10
\quad \text{and} \quad
\text{caution_center} > 0.60
\]

In direct code-aligned notation, the risk accumulator is:

\[
\text{risk} = 0
\]
\[
\text{if } p > 0.002:\ \text{risk} \mathrel{+}= 0.55 + 18.0\,p
\]
\[
\text{if } a > 0.002:\ \text{risk} \mathrel{+}= 0.55 + 18.0\,a
\]
\[
\text{if } \pi > 0.002:\ \text{risk} \mathrel{+}= 0.35 + 10.0\,\pi
\]
\[
\text{if } w > 0.010:\ \text{risk} \mathrel{+}= 0.30 + 8.0\,w
\]
\[
\text{if } (d < 0.10 \land c > 0.60):\ \text{risk} \mathrel{+}= 0.45 + 0.50\,(c-0.60)
\]

where \(p,a,\pi,w,d,c\) are center-region fractions for person, animal, pole, wall,
drivable, and caution respectively.

## Semantic Angular Bias Formula

The free-score used for left/right steering preference is:

\[
F = d + 0.30\,n - 0.60\,c
\]

and in hard mode:

\[
F = d + 0.30\,n - 0.60\,c - 4.0\,(p+a) - 3.0\,(\pi+w)
\]

The bias then uses normalized free-score difference:

\[
\Delta = F_{\text{left}} - F_{\text{right}},\quad
S = \max(0.25,\ |F_{\text{left}}| + |F_{\text{right}}|)
\]
\[
\text{bias} = \operatorname{clip}\!\left(\frac{\Delta}{S},\ -0.50,\ +0.50\right)
\]

## Traversability Geometry

Traversability slices the image into angular bins and looks at a middle vertical
band rather than the bottom ground-heavy region. That is a geometric choice, not
just a coding choice. It says: the useful obstacle evidence is at rover-eye level,
not in the patch of ground right under the camera.

## IMU Tilt Logic

The IMU safety logic calibrates a reference gravity direction and then measures the
angle between the current accelerometer direction and that reference direction. That
is how the system estimates tilt.

In math form:

\[
\hat{a} = \frac{a}{\|a\|},\quad
\hat{g}_{\text{ref}} = \text{calibrated gravity unit vector}
\]
\[
\theta_{\text{tilt}} = \cos^{-1}\!\left(\operatorname{clip}\left(\hat{a}\cdot\hat{g}_{\text{ref}}, -1, 1\right)\right)\cdot\frac{180}{\pi}
\]

The code anchor in `src/imu_safety.py` is:

```
dot = float(np.clip(np.dot(accel_unit, gravity_ref), -1.0, 1.0))
tilt_deg = math.degrees(math.acos(dot))
```

## Temporal Localization Penalty Formula (Explicit)

The temporal-localization score used in the code can be written as:

\[
\text{score}_i
=
w_d\,d_i
+ \lambda_j\,\max(0,\ |\Delta s_i| - S_{\max})
+ \lambda_b\,\max(0,\ -\Delta s_i)
+ \lambda_h\,|\Delta \psi_i|
\]

with:

- \(d_i\): descriptor distance for candidate \(i\),
- \(\Delta s_i\): step jump from previous localized step,
- \(S_{\max} = 20\): allowed jump before extra penalty,
- \(\lambda_j = 0.05\), \(\lambda_b = 0.15\), \(\lambda_h = 0.002\),
- confidence \(= 1/(1+\text{best score})\).

# Code Anchors For The Behaviors People Ask About Most

One of the weaknesses of many project guides is that they stay one level too high.
They describe behavior, but they do not show where that behavior actually lives. The
sections below fix that for the most commonly misunderstood mechanisms in this repo.

## Indoor skip-past-checkpoint logic

This is the logic that stopped the rover from freezing forever when it had already
localized past a checkpoint in a forward-only graph:

```
if (
    is_checkpoint_mode
    and args.auto_advance_checkpoints
    and not path_found
    and cur_step is not None
    and tgt_step is not None
    and int(cur_step) > int(tgt_step)
    and confidence >= 0.45
):
    skipped = tgt_step
    next_cp = runtime.planner.advance_checkpoint()
```

Why it matters:

- the planner was not wrong when it said "no path",
- the runtime was wrong for still asking it to go backward,
- this logic converts that situation into progress instead of deadlock.

## Indoor stale-context / no-progress reset

This is the practical indoor fix that says "if MBRA is staring at effectively the
same place for too long, stop trusting the old context":

```
if no_progress_count > 0 and no_progress_count % NO_PROGRESS_RESET_TICKS == 0:
    if hasattr(controller, 'reset'):
        controller.reset()
        print(f"[{iteration:04d}] no-progress reset after {no_progress_count} ticks at step {cur_step}")
```

The point is not magic recovery. The point is to stop the controller from dragging
forward stale visual context that no longer matches what the robot needs to do next.

## Why indoor localization confidence is not a mysterious black box

The temporal localizer confidence is currently defined as:

\[
\text{confidence} = \frac{1}{1 + \text{best score}}
\]

That matters because it means confidence is not some opaque neural-network
probability. It is a transformed retrieval score after continuity penalties have
already shaped the candidate ranking.

## Outdoor route corridor stop logic

The route-corridor stop is one of the clearest examples of the runtime being more
important than any single controller:

```
if route_corridor_distance is not None and route_corridor_distance > route_corridor_stop_threshold:
    route_corridor_violation_ticks += 1
...
if route_corridor_violation_ticks >= args.route_corridor_confirm_ticks:
    safe_stop()
    print(f"[{iteration:04d}] route_corridor_stop dist={route_corridor_distance:.1f}m ...")
    _rerouted = reroute_from_current_pose(...)
```

This is what turns "OSM routing is a good idea" into an executable runtime rule.
Without this, the route is only a suggestion.

## Outdoor semantic yield and stop logic

The runtime does not stop on every semantic signal. It uses specific thresholds and
specific label families. For example:

```
_semantic_stop_active = bool(_alerts & {"person", "animal"}) \
    and last_sem_result.risk_score >= args.semantic_stop_risk
```

and yield can activate if any of these are true:

- semantic risk is high enough,
- a person fraction is present,
- an animal fraction is present,
- or the center looks road-dominant without enough sidewalk/path support.

This is important because many people imagine the semantic layer as "if model sees
danger, stop." The actual logic is more conditional and more careful than that.

## Outdoor dynamic intermediate waypoint radius

The repo used to treat routed waypoints too much like real checkpoints. One of the
important fixes was to make intermediate routed waypoint handoff depend on segment
geometry:

```
if bool(target.get("mission_checkpoint", False)):
    active_goal_radius_m = args.goal_radius_m
else:
    _dynamic_radius_m = float(args.intermediate_goal_radius_m)
    if math.isfinite(_segment_distance_m) and _segment_distance_m > 0.0:
        _dynamic_radius_m = min(_dynamic_radius_m, max(2.5, 0.35 * _segment_distance_m))
    active_goal_radius_m = min(args.goal_radius_m, _dynamic_radius_m)
```

This is one of the cleanest examples of a small code change reflecting a deep
runtime insight: not all targets deserve the same semantics.

## Outdoor align gating

The LogoNav alignment behavior now depends not only on bearing error but also on
distance and target type:

```
_align_allowed = (
    math.isfinite(distance_to_goal)
    and (
        distance_to_goal <= args.logonav_align_distance_m
        or (
            not bool(target.get("mission_checkpoint", False))
            and _bearing_error_deg >= args.logonav_align_extreme_deg
        )
    )
)
```

That code matters because it captures the later outdoor lesson exactly:
alignment is not always bad, but blind alignment is bad.

# Training and Experimentation History

## What Counted as "Training" Here

This repository is not mainly a training repo. Much more of the work was:

- runtime integration,
- wrapper design,
- offline analysis,
- parameter sweeps,
- field debugging,
- and deciding what not to trust.

## Indoor Experimental Progression

Indoor passed through several conceptual versions:

1. baseline corridor localization and graph planning,
1. confusion about local control and backup behavior,
1. MBRA-first clarification,
1. checkpoint-step mode with skip logic,
1. no-reverse / stale-context cleanup.

## Outdoor Experimental Progression

Outdoor also passed through several versions:

1. plain mission-following baseline,
1. controller-first outdoor framing,
1. depth/traversability investigation,
1. semantic investigation,
1. marathon and night safety hardening,
1. post-run fixes for waypoint semantics and transition behavior.

## Mistakes That Looked Like Improvements At First

- adding heavy recovery behavior around MBRA,
- treating depth as if it were already a reliable hard-stop outdoor backbone,
- treating semantic labels as if they already captured the terrain semantics we
cared about,
- assuming more caution automatically meant more stability.

# How To Interpret The Results Honestly

## What The Indoor Result Proves

It proves that the indoor stack had a real, working backbone. It does *not*
prove that indoor navigation was fully solved under every corridor ambiguity or
recovery case.

## What The Outdoor Result Proves

It proves that the outdoor stack was capable of a full mission completion at least
once. It does *not* prove robust repeatability.

## What The Marathon Result Proves

It proves that the runtime and platform were still too fragile during target
transitions for long-run confidence. It does *not* prove that the safety work
was pointless; in fact it made the failure easier to interpret.

# Operations: How To Run and What a Healthy Run Looks Like

## Before You Run Anything

Do these first:

1. verify SDK connectivity,
1. verify camera frames are live,
1. verify telemetry timestamps are advancing,
1. verify correct controller weights are present,
1. start with dry-run if behavior is not already trusted.

## What A Healthy Indoor Run Looks Like

- localization is stable over multiple ticks,
- confidence does not collapse repeatedly,
- checkpoint progression is legible in the logs,
- MBRA keeps moving toward the subgoal instead of fighting a backup controller.

## What A Healthy Outdoor Run Looks Like

- mission checkpoints load correctly,
- the current waypoint makes sense,
- the rover is not endlessly re-aligning,
- route corridor deviation stays bounded,
- safety stops, if they happen, are interpretable from the logs.

## What Not To Touch Too Early

- do not start by rewriting the whole controller,
- do not trust a new safety layer before offline sanity checks,
- do not mix indoor and outdoor assumptions,
- do not change multiple interacting thresholds at once if you want to learn
anything from the run.

# Troubleshooting

```text
\toprule
\textbf{Symptom} & \textbf{Likely cause} & \textbf{What to inspect first} \\
\midrule
Indoor says no path repeatedly & rover already past target in forward-only graph & checkpoint step, current step, skip logic \\
Indoor moves but never steers well & simple controller active or MBRA context not healthy & controller flag, MBRA warmup, subgoal image path \\
Indoor oscillates after a stop & stale context or backup behavior fighting local control & no-progress reset, recovery logic, depth stop logs \\
Outdoor keeps rerouting & corridor threshold too aggressive or route geometry unstable & corridor stop threshold, reroute trigger, waypoint radius \\
Outdoor follows the wrong local target & routed waypoint semantics broken or waypoint behind rover & route expansion, behind-waypoint prune logic \\
Outdoor spins in place & over-aggressive alignment on the wrong target type & align gating, mission checkpoint vs route waypoint \\
Night run freezes & image-quality gate or conservative safety stack & lamp, vision safety debug, traversability thresholds \\
Marathon run feels too hesitant & safety stack too cautious or transition logic unstable & composite flags, speed caps, route guard, post-waypoint behavior \\
\bottomrule
```

# Known Failure Modes and Current Mitigation Status

```text
\toprule
\textbf{Failure mode} & \textbf{Observed impact} & \textbf{Current mitigation} & \textbf{Status} \\
\midrule
Indoor ambiguous localization jump & wrong step hypothesis, unstable short-term steering & temporal stabilization + jump rejection + checkpoint-step progression & Partially controlled \\
Indoor forward-only no-path condition & rover appears ``stuck'' near passed checkpoint & skip-past-checkpoint advancement logic & Controlled in known cases \\
Indoor stale MBRA context while stationary & repeated low-value commands with low net progress & no-progress context reset for MBRA & Partially controlled \\
Outdoor intermediate-waypoint handoff instability & ALIGN-heavy loop and delayed forward recovery & dynamic waypoint radius + behind-waypoint pruning + align gating & Partially controlled \\
Outdoor route corridor drift deadlock & repeated stops without meaningful progress & corridor-stop reroute from current pose & Partially controlled \\
Outdoor over-cautious intervention stack & excessive hesitation / slow crawl in some scenes & profile-specific caps + guarded intervention tuning & Not fully resolved \\
Marathon post-transition spin + physical tip risk & run termination after checkpoint progress & stricter profile + IMU safety + route discipline; platform dynamics still limiting & Open \\
\bottomrule
```

# Interview Mode

## Short clean answer: What is this project?

It is a split indoor/outdoor rover autonomy stack. Indoors, the system uses visual
place recognition, temporal smoothing, graph planning, and MBRA as a local
controller. Outdoors, the system uses mission checkpoints, OSM-expanded waypoints,
LogoNav as a local controller, and a layered runtime safety envelope.

## Short clean answer: What is the hardest part?

Transitions. Indoors it is checkpoint-step and stale-context transitions. Outdoors
it is waypoint and checkpoint handoff stability. Marathon proved that transition
stability is the biggest remaining outdoor weakness.

## Blunt answer: Did it work?

Yes, but unevenly. Indoor worked best and reached 8 of 11 checkpoints. Outdoor
worked well enough to complete a run once, but not robustly enough to claim solved
reliability. Marathon failed in a way that exposed the remaining transition and
stability problem very clearly.

## Aggressive answer: Why is the design defensible?

Because the final architecture follows the actual structure of the problem instead
of pretending there is one universal controller. The indoor and outdoor problems are
different, so the stack is different. That is the honest and technically defensible
choice.

## Strong answer: What are you most proud of?

That the project became understandable. By the end, the roles of localization,
planning, control, safety, recovery, and mission logic were much clearer and more
defensible than they were when the work started.

# Repo Map By Question

## If I want to understand indoor localization

Read:

- `baseline.py`
- `src/corridor_localizer.py`
- `src/temporal_localization.py`

## If I want to understand indoor planning

Read:

- `src/graph_planner.py`
- `src/navigation_runtime.py`
- `live_indoor_runtime.py`

## If I want to understand MBRA integration

Read:

- `src/mbra_controller.py`
- `live_indoor_runtime.py`
- [[docs/our_mbra_discoveries]]

## If I want to understand outdoor control

Read:

- `live_outdoor_runtime.py`
- `src/outdoor_logonav_controller.py`
- `src/outdoor_gps_controller.py`

## If I want to understand outdoor safety

Read:

- `src/outdoor_traversability.py`
- `src/semantic_risk_estimator.py`
- `src/imu_safety.py`
- `src/vision_safety_monitor.py`

## If I want the project story

Read:

- [[live_indoor_runtime_story]]
- [[live_outdoor_ultra_marathon_story]]
- [[outdoor_perception_review]]

# Glossary

- **VPR**: Visual place recognition. In this project, that means matching a live
corridor frame against stored corridor descriptors.
- **CosPlace**: The descriptor model used for corridor retrieval. People often
misunderstand it as a full navigation method. It is only the localization part.
- **Temporal stabilization**: The continuity-aware layer that stops localization
from jumping too aggressively between plausible corridor matches.
Also called temporal filtering in some notes, though the implementation here is
lighter-weight and more hand-structured than what many people imagine from that
phrase.
- **Checkpoint-step mode**: Indoor competition mode where checkpoints are exact
corridor steps rather than vague target images.
Older notes may still talk in terms of checkpoint images; the decisive later
improvement was to collapse those ideas into explicit step semantics.
- **MBRA**: Model-Based Re-Annotation. In this project, used as a short-horizon
indoor local controller, not as a planner. A common misconception is that MBRA
was the indoor system. It was only one layer.
- **Sole motion authority**: The idea that when MBRA is active, backup motion logic
should not constantly fight it.
- **LogoNav**: The learned outdoor local motion policy. Not the whole outdoor
autonomy stack. A common misunderstanding is to treat LogoNav as the outdoor
equivalent of "the whole brain." It is not.
- **OSM-expanded waypoint**: An intermediate route point created from
OpenStreetMap path expansion, not a real mission checkpoint.
- **Mission checkpoint**: A real task-level target from the SDK. Not the same
thing as a local routed waypoint.
- **Traversability**: The depth-based local estimate of whether the forward region
looks open enough to drive through. In this repo it became a middle-band,
obstacle-height interpretation rather than an underfoot ground detector.
- **Semantic risk**: A compact score built from semantic label fractions in key
regions of the image.
- **Route corridor guard**: A runtime rule that watches how far the rover drifts
from the intended routed path.
- **Stale context**: Old controller context that no longer describes the current
motion situation well enough to trust.
- **Jump rejection**: The indoor runtime rule that rejects very large localization
jumps unless confidence is high enough to justify them.
- **Held on ambiguity**: A temporal-localization outcome where the filter keeps the
previous node because the new best candidate is not decisively better.
- **Forward-only graph behavior**: The indoor phenomenon where the planner can
correctly return no backward path once the rover has already moved past a
checkpoint.
- **Semantic yield**: An outdoor behavior where motion is strongly slowed rather
than fully stopped because the scene looks risky or socially occupied.
- **Semantic sidewalk stop**: An outdoor stop condition triggered when the center
corridor looks road-dominant and insufficiently sidewalk-like.
- **Health gate**: A pre-run or inter-leg validation step that checks telemetry,
GPS, camera, and battery before motion is allowed.
- **Nav-ready gate**: A runtime gate requiring several consecutive healthy ticks
before motion resumes after startup or hard stops.
- **Ultra-marathon mode**: The stricter outdoor safety profile that combines IMU
protection, route discipline, speed caps, recovery restraint, and readiness
gating.
- **Night-safe mode**: The night-time outdoor profile that reduces speed, enables
image-quality safety, and makes the runtime more conservative overall.

# Common Misconceptions That Waste Time

## "The project is basically MBRA indoors and LogoNav outdoors"

That sentence is too shallow to be useful. It hides localization, graph planning,
target semantics, safety layers, and runtime transitions. If someone says that in
an interview, they sound like they only know the names of the controllers.

## "Indoor failed because MBRA was weak"

Not mainly. Indoor became strong only after the team stopped blaming the controller
for problems that were really about localization drift, forward-only graph logic, or
stale runtime state.

## "Outdoor failed because LogoNav was bad"

Also too shallow. The more honest statement is that the outdoor runtime was brittle
around target transitions, especially when intermediate routed waypoints were handed
off badly or alignment logic overreacted.

## "More safety would have saved marathon"

Not automatically. Safety only helps if it preserves a calm loop. Marathon showed
that a stack can become safer on paper and still become less stable in practice if
transitions become stop-turn-stop-turn loops.

## "Depth and semantics did not matter because they were not the main controller"

Wrong in the other direction. They mattered a lot. They just did not earn the right
to become the whole truth layer. That is different from being useless.

# Current Status

## Production-Ready vs Experimental Capability Matrix

```text
\toprule
\textbf{Capability} & \textbf{State} & \textbf{Notes} \\
\midrule
Indoor checkpoint-step runtime & Production-ready (competition use) & Delivered 8/11 checkpoints; behavior understood and debuggable. \\
Indoor MBRA local-control path & Production-ready (with guardrails) & Stable when used as sole motion authority with checkpoint-step semantics. \\
Outdoor mission runtime with LogoNav + OSM & Operational / partially reliable & One full completion achieved; repeatability still mixed across runs. \\
Outdoor traversability intervention & Operational / tuning-sensitive & Useful for local obstacle handling; still environment-dependent. \\
Outdoor semantic risk layer & Experimental support layer & Valuable bias/intervention signal, but not trusted as primary authority. \\
Marathon ultra-safe profile & Operational but not race-robust & Safety layering is substantial; transition and platform stability remain open. \\
\bottomrule
```

## What works now

- indoor localization backbone,
- indoor graph progression,
- indoor MBRA-first runtime path,
- outdoor mission runtime skeleton,
- outdoor LogoNav path,
- multiple outdoor safety and intervention layers.

## What is partially working

- outdoor repeatability,
- semantic runtime value in the field,
- traversability calibration across many scene types,
- long-run outdoor stability.

## What is still missing

- truly calm waypoint / checkpoint transition behavior outdoors,
- better physical stability for marathon conditions,
- stronger confidence that the runtime will not fall into repeated align-turn
loops under field stress.

## What the next clean milestone is

The next clean milestone is not "add more models." It is:

> make the outdoor runtime boring during transitions.

That means fewer surprising waypoint handoffs, less spinning, better route
semantics, and calmer platform behavior.

# Source of Truth, Historical Drift, and How Not To Get Lied To By The Repo

One of the reasons this repository feels harder than it should is that it contains
multiple generations of thought at once. Some files are active runtime truth. Some
files are historical explanations. Some files are experiments that were valuable,
but never became final authority.

## What counts as source of truth

If two documents disagree, trust the active runtime and the active wrappers first.
For this project that means:

- `live_indoor_runtime.py`
- `live_outdoor_runtime.py`
- `src/temporal_localization.py`
- `src/graph_planner.py`
- `src/mbra_controller.py`
- `src/outdoor_logonav_controller.py`
- `src/outdoor_traversability.py`
- `src/semantic_risk_estimator.py`
- `src/imu_safety.py`

## What is historical but still valuable

The long story and review documents remain valuable because they preserve the
reasoning path:

- [[live_indoor_runtime_story]]
- [[live_outdoor_ultra_marathon_story]]
- [[outdoor_perception_review]]
- [[semantic_segmentation_research_review]]
- [[docs/our_mbra_discoveries]]

These are not junk. They explain why certain ideas were promoted, demoted, or
left half-integrated. But they should not override the current code.

> [!warning] Important reading rule
> Many misunderstandings in this repo come from treating a historical document as if
> it were a live configuration file. The stories tell you how the team learned. The
> runtime files tell you what the rover actually does now.

## Examples of historical drift that matter

- Older indoor discussions often revolve around 4-hop MBRA subgoals. The
current competition-facing indoor runtime uses 8 hops in MBRA mode.
- Older docs sometimes talk about depth or semantics as if they were about to
become the main outdoor truth layer. The current code keeps them as support layers
with bounded authority.
- Older marathon notes may sound like more safety automatically meant a better
run. The field result made it obvious that over-cautious logic can still produce a
worse control loop if transitions become unstable.

# If You Only Remember Ten Things, Remember These

1. Indoor and outdoor are different systems that happen to share the same rover.
1. Indoor became solvable only after it was reframed as a known-corridor graph
problem.
1. Outdoor became understandable only after it was reframed as a mission-runtime
problem rather than a single-controller problem.
1. MBRA was never the whole indoor stack; it was the short-horizon indoor local
controller.
1. LogoNav was never the whole outdoor stack; it was the outdoor local motion
policy inside a much larger runtime.
1. The best indoor insight was that localization and graph progression were the
backbone.
1. The best outdoor insight was that target semantics and transition behavior
were the real bottleneck.
1. Depth and semantics were worth keeping, but not worth worshipping.
1. The marathon failure was not random. It was the most compressed possible
demonstration of the project’s remaining weakness: unstable outdoor transitions.
1. The next milestone is not a new model. It is a calmer outdoor runtime.

# Indoor Runtime: A Tick-By-Tick Walkthrough

The easiest way to really understand indoor is to watch one iteration of the loop as
if you were the code.

## Step 1: a camera frame arrives

The runtime receives a live RGB frame from the front camera. At this moment the
system still does not know where it is. It only knows that it has an image.

## Step 2: corridor localization produces candidates

The corridor localizer embeds the live frame and compares it against the stored
corridor descriptors. The raw output is not just one definitive answer. It is a set
of plausible corridor positions ranked by descriptor distance.

What a newcomer often misses is that this is the first place where the project
became tractable. Without a corridor memory, indoor would have been vague. With a
corridor memory, indoor becomes a repeated recognition task.

## Step 3: temporal localization decides whether to trust the jump

The temporal localizer then scores the candidates again using continuity penalties.
In the current code this is literally:

```
score = distance_weight * distance
score += continuity_cost(candidate_index)
score += heading_cost(candidate_heading, observation_heading)
```

The continuity cost itself does two important things:

- penalize large jumps beyond `max_step_jump = 20`
- penalize backward movement using `backward_penalty = 0.15`

If the best and second-best candidates are too close in score, and switching would
require a sufficiently large jump, the filter can hold the previous state instead of
jumping on weak evidence.

> [!note] Why this mattered so much
> Indoor corridor images repeat. Without continuity logic, the rover can look at one
> wall segment and suddenly believe it is 50 steps away in a visually similar place.
> Temporal stabilization is what turns place recognition into usable localization.

## Step 4: the graph planner asks the right question

Once the current node is stable enough, the planner asks:

> What checkpoint am I actually trying to reach now, and what nearby subgoal should I
> use so the local controller is not forced to reason too far ahead?

That is why `GraphPlanner` does shortest-path reasoning and then selects a
subgoal node several hops along that path.

In the generic planner config the default is:

- `max_subgoal_hops = 3`
- `min_confidence_to_advance = 0.55`
- `checkpoint_reach_tolerance = 3`

But in the actual indoor competition runtime, MBRA mode overrides the hop count to
8. That is an important example of runtime truth outranking generic library
defaults.

## Step 5: MBRA gets a much smaller problem than global navigation

At this point MBRA is not being asked to decide where the rover is in the building.
It is not being asked which checkpoint comes next. It is being asked a much more
reasonable question:

> Given the current image and the chosen subgoal image, what should the next short
> motion command be?

That is why the wrapper description in `src/mbra_controller.py` is so
important. It explicitly says MBRA is being kept in the only role that makes sense
for this project: short-horizon local control between nearby graph subgoals.

## Step 6: depth can veto motion, but it does not own the loop

Only after localization, temporal smoothing, planning, and controller inference does
indoor depth safety get a say. In MBRA mode the depth thresholds are:

- stop below 0.25 m
- slow below 0.60 m

This ordering matters. If depth had been turned into the central intelligence layer,
the indoor system would have become much harder to reason about. Instead, depth
stayed a local safety veto.

## Step 7: checkpoint advancement is not the same thing as image matching

The runtime then decides whether the current checkpoint should be considered reached.
That uses localized step, confidence, and checkpoint-step tolerance rather than a
romantic idea of "the image looks close enough." This was one of the most
important clarity improvements in the whole project.

# Outdoor Runtime: A Tick-By-Tick Walkthrough

Outdoor needs the same treatment because people often collapse it into "GPS +
LogoNav," which is not even close to enough.

## Step 1: the SDK defines the mission world

The outdoor runtime begins by talking to the EarthRover SDK. That means:

- fetching mission checkpoints,
- reading telemetry,
- receiving camera frames,
- sending motion commands,
- optionally reporting checkpoint completion and interventions.

This is why the outdoor system is inseparable from the mission runtime. It is not a
controller demo script.

## Step 2: a mission checkpoint may become a routed corridor of local targets

If OSM routing is enabled, the active checkpoint is not followed directly. Instead,
the runtime queries OpenStreetMap and expands the leg into a sequence of local
waypoints that roughly respect pedestrian structure.

This is why the outdoor problem became subtle. The rover now has to understand the
difference between:

- the real mission checkpoint, which matters for task completion,
- the temporary routed waypoint, which only exists to guide local progress.

> [!warning] Outdoor misconception
> Many outdoor bugs looked like controller weakness. In reality they came from the
> runtime confusing target types. A routed intermediate waypoint is not a mission
> checkpoint, and treating them the same created bad handoffs, bad alignment, and bad
> reroutes.

## Step 3: LogoNav proposes motion

LogoNav then turns the current perception state and target geometry into a local
command. That command is valuable, but it is never the final truth. The rest of the
runtime still has a chance to reject, clip, or redirect it.

## Step 4: traversability, semantics, and guards decide whether motion is sane

The learned controller output then passes through multiple checks:

- traversability can clip linear speed or suggest a safer heading,
- semantics can apply a soft bias or, in stricter modes, participate in stop
or yield logic,
- IMU safety can hard-stop on dangerous motion,
- vision safety can halt motion if image quality collapses,
- GPS safety can block motion if signal or position consistency looks bad,
- route corridor logic can stop or reroute if the rover drifts too far from
the intended corridor.

## Step 5: the runtime asks whether the target should still be trusted

This is where many of the late outdoor fixes live. The system must ask:

- is the current routed waypoint already effectively reached?
- is the next routed waypoint behind the rover?
- should a reroute happen from the live pose rather than stale route memory?
- is the current bearing error large enough to justify alignment behavior?

The reason the outdoor runtime grew so much is that these questions are not
optional. If you do not answer them explicitly, the rover answers them badly in the
field.

# What Older Docs Still Get Wrong, or At Least Leave Ambiguous

## "MBRA was the indoor system"

No. MBRA was the indoor local controller. The indoor system was corridor memory,
localization, temporal stabilization, graph planning, checkpoint-step semantics,
MBRA, and depth veto.

## "LogoNav was the outdoor system"

Also no. LogoNav was the outdoor learned motion policy inside a mission runtime
that had to manage checkpoint semantics, route semantics, safety, and recovery.

## "Depth was almost the full outdoor safety answer"

No. Depth helped, especially once the old bottom-heavy crop was replaced by the
middle-band traversability interpretation. But the repo itself warns against
treating old outdoor depth safety as fully reliable on this platform.

## "Semantics was about to become the full scene understanding layer"

Not honestly. Semantics helped the team reason about people, vegetation, sidewalk
structure, and caution zones, but the runtime never had enough evidence to hand it
full authority.

## "More recovery logic is always better"

Indoor disproved this with MBRA, and marathon disproved it again outdoors. Recovery
logic can save you from real failure, but it can also keep injecting its own
failure mode if it fights the controller or destabilizes a transition.

# Concrete Configuration Appendix

This section exists because newcomers often want the exact parameters after they
finally understand the story. That is a good instinct. The project becomes much
easier to reason about once the numbers are visible.

## Temporal localization defaults

```text
\toprule
\textbf{Parameter} & \textbf{Value} \\
\midrule
\texttt{top\_k} & 10 \\
\texttt{max\_step\_jump} & 20 \\
\texttt{distance\_weight} & 1.0 \\
\texttt{jump\_penalty} & 0.05 \\
\texttt{backward\_penalty} & 0.15 \\
\texttt{heading\_penalty} & 0.002 \\
\texttt{ambiguity\_margin} & 0.05 \\
\texttt{hold\_on\_ambiguity} & True \\
\texttt{ambiguity\_hold\_min\_jump} & 4 \\
\bottomrule
```

## Graph planner defaults

```text
\toprule
\textbf{Parameter} & \textbf{Value} \\
\midrule
\texttt{max\_subgoal\_hops} & 3 (generic planner default) \\
\texttt{min\_confidence\_to\_advance} & 0.55 \\
\texttt{checkpoint\_reach\_tolerance} & 3 steps \\
\bottomrule
```

## Indoor competition runtime overrides

```text
\toprule
\textbf{Parameter} & \textbf{MBRA mode} & \textbf{Simple mode} \\
\midrule
\texttt{max\_subgoal\_hops} & 8 & 15 \\
\texttt{tick\_hz} & 3.0 & 2.0 \\
\texttt{depth\_stop\_m} & 0.25 & 0.40 \\
\texttt{depth\_slow\_m} & 0.60 & 0.80 \\
\texttt{NO\_PROGRESS\_RESET\_TICKS} & 10 & 10 \\
\bottomrule
```

## MBRA wrapper deployment values

```text
\toprule
\textbf{Parameter} & \textbf{Value} \\
\midrule
\texttt{context\_size} & 6 frames of history \\
\texttt{vel\_past\_linear} & 0.5 \\
\texttt{vel\_past\_angular} & 0.0 \\
\texttt{max\_linear} & 0.40 \\
\texttt{min\_linear} & 0.18 \\
\texttt{max\_angular} & 0.34 \\
\texttt{min\_confidence} & 0.45 \\
\bottomrule
```

## Traversability defaults

```text
\toprule
\textbf{Parameter} & \textbf{Value} \\
\midrule
\texttt{num\_bins} & 16 \\
\texttt{fov\_horizontal\_deg} & 90.0 \\
\texttt{crop\_top\_frac} & 0.15 \\
\texttt{crop\_bot\_frac} & 0.60 \\
\texttt{obstacle\_distance\_m} & 1.5 \\
\texttt{stop\_distance\_m} & 0.60 \\
\texttt{slow\_distance\_m} & 1.20 \\
\texttt{memory\_frames} & 4 \\
\bottomrule
```

## Semantic runtime thresholds

```text
\toprule
\textbf{Parameter} & \textbf{Value} \\
\midrule
\texttt{roi\_top\_frac} & 0.40 \\
\texttt{roi\_bottom\_frac} & 0.80 \\
\texttt{roi\_left\_frac} & 0.30 \\
\texttt{roi\_right\_frac} & 0.70 \\
\texttt{person\_thresh} & 0.002 \\
\texttt{animal\_thresh} & 0.002 \\
\texttt{pole\_thresh} & 0.002 \\
\texttt{wall\_thresh} & 0.010 \\
\texttt{drive\_thresh} & 0.10 \\
\texttt{caution\_thresh} & 0.60 \\
\texttt{max\_bias} & 0.50 \\
\bottomrule
```

## IMU safety thresholds

```text
\toprule
\textbf{Parameter} & \textbf{Value} \\
\midrule
\texttt{max\_tilt\_deg} & 40.0 \\
\texttt{max\_pitch\_roll\_rate\_dps} & 150.0 \\
\texttt{gyro\_min\_tilt\_deg} & 12.0 \\
\texttt{gyro\_min\_vibration} & 1.0 \\
\texttt{vibration\_limit} & 3.0 \\
\texttt{consecutive\_trips\_to\_stop} & 2 \\
\texttt{calibration\_samples} & 8 \\
\bottomrule
```

# Layer-by-Layer Debug Playbooks

This section is deliberately more procedural than the rest of the document. It is
for the moment when the rover does something dumb and you do not want philosophy;
you want order.

## Indoor debug playbook

1. Confirm the controller mode first. If `--controller simple` is active,
do not expect MBRA-like steering behavior.
1. Look at localized step, confidence, and stability before blaming control.
1. If the planner says `no_path`, check whether the rover has already
passed the active checkpoint in the forward graph.
1. If MBRA is active but behavior looks random, ask whether stale context or a
safety backup just destroyed its visual history.
1. Only after those checks should you start tuning controller behavior.

## Outdoor debug playbook

1. Confirm the mission state loaded correctly and the active checkpoint is real.
1. Confirm whether the active target is a mission checkpoint or an intermediate
routed waypoint.
1. If the rover keeps stopping or rerouting, inspect route-corridor deviation
and target handoff radius before blaming LogoNav.
1. If it spins, inspect align gating and the bearing regime of the new target.
1. If it slows strangely, check traversability and semantic intervention before
concluding the controller lost confidence.

## Marathon debug playbook

1. Ask whether the system is too cautious, not just whether it is too bold.
1. Inspect every post-checkpoint transition because that is the most dangerous
regime.
1. Check whether repeated alignment is happening on a target that should have
been handed off more gently.
1. Check whether IMU or route-corridor stops are revealing a real instability
or merely reacting to an already broken target interpretation.
1. Remember that physical stability is part of the diagnosis, not an unrelated
hardware footnote.

# Harder Interview Questions and Better Answers

## Why did the project split instead of converging to one universal stack?

Because the data geometry and operational questions are different. Indoor has a
known corridor, recorded imagery, and exact checkpoint-step structure. Outdoor has
live GPS mission semantics, route semantics, and uncontrolled terrain. Pretending
they were one problem would have produced worse engineering.

## Why not just use one end-to-end controller for everything?

Because the repo already showed that controller quality was only part of the story.
Localization, target semantics, route structure, safety, and recovery all had to be
made explicit. Hiding them inside one slogan would have reduced explainability
without actually fixing the runtime.

## What is the main mathematical idea indoors?

Nearest-neighbor place recognition becomes usable only when continuity penalties are
added over time and the resulting stable node estimate is passed through a graph
planner instead of directly into control.

## What is the main mathematical idea outdoors?

Outdoor is not dominated by one equation. It is dominated by layered gating:
controller proposals are filtered by geometric route logic, local depth structure,
semantic fractions, IMU thresholds, and mission state.

## What did the marathon failure really teach you?

That transition stability matters more than piling on additional safety flags.
Safety helps only if the system still interprets the next target calmly.

# A One-Week Onboarding Plan For A New Team Member

If someone had one week to become useful in this repo, the fastest route would be:

## Day 1

Read this document up to the end of the indoor and outdoor story sections. Do not
edit code yet.

## Day 2

Read:

- `live_indoor_runtime.py`
- `src/temporal_localization.py`
- `src/graph_planner.py`
- `src/mbra_controller.py`

Goal: understand the indoor loop end to end.

## Day 3

Read:

- `live_outdoor_runtime.py`
- `src/outdoor_logonav_controller.py`
- `src/osm_router.py`
- `src/outdoor_traversability.py`

Goal: understand the outdoor loop end to end.

## Day 4

Read the long story and review docs. Goal: understand what failed and why the
current architecture looks the way it does.

## Day 5

Run one indoor dry run and one outdoor dry run with logs enabled. Goal: learn what
healthy behavior and unhealthy behavior look like.

## Day 6

Pick one subsystem and trace it from CLI flag to runtime effect. Good choices:
semantic risk, traversability, or IMU safety.

## Day 7

Explain the full project out loud without looking at notes. If you still collapse it
into "MBRA and LogoNav," repeat the week.

# What I Would Tell A New Teammate In The First Hour

This is the informal version, because this is usually what people need most.

1. Indoor and outdoor are different problems. If you mix assumptions between
them, you will waste days.
1. Indoors, do not touch controller tuning first. Check localization confidence,
checkpoint-step semantics, and graph progression first.
1. MBRA is not a planner. It is the short-horizon controller once state
semantics are already clean.
1. If indoor says `no_path`, first ask whether the rover is already past
the active checkpoint in a forward-only graph.
1. Outdoors, controller output is only one vote. Route semantics and safety
gates can and should override it.
1. Most ugly outdoor failures are transition bugs, not raw speed bugs.
1. A routed intermediate waypoint is not a mission checkpoint. Treating them the
same is how you create loops.
1. Marathon failure did not mean "all safety was wrong." It meant transition
stability and physical stability were still underpowered.
1. If you want one thing to improve next, make outdoor transitions boring.
1. Read the logs like a state-machine trace, not like a random stream.

# Ground-Truth Evidence Appendix

This appendix ties headline claims back to concrete evidence artifacts so results
can be defended without hand-waving.

## Outcome Provenance Table

```text
\toprule
\textbf{Track} & \textbf{Claim used in this doc} & \textbf{Evidence level} & \textbf{Primary source} \\
\midrule
Indoor & 8 / 11 checkpoints reached & Direct run result summary & \texttt{competition\_results.tex}, Section ``Indoor Run: 8 of 11 Checkpoints'' \\
Outdoor standard & One full completion; other runs partial (about 50\% overall) & Direct run-result summary + repeated notes & \texttt{competition\_results.tex}, Section ``Outdoor Standard Run'' \\
Marathon & Reached one checkpoint, then toppled & Direct run-result summary + operator narrative & \texttt{competition\_results.tex}, Section ``Marathon Attempt'' \\
Outdoor transition issue & Route-corridor stop and reroute loops happened in field logs & Direct log signature & \texttt{live\_outdoor\_ultra\_marathon\_story.tex}, appendix ``Representative Failure Logs'' \\
Waypoint handoff issue & Intermediate-waypoint handoff sometimes entered ALIGN-heavy behavior & Direct log signature & \texttt{live\_outdoor\_ultra\_marathon\_story.tex}, appendix ``Representative Failure Logs'' \\
\bottomrule
```

## Representative Field Log Signatures

The exact values below are from operator terminal logs and are preserved here
because they explain why specific runtime guardrails were added.

### A. Corridor-stop triggered reroute

```
[0416] route_corridor_stop dist=6.0m threshold=6.0m ticks=3
[routing] re-routing from current pose due to route_corridor_stop...
[routing] building leg 2/47 reroute via OSM for 1 checkpoint(s)...
```

Interpretation: route-corridor enforcement was active and forced a reroute when
deviation persisted.

### B. Intermediate waypoint handoff entered ALIGN mode

```
[0290] intermediate waypoint reached at (37.869384, -122.259327)
[0293] DONE=1/47 LEG=2/47 WP=3/8 dist=13.7m ... mode=ALIGN ...
```

Interpretation: after a local waypoint completion, target transition entered an
alignment-dominant phase. This is exactly the handoff region that remained fragile.

### C. ALIGN saturation signature

```
[0318] ... mode=ALIGN lin=+0.000 ang=-0.279 ...
[0320] ... mode=ALIGN lin=+0.000 ang=-0.280 ...
[0322] ... mode=ALIGN lin=+0.000 ang=-0.280 ...
```

Interpretation: repeated high angular command with zero linear motion indicates
turn-priority lock behavior under transition stress.

### D. Post-reroute recovery to forward motion

```
[0450] ... mode=logonav_ lin=+0.160 ang=-0.277 ...
[0453] ... mode=logonav_ lin=+0.201 ang=-0.239 ...
[0460] ... mode=logonav_ lin=+0.190 ang=-0.232 ...
```

Interpretation: after the ALIGN-heavy period, the controller recovered forward
motion, but still with substantial steering load.

## How To Use This Evidence Correctly

- Treat indoor 8/11 as proof of a working indoor backbone, not proof that
all corridor ambiguity is solved.
- Treat outdoor full completion as proof the stack can work, not proof it is
robustly repeatable yet.
- Treat the marathon failure as a transition-and-platform stability failure,
not as proof that every added safety layer was wrong.
- When debating architecture choices, use failure signatures above and map
them back to specific guards in `live_outdoor_runtime.py`.

# Math Companion File

This document now has a matching intuition-first math teaching file:

`erc3_mathematics_guide.txt`

That file is written for someone who does not yet think fluently in robotics math.
It explains the core quantitative ideas in this project from intuition upward and
ends with quiz questions so the reader can test whether the ideas actually stuck.

# Conclusion

The project ended with one strong completed system and one partially successful but
still evolving system.

The strong completed system was the indoor corridor stack:

- corridor localization,
- temporal smoothing,
- graph progression,
- MBRA local control,
- 8 of 11 checkpoints reached.

The outdoor system proved it could work, but also showed exactly where it still
breaks:

- one run reached all checkpoints,
- several others only partially succeeded,
- marathon exposed transition instability and physical tipping risk clearly.

The biggest success was not just that the rover moved. The biggest success was that
the architecture became understandable. By the end, the roles of localization,
planning, controller behavior, safety, recovery, and mission logic were much
clearer than they had been at the start.

# Related Documents

This master document is part of a larger documentation system. The companion documents
add depth in specific areas:

- **`CLAUDE.md`**: The ground-truth quick-reference system guide. If this
master document and CLAUDE.md disagree on a parameter value, check the code — one of
them is stale.
- **[[live_indoor_runtime_story]]**: The full indoor narrative — from
blank slate through MBRA debugging to working system. Deeper indoor history than this
document provides.
- **[[live_outdoor_ultra_marathon_story]]**: The full outdoor/marathon
narrative — three implementation stages, every safety feature, what was explored but
not shipped.
- **[[outdoor_perception_review]]**: The depth and semantic investigation
story — what was tried, what failed, what shipped, and what the probe results actually
showed.
- **[[semantic_segmentation_research_review]]**: The semantic research
from first offline probe through production estimator — every recommendation and how
each was addressed.
- **[[docs/our_mbra_discoveries]]**: The MBRA bug post-mortem — 7
discoveries that shaped both indoor and outdoor systems (vel_past, no-reverse, sole
motion authority).
- **[[docs/march19]]**: The mid-March retrospective — the longest single
narrative document (1158 lines), covering both tracks reaching working state.
- **[[docs/current_codebase_deep_read]]**: Code analysis snapshot — what
every file does, with indoor and outdoor coverage.
- **[[docs/discoveries]]**: Literature review — ViNT, PlaceNav, MBRA papers
fact-checked against what was actually built.
- **[[live_outdoor_runtime_explained]]**: Plain-language outdoor runtime
walkthrough with safety layer details.
