# ERC-3 Competition Results - What Actually Happened on Race Day

> Source: `competition_results.tex`
> Master Note: [[erc3_full_documentation]]

# Why This Document Exists

This is the race-day document. It is not meant to be a glossy victory report and
it is not meant to be a blame document either. The useful version of competition
results is the honest one: what system we ran, what it managed to do, what broke
under pressure, and what those failures taught us.

The project competed in three practical modes:

- **Indoor known-corridor navigation**, where checkpoints were tied to a
fixed corridor dataset and the system could exploit repeated structure.
- **Outdoor GPS mission navigation**, where the rover had to move through
a live environment using mission checkpoints and waypoint routing.
- **Marathon endurance mode**, which stressed not just navigation quality
but runtime stability, platform stability, and operational robustness over a
longer run.

# Competition Format

The indoor rules are summarized in [[docs/nyu_indoor_track]]. In practice
the indoor track behaved like a known-corridor checkpoint problem: the route
structure was fixed, the corridor had already been recorded, and the goal was to
reach a sequence of image-goal checkpoints without losing the path structure.

The outdoor track was structurally different. The rover received GPS checkpoints
through the SDK, had to move through a live outdoor environment, and had to do so
under real-world safety constraints. That immediately made the problem more than a
controller question. It became a runtime question: routing, waypoint transitions,
telemetry quality, safety gating, and recovery all mattered.

The marathon mode stretched that same outdoor system over a longer and harsher run.
That changed the engineering objective. The goal was no longer just "reach the
next checkpoint." The goal became: do that while staying physically stable,
avoiding brittle recovery behavior, and keeping the operator in a position to
intervene early.

# Indoor Run: 8 of 11 Checkpoints

The indoor system we took into competition was:

- CosPlace-based visual place recognition in `src/corridor_localizer.py`,
- temporal stabilization in `src/temporal_localization.py`,
- topological graph progression in `src/graph_planner.py`,
- MBRA as the sole motion authority through `src/mbra_controller.py`,
- and depth veto logic in `live_indoor_runtime.py`.

The run command was:

```
python live_indoor_runtime.py \
  --checkpoint-steps 45 480 761 821 1094 1208 1345 1430 1544 1638 1764 \
  --auto-advance-checkpoints --send-control --controller mbra --depth-safety
```

The key runtime defaults for that MBRA-based indoor run were:

- `tick_hz = 3.0`
- `max_subgoal_hops = 8`
- `depth_stop_m = 0.25`
- `min_confidence_to_advance = 0.55` inside the graph planner
- skip-past-checkpoint handling when localization confidence was high enough

## What Went Right

What worked indoors was the structural backbone. Localization was stable enough to
make the corridor legible. Temporal smoothing prevented one-frame VPR flicker from
constantly rewriting the current pose estimate. The graph planner gave MBRA a
cleaner target than "go to some vaguely similar image." That was the real shift:
once the problem was reframed as exact checkpoint-step progression in a known
corridor, MBRA became much easier to use honestly.

The best evidence for that is simple: the rover reached **8 out of 11**
checkpoints. That is not perfect, but it is far beyond a toy demo. It means the
indoor system had a real working backbone.

## What Went Wrong

The missed indoor checkpoints were not random. They live exactly where the
architecture is weakest:

- visually similar corridor segments can still create VPR ambiguity,
- depth vetoes can be conservative enough to interrupt otherwise viable motion,
- forward-only graph behavior can create awkward transition cases when the
rover has already drifted past a target step,
- and stale controller context can keep MBRA trying to solve the wrong local
problem for a few ticks too long.

## What The Indoor Result Means

The indoor result says the project's strongest completed stack is still the
known-corridor system. The key lesson was not just "MBRA worked." The more
important lesson was: MBRA worked when it was used in the role it was good at,
namely short-horizon visual control on top of a stronger localization and graph
progression backbone.

# Outdoor Standard Run: One Full Success, Roughly Half Overall

The outdoor system was more operationally complex:

- mission checkpoints came through the SDK,
- `live_outdoor_runtime.py` handled mission control,
- LogoNav in `src/outdoor_logonav_controller.py` acted as the learned
local motion policy,
- optional OSM routing expanded a mission checkpoint into intermediate local
waypoints,
- and the runtime layered safety and recovery logic around the controller.

The standard command path was:

```
python live_outdoor_runtime.py --mission --send-control --traversability
```

The controller default in the outdoor runtime is `logonav`, so the plain
command above already selects the learned outdoor controller.

## What Worked

The most important outdoor fact is that the stack was real enough to finish a run:
**one run reached all checkpoints**. That matters because outdoor success is
not just a matter of one network prediction looking reasonable on one frame. It
means the entire stack held together long enough for checkpoint management,
controller output, routing, and safety logic to coexist.

## Why The Overall Rate Was Still Only About Half

The weak point outdoors was not the existence of a controller. It was what
happened around waypoint transitions, target interpretation, and recovery under
noise.

The recurring failure mode was waypoint transition instability:

- intermediate routed waypoints were sometimes handed off too loosely,
- the next target could be structurally behind the rover after a reroute,
- alignment logic could spin too aggressively when applied to the wrong kind
of target,
- and route corridor logic could overreact in ways that reset useful forward
progress.

## What The Outdoor Result Means

The outdoor result says the controller choice was not the whole problem. LogoNav
was good enough to be the right starting point, but the runtime around it was the
real engineering burden. Outdoor autonomy here was mostly a systems problem:
mission semantics, reroute semantics, target semantics, and safe behavior under
partial uncertainty.

# Marathon Attempt: One Checkpoint, Then The Rover Toppled

The marathon run used the outdoor runtime in its most defensive configuration:

```
python live_outdoor_runtime.py \
  --mission --send-control --controller logonav --osm-route --ultra-marathon
```

The intended logic was sound. Marathon mode tightened recovery behavior, added
IMU safety, health gating, leg pauses, stricter speed caps, route corridor
guarding, and stronger operator-facing visibility into what the runtime was doing.

But the marathon failure made an uncomfortable point very clearly: software safety
layers do not automatically produce physical stability.

## What Happened

The rover reached one checkpoint. After that, the system entered a bad transition
regime. The next-target interpretation and repeated alignment behavior did not
settle quickly enough. The rover kept turning aggressively on real terrain, and
the platform physically toppled.

This is why the right explanation is **not** "the software hit a random
bug and fell over." The better explanation is:

- checkpoint progress happened,
- the runtime did not stabilize fast enough after that transition,
- turning behavior remained too persistent,
- and the physical platform was sensitive enough that repeated turning on
uneven ground became a real mechanical problem.

## What IMU Safety Would Have Said

The IMU layer in the marathon stack was designed to notice dangerous tilt and
gyro-rate behavior. That is still useful, but it is a last-resort safety layer.
It can stop motion after the platform becomes unsafe. It cannot undo the fact that
an unstable controller-to-platform interaction has already started.

## What We Would Change

The marathon result points to three changes immediately:

- lower transition speed after checkpoint or waypoint completion,
- widen the turning radius and reduce aggressive in-place alignment,
- and improve physical stability with lower center of gravity or ballast.

# Summary Table

```text
\toprule
\textbf{Track} & \textbf{Target} & \textbf{Achieved} & \textbf{Key Limitation} \\
\midrule
Indoor & 11 checkpoints & 8 checkpoints & Ambiguity / transition / recovery edge cases \\
Outdoor & Full checkpoint mission & 1 full success, others partial & Waypoint transition instability \\
Marathon & Endurance run & 1 checkpoint, then toppled & Post-transition stability and platform dynamics \\
\bottomrule
```

# What We Learned

- Waypoint and checkpoint transitions are the weakest point in the current
outdoor system.
- Physical stability matters as much as software intelligence once the robot
leaves a clean corridor and enters a long live run.
- The safety stack was still valuable. The project did not fail because it had
no safety thinking; it failed because safety and stable behavior are not the same
thing.
- The documentation set became one of the strongest deliverables from the
project because it captures not just the final architecture, but the route that
led to it.

# Interview-Ready Answers

## How would you explain these results quickly?

Indoors, we had a genuine working stack and reached 8 of 11 checkpoints. Outdoors,
we proved the system could complete a full run once, but reliability was still
limited by transition behavior around waypoints. Marathon mode exposed the last big
gap very clearly: stable navigation across long live runs still depends on better
post-transition behavior and better physical stability.

## What would you do differently?

I would reduce the number of uncontrolled transition cases much earlier, especially
for outdoor waypoint handoff. I would also test marathon-style stability on the
physical platform sooner instead of assuming that a more cautious runtime alone was
enough.

## What are you most proud of?

The most defensible result is that the project became understandable. By the end,
the indoor stack had a clear backbone, the outdoor stack had a clear runtime
envelope, and the failure modes were explicit enough to be debugged instead of
being mystified.

## What is the honest weakness?

The honest weakness is that the outdoor system still had too many ways to become
confused during target transitions. Marathon mode made that impossible to ignore.
