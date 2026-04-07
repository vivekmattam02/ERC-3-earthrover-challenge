# Indoor Navigation Strategy for This Project - [0.3em] A clear plan for what we are solving, where we are now, and what we should do next

> Source: `docs/indoor_navigation_strategy.tex`
> Master Note: [[erc3_full_documentation]]

\fbox{\parbox{0.9\textwidth}{
**HISTORICAL DOCUMENT** \\[4pt]
Initial strategy note from March 2026. The overall direction was mostly right,
but the real implementation details evolved significantly once the live runtime,
checkpoint-step mode, and MBRA-specific constraints became clearer. \\[4pt]
\textit{For the current system, see: [[live_indoor_runtime_story]],
`CLAUDE.md`}
}}

# What This Document Is

This is not a paper and not a textbook. It is a decision document.

\begin{mdframed}
**Purpose:** explain, in plain technical language, what this project
should do for **indoor navigation**, what is already available in the
repo, what is missing, which approach we should prefer, which approaches we
should *not* trust by default, and how all pieces should be integrated.
\end{mdframed}

The goal is that after reading this document, you should be able to say:
- what problem we are actually solving,
- what architecture we currently believe is strongest,
- where MBRA fits,
- what backup options we should keep open,
- and what the next engineering step should be.

# The Problem We Are Actually Solving

The indoor problem is **not** "make outdoor GPS navigation work indoors."
That would be the wrong framing.

The indoor competition problem is:
- the robot must navigate inside NYU buildings such as 6MTC and 5MTC,
- the goals are **images**, not GPS coordinates,
- the robot still runs through the FrodoBots SDK over **4G**,
- latency and low effective FPS are still part of the problem,
- and success must be judged visually at the goal, not by coming within
a GPS radius.

\begin{mdframed}
**The real indoor task:** given the current camera stream and an ordered
sequence of checkpoint images, move the robot through the building safely and
reliably until each checkpoint image is reached.
\end{mdframed}

That means the system must answer four questions repeatedly:
1. Where am I?
1. Where should I go next?
1. How do I move there safely?
1. How do I know I actually reached it?

# Where We Are Right Now

The repo already contains useful pieces, but it does *not* yet contain a
finished indoor stack.

## What is already strong

- **Robot I/O path:** the EarthRover SDK and Python interface are already
present and useful.
- **MBRA codebase:** there is a real MBRA/LogoNav research repo in
`mbra_repo/`.
- **Indoor competition framing:** the NYU indoor track is well described in
the repo docs.
- **Safety ideas:** runtime depth-based safety is available as
concepts and partial code.

## What is not yet strong

- no clean indoor deployment script using MBRA on EarthRover,
- no topological graph builder,
- no indoor visual localizer ready for competition use,
- no formal goal-reached visual verifier,
- and no fully integrated planner-controller-safety loop for indoor use.

\begin{mdframed}
**Current status in one sentence:** we have useful building blocks and a
good direction, but not yet a reliable end-to-end indoor system.
\end{mdframed}

# What We Want From the Final System

The final system should be judged by reliability first, not elegance.

## What matters most

- **Reliability:** works repeatedly in indoor corridors and junctions.
- **Interpretability:** when it fails, we know why.
- **Recoverability:** can relocalize and continue after mistakes.
- **Safety:** does not clip walls, chairs, doors, or pedestrians.
- **Competition realism:** tolerates 4G latency and modest frame rate.

## What does not matter enough to justify risk

- a fashionable end-to-end system with no debuggable structure,
- perfect metric geometry if it is fragile,
- or using one model for everything just because it looks simpler on paper.

# Main Decision: The Best Baseline Architecture

Our strongest default architecture for indoor navigation is:

\begin{mdframed}
\textbf{Topological graph + visual localization + graph planning + MBRA local
controller + depth safety override.}
\end{mdframed}

## Why this is the best default

This choice matches the indoor problem better than GPS-style navigation or pure
visual odometry:
- the competition uses **image goals**,
- buildings are naturally suited to **place-based navigation**,
- a graph is easier to debug than a pure learned policy,
- MBRA is naturally useful for **short-horizon image-goal control**,
- and a separate safety layer keeps the system from trusting the learned
controller too much.

## What each layer does

```text
\toprule
\textbf{Layer} & \textbf{Role} \\
\midrule
\textbf{Perception} & Read front-camera frames and maintain a short context buffer. \\
\textbf{Localization} & Match current view to a topological map of reference images. \\
\textbf{Planning} & Find a path in that graph from current node to goal node. \\
\textbf{Subgoal selection} & Choose the next nearby node image along the path. \\
\textbf{MBRA} & Turn the recent camera history and the subgoal image into short-horizon controls. \\
\textbf{Safety} & Override or clip unsafe commands using depth/clearance checks. \\
\textbf{Goal checking} & Decide when the current checkpoint image has actually been reached. \\
\bottomrule
```

# Where MBRA Fits

This is the most important conceptual point in the whole indoor plan.

\begin{mdframed}
\textbf{MBRA should be treated as a local image-goal controller, not as the
entire indoor navigation system.}
\end{mdframed}

## What MBRA is good at

MBRA is a short-horizon policy. In practice that means it is best at:
- taking the robot from its current view to a **nearby subgoal image**,
- handling local turns and immediate motion choices,
- and converting image context into motion in a direct way.

## What MBRA is not good enough to do alone

MBRA should not be expected to:
- solve long multi-turn routes from one faraway final goal image,
- replace localization,
- replace planning,
- or decide globally where in the building the robot currently is.

## The right use of MBRA indoors

The right use is:
1. localize current place in the graph,
1. find the next node on the planned path,
1. feed that node image as the current goal image to MBRA,
1. let MBRA produce local controls toward that nearby target,
1. and repeat.

\begin{mdframed}
**Short version:** the graph tells the robot *which nearby place to go to next*; MBRA tells the robot *how to move there right now*.
\end{mdframed}

# Why Not These Other Approaches by Default

## Why not A* on a metric occupancy map?

In principle this is a valid robotics approach. In practice, for this project,
it is not the strongest default because:
- we do not currently have a trusted metric indoor map,
- we do not currently have trusted indoor metric localization,
- we do not have LiDAR or a reliable indoor depth stack already operating
as a mapping system,
- and if the metric map or the pose is wrong, A* gives a beautiful plan
in the wrong place.

**Conclusion:** metric planning is not forbidden, but it should not be our
first competition baseline.

## Why not pure PID toward global coordinates?

PID is not the issue. The issue is the coordinate frame.

PID works only after a good target in the robot frame is already known. Indoors:
- there is no GPS,
- metric localization is uncertain,
- and coordinate drift turns a simple controller into a confidently wrong controller.

**Conclusion:** PID may still exist inside low-level control, but it cannot
be the main indoor navigation method.

## Why not visual odometry as the planner?

Visual odometry and planning solve different problems.

Visual odometry estimates short-term motion. It does not decide where to go.
It also drifts, especially under:
- repeated corridors,
- texture-poor walls,
- motion blur,
- frame drops,
- and low effective FPS.

**Conclusion:** visual odometry may help as a support signal, but it should
not be the global planning layer.

## Why not ORB-SLAM3 as the main backbone?

ORB-SLAM3 is real, serious, and respected. The problem is not that it is bad.
The problem is that our setting is harsh for it:
- likely monocular only,
- no clean stereo depth setup,
- indoor repetition,
- and 4G-driven low frame rate.

That means ORB-SLAM3 is better treated as:
- an optional support module,
- or an offline mapping aid,
- not the default competition backbone.

# Keep the Options Open: What We Should Test in Parallel

We should be opinionated, but not narrow-minded.

\begin{mdframed}
**Default plan:** topological graph + MBRA.\\
**Open option:** ORB-SLAM3 or VO can still be tested as a supporting module.\\
**Open option:** metric methods can be revisited if they prove unusually stable on real NYU data.
\end{mdframed}

The correct attitude is:
- do not lock the entire project to one fragile assumption,
- but also do not chase every robotics idea at once.

So our default should be the graph-based approach, while still testing:
- a minimal ORB-SLAM3 support experiment,
- a visual-odometry support experiment,
- and multiple visual-localization methods for node retrieval.

# How the Whole Indoor System Should Integrate

This is the integration story from end to end.

## Stage 1: Data collection

We teleoperate the robot through the indoor routes and save:
- front camera frames,
- timestamps,
- route order,
- and route connectivity information.

This gives us the raw material for a topological graph.

## Stage 2: Topological graph construction

From the collected runs we build:
- nodes = representative images,
- edges = valid transitions between nearby places,
- optional labels = corridor, turn, door, lobby, intersection.

This graph becomes the global indoor navigation structure.

## Stage 3: Runtime localization

At runtime:
1. get the current front camera frame,
1. compare it against graph node images,
1. retrieve likely matches with a visual descriptor,
1. optionally verify top candidates with local feature geometry.

Now we have an estimate of where in the graph the robot is.

## Stage 4: Runtime planning

Given:
- current node,
- target checkpoint node,

the planner computes a graph path.

Then it selects the **next nearby node** as the current subgoal.

## Stage 5: MBRA local control

MBRA receives:
- recent image context,
- the current subgoal image,
- delay estimate,
- recent command history,
- and fixed inference-time constants such as robot size.

MBRA outputs short-horizon controls.

## Stage 6: Safety override

Before commands are sent:
- estimate depth or clearance,
- check whether the proposed motion is safe,
- clip, redirect, or slow if necessary.

## Stage 7: Goal reached verification

When the robot appears close to the target image:
- use global similarity to propose a match,
- use geometric feature verification to confirm it,
- require consistency across multiple frames if needed.

Then move to the next checkpoint.

# What We Should Build First

We should not try to build the entire competition system at once.

## Milestone 1: MBRA local goal-following proof of concept

Build the smallest useful thing:
- use the existing EarthRover SDK path,
- load MBRA,
- manually choose one nearby goal image,
- run MBRA to move toward that one local image goal,
- add a minimal safety override.

**Why first?** It proves that MBRA can actually drive the robot locally on
our platform.

## Milestone 2: Build the topological graph tooling

Next:
- record indoor runs,
- choose node images,
- define edges,
- and build a simple graph format.

## Milestone 3: Visual localization

Implement node retrieval from current frame to graph nodes.

## Milestone 4: Full graph + MBRA loop

Combine localization, graph path, subgoal switching, MBRA, and safety.

## Milestone 5: Goal verification and competition hardening

Only after the full loop works should we focus on:
- strict checkpoint verification,
- relocalization after failure,
- and route-level robustness under real building traffic.

# What the Next Thing Should Be

\begin{mdframed}
**Immediate next step:** build an MBRA-based local indoor prototype for
EarthRover that follows one manually selected nearby goal image.
\end{mdframed}

Not a full map. Not final competition logic. Not retraining first.

The next step should answer one simple question:
*Can MBRA, on this robot and this SDK path, reliably drive toward a nearby indoor goal image?*

If the answer is yes, the graph layer is worth building. If the answer is no,
we revise early instead of wasting weeks.

# What I Would Tell Ourselves Right Now

\begin{mdframed}
**We should not try to be clever before we are reliable.**
\end{mdframed}

That means:
- choose the architecture that matches the competition problem,
- keep alternatives open but secondary,
- test MBRA in the exact role it is best suited for,
- and build the system in a layered way so each piece can be validated.

The strongest current plan is still:
**topological graph + visual localization + graph planning + MBRA local control + safety override**

This is the plan that is most consistent with the task, the repo contents, and
the requirement to use methods that are actually testable and trustworthy.

# Final Summary

- The indoor task is image-goal navigation inside NYU buildings.
- The repo does not yet contain a complete indoor stack.
- MBRA is useful, but as a **local image-goal controller**, not as the entire system.
- The best default architecture is graph-based, not GPS-based and not VO-only.
- ORB-SLAM3 and metric methods should remain optional experiments, not the default backbone.
- The next engineering step is an **MBRA local prototype** on EarthRover with one nearby goal image and a safety layer.

\begin{mdframed}
**If we stay disciplined, the project becomes manageable:**
first prove MBRA locally, then add the graph, then add localization, then harden
for competition.
\end{mdframed}
