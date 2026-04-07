# ERC Indoor Progress Note: March 19 - What We Built, Why We Built It, and Where We Are Now

> Source: `docs/march19.tex`
> Master Note: [[erc3_full_documentation]]

# What This Document Is

This document explains, in a simple but technically honest way, what has been
built so far for the ERC indoor project inside this repository.  The goal is
not just to list files.  The goal is to explain:

- what each part does,
- why we chose it,
- what alternatives existed,
- why those alternatives were not chosen as the main baseline,
- and where the project stands right now.

The intended result is that after reading this, a teammate should understand the
current project structure clearly enough to contribute to it and defend the main
engineering choices.

# The Actual Problem We Are Solving

We are building an **indoor navigation system** for the
**EarthRover Challenge** in a \textbf{known and repeated corridor
environment}.

That last point matters a lot.

This is **not** the same as general indoor navigation in an unknown
building.  We are not trying to make a robot that can enter any building in the
world and navigate from scratch.  We are solving a more specific and more
practical problem:

- the corridor or building family is known,
- we can collect data in advance,
- we can test in the same environment repeatedly,
- and the competition checkpoints are image-based, not GPS-based.

This changes the whole strategy.  It means we are allowed to build a
corridor-specific baseline and tune it carefully for that environment.

# Why The Final Direction Changed

At the start, many possible directions existed:

- metric SLAM,
- pure visual odometry,
- a large end-to-end controller,
- CityWalker-style reuse,
- MBRA as the whole stack,
- or a graph-based visual localization system.

After looking at the repo and the actual competition setting, the most sensible
backbone became:

\fbox{\parbox{0.9\textwidth}{
**known-corridor visual localization + graph planning + local controller**
}}

Why this direction won:

- goals are image-defined,
- the corridor is repeated and known,
- monocular RGB is available,
- metric localization is weak,
- latency and frame rate are poor,
- and we needed something debuggable and reliable.

This does *not* mean the other ideas are useless.  It means they are not
the cleanest **main backbone** for this exact problem.

# The Most Important Decision

The most important decision we made is:

> Use a **corridor-specific perception/planning backbone** first, then plug
> in a controller on top of it.

That means:

- first answer "where am I?",
- then answer "where should I go next?",
- then answer "how do I move there safely?".

We did **not** start from the controller and hope it solves everything.

# What We Built

## 1. A Clean Data Pipeline From The Teleoperation H5 File

We found that the teleoperation recording file
`data/corrider.h5` is extremely useful.

It contains:

- front camera frames,
- control commands,
- telemetry,
- orientation,
- gyros,
- accelerometers,
- magnetometer data,
- and RPM samples.

This was better than expected.  It means we do not need to start by collecting
everything again from zero.

To make this usable, we built:

- `tools/extract_h5_dataset.py`

What it does:

- extracts front frames into an ordered image folder,
- exports CSV metadata,
- builds a baseline-friendly `data_info.json`,
- and aligns the different streams using relative time.

Why this helps:

- the localization system wants images,
- the planning system wants ordered steps,
- and the control / state-estimation side needs actions and motion
signals.

Why we chose this over ad hoc notebooks:

- it is reusable,
- it is shareable,
- and it reduces the chance of hidden one-off mistakes.

## 2. A Real Baseline Database Builder

We turned the earlier notebook-style corridor work into a usable script:

- `baseline.py`

What it does:

- extracts CosPlace descriptors,
- builds a place graph,
- builds a navigation/action graph when `data_info.json` is
available,
- and supports query-time localization.

We used it to build:

- `data/corrider_db/`

This contains:

- `descriptors.npz`
- `config.json`
- `place_graph.json`
- `navigation_graph.json`

Why this matters:

- this is the corridor memory of the system,
- without it, we do not have a known-environment localization backbone.

Why we chose CosPlace + graph:

- it matches the known-corridor setting,
- it is explainable,
- it can be debugged visually,
- and it is already close to the older project code direction.

Alternative option:

- use ORB-SLAM3 or direct SLAM as the main backbone

Why we did not choose that as the baseline:

- the problem is image-goal and corridor-specific, not metric-map-first,
- repeated corridor appearance is risky for monocular SLAM,
- and the graph baseline better matches what we already had.

## 3. Temporal Localization

Single-frame retrieval is not enough in a repeated corridor.  Even when the
database is good, one frame can still be ambiguous.

So we built:

- `src/temporal_localization.py`

What it does:

- takes top retrieval candidates,
- penalizes impossible jumps,
- penalizes unreasonable backwards motion,
- optionally uses heading consistency,
- and resolves ambiguity conservatively.

Why it helps:

- a robot does not teleport,
- so localization should respect continuity,
- and repeated corridor segments need temporal context.

We then built:

- `tools/evaluate_temporal_localization.py`

This let us test temporal localization on held-out frames.

The most important outcome:

- overall exact match was strong,
- but more importantly, moving-frame performance was extremely strong,
- and the worst-looking error cluster turned out to be mostly a repeated
stationary segment, not true moving localization failure.

This is a big result because it means the perception/planning backbone is
actually stronger than the naive error count first suggested.

## 4. Runtime Corridor Localizer

The database and temporal localizer were not enough by themselves because they
were still pieces, not a runtime API.

So we built:

- `src/corridor_localizer.py`

What it does:

- loads the built DB once,
- encodes a new frame,
- retrieves candidates,
- applies temporal localization,
- and returns a planning-ready result:
node, step, confidence, and candidate list.

Why this helps:

- planning should not need to know how descriptor extraction works,
- it should just ask: "where are we right now?".

## 5. Runtime Graph Planner

Once localization exists, the next question is:

> Given the current node and a target checkpoint, what nearby subgoal should the
> controller follow right now?

So we built:

- `src/graph_planner.py`

What it does:

- loads the runtime graph,
- resolves targets by node, step, or image name,
- computes shortest path,
- chooses a nearby subgoal node,
- and supports checkpoint progression.

Why it helps:

- the controller should not chase the final goal from far away,
- it should chase a nearby subgoal on a valid corridor path.

Alternative option:

- let a learned controller implicitly handle the whole route

Why we did not choose that as the baseline:

- it would be much harder to debug,
- and the graph planner already gives us route structure cleanly.

## 6. Controller-Facing Runtime Coordinator

At that point, we had localization and planning, but control still had to
manually call several pieces.

So we built:

- `src/navigation_runtime.py`

What it does:

- combines localization and planning,
- returns a controller-facing bundle,
- and can optionally load the subgoal image for the controller.

Why this helps:

- control should receive one clean package:
current state, target, subgoal, confidence, and subgoal image.

## 7. A First Local Controller Baseline

We still needed a controller that can turn a nearby subgoal into a motion
command.

So we built:

- `src/local_controller.py`

What it does:

- takes the controller bundle,
- uses current step, subgoal step, current orientation, subgoal
orientation, and confidence,
- and outputs a simple baseline:
`linear`, `angular`, and a reason.

Why this helps:

- it gives us a real executable baseline before MBRA integration,
- it is simple enough to debug,
- and it gives the control side a starting point immediately.

Why this is not the final answer:

- it is heuristic,
- not learned,
- and not yet safety-aware.

But it is still valuable because it proves the runtime pipeline can already
produce commands.

## 8. A Conservative Live Runner

Finally, we connected the pieces into a live loop:

- `live_indoor_runtime.py`

What it does:

- connects to the SDK,
- gets camera and heading,
- runs localization,
- runs planning,
- runs the local controller,
- and optionally sends control to the robot.

Important safety choice:

- it defaults to **dry-run**,
- and only sends real control if explicitly requested.

Why this matters:

- we should not accidentally move the robot during first tests,
- especially when the runtime stack is still being validated.

## 9. A Lightweight Motion-Prior Layer

One thing that was missing in the first version of the stack was a proper
middle layer between raw robot telemetry and the controller.

So we added:

- `src/sensor_state.py`

What it does:

- smooths heading from SDK orientation,
- estimates turn rate from gyro $z$,
- summarizes RPMs into a weak motion hint,
- and passes this filtered motion state into localization and control.

Why this matters:

- a live robot should not use raw noisy heading directly if we can avoid
it,
- gyro information helps tell whether the robot is already turning,
- and this makes the controller less twitchy.

Why we chose this instead of jumping straight to an EKF:

- the visual place-localization backbone is already the main state
estimate,
- we do not yet have a trustworthy metric motion model for the whole
indoor problem,
- and a small filter gives us most of the immediate benefit without
pretending we solved full state estimation.

So the correct interpretation is:

> We are using a **lightweight motion prior**, not replacing the visual
> backbone with a Kalman-filtered metric localization system.

# What About IMU, Orientation, and RPMs?

One correction that became important during this work:

> The project should not be described as vision-only.

The correct statement is:

> **vision-primary with IMU / heading support**

What that means:

- camera frames are still the main localization input,
- but orientation and IMU should help temporal filtering, recovery, and
controller smoothing,
- and RPM data should be used only if it proves trustworthy on the real
robot.

Why we did not make IMU the main baseline:

- IMU alone drifts,
- the task is image-goal and corridor-specific,
- and the strongest existing code already centered on visual place-based
reasoning.

So the right role of IMU is:

- support signal,
- not sole localization backbone.

The practical version of that sentence is:

- filtered heading helps the localizer,
- gyro turn rate helps the controller,
- RPMs can give weak motion hints,
- but visual localization still decides where we are in the corridor.

# What About MBRA?

This is one of the most important conceptual clarifications.

The short answer is:

> `mbra_repo` is **not** the current deployed indoor backbone.

What it is:

- a research reference repo,
- containing MBRA / LogoNav-related model and deployment code,
- useful as a candidate controller direction.

What it is **not** currently:

- not the active perception/planning backbone,
- not fully integrated into our corridor runtime,
- not already a clean indoor drop-in controller in this project.

Why we did not force MBRA into the stack immediately:

- the corridor perception/planning side needed to be solid first,
- and the repo itself does not automatically provide a finished indoor
controller plug-in for our exact setup.

So the current status of MBRA is:

- important candidate for the controller side,
- not yet the active controller in the baseline loop.

# Localization Explained Very Clearly

This is the most important part to understand, because this is the part that is
currently working well.

## What localization means here

Localization means:

> Given the current live camera frame, estimate which stored corridor frame or
> graph node the robot is closest to.

We are **not** doing metric $(x, y)$ localization on a floorplan.
We are doing **place localization in a known visual corridor graph**.

## What information goes in

The localization pipeline takes:

- a live RGB frame from the robot,
- the built database of reference corridor images,
- the descriptor vectors for those reference images,
- optional heading information,
- and temporal history from previous frames.

## What information comes out

The localization pipeline outputs:

- the best-matching graph node,
- the corresponding corridor step,
- a confidence score,
- and a shortlist of alternative candidates.

## The actual logic

The logic has three layers.

\paragraph{Layer 1: image embedding.}

Each reference image is passed through CosPlace and converted into a feature
vector of dimension $512$.

Call the reference descriptors:
\[
d_1, d_2, \dots, d_N
\]

and the live query descriptor:
\[
q
\]

The first retrieval step asks:

> Which stored descriptor $d_i$ is most similar to $q$?

That gives top candidate nodes.

\paragraph{Layer 2: retrieval score.}

At the simplest level, retrieval is just nearest-neighbor search in descriptor
space.

Conceptually, we rank candidates by a similarity score such as:
\[
\mathrm{sim}(q, d_i)
\]

Higher similarity means the current live view looks more like reference frame
$i$.

\paragraph{Layer 3: temporal filtering.}

This is the part that made the system actually usable.

A single frame can be ambiguous in a corridor.  So we do not trust only the raw
best match.  We also penalize unreasonable jumps.

Conceptually, the score becomes:
\[
\mathrm{score}(i) =
\mathrm{sim}(q, d_i)
- \lambda_{\text{jump}} \cdot \mathrm{jump_cost}
- \lambda_{\text{back}} \cdot \mathrm{backward_cost}
- \lambda_{\text{heading}} \cdot \mathrm{heading_cost}
\]

This is not meant as theory for theory's sake.  It is exactly the practical
idea implemented in `src/temporal_localization.py`.

In plain terms:

- if a candidate would require the robot to teleport many steps, punish it,
- if a candidate implies weird backward motion, punish it,
- if a candidate disagrees strongly with heading, punish it,
- if two candidates are too close in score, prefer stability over jumping.

## Why this worked well

Localization worked well for us because several things lined up correctly:

- the corridor is known and repeated,
- the database was built from the same environment family,
- CosPlace is good at place-level visual retrieval,
- temporal filtering prevents silly jumps,
- and the graph/node representation keeps the problem simple.

This is why the system can look at the live frame and say:

> "This is probably step 313, not step 850 and not step 40."

## What exactly we proved

We already proved the following:

- the teleop H5 can be converted into a clean localization dataset,
- the corridor database can be built,
- the localizer retrieves the correct corridor neighborhood,
- temporal filtering makes it stable,
- and in live dry-run testing, localization can lock to the correct region
when the robot is placed in the corridor.

So the current strongest statement is:

> The **place-localization backbone is real and useful**.  This is no longer
> the vague part of the project.

# Planning Explained Very Clearly

Now that localization is working, the next question is:

> Once we know where we are, how do we decide where to go next?

## What planning means here

Planning here does **not** mean occupancy-grid A* on a metric map.

Planning here means:

> Take the current localized graph node and the desired target node, then compute
> a path through the corridor graph and choose a nearby subgoal.

## What goes into planning

The planner needs:

- current node from localization,
- current step,
- target step or target node,
- graph connectivity from `navigation_graph.json`.

## What comes out of planning

The planner outputs:

- a path from current node to target node,
- a path in step numbers,
- a subgoal node a few hops ahead,
- and the corresponding subgoal image.

## Why we need a subgoal

We do not want the controller to chase frame $400$ directly from frame $313$ as
one giant jump.  That is too much.

Instead, if the path is:
\[
313 \rightarrow 314 \rightarrow 315 \rightarrow 316 \rightarrow \dots \rightarrow 400
\]

the planner picks a nearby subgoal such as step $316$.

So planning is doing:

> "Do not think about the whole route at once.  Think about the next small piece
> of the route."

## What is right and wrong in planning right now

What is already right:

- graph construction works,
- shortest-path planning works,
- subgoal selection works,
- and checkpoint-style planning structure exists.

What is still weak:

- the graph is mainly one-way because it follows teleop order,
- planner recovery when there is no valid path is still basic,
- and control handoff is stronger than the controller that receives it.

## What planning needs next

The next planning-side work is not inventing a new planner from scratch.

It is:

- making target selection cleaner,
- handling no-path cases more gracefully,
- deciding whether we want bidirectional corridor edges,
- and integrating checkpoint progression more carefully.

# What Needs To Happen Next

Now that localization is the strongest part, the next weak part is the
**controller and runtime behavior**.

The order of work should be:

1. verify that manual teleoperation works cleanly,
1. improve the simple controller so it stops spinning in place,
1. keep localization and planning fixed while debugging control,
1. add safer runtime behavior for low confidence and no-path cases,
1. only after that, decide whether MBRA should replace the simple controller.

So the main project is no longer:

> "Can we localize at all?"

The main project is now:

> "Can we turn good localization and good subgoals into stable robot motion?"

# Why We Chose This Path Instead of Other Paths

## Why not pure SLAM as the main backbone?

- The task is image-goal based.
- The corridor is known, so corridor-specific place recognition is very
attractive.
- The existing graph localization work was already closer to the actual
problem.

## Why not a giant end-to-end policy first?

- It would be harder to debug.
- We already had enough structure to do better than that.
- Reliability matters more than elegance here.

## Why not MBRA as the entire stack from day one?

- MBRA is not the same thing as the full runtime system.
- The repo evidence suggested MBRA/LogoNav needed careful interpretation,
not blind reuse.
- The graph-localization side was already much more concrete for this
corridor task.

## Why this current path makes sense

- It uses the known corridor advantage properly.
- It is modular.
- It is debuggable.
- It gives each teammate a clear layer to work on.
- It lets us plug in a better controller later without rebuilding the
whole stack.

# Where We Are Right Now

Right now, the project has:

- a usable dataset extraction pipeline,
- a built corridor database,
- a working temporal localizer,
- a working runtime localizer,
- a working runtime graph planner,
- a controller-facing coordinator,
- a first local-controller baseline,
- and a live loop script that can run in dry-run mode.

In one line:

\fbox{\parbox{0.92\textwidth}{
\texttt{camera / heading -> corridor localizer -> graph planner -> subgoal ->
simple local controller -> optional live loop}
}}

That means the perception/planning/runtime backbone is now real.

# What Is Still Missing

Important things are still missing:

- final safety integration inside the live loop,
- a controller that can follow subgoals without spinning in place,
- reliable teleoperation and manual recovery tooling,
- cleaner no-path and low-confidence runtime behavior,
- final checkpoint sequencing tests on the robot,
- and the decision of whether MBRA should replace or augment the simple
local controller.

So the project is **not finished**.  But it is no longer vague.  It now
has a real technical spine.

# What The Next Best Step Is

The next best step is:

1. make manual teleoperation reliable,
1. improve the simple controller so it follows nearby subgoals without
spinning,
1. validate short forward targets such as "current step $\rightarrow$
current step + 50",
1. then decide whether to keep upgrading the heuristic controller or begin
MBRA integration.

In practical terms:

> The project has moved from "can we localize and plan in the corridor?" to
> "how do we make the controller behave correctly on the real robot?"

# Bottom Line

What we built so far is not random glue code.  It is a deliberate indoor
navigation baseline for a known corridor.

The logic of the system is:

- build memory of the corridor,
- localize against that memory,
- plan a route through the graph,
- choose a nearby subgoal,
- and produce a local motion command.

The reason this is the right direction right now is simple:

- it matches the actual competition setting,
- it uses the strongest existing code in the repo,
- it is easier to debug than a fully end-to-end approach,
- and it keeps the controller layer open for improvement later.

So if you want the cleanest summary:

> We have already built the corridor-specific indoor perception/planning backbone.
> The project is now centered on validating the live loop and deciding how much
> the controller should remain simple versus how much MBRA should be integrated.

# Related Documents

- [[erc3_full_documentation]] --- single master guide for the complete project story and current architecture.
- [[live_indoor_runtime_story]] --- indoor evolution, MBRA integration, and checkpoint-step runtime behavior.
- [[live_outdoor_ultra_marathon_story]] --- outdoor and marathon runtime evolution with safety-layer reasoning.
- [[outdoor_perception_review]] --- depth/semantic perception findings and their runtime implications.
