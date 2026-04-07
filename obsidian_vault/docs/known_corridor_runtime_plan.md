# ERC Indoor Navigation Plan - [0.3em] System choice, module roles, and implementation plan

> Source: `docs/known_corridor_runtime_plan.tex`
> Master Note: [[erc3_full_documentation]]

\fbox{\parbox{0.9\textwidth}{
**HISTORICAL DOCUMENT** \\[4pt]
Implementation plan from March 2026. Most of this plan was executed, but the
final result lives in the actual runtime code and later indoor narrative
documents. \\[4pt]
\textit{For the current system, see: [[live_indoor_runtime_story]],
`CLAUDE.md`}
}}

# Read This First

This document is written so that a teammate can understand the indoor system
without needing prior context from earlier discussions or code exploration.

\begin{mdframed}
**Short version:** we are solving ERC indoor navigation in a \textbf{known
corridor}. The system we will use is:

- a **CosPlace-based visual place recognition pipeline** to recognize locations,
- a **graph of reference images and recorded transitions** to represent the corridor,
- **SuperPoint/SuperGlue-style geometric verification** to reject weak matches,
- **graph planning** to choose the next nearby target place,
- a **validated online local controller** to drive from the current view to that nearby target image,
- a separate **safety and checkpoint verification layer**.

\end{mdframed}

# What Problem Are We Solving?

The indoor ERC task is **image-goal navigation**.

That means:

- the robot is **not** given GPS coordinates,
- the robot is told to go to places that are represented by **images**,
- the robot may have to visit **multiple checkpoints in sequence**,
- the robot runs through the EarthRover / FrodoBots SDK over 4G,
- the robot sees only a few frames per second and commands arrive with delay.

So the real challenge is:

\fbox{\parbox{0.9\textwidth}{
From a delayed low-FPS camera stream, figure out **where the robot is**,
decide **which place it should go to next**, move there safely, and confirm
that the required checkpoint was actually reached.
}}

# Why Our Situation Is Better Than a General Indoor Navigation Problem

This is important.

We are **not** dealing with a completely new and unseen environment.

We now know:

- we will test in the same corridor repeatedly,
- the competition corridor/environment family will also be the same one,
- we can collect data there in advance,
- we can tune our system for that exact corridor.

This changes everything.

## What this allows us to do

- build a database of images from that exact corridor,
- choose good reference locations manually,
- tune descriptor thresholds for that corridor,
- tune image cropping for white walls / floor reflections / bright doors,
- map each checkpoint image to known graph nodes ahead of time,
- accept some environment-specific overfitting on purpose.

## What still does \underline{not} become easy

- low FPS,
- 500 ms-ish latency,
- pedestrians,
- repeated corridor appearance,
- bright doorway / reflective floor effects,
- stable control on the real robot.

# What Existing Work We Already Have

We already have code for a **CosPlace + SuperGlue corridor localization and graph-navigation pipeline**.

This work was focused on the **perception and planning** side of the problem.
It was not yet a full EarthRover runtime system, but it already implemented the
core logic for recognizing places and building a navigation graph from corridor data.

## What that pipeline was designed to do

The pipeline does the following:

1. record images while moving through the corridor,
1. represent each place using a visual descriptor,
1. connect those places into a graph,
1. when a new query image comes in, find the most similar places in the graph,
1. verify the match using geometry,
1. then use the graph to reason about where the robot is and how to move.

## What technical words mean in plain English

**Visual descriptor**

A compact vector that represents what a place looks like.
If two images are visually similar, their descriptors should be similar.

**Retrieval**

Given the current camera frame, search the stored database and find the most
similar reference images.

**Graph**

A set of nodes and edges.
Here:

- a **node** means a saved reference image / place,
- an **edge** means two places are connected or reachable.

**Geometric verification**

After retrieval says "these two images look similar," geometric verification
checks whether they are actually consistent in terms of visual keypoint matches,
instead of being a false match caused by a repetitive corridor.

**Action graph**

A directed graph that says not just which places are similar, but how recorded
motion connected one place to the next.

**MBRA**

A learned visual controller that takes recent observation images and a goal image
and predicts short-horizon motion.

In the MBRA paper/codebase, its primary role is the \textbf{short-horizon expert
used for relabeling or annotation}. The deployed-policy side of that project is
**LogoNav**.

# What The CosPlace + SuperGlue Code Already Does

The file `baseline.py` already gives us a real
**perception/planning baseline**.

It now serves as a shared baseline script with a database-building workflow and
a query/localization workflow.

## It already does these things

- computes global visual descriptors for corridor images,
- builds a database of those descriptors,
- retrieves nearest visual matches for a query image,
- builds a place graph over image nodes,
- builds an action / connectivity graph from recorded route data,
- runs SuperPoint/SuperGlue-style geometric verification,
- visualizes graphs and candidate matches for debugging.

## This means the current corridor pipeline already solves

- how to represent places,
- how to compare a live image to stored places,
- how to connect those places into a graph,
- how to reject weak or fake matches,
- how to inspect what the system thinks is happening.

## But it does \underline{not} yet solve

- live EarthRover SDK runtime,
- real-time command loop,
- multi-checkpoint mission execution,
- selection and validation of the final online local controller,
- safety override,
- recovery behavior if localization becomes uncertain.

\begin{mdframed}
**Correct conclusion:** the **CosPlace + SuperGlue corridor pipeline**
should be treated as the actual baseline for **perception and planning**,
but not as the full competition system.
\end{mdframed}

# What We Are Building Now

We are building a full system around that **CosPlace + SuperGlue + graph**
backbone.

The complete system has five parts:

1. perception / localization,
1. planning,
1. online local control,
1. safety,
1. checkpoint verification and recovery.

# Big Picture: Two Phases

## Phase A: before competition day

Before runtime, we prepare the environment model.

1. drive the robot or manually collect images through the corridor,
1. save those images as reference images,
1. compute CosPlace descriptors for them,
1. build a graph of the corridor,
1. attach recorded route/action information,
1. decide which graph nodes correspond to likely checkpoint regions,
1. test retrieval and SuperGlue-based verification on repeated runs.

## Phase B: during runtime

During competition/runtime, the robot repeatedly does this:

1. read current front-camera frame,
1. localize in the corridor graph,
1. plan path to the current checkpoint target node,
1. choose the next nearby subgoal node,
1. give that subgoal image to the chosen online controller,
1. get a local control command,
1. run safety checks,
1. send safe command to the rover,
1. check whether the current checkpoint has been reached,
1. if yes, switch to the next checkpoint.

# Each Stack Explained Clearly

## 1. Perception / Localization Stack

**Job:** estimate where the robot is in the corridor graph.

**Input:**

- current live camera frame,
- reference database of corridor images and descriptors,
- previous localization state.

**What it does step by step:**

1. compute a CosPlace descriptor for the current frame,
1. retrieve top matching reference images,
1. run SuperGlue-based geometric verification on the best candidates,
1. combine this with temporal continuity,
1. output current best node estimate and confidence.

**Output:**

- current node estimate,
- candidate nodes,
- confidence score,
- localization diagnostics.

**What team members should understand:**

This stack mainly reuses the existing CosPlace + SuperGlue code, but it must be
turned from notebook-style analysis code into a **live runtime module**.

## 2. Planning Stack

**Job:** choose the next place image the robot should move toward.

**Input:**

- current node estimate,
- graph of corridor nodes,
- current target checkpoint,
- mission progress state.

**What it does step by step:**

1. map the current checkpoint to a target node or target region,
1. compute a path on the corridor graph from the current node to that target,
1. choose a nearby node on that path as the current subgoal,
1. keep that subgoal stable until there is enough evidence to switch,
1. once the checkpoint is confirmed, move to the next checkpoint.

**Output:**

- current graph path,
- next subgoal node,
- next subgoal image,
- checkpoint progression state.

**Important rule:**

The clean starting point is for planning to give the online controller a **nearby graph-node image**.
If someone proposes a stronger alternative, it should be evaluated against this baseline on the real corridor.

## 3. Online Local Control Stack

**Job:** move from the current camera view to the next nearby graph-node image.

**Input:**

- recent camera frames,
- current subgoal image from the planner,
- past command history,
- delay-related inputs if needed by the chosen controller.

**What it does step by step:**

1. collect the recent visual context,
1. package the current subgoal image,
1. run the chosen online controller,
1. convert controller output into rover linear/angular commands,
1. smooth or limit commands if needed.

**Output:**

- proposed motion command.

**Important boundary:**

The online local controller is not the full navigation system.
It is only the short-horizon movement module between nearby nodes.

**Important clarification about MBRA / LogoNav**

- In the paper, **MBRA** is mainly the short-horizon expert used for relabeling.
- **LogoNav** is the deployed-policy side of that project.
- In this repo, the provided deployment scripts are for **LogoNav**, not for an indoor image-goal MBRA controller.
- LogoNav here is GPS/pose-conditioned, so it is not a drop-in indoor image-goal controller.

So one of the current engineering tasks is to determine what the actual online
local controller will be for this indoor stack.

## 4. Safety Stack

**Job:** answer the question "Is the proposed motion safe enough to execute?"

**Input:**

- current camera frame,
- proposed local-control command,
- optional depth/clearance estimate,
- optional pedestrian detector,
- localization confidence.

**What it does step by step:**

1. check whether the command looks safe,
1. slow or stop if a person is ahead,
1. slow down if localization confidence is weak,
1. veto dangerous commands,
1. send only a safe final command.

**Output:**

- final command sent to the robot,
- safety state such as OK / slow / stop.

## 5. Checkpoint Verification and Recovery Stack

**Checkpoint verification job:** decide whether the required checkpoint has actually been reached.

**What it should do:**

- require repeated agreement over multiple frames,
- use geometric verification for final confirmation,
- avoid declaring success from one accidental similar corridor view.

**Recovery job:** answer the question "What do we do if we are not sure where we are?"

**What it should do:**

- stop or slow down instead of continuing blindly,
- try relocalizing from recent frames,
- optionally perform a small search motion,
- resume only after confidence improves.

# How All Stacks Connect Together

\fbox{\parbox{0.94\textwidth}{
Camera frame $\rightarrow$
Perception says current node/confidence $\rightarrow$
Planner chooses path and nearby subgoal $\rightarrow$
online controller proposes local command $\rightarrow$
Safety accepts/modifies/rejects command $\rightarrow$
Robot moves $\rightarrow$
Checkpoint verifier decides whether to advance mission state
}}

## In one sentence

- Perception says where we are in the corridor graph.
- Planning says which nearby graph node we should move toward next.
- The online controller says how to move locally toward that node image.
- Safety says whether that movement is safe enough to execute.
- Checkpoint verification says whether the current checkpoint is complete.

# Why We Are Not Using Other Approaches As the Main Backbone

## Why not pure end-to-end policy?

Because we already have a more debuggable and corridor-specific system based on
CosPlace retrieval, geometric verification, and graph planning. A pure end-to-end
policy would hide localization, routing, and failure recovery inside one black box.

## Why not ORB-SLAM3 as the main backbone?

Because even in a known corridor, our real runtime still has:

- low FPS,
- latency,
- visually repetitive corridor structure,
- reflective floor,
- monocular RGB limitations.

ORB-SLAM3 can still be tested, but it should not replace a corridor-specific
CosPlace + graph baseline unless it proves clearly better on our real data.

## Why not retrieval-only without a local controller?

Because retrieval tells us **which place looks similar**, but it does not
by itself provide a strong short-horizon controller under delay and low frame rate.

# What Becomes Easier Because the Corridor Is Known

- node spacing can be tuned on the exact route,
- descriptor preprocessing can be corridor-specific,
- bright-door and reflective-floor cases can be tested directly,
- checkpoint thresholds can be calibrated on real repeated data,
- ambiguous corridor segments can be identified in advance,
- some overfitting is acceptable because this is the actual target environment.

## Examples of acceptable overfitting

- cropping ceiling/floor bands,
- putting more nodes near doors or signs,
- checkpoint-specific verification thresholds,
- graph pruning for impossible transitions.

# What Can Still Fail

Even with a known corridor, these are the main remaining risks:

1. The chosen online controller fails to follow a nearby subgoal cleanly.
1. Node switching becomes unstable in visually repetitive areas.
1. Checkpoint verification declares success too early.
1. Bright doorway / reflective floor causes bad visual matching.
1. Pedestrians block the visual cues we need.
1. Low FPS and latency cause overshoot or slow correction.

# Who Should Work On What

\begin{tabularx}{\textwidth}{p{0.17\textwidth} p{0.31\textwidth} X}

**Area** & **Main responsibility** & **What they should deliver** \\

Perception & Turn the CosPlace + SuperGlue corridor code into a live localization module & Current-node estimate, confidence score, candidate matches, runtime localization API \\
Planning & Build path query and sequential checkpoint logic on the graph & Target-node lookup, path planner, subgoal selector, mission state logic \\
Control & Select and validate the online local controller on the rover & Controller input pipeline, local command generation, smoothing / limits \\
Safety & Prevent unsafe motion at runtime & Slow/stop logic, pedestrian-aware checks, confidence-aware speed limiting \\
Checkpoint / Recovery & Decide when a checkpoint is really reached and what to do when uncertain & Goal verifier, recovery state logic, relocalization behavior \\
Integration & Connect everything end-to-end with the SDK & One full runnable ERC indoor loop \\
Testing & Repeated corridor experiments and tuning & Metrics, debug plots, threshold tuning, failure-case reports \\

\end{tabularx}

# What We Should Do Next In Order

1. Reuse the existing CosPlace + SuperGlue corridor code directly.
1. Clean it up into reusable modules instead of notebook-style code.
1. Validate live localization on the exact corridor.
1. Determine which online local controller is actually deployable.
1. Validate that controller on nearby graph-node subgoals.
1. Integrate planner $\rightarrow$ subgoal $\rightarrow$ controller.
1. Add safety and checkpoint verification.
1. Add recovery behavior.
1. Then run full multi-checkpoint tests.

**Lower-priority directions right now:**

- rebuilding the CosPlace + graph backbone from scratch,
- replacing the full stack before the current baseline is validated end-to-end,
- making SLAM the default backbone before it proves better on our corridor,
- training a big new policy before the known-corridor baseline works.

Improvements are welcome, but the standard should be clear:
**a change is worth adopting if it makes the real robot more reliable on the real ERC indoor route.**

# Final Team Verdict

\begin{mdframed}
- The **CosPlace + SuperGlue corridor-localization pipeline** is the real baseline for perception and planning.
- The main new work is **runtime integration around that baseline**.
- The **online local controller** still has to be selected and validated carefully.
- Safety, checkpoint verification, and recovery must be separate modules.
- The biggest risk is **runtime stability on the real robot**, not lack of architecture ideas.

\end{mdframed}
