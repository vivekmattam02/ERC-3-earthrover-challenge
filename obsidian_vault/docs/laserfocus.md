# ERC Indoor Laser Focus - [0.3em] What we implement next and what we deliberately postpone

> Source: `docs/laserfocus.tex`
> Master Note: [[erc3_full_documentation]]

\fbox{\parbox{0.9\textwidth}{
**HISTORICAL DOCUMENT** \\[4pt]
Scope document from March 2026. The scope decisions here were either implemented,
dropped, or folded into later runtime work. SuperGlue appears in this file but
was never part of the final indoor backbone; CosPlace alone proved sufficient. \\[4pt]
\textit{For the current system, see: `CLAUDE.md`}
}}

# One-Sentence Project Definition

We are building a **known-corridor indoor navigation system** for ERC using:

- **CosPlace + SuperGlue + graph** for perception and planning,
- a **motion-prior layer** built from orientation / IMU support,
- a **validated online local controller** for nearby graph-node movement,
- a separate **safety layer**,
- a separate **checkpoint verification and recovery layer**.

# What We Are Using Right Now

## Core baseline

- `baseline.py` is the perception/planning baseline.
- It is responsible for descriptor extraction, retrieval, graph construction, and query-time localization support.
- It is the right starting point because our corridor is known and repeated.

## Robot interface

- `src/earthrover_interface.py` gives us the live robot interface.
- The front camera frame is still the main localization signal.
- The SDK also exposes orientation, speed, and IMU arrays (`accels`, `gyros`, `mags`).
- For indoor ERC, those motion signals should be used as **supporting priors** for heading consistency, node-belief smoothing, and recovery.
- GPS is available in the SDK, but it is not part of the indoor backbone.

## Depth safety

- `src/depth_estimator.py` and `src/depth_safety.py` are optional safety tools.
- They are useful after the baseline localization/planning loop is working.

## MBRA / LogoNav code

- `mbra_repo/` is a research reference, not yet our deployed indoor controller.
- MBRA is the relabeling expert side of that project.
- LogoNav is the deployed-policy side, but in this repo it is GPS/pose-conditioned, so it is not a drop-in indoor image-goal controller.

# What We Are \underline{Actually} Doing First

\begin{mdframed}
**Immediate focus:** stop thinking about the whole final system at once.
The next job is to make the corridor baseline real on our actual data.
\end{mdframed}

## Step 1: Collect corridor data

- Run the robot through the real corridor.
- Save front-camera images in order.
- Save orientation, speed, IMU samples, timestamps, and any available command / teleoperation metadata per step.
- Collect multiple runs of the same route.

**Deliverable:** one clean reference dataset from the real ERC corridor with both images and motion-state logs.

## Step 2: Build the corridor database

- Run `baseline.py build-db`.
- Build descriptors.
- Build the place graph.
- Build the action graph if route metadata is available.

**Deliverable:** database artifacts that can be queried repeatedly.

## Step 3: Validate localization

- Run `baseline.py query` on real corridor images.
- Check whether retrieval gives the correct corridor nodes.
- Identify ambiguous segments, weak landmarks, and failure cases.
- Tune graph density and image preprocessing.
- Add heading-consistency checks so retrieval cannot jump to impossible nodes too easily.

**Deliverable:** a corridor graph and localization pipeline that is stable enough to trust, including temporal and heading-aware filtering.

## Step 4: Decide the online local controller

- Decide what actually runs online for nearby graph-node movement.
- MBRA / LogoNav is a reference here, but should not be assumed correct by default.
- The decision must be made from real corridor testing, not from paper naming.
- The chosen controller should use heading / yaw-rate feedback for smoothing if that improves stability.

**Deliverable:** one local controller candidate that we can actually test online.

## Step 5: Integrate runtime loop

- camera frame,
- localization in graph,
- graph path / next subgoal,
- online local controller,
- safety override,
- checkpoint verification.

**Deliverable:** one full indoor loop for a single checkpoint.

# What We Should \underline{Not} Split Attention On Right Now

- training a brand new large model,
- replacing the graph-localization baseline before it is tested,
- making SLAM the backbone before the current baseline is evaluated on the real corridor,
- polishing multi-checkpoint logic before single-checkpoint execution works,
- deep safety tuning before localization and control are running.

# Sensors and Signals: What Matters Now

\begin{tabularx}{\textwidth}{p{0.22\textwidth} p{0.18\textwidth} X}

**Signal** & **Priority** & **How we use it now** \\

Front camera & Highest & Main signal for localization, graph matching, checkpoint verification, and local control input \\
Orientation / heading & High & Motion prior for heading consistency, candidate-node gating, subgoal alignment, and recovery behavior \\
IMU arrays & High-medium & Short-term yaw / motion support for temporal filtering, controller damping, and uncertainty handling \\
GPS & Low for indoor & Mostly irrelevant for the main indoor corridor task \\
Wheel encoders / RPMs & Validate first & Use only if live tests prove they are populated and stable; do not assume odometry quality without verification \\
Depth estimate & Medium-later & Safety support after baseline localization/control is working \\

\end{tabularx}

**Practical answer:** the first working indoor baseline should be
**vision-primary with IMU / heading support**. It should not be vision-only,
and it should not assume encoder-based odometry until the robot proves that signal is real.

# The Main Open Question

The biggest unresolved implementation question is:

\fbox{\parbox{0.85\textwidth}{
**What is our actual online short-horizon controller for moving between nearby graph nodes?**
}}

Everything else has a cleaner answer already:

- perception/planning baseline = `baseline.py`
- robot interface = `src/earthrover_interface.py`
- motion prior = orientation / IMU support with temporal filtering
- safety support = depth and simple runtime veto logic

# Immediate Milestones

## Milestone 1

**Collect corridor data and build the reference database.**
**At the same time, verify orientation / IMU / RPM availability on the real robot.**

## Milestone 2

**Prove that query-time localization works reliably on repeated corridor runs.**
**Add temporal and heading-aware filtering to the node belief.**

## Milestone 3

**Choose and validate one online local controller on nearby graph-node subgoals.**

## Milestone 4

**Run one single-checkpoint indoor loop with localization + planning + controller + safety.**

## Milestone 5

**Add checkpoint verification and multi-checkpoint execution.**

# What Success Looks Like Right Now

Success in the next phase is **not**:

- a polished final competition stack,
- a fully trained new model,
- a fancy architecture diagram.

Success in the next phase **is**:

- we can collect corridor data cleanly,
- we can log the motion signals we actually trust,
- we can build a graph database from it,
- we can localize repeated query frames correctly,
- we can stabilize localization using heading / IMU support,
- we can pick the next nearby graph node,
- we have one real candidate for the online local controller.

# Bottom Line

\begin{mdframed}
- The baseline is ready enough to start real work.
- The next step is **data collection on the actual corridor**.
- The project should be **vision-primary with IMU / heading support**.
- We must verify which onboard motion signals are actually trustworthy.
- The biggest open engineering decision is the **online local controller**.
- Everything else should stay focused on making the known-corridor baseline work end-to-end.

\end{mdframed}
