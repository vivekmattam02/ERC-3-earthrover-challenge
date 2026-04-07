# ERC Indoor Team Start Here - [0.3em] High-level overview for everyone working on the project

> Source: `docs/team_start_here.tex`
> Master Note: [[erc3_full_documentation]]

\fbox{\parbox{0.9\textwidth}{
**HISTORICAL DOCUMENT** \\[4pt]
Early onboarding note from March 2026. Warning: several statements here are now
outdated. SuperGlue was never part of the final stack, and MBRA became the
primary indoor controller rather than a side experiment. \\[4pt]
\textit{For the current system, see: `CLAUDE.md`,
`docs/INDEX.md`}
}}

# What We Are Building

We are building an indoor navigation system for the EarthRover Challenge in a
**known corridor environment**.

That means:

- the robot is not navigating an unknown building,
- we can collect data in the same corridor ahead of time,
- we can tune the system specifically for that corridor,
- our goal is a reliable competition system, not a general research demo.

\begin{mdframed}
**System choice:** we will use a **CosPlace + SuperGlue + graph**
pipeline for perception and planning, a **validated online local controller**
for short-horizon movement, and separate modules for **safety**,
**checkpoint verification**, and **recovery**.

The MBRA / LogoNav repository remains an important reference:

- **MBRA** is primarily the short-horizon expert / relabeling model,
- **LogoNav** is the deployed-policy side of that project,
- neither should be assumed to be our indoor runtime controller without validation.

\end{mdframed}

# The Most Important File To Know

`baseline.py` is the current baseline perception/planning implementation.

This file already contains the main ideas we are building on:

- CosPlace descriptors for place recognition,
- retrieval of matching corridor images,
- graph construction over reference images,
- action / connectivity graph generation,
- SuperPoint / SuperGlue geometric verification,
- debugging visualizations for graph and localization behavior.

\textbf{What `baseline.py` is:}

- the baseline for **perception and planning**,
- the main file the perception and planning people should study first,
- a shared CLI script with two starting workflows:
- `build-db` for creating the reference database and graph artifacts,
- `query` for testing localization against that database.

\textbf{What `baseline.py` does not cover yet:}

- the full EarthRover runtime system,
- the final online controller for indoor edge-following,
- the safety layer,
- the sequential checkpoint mission loop.

# How The Full System Works

At a high level, the robot loop is:

1. get the current front-camera image,
1. localize that image in the corridor graph,
1. plan which graph node to move toward next,
1. give that nearby node image to the chosen online controller,
1. let that controller propose a local motion command,
1. run safety checks,
1. send the safe command to the robot,
1. verify whether the current checkpoint has been reached,
1. if yes, switch to the next checkpoint.

# Who Owns What

\begin{tabularx}{\textwidth}{p{0.17\textwidth} p{0.28\textwidth} X}

**Stack** & **Main job** & **Where to start** \\

Perception & Figure out where the robot is in the corridor graph & Start with `baseline.py`; turn retrieval + verification into a live module \\
Planning & Decide which nearby graph node the robot should go to next & Start with graph construction and action-graph logic in `baseline.py` \\
Control & Move toward the chosen nearby node image & Start by deciding which online controller we will actually deploy, then build local command generation around it \\
Safety & Prevent unsafe commands from being sent & Build slow/stop/veto logic on top of control outputs \\
Checkpoint / Recovery & Decide when a checkpoint is reached and what to do when uncertain & Build goal confirmation and relocalization logic around the perception output \\
Integration & Connect all stacks into one runtime loop & Connect SDK, localization, planning, MBRA, safety, and mission state \\

\end{tabularx}

# What Each Team Should Keep In Mind

## Perception

Your job is to take the existing CosPlace + SuperGlue corridor pipeline and make
it work reliably in a live loop.

## Planning

Your job is to use the graph to choose the next nearby subgoal image.
Planning should keep global routing in the graph layer and hand off only nearby subgoals to the online controller.

## Control

Your job is to determine which controller we can actually run online for nearby
graph-node movement.

The MBRA / LogoNav codebase is the main reference here, but it should be read carefully:
- MBRA is the relabeling expert in the paper,
- LogoNav is the deployed-policy side,
- LogoNav in this repo is GPS/pose-conditioned, so it is not a drop-in indoor image-goal controller.

## Safety

Your job is to stop the robot from doing something dumb when the environment or
localization is uncertain.

## Checkpoint / Recovery

Your job is to prevent silent failure:
- prevent the system from claiming a checkpoint too early,
- prevent it from continuing confidently when it is lost.

# How To Build On This Baseline

- Start from the existing baseline instead of rebuilding perception/planning from zero.
- Use the MBRA / LogoNav codebase as the main controller reference and adopt the online controller that proves strongest in real corridor testing.
- Treat monocular SLAM and other alternatives as possible enhancements if they prove stronger on real corridor data.
- Prioritize changes that improve reliability, recoverability, and safety on the robot.
- If someone has a better idea, the right standard is simple: show that it works better than the baseline on the real indoor route.

# Immediate Priority Order

1. Understand `baseline.py`.
1. Extract the reusable perception/planning pieces from it.
1. Validate live localization on the corridor.
1. Determine which online short-horizon controller is actually deployable.
1. Validate that controller on nearby graph-node subgoals.
1. Integrate planner $\rightarrow$ controller $\rightarrow$ safety.
1. Add checkpoint verification and recovery.
1. Then run full multi-checkpoint tests.

# Bottom Line

\begin{mdframed}
- `baseline.py` is the baseline perception/planning system.
- The online local controller still has to be selected and validated.
- Safety, checkpoint verification, and recovery are separate stacks.
- The main work now is integration and reliability on the real robot.

\end{mdframed}

# Related Documents

- [[erc3_full_documentation]] --- single master guide for the complete project story and current architecture.
- [[live_indoor_runtime_story]] --- indoor evolution, MBRA integration, and checkpoint-step runtime behavior.
- [[live_outdoor_ultra_marathon_story]] --- outdoor and marathon runtime evolution with safety-layer reasoning.
- [[outdoor_perception_review]] --- depth/semantic perception findings and their runtime implications.
