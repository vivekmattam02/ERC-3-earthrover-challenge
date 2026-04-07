# Our Indoor Navigation Approach - [0.3em] High-level overview for team planning and work assignment

> Source: `docs/our_approach.tex`
> Master Note: [[erc3_full_documentation]]

\fbox{\parbox{0.9\textwidth}{
**HISTORICAL DOCUMENT** \\[4pt]
High-level overview from March 2026. Useful for seeing the early framing, but it
has been superseded by the later narrative documents and the current system guide. \\[4pt]
\textit{For the current system, see: `CLAUDE.md`,
[[final_system_overview]]}
}}

# Goal

Build a **reliable indoor navigation system** for the EarthRover that can
move through NYU indoor spaces using **image goals** rather than GPS.

\begin{mdframed}
**Core idea:** use a **topological graph** for global navigation,
**MBRA** for local image-goal control, and a **safety layer** to
prevent unsafe actions.
\end{mdframed}

# High-Level System

Our system has five main parts:

1. **Perception** --- read camera frames and identify where the robot is.
1. **Map / Graph** --- store indoor places as connected image nodes.
1. **Planning** --- compute which node the robot should go to next.
1. **Control** --- use MBRA to move toward the next subgoal image.
1. **Safety** --- override unsafe commands before sending them.

## Data flow

\fbox{\parbox{0.93\textwidth}{
Camera stream $\rightarrow$ localization in graph $\rightarrow$ path to target
node $\rightarrow$ next subgoal image $\rightarrow$ MBRA local control
$\rightarrow$ safety check $\rightarrow$ robot command
}}

# Why This Approach

- The competition is **image-goal based**, so a place-based graph is
more natural than GPS-style navigation.
- MBRA is best used as a **short-horizon local controller**, not as
the whole long-range navigation system.
- A separate planning layer makes the system easier to debug and more reliable.
- A separate safety layer reduces the risk of trusting the learned model too much.

# Module Breakdown

## Perception

**Purpose:** understand what the robot currently sees and localize it in
the indoor environment.

**Responsibilities:**
- capture and buffer front camera frames,
- extract visual descriptors for current frames,
- match current view to graph node images,
- provide current node estimate and confidence,
- support visual checkpoint verification.

**Outputs:**
- current node estimate,
- candidate nearby nodes,
- similarity scores / verification results.

## Planning

**Purpose:** decide where the robot should go next at the place level.

**Responsibilities:**
- maintain the topological graph,
- map checkpoint images to target nodes,
- compute graph paths from current node to target node,
- choose the next local subgoal node on that path,
- handle node switching and simple recovery logic.

**Outputs:**
- target path,
- current subgoal image,
- path progress state.

## Control

**Purpose:** move the robot from its current view to the next nearby
subgoal image.

**Responsibilities:**
- maintain recent image context,
- prepare MBRA inputs,
- run MBRA inference,
- convert MBRA outputs into rover commands,
- manage local command smoothing and rate limits.

**Outputs:**
- proposed linear/angular commands,
- controller state and diagnostics.

## Safety

**Purpose:** prevent collisions and unsafe actions.

**Responsibilities:**
- estimate depth or clearance from the current frame,
- evaluate whether the MBRA action is safe,
- clip, redirect, or stop when needed,
- expose safety status to the rest of the system.

## Integration

**Purpose:** connect everything into one runnable indoor system.

**Responsibilities:**
- EarthRover SDK interface,
- runtime orchestration between modules,
- configuration and logging,
- debugging tools,
- evaluation scripts and test runs.

# Suggested Team Workstreams

\begin{tabularx}{\textwidth}{p{0.18\textwidth} p{0.35\textwidth} X}

**Area** & **Main task** & **Expected deliverable** \\

Perception & Visual localization and checkpoint matching & Current-node estimator and goal verification pipeline \\
Planning & Graph construction and graph search & Indoor topological graph format, path planner, subgoal selector \\
Control & MBRA runtime controller & MBRA wrapper and local goal-following controller on EarthRover \\
Safety & Runtime obstacle override & Depth/clearance-based safety module \\
Integration & End-to-end system glue & One script or service that runs the full pipeline \\
Testing & Evaluation and failure analysis & Repeatable test routes, metrics, debug reports \\

\end{tabularx}

# Immediate Next Tasks

## Perception team

- decide the first localization baseline,
- define how node images are stored,
- define the first goal verification method.

## Planning team

- define graph node and edge format,
- build a simple graph from one short indoor route,
- implement shortest-path search and subgoal selection.

## Control team

- load MBRA,
- buffer the last few images,
- feed a manually selected nearby goal image,
- generate rover commands from MBRA outputs.

## Safety team

- build a minimal runtime safety check,
- define stop / slow / override behavior.

## Integration team

- connect SDK + perception + planning + control + safety,
- make a minimal runnable indoor demo.

# First Milestone

\begin{mdframed}
**Milestone 1:** prove that the robot can move toward a nearby indoor
subgoal image using MBRA while the rest of the system provides only minimal
support.
\end{mdframed}

This is the right first milestone because it validates the local controller
before we invest heavily in graph infrastructure.

# Second Milestone

\begin{mdframed}
**Milestone 2:** connect graph localization and path planning so MBRA no
longer receives a manually selected goal image, but an automatically selected
subgoal node image.
\end{mdframed}

# What Success Looks Like

We should consider the approach healthy if:
- the robot can localize itself in the graph,
- the planner can select the correct next node,
- MBRA can drive to that local node image,
- the safety layer prevents obvious collisions,
- and the full loop can repeat over multiple checkpoints.

# Short Summary

\begin{mdframed}
**Our approach in one sentence:** use perception to localize in a
topological graph, planning to choose the next node, MBRA to drive toward that
node image, and safety to prevent bad local actions.
\end{mdframed}

That gives us a clear division of labor:
- **Perception** finds where we are,
- **Planning** decides where we go next,
- **Control** gets us there with MBRA,
- **Safety** keeps us from doing something stupid,
- **Integration** makes the whole system real.
