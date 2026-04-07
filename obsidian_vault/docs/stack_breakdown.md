# Indoor Navigation Stack Breakdown - [0.3em] Exact module responsibilities, interfaces, and work allocation

> Source: `docs/stack_breakdown.tex`
> Master Note: [[erc3_full_documentation]]

# Why This Document Exists

This document is the **operational version** of the indoor strategy.
It answers:
- what each stack is supposed to do,
- what goes into each stack,
- what comes out of each stack,
- what technical methods we plan to use first,
- who should own which stack,
- and how the stacks connect into one system.

\begin{mdframed}
**System goal:** indoor image-goal navigation on EarthRover using a
topological graph for global navigation, MBRA for local image-goal control, and
a safety layer for collision prevention.
\end{mdframed}

# System in One Pass

## End-to-end runtime loop

1. read the latest front camera frame from the EarthRover SDK,
1. update the recent image buffer,
1. localize the current view in the indoor topological graph,
1. map the mission checkpoint image to a target graph node,
1. plan a graph path from current node to target node,
1. choose the next nearby node on that path as the current subgoal image,
1. run MBRA on recent context plus the subgoal image,
1. run safety checks on the proposed control,
1. send the safe control command to the robot,
1. repeat until checkpoint image verification says the target is reached.

## Main stacks

1. SDK / robot interface
1. Perception
1. Map / graph
1. Planning
1. Control
1. Safety
1. Integration / orchestration
1. Testing / evaluation

# Ownership Model

If people need to be assigned immediately, use the following work split.
Replace the owner placeholders with actual names.

\begin{tabularx}{\textwidth}{p{0.18\textwidth} p{0.22\textwidth} X}

**Stack** & **Owner** & **Primary responsibility** \\

SDK / Interface & Owner A & camera ingestion, control send, runtime I/O reliability \\
Perception & Owner B & visual localization, checkpoint verification, pedestrian detection \\
Map / Graph & Owner C & graph data model, graph builder, node/edge storage \\
Planning & Owner D & graph search, subgoal selection, progress logic \\
Control & Owner E & MBRA runtime wrapper and local control loop \\
Safety & Owner F & depth / detector safety overrides and stop logic \\
Integration & Owner G & connect all modules into one process or service \\
Testing & Owner H & route tests, metrics, failure analysis, logging \\

\end{tabularx}

# Stack 0: SDK / Robot Interface

## Purpose

Provide reliable robot I/O for the rest of the system.

## What this stack receives

- browser-fed EarthRover camera stream,
- EarthRover telemetry,
- command requests from the navigation system.

## What this stack outputs

- latest front camera frame,
- timestamps and any usable telemetry,
- confirmed command send path to the robot.

## What we are going to do

1. use the existing EarthRover SDK server,
1. use the existing Python interface to fetch frames and send commands,
1. add any missing timestamping or logging needed for indoor experiments,
1. standardize a clean API for the rest of the indoor pipeline.

## Implementation details

**Primary functions needed:**
- `get_camera_frame()`
- `send_control(linear, angular)`
- optional frame timestamp retrieval
- optional telemetry logging

## Definition of done

- frame retrieval is stable,
- command sending is stable,
- all indoor modules can consume the same camera and control API.

# Stack 1: Perception

## Purpose

Understand what the robot is currently seeing.

## Submodules

1. **Context buffer**: keep the last $N$ frames for MBRA.
1. **Visual localization**: match the current view to graph nodes.
1. **Checkpoint verification**: determine whether a goal image is reached.
1. **Pedestrian detection**: detect people for safety support.

## Inputs

- latest front camera frame from SDK,
- graph node images from the map stack,
- target checkpoint image from mission definition.

## Outputs

- current node estimate,
- top-$k$ candidate matching nodes,
- localization confidence,
- checkpoint reached / not reached,
- pedestrian present / not present in danger zone.

## What we are going to do first

1. implement a clean frame buffer,
1. choose a first visual descriptor baseline for node retrieval,
1. implement a first checkpoint verification baseline,
1. run a pretrained person detector for corridor safety.

## Preferred first methods

**Visual localization:**
- start with global descriptor retrieval against graph nodes,
- add temporal smoothing,
- add local geometric verification for strong matches.

**Checkpoint verification:**
- stage 1: descriptor similarity threshold,
- stage 2: feature matching + geometric verification,
- require multiple consecutive confirmations if needed.

**Pedestrian detection:**
- start with pretrained YOLO `person` detection,
- define a forward danger zone in image coordinates,
- pass danger flags to the safety stack.

## Definition of done

- current node can be estimated from live camera frames,
- checkpoint verification can confirm a reached goal image,
- pedestrian detector works well enough to support indoor safety.

# Stack 2: Map / Graph

## Purpose

Represent the indoor environment as a place graph instead of a GPS map.

## Inputs

- teleoperated indoor camera recordings,
- route ordering,
- manually or semi-automatically selected keyframes.

## Outputs

- graph node set,
- graph edge set,
- node images and metadata,
- target checkpoint-to-node mapping.

## What we are going to do

1. teleoperate the robot through indoor routes,
1. save reference images,
1. choose representative keyframes as nodes,
1. define traversable connections as edges,
1. store the graph in a format the planner can query directly.

## Minimum graph schema

Each node should contain:
- node id,
- image path,
- route id,
- ordering index,
- optional semantic label,
- optional descriptor embedding.

Each edge should contain:
- source node id,
- destination node id,
- traversal cost,
- optional direction label.

## Definition of done

- one indoor route can be represented as a queryable graph,
- the planner can load it and search it,
- the perception stack can localize against it.

# Stack 3: Planning

## Purpose

Decide what place the robot should go to next.

## Inputs

- current node estimate from perception,
- target node from mission checkpoint mapping,
- graph from map stack.

## Outputs

- graph path from current node to target node,
- next subgoal node,
- subgoal image for MBRA,
- path progress state.

## What we are going to do

1. compute shortest path in the graph,
1. pick the next nearby node on that path,
1. expose the next node image as the current local goal,
1. update path state as localization changes,
1. add simple recovery if localization confidence drops.

## Important design rule

\begin{mdframed}
\textbf{MBRA should receive a nearby subgoal image, not the final faraway goal
image for the whole mission.}
\end{mdframed}

## Definition of done

- planner can produce a valid path,
- planner can switch subgoals correctly,
- control stack can always request the current subgoal image.

# Stack 4: Control

## Purpose

Move the robot from the current view toward the next nearby subgoal image.

## Inputs

- recent frame buffer from perception,
- current subgoal image from planner,
- optional delay estimate,
- recent command history,
- robot-size inference constant if required by MBRA.

## Outputs

- proposed linear command,
- proposed angular command,
- controller diagnostics.

## What we are going to do

1. build an MBRA runtime wrapper,
1. prepare MBRA inputs from live EarthRover frames,
1. run MBRA inference on recent context plus the subgoal image,
1. convert MBRA outputs into EarthRover commands,
1. clip and smooth commands before safety checks.

## Exact role of MBRA

\begin{mdframed}
**MBRA is the local controller.** It solves the short-horizon problem:
"from what I see now, how do I move toward this nearby goal image?"
\end{mdframed}

## Definition of done

- robot can be driven toward a manually selected nearby indoor goal image,
- MBRA commands are correctly transformed into rover control commands,
- local goal-following is stable enough for integration with the planner.

# Stack 5: Safety

## Purpose

Prevent local control from producing unsafe motion.

## Inputs

- current frame,
- MBRA proposed command,
- pedestrian detection flags from perception.

## Outputs

- safe final linear command,
- safe final angular command,
- stop / slow / override reason.

## What we are going to do

1. add a depth or clearance estimation step,
1. define a forward safety check for commanded motion,
1. stop or slow down when people are detected in the danger zone,
1. provide a final safe command to the integration layer.

## Preferred first safety rules

- if a person is detected in the forward danger zone, slow or stop,
- if predicted clearance in commanded direction is too low, stop or redirect,
- if confidence is low or sensor state is stale, fail safe.

## Definition of done

- unsafe MBRA commands are prevented,
- pedestrian-aware slowing or stopping works in corridor tests.

# Stack 6: Integration / Orchestration

## Purpose

Connect all stacks into one real system.

## Inputs

- outputs from all other stacks.

## Outputs

- one running indoor navigation process,
- logs,
- debug traces,
- replayable experiment data.

## What we are going to do

1. define clear module interfaces,
1. build one runtime loop that calls each stack in order,
1. standardize config files and logging,
1. provide debug views for localization, path, MBRA output, and safety.

## Definition of done

- all stacks can run together in one loop,
- end-to-end indoor tests are possible.

# Stack 7: Testing / Evaluation

## Purpose

Turn a collection of modules into a reliable system.

## What we are going to measure

- localization accuracy against node labels,
- subgoal switching correctness,
- local goal-following success rate,
- safety interventions,
- checkpoint reach success rate,
- number and type of failures.

## What we are going to do

1. define a few standard indoor routes,
1. run repeated tests on those routes,
1. log failures by stack,
1. decide whether failures are from perception, planning, control, or safety.

# Exact Handoffs Between Stacks

```text
\toprule
\textbf{From} & \textbf{To} & \textbf{Handoff} \\
\midrule
SDK & Perception & latest frame, timestamps, live stream access \\
Perception & Planning & current node estimate, candidate nodes, confidence \\
Map / Graph & Planning & graph structure, node images, target node ids \\
Planning & Control & current subgoal node id and subgoal image \\
Perception & Safety & pedestrian detections, checkpoint verification flags if relevant \\
Control & Safety & proposed rover command \\
Safety & Integration & final safe rover command \\
Integration & SDK & command send request \\
\bottomrule
```

# Immediate Engineering Order

This is the order we should build in.

1. **SDK stability** --- confirm live frame and control path are solid.
1. **MBRA local prototype** --- prove the robot can move toward one manually selected nearby indoor goal image.
1. **Graph build tooling** --- create one short indoor graph from collected data.
1. **Visual localization** --- localize live frames into that graph.
1. **Planner** --- choose subgoal nodes from current node to target node.
1. **Safety layer** --- stop or slow on people and low clearance.
1. **Full integration** --- run the whole loop.
1. **Repeat testing** --- measure failure modes and refine.

# What Each Team Member Should Produce

## SDK / Interface owner

- stable camera and control API,
- frame logging utility,
- command logging utility.

## Perception owner

- node retrieval module,
- checkpoint verification module,
- pedestrian detector wrapper.

## Map / Graph owner

- graph schema,
- graph builder from recorded routes,
- graph serialization format.

## Planning owner

- graph search module,
- subgoal selection module,
- path progress tracker.

## Control owner

- MBRA wrapper,
- recent-frame buffer interface,
- MBRA-to-rover command conversion.

## Safety owner

- depth / clearance check,
- person-aware slow / stop logic,
- final command override policy.

## Integration owner

- one main runtime loop,
- stack wiring,
- unified configuration and logging.

## Testing owner

- test route definitions,
- evaluation metrics,
- failure reports and experiment tracking.

# What We Are Doing Exactly

\begin{mdframed}
**Perception** tells us where we are and whether a checkpoint is reached.\\
**Map / Graph** stores the building as connected places.\\
**Planning** decides which place we should go to next.\\
**Control** uses MBRA to move toward that nearby place image.\\
**Safety** prevents bad local actions.\\
**Integration** makes everything run as one system.\\
**Testing** tells us whether any of this is actually reliable.
\end{mdframed}

\begin{mdframed}
**The single most important architectural rule:**
MBRA is our **local controller**, not our global planner.
\end{mdframed}

If all team members build their stack with that rule in mind, the project
stays coherent.
