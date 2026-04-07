# ERC-3 EarthRover Competition Technical Report

> Source: `competition_technical_report_2026_03_31.tex`
> Master Note: [[erc3_full_documentation]]

# Executive Summary

This report summarizes the current state of the ERC-3 EarthRover project as of March 31, 2026, with emphasis on the engineering work most relevant to competition operation. The most important high-level conclusion is that the project naturally split into two distinct autonomy problems:
- **indoor corridor navigation**, where the strongest structure is dataset-based visual localization and corridor graph planning; and
- **outdoor mission runtime control**, where the strongest structure is checkpoint management, route expansion, and layered runtime safety around a learned local controller.

The project therefore should not be described as one monolithic autonomy stack. It is better understood as two related but different systems that share supporting infrastructure and engineering discipline.

Indoors, the current direction is MBRA-first navigation with exact checkpoint-step targets, graph-based subgoal progression, depth-based veto logic, no-reverse behavior for MBRA, skip-past-checkpoint handling for the forward-only corridor graph, and explicit no-progress context reset. Outdoors, the current direction is LogoNav-based checkpoint navigation wrapped in a stricter mission runtime including route corridor guarding, rerouting from live GPS, stronger traversability intervention, carefully gated semantic hooks, IMU safety, camera and health checks, and operator-visible intervention logic.

The project has made real progress, but the right description is still **supervised field robotics**, not solved autonomy. The major remaining outdoor weakness is reliable handling of curbs, steps, drops, and other low-lying hazards under repeated live testing.

# Competition Context

The competition setting matters because it defines the correct engineering objective. The indoor and outdoor events impose different assumptions:
- indoors, the rover operates in a constrained and partially known visual corridor environment;
- outdoors, the rover must complete checkpoint-driven missions in a live pedestrian environment under safety constraints.

This means the optimization target is not simply "maximize autonomy." The real target is:
**safe, explainable, and repeatable supervised operation under competition conditions.**

For outdoor competition use, this strongly prioritizes:
- route discipline,
- prevention of obviously unsafe behavior,
- operator intervention readiness,
- protection against brittle deadlock modes,
- and honest accounting of what remains unresolved.

# System Decomposition

## Indoor System

The indoor stack is best described as:
dataset images $\rightarrow$ visual localization $\rightarrow$ temporal filtering $\rightarrow$ corridor graph planning $\rightarrow$ short-horizon controller

In practice, the most important indoor realization was that the problem was not generic exploration. It was a **known-corridor visual navigation problem**. This reframing clarified which components were already structurally strong and which needed most of the attention.

## Outdoor System

The outdoor stack is best described as:
SDK mission checkpoints $\rightarrow$ optional OSM pedestrian routing $\rightarrow$ intermediate waypoints $\rightarrow$ LogoNav controller $\rightarrow$ runtime safety envelope

The strongest outdoor lesson was that the controller alone is not the system. The real engineering work lies in the mission runtime: route interpretation, waypoint handoff, telemetry sanity, corridor enforcement, perception support, and intervention discipline.

# Indoor Navigation: Current Understanding

## Core Indoor Lessons

Several indoor discoveries became foundational:
- exact checkpoint-step targets are much cleaner than vague target images or late-stage guesswork;
- indoor compass heading was harmful for the localization path that was actually being used;
- MBRA should be treated as a short-horizon image-goal controller, not as a global planner;
- reverse-heavy recovery behavior was counterproductive for MBRA;
- stale visual context could trap the rover in self-reinforcing no-progress loops.

Another important indoor lesson was structural rather than model-specific: the corridor graph behaves like a forward-only graph in practice. That means the runtime needs explicit skip-past-checkpoint handling when localization shows that the rover has already moved beyond a target step with enough confidence. Otherwise, the runtime can stop forever on a path-planning "no path" state even though the robot has already progressed past the intended checkpoint.

## Current Indoor Runtime Direction

The current indoor recommendation is MBRA-first operation:
- MBRA used as the primary indoor local controller;
- exact checkpoint-step competition mode rather than vague target selection;
- graph-based forward subgoal progression;
- depth used as a veto or slow/stop layer rather than the planner itself;
- no reverse for MBRA recovery;
- skip-past-checkpoint handling for the forward-only graph;
- no-progress reset to force context refresh.

This is not a claim that indoor operation is fully solved. It is a claim that the indoor system is now technically coherent and much better understood than before.

\placeholderfigure{Placeholder for indoor pipeline diagram or localization-to-subgoal figure}{2.0in}

# Outdoor Mission Runtime

## Starting Point

The outdoor runtime already had meaningful capabilities before the latest hardening work:
- mission start and checkpoint handling through the SDK,
- checkpoint resume,
- LogoNav as the main outdoor controller,
- optional OSM pedestrian route expansion,
- traversability and semantic support layers,
- stuck detection and telemetry freeze awareness.

The right outdoor engineering question therefore was not "what brand-new controller should replace everything?" It was:
**How do we stop the runtime from doing obviously dumb things in the real world?**

## Main Outdoor Runtime Additions

The major additions and corrections include:
- IMU safety integration and later refinement to reduce false anti-flip behavior,
- route corridor guard,
- rerouting from live GPS after corridor violations,
- startup auto-claim of already-reached checkpoints,
- dynamic intermediate waypoint radius,
- behind-the-rover waypoint pruning,
- distinction between routed intermediate waypoints and mission checkpoints,
- night-safe and ultra-marathon profiles,
- stronger traversability intervention behavior,
- improved operator-facing logs,
- preflight validation.

One subtle but important outdoor correction was recognizing that **mission checkpoints and routed intermediate waypoints should not be treated as the same kind of target**. The deeper outdoor runtime writeups make this explicit: the route-expanded waypoints need tighter reach semantics and more aggressive pruning, while the mission checkpoints remain larger-scale objectives.

## Representative Failure-Driven Corrections

```text
\toprule
\textbf{Observed issue} & \textbf{Correction} & \textbf{Intended effect} \\
\midrule
False IMU emergencies while upright & IMU self-calibration and less naive triggering & Reduce bogus emergency halts during normal turning \\
\addlinespace
Corridor stop deadlock & Reroute current leg from live pose & Prevent endless stop-loop on stale route state \\
\addlinespace
Behind-the-rover reroute points & Prune backward starter waypoints & Prevent useless spins and regressions after reroute \\
\addlinespace
Intermediate waypoint handoff too early & Dynamic route waypoint radius & Reduce premature waypoint advances \\
\addlinespace
Over-eager align-turn behavior & Distance-aware and target-aware align gating & Reduce spinning on far or poorly chosen targets \\
\addlinespace
Weak soft obstacle bias & Stronger traversability override logic & Make local obstacle handling more meaningful \\
\bottomrule
```

\placeholderfigure{Placeholder for outdoor runtime block diagram or route-corridor figure}{2.1in}

# Perception and Safety Layers

## Depth and Traversability

Metric depth and derived traversability became more useful after scale assumptions were revisited. Earlier work demoted traversability to a soft bias because the raw depth signal was too weak to trust as a standalone hard safety layer. The later runtime, however, gave traversability more authority when the forward corridor is clearly blocked. In the current project state it is best described as a **local override layer with narrow, geometry-limited authority**, not as a complete scene-understanding system.

Its most useful roles are:
- blocked-corridor detection,
- local slowdown or stop behavior,
- limited steering bias away from clearly bad local structure.

They should not be oversold as a complete outdoor obstacle-avoidance solution. Low-lying hazards remain a major open issue.

## Semantic Segmentation

Semantic segmentation is more promising than raw depth for category-level understanding:
- people and animals,
- road versus sidewalk interpretation,
- non-drivable or caution-heavy regions.

However, semantic logic depends strongly on:
- ROI design,
- label grouping,
- thresholding policy,
- and how much authority the semantic layer is allowed to have.

The current correct description is that semantics is a **gated support layer with experimental hard-stop and sidewalk-stop hooks**, not a magic safety oracle.

## Safety Authority

The current outdoor runtime is best described as LogoNav inside a layered safety envelope. The important authorities are:
1. mission logic selects the active objective,
1. OSM routing defines the preferred corridor,
1. LogoNav proposes local action,
1. traversability and semantics may bias, slow, or stop,
1. route corridor, IMU, vision, and telemetry checks may veto or force intervention,
1. the operator remains the final authority.

# Current Status

## Status Table

```text
\toprule
\textbf{Area} & \textbf{Working} & \textbf{Partially working} & \textbf{Still open} \\
\midrule
Indoor localization and planning & Corridor localization and graph path structure & Sensitivity to scene mismatch and stale state & Broader stress testing \\
\addlinespace
Indoor control & MBRA-first path is coherent & Still needs stronger repeatability evidence & Formal evaluation campaign \\
\addlinespace
Outdoor runtime & Mission handling, resume, reroute, safety wrappers & Still supervised, still field-sensitive & Stable repeated long-form runs \\
\addlinespace
Outdoor perception support & Traversability and semantics exist and help & Useful as support, not full avoidance & Curbs, steps, drops, thin hazards \\
\addlinespace
Documentation & Strong report layer already exists & Current-truth layer is being organized & Full canonical operator docs \\
\bottomrule
```

## Most Honest Single-Sentence Summary

Indoors, the system is now much better framed and technically coherent. Outdoors, the runtime is much stronger than before, but it is still a supervised field robotics system rather than a solved autonomous navigation stack.

# Competition Operating Plan

For competition-facing use, the correct operational approach is conservative:
- use the current runtime with explicit safety profiles rather than introducing speculative new autonomy modules;
- use preflight checks before operation;
- keep route discipline and intervention discipline explicit;
- prefer stability and explainability over aggressive behavior;
- do not overclaim obstacle avoidance capability outdoors.

The main outdoor risk concentration is still low-lying hazards and repeated real-world validation on the current runtime.

# Documentation and Handoff

The project now has substantial documentation content already produced in reports, stories, and reviews. The next step is to organize it into a maintainable system:
- current truth,
- how-to guides,
- reference documentation,
- architecture documentation,
- preserved deep technical reports.

This matters because the project has now accumulated enough engineering knowledge that undocumented decisions would create real technical debt.

# Next Steps

The most useful next work items are:
1. repeated outdoor validation on the current runtime,
1. targeted curb / step / drop handling work,
1. continued cleanup of waypoint and reroute behavior only where evidence justifies it,
1. conversion of the documentation set into a canonical current/reference/report structure,
1. clearer quantitative evaluation for both indoor and outdoor progress claims.

# Conclusion

The strongest result of the project so far is not a single model or single feature. It is the emergence of a more honest and technically grounded system understanding:
- indoor and outdoor are different problems,
- controllers must sit inside runtimes,
- field failures are a primary source of design truth,
- and documentation is part of the engineering system, not an afterthought.

That understanding is what now makes the next phase of work more focused and more credible.
