# ERC-3 Outdoor Perception Review - What We Tried, What Worked, What Failed, and Why

> Source: `outdoor_perception_review.tex`
> Master Note: [[erc3_full_documentation]]

> [!abstract] Note Header
> **Purpose:** Record of the outdoor perception investigation, especially depth and semantics.
> **Read when:** I need to understand why perception layers were kept as bounded support rather than promoted to full authority.
> **Authority:** Investigation note. Trust current runtime behavior and the master note for final system truth.

> [!important] Current Status
> This note is still relevant because the current off-road branch inherited its main perception lesson:
> depth and semantics are bounded support layers, not the main authority.
> In the current no-GPS branch, visual route evidence and terrain-aware control matter more than promoting monocular depth into a hard decision-maker.

# Purpose of This Document

This document records, in detail, the outdoor perception and safety work carried out for the ERC-3 EarthRover Challenge codebase. The goal is not only to remember the final code state, but also to preserve the reasoning process:
- what problem we were trying to solve,
- why we thought certain ideas were promising,
- what experiments we ran,
- what we observed on real recorded data,
- what conclusions we reached,
- and what should happen next.

The motivation for writing this is simple: this was first-principles engineering work. If we later discuss this system with a professor, teammate, interviewer, or judge, we should be able to explain not just the final code, but the investigation itself.

# What Carried Forward Into The Current Off-Road Branch

The current no-GPS off-road branch kept one core lesson from this note:

- do not let a weak perception signal pretend to be a strong authority

That means:

- depth can support slowdown or bias
- semantics can support caution
- but the main system should still be built around teach-route evidence,
  relocalization stability, and terrain-aware control

# Initial Outdoor System and Motivation

The outdoor stack had already reached a partially working state. In its useful form, it consisted of:
- GPS waypoint navigation for global checkpoint direction,
- LogoNav as the primary learned local motion policy,
- startup hardening and mission-state handling,
- reactive recovery logic for stuck or wall-hit situations.

This version was described informally as "amazing" or "partially working" because it was already moving and making progress. However, it still had an important failure mode:
- it could still drift into trees,
- it could still approach bad road edges or shoulders,
- and it lacked an explicit local safety representation.

The core architectural gap was identified as:
\[
`camera + GPS -> direct command`
\]
instead of
\[
`camera + GPS -> local traversability estimate -> safer command`.
\]

# High-Level Hypothesis

The first practical hypothesis was:
> Use the existing monocular depth system as a lightweight local traversability layer, without rewriting the whole outdoor stack.

This was attractive because:
- a depth estimator already existed in the repo,
- a full semantic perception stack was not yet implemented,
- and the smallest safe improvement seemed to be adding a local obstacle signal on top of the already-working LogoNav path.

The intended design was:
1. estimate depth from the front RGB camera,
1. convert depth into angular clearances,
1. infer whether the forward path is clear,
1. and bias or veto motion if the scene appears blocked.

# What We Built

## Traversability Module

We added a dedicated module:
- `src/outdoor_traversability.py`

Its purpose was to transform a depth map into a short-horizon obstacle signal.

The main design features were:
- a middle-band crop, rather than a bottom-only crop,
- per-bin clearance estimation over a front field of view,
- a short temporal memory over several frames,
- a safe heading selection biased toward the GPS goal direction.

## Runtime Integration

We integrated the traversability layer into:
- `live_outdoor_runtime.py`

The integration was intentionally conservative:
- it was behind a flag, `--traversability`,
- it did not replace GPS or LogoNav,
- and later, after analysis, it was reduced to a soft steering bias only.

## Offline Analysis Scripts

To avoid changing runtime behavior blindly, we created offline tools:
- `scripts/calibrate_traversability.py`
- `scripts/sweep_trav.py`
- `scripts/probe_semantics.py`

These scripts were analysis-only. They changed no runtime behavior.

# Recorded Data Used for Analysis

We had four recorded outdoor datasets in HDF5 format:
- `test_outdoor_1.h5`
- `test_outdoor_2.h5`
- `test_outdoor_3.h5`
- `test_outdoor_4.h5`

These recordings contained at least:
- front camera frames,
- telemetry,
- GPS,
- orientation,
- accelerometer data,
- and additional logged channels.

The front frames were stored as encoded images and decoded offline for analysis.

# Phase 1: Why the Original Depth Safety Failed

## Original Assumption

Initially, the major suspicion was that depth safety failed because the crop was wrong. The old logic effectively looked at the bottom region of the image, which on this rover is heavily influenced by nearby ground.

This led to a reasonable early hypothesis:
> The model might be fine, but the crop region is reading the floor or ground instead of obstacle body.

## What We Tried

We built the traversability layer around a middle-band crop:
- crop top fraction around $15%$,
- crop bottom fraction around $60%$,
- in other words, rows intended to capture trunks, walls, bushes, and barriers, while ignoring sky and the nearest ground patch.

This was a good idea in principle, and it remained part of the final design.

# Phase 2: Offline Calibration Revealed a Deeper Problem

## Calibration Script

We then ran `scripts/calibrate_traversability.py` on recorded outdoor frames. This script:
1. loaded frames from the `test_outdoor` recordings,
1. ran `DepthEstimator`,
1. ran `OutdoorTraversability`,
1. summarized forward clearance and trigger rates.

## Initial Failure Mode

The first calibration result was alarming:
- the traversability layer effectively triggered stop on every frame,
- the depth estimates in all bands looked compressed into a very small range,
- open outdoor scenes looked like near obstacles.

At that stage, it looked as if monocular metric depth might simply be unusable on this camera.

# Phase 3: Root Cause --- Incorrect `max_depth`

## What We Realized

Further inspection showed that the depth model itself was not necessarily broken. Instead, the runtime was instantiating the metric-depth model with the wrong scale.

The critical fact was:
- the repo was using the `vkitti` outdoor metric checkpoint,
- but the estimator was being constructed with a small `max_depth`,
- compressing the model's intended output range.

The Depth Anything V2 metric implementation multiplies the model's prediction by `max_depth`. For `vkitti`, the correct scale is $80\text{ m}$, not $5\text{ m}$.

## Fix

We corrected the estimator so that:
- `vkitti` checkpoints infer $`max_depth` = 80.0$,
- `hypersim` checkpoints infer $`max_depth` = 20.0$,
- the runtime no longer hardcodes an incorrect `max_depth=5.0`.

## Observed Effect

After the fix:
- forward clearance on open terrain became plausible,
- the system no longer froze on every frame,
- and thresholds like obstacle $1.5\text{ m}$, slow $1.2\text{ m}$, stop $0.6\text{ m}$ looked at least numerically sane.

# Phase 4: Sweeping the Traversability Parameters

## Why a Sweep Was Necessary

Even after fixing `max_depth`, we still needed to answer:
- is the chosen image resolution sufficient?
- does a different percentile improve obstacle detection?
- do obstacle and open scenes separate enough for reliable use?

## What `sweep_trav.py` Tested

The parameter sweep script compared:
- resolutions $120 \times 160$ and $240 \times 320$,
- percentiles $p1, p5, p10, p25$,
- curated obstacle frames versus open frames.

## Key Findings

The sweep led to several important conclusions:
1. Resolution change did not materially improve the results.
1. Percentile choice changed values slightly, but did not fundamentally solve the problem.
1. Thin obstacles such as people remained effectively invisible to the metric depth system.
1. Broad vegetation or walls sometimes showed some separation from open terrain, but the gap was narrow.

This was one of the most important turning points in the investigation:
> The problem was no longer threshold tuning. The problem was that the depth modality itself did not robustly capture the obstacle types we care about.

# What We Learned About Metric Depth

The strongest conclusion from calibration and sweep work was:
- metric depth on this camera can provide a weak signal for broad scene structure,
- but it is not reliable enough for hard stop or slow decisions,
- especially for thin obstacles such as people, poles, and fine trunks.

This was a critical engineering insight. It prevented us from continuing to tune a fundamentally weak signal into a false sense of safety.

# Design Decision: Demote Traversability to a Soft Bias

Once the depth limitations were clear, the right move was not to delete the work, but to reduce its authority.

We changed the role of traversability from:
- stop or slow the robot,
- possibly hard-override steering,

to:
- a small angular bias only,
- used as a gentle nudge away from broad low-clearance regions,
- with no ability to stop the rover.

This was the safest compromise because:
- it preserved the already-working LogoNav behavior,
- it prevented catastrophic freezing from noisy depth,
- and it still allowed depth to contribute a weak broad-obstacle hint.

# Current Outdoor Stack After This Work

At the end of this phase, the outdoor system can be summarized as follows:

```text
\toprule
\textbf{Layer} & \textbf{Role} & \textbf{Status} \\
\midrule
GPS waypoints & Global checkpoint direction & Working \\
LogoNav & Primary local motion policy & Working \\
Depth traversability & Soft angular nudge only & Working in reduced role \\
Recovery logic & Reactive backup behavior & Working \\
\bottomrule
```

The important point is that depth no longer has the power to freeze the robot based on weak evidence.

# Phase 5: Offline Semantic Segmentation Prototype

## Why We Switched to Semantics

After the depth sweep, the next logical direction was semantic perception. The reasoning was:
- depth struggled with thin or semantically important obstacles,
- but a semantic model might detect categories like person, tree, plant, path, earth, pole, or wall,
- which are closer to the real decision boundary we care about.

## Research and Model Choice

We researched lightweight segmentation directions and concluded that a small ADE20K-style model was the best immediate offline prototype, because it already contains useful general scene labels.

The chosen model was:
- `nvidia/segformer-b0-finetuned-ade-512-512`

Relevant external references:
- SegFormer ADE20K model card: [https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)
- RELLIS-3D off-road dataset: [https://github.com/unmannedlab/RELLIS-3D](https://github.com/unmannedlab/RELLIS-3D)
- Depth Anything V2 metric-depth README in local repo, which informed earlier depth calibration.

## Why This Model Looked Promising

We specifically confirmed that the label space includes relevant categories such as:
- `tree`,
- `road`,
- `grass`,
- `sidewalk`,
- `person`,
- `earth`,
- `path`,
- `pole`,
- `animal`,
- `wall`,
- `plant`,
- `sky`.

This was enough to justify an offline test.

## Offline Probe

We created:
- `scripts/probe_semantics.py`

This script:
1. loads selected frames from the outdoor recordings,
1. runs the SegFormer model,
1. analyzes a forward region of interest,
1. computes coarse fractions of:
- drivable labels,
- obstacle labels,
- caution labels,

1. and optionally saves overlays.

# Semantic Probe Observations

The probe produced both encouraging and discouraging results.

## What Was Encouraging

The model did identify meaningful classes on the recorded data. For example:
- on a person-plus-dog frame, the forward region was mostly `path`, `tree`, and `earth`, with `person` also appearing,
- on a bushes frame, `plant` and `tree` were strongly represented,
- on a person-on-trail frame, `earth` and `tree` dominated, with `person` still appearing.

This showed that semantics is not a dead end. Unlike metric depth, it can at least expose the categories we care about.

## What Was Discouraging

However, one open-baseline frame was classified mostly as `plant/tree/grass`, with relatively little `earth/path`.

This meant:
- raw ADE20K segmentation is not plug-and-play for our camera and terrain,
- it can over-call vegetation,
- and a naive use of class fractions would produce false alarms.

## Interpretation

The semantic prototype was therefore:
- more promising than metric depth for semantically meaningful obstacles,
- but not ready to enter the runtime unchanged.

This is an important result. It means:
> The next likely good direction is semantic perception, but it still needs task-specific offline iteration before being trusted online.

# What We Researched

For completeness, the following themes were explicitly investigated or reasoned through during this work:
- metric monocular depth as a local safety layer,
- crop-band selection for obstacle-relevant image regions,
- percentile versus hard minimum in per-bin depth aggregation,
- short temporal memory over obstacle signals,
- whether metric depth could support hard stop/slow decisions,
- whether threshold tuning could salvage the depth signal,
- lightweight semantic segmentation models,
- suitability of ADE20K-style labels for off-road traversability,
- the possibility of later using accelerometer data for reactive collision detection.

# What We Did Not Yet Implement

Several ideas were considered but deliberately deferred:
- no semantic model was integrated into the runtime,
- no accelerometer-based collision detector was added,
- no major control rewrite was done,
- no planner or local costmap was added beyond the lightweight traversability exploration.

These were delayed intentionally in order to avoid uncontrolled changes before evidence was available.

# Main Lessons

The main lessons from this entire investigation are:
1. **Do not tune around a broken scale.** The wrong `max_depth` initially made the depth model look much worse than it really was.
1. **Offline replay is essential.** Without the recorded `.h5` data, we could easily have deployed unsafe logic or kept tuning a dead end.
1. **Weak signals should have weak authority.** Once we learned depth was unreliable for thin obstacles, it was correct to demote traversability to a soft bias.
1. **Category-level understanding matters.** The semantic probe showed why a label like `person` or `pole` is fundamentally more useful than a noisy metric threshold.
1. **The right answer is often narrowing scope, not adding more complexity.** Many possible changes were considered, but the safest path was to reduce authority and continue learning from offline analysis.

# Current Final Position

The current position after this work is:
- the outdoor runtime remains primarily driven by GPS plus LogoNav,
- depth-based traversability remains in the system only as a soft bias,
- metric depth should not be used as a hard safety layer on this camera,
- semantic segmentation is the most promising next research direction,
- and obstacle-positive data plus additional offline evaluation should precede any runtime semantic integration.

# Artifacts Produced

The following files and artifacts now exist and document this process:
- `scripts/calibrate_traversability.py`
- `scripts/sweep_trav.py`
- `scripts/probe_semantics.py`
- `scripts/trav_debug/`
- `scripts/semantic_debug/`

These are valuable because they preserve not only conclusions, but the actual experimental tooling that led to those conclusions.

# Why This Work Matters

This work is worth being proud of because it was not superficial tuning. It was an engineering investigation:
- a hypothesis was formed,
- tooling was written,
- the hypothesis was tested on real data,
- initial interpretations were corrected,
- a hidden root-cause bug was found,
- another limitation was exposed,
- and the system was redesigned to match the real evidence.

That is exactly the kind of process that serious robotics and autonomy work requires.

# Code-Verified Addendum: What Actually Entered The Runtime

The earlier sections of this review correctly describe the sequence of ideas, but the present repo state is now specific enough that a code-grounded addendum is useful.

## Perception Modules And Their Real Authority Levels

The current outdoor stack does not treat every perception signal equally. The easiest way to understand the present design is as an authority table.

```text
\toprule
\textbf{Module} & \textbf{Primary file} & \textbf{Runtime authority} & \textbf{Important limitation} \\
\midrule
Depth Anything V2 & \texttt{src/depth\_estimator.py} & provides metric depth only; not itself a motion policy & scale and geometry are only as good as the checkpoint and camera setup \\
Outdoor traversability & \texttt{src/outdoor\_traversability.py} & later upgraded to a hard local override in outdoor runtime & middle-band crop is still weak for curbs, drops, and some low obstacles \\
Semantic risk estimator & \texttt{src/semantic\_risk\_estimator.py} & provides hard stop, yield, sidewalk-stop, and weak bias hooks in runtime & still marked experimental and depends on label grouping quality \\
Vision safety monitor & \texttt{src/vision\_safety\_monitor.py} & hard image-quality gate in night-safe usage & it detects bad visibility, not scene semantics \\
\bottomrule
```

That table is important because it captures one of the main later design lessons: weak or noisy signals were demoted, while the signals that proved operationally useful were granted stronger but still bounded authority.

## Exact Traversability Parameters In The Current Code

The present traversability module uses the following default configuration:
```
crop_top_frac      = 0.15
crop_bot_frac      = 0.60
obstacle_distance_m = 1.5
stop_distance_m     = 0.60
slow_distance_m     = 1.20
slow_linear_min     = 0.35
forward_bin_half_window = 1
memory_frames       = 4
```

This matters for interpretation. The module is not looking at the whole frame. It is explicitly analyzing a middle image band to avoid the sky and the ground patch immediately under the rover. That design is good for trunks, walls, and barriers, but it also explains why curb-like or step-like hazards remain a documented weakness.

The compute path also reveals why traversability later became stronger in the runtime:
```
if fwd_clearance < stop_distance_m or all_blocked:
    linear_scale = 0.0
    angular_override = safe_heading
elif fwd_clearance < slow_distance_m:
    linear_scale = ...
    if fwd_blocked:
        angular_override = safe_heading
elif fwd_blocked:
    linear_scale = max(slow_linear_min, 0.70)
    angular_override = safe_heading
```

So the module was always capable of producing a strong local obstacle signal; the later runtime change was mainly about granting that signal more authority over LogoNav when the forward corridor was clearly blocked.

## Exact Night-Time Vision Gate Parameters

The current vision safety monitor is deliberately simple, but it is more specific than the earlier narrative summary might suggest. Its configuration is:
```
min_brightness = 42.0
max_dark_fraction = 0.65
max_glare_fraction = 0.12
min_texture_score = 8.0
consecutive_bad_ticks_to_stop = 3
consecutive_clear_ticks_to_reset = 1
```

The logic is also concrete:
- `too_dark` if mean brightness is low and the dark-pixel fraction is high,
- `glare` if bright saturation is high while texture is low,
- `low_detail` if the frame is low-texture and also fairly dark.

This means the night-time image gate is not a learned model at all. It is a deliberately conservative heuristic that asks a narrow question: is the frame visually trustworthy enough to continue using a visual controller?

## What The Semantic Layer Became After The First Probe

One thing that changed after the first round of semantic experiments is that the runtime no longer thinks of semantics only as a vague future idea. It now has an explicit runtime wrapper through `src/semantic_risk_estimator.py`, and the outdoor runtime can use that estimator in several ways:
- hard stop for person or animal alerts once risk confirms for enough ticks,
- yield behavior that caps linear speed,
- sidewalk-stop behavior if the center ROI becomes road-dominant with too little sidewalk/path evidence,
- and a weak angular bias path.

That said, the present code still treats these layers carefully. Even now, the semantic printouts in the runtime explicitly mark hard stop and sidewalk stop as experimental.

## Failure-Mode Analysis For The Present Perception Stack

The current perception stack only makes sense if its failure modes are stated plainly.

\paragraph{Metric depth as a standalone hard stop.}
This is still not trustworthy enough on this camera to be the only safety mechanism. The earlier analysis in this document remains correct: geometry alone missed semantically critical obstacles and could be distorted by camera geometry, scale choices, and texture.

\paragraph{Traversability as the only obstacle layer.}
Traversability is much more useful than the old bottom-band depth stop, but it is not a complete semantic safety system. It may steer away from a blocked forward corridor while still lacking a full understanding of curbs, stairs, or socially relevant hazards.

\paragraph{Semantics as the only local safety layer.}
Semantics sees categories that depth often misses, but it inherits the weaknesses of the label space and the ROI policy. False positives in rough vegetation-heavy scenes remain a real concern, which is why the present runtime still combines semantics with other layers rather than promoting it to sole authority.

\paragraph{Vision safety as an obstacle detector.}
That is not what it is. The night-time vision monitor protects against bad image quality; it does not classify obstacles and it does not understand where the path is.

## What The Later Outdoor Logs Forced Us To Admit

The later real-world logs pushed the perception interpretation beyond the original offline conclusions.

They showed that the main practical problems were not only "is the model good enough?" but also:
- whether the runtime gives the perception layer enough authority when the forward corridor is obviously bad,
- whether waypoint transitions can hand the controller a target that fights with local perception,
- whether rerouting and corridor logic interact sanely with what the camera sees,
- and whether the operator can tell, from the terminal, which layer is currently making the decision.

That is why the present outdoor perception story is no longer just about model quality. It is about perception authority, state transitions, and operator observability.

## The Honest Current Position

The best current reading of the repo is therefore more nuanced than either "depth failed" or "semantics solved it." The real position is:
- metric depth became useful only after scale correction and after being demoted from sole hard-stop logic,
- traversability became useful when it was allowed to act as a real local override rather than a weak hint,
- semantic segmentation became promising enough to justify real runtime hooks, but still requires careful gating and remains experimental,
- and night-time image quality needed its own explicit non-semantic safety gate.

That is the code-grounded state of the outdoor perception stack now.

# Related Documents

- [[erc3_full_documentation]] --- single master guide for the complete project story and current architecture.
- [[live_indoor_runtime_story]] --- indoor evolution, MBRA integration, and checkpoint-step runtime behavior.
- [[live_outdoor_ultra_marathon_story]] --- outdoor and marathon runtime evolution with safety-layer reasoning.
- [[outdoor_perception_review]] --- depth/semantic perception findings and their runtime implications.
