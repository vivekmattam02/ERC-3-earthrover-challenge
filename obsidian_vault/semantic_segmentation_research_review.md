# Semantic Segmentation Research Review for ERC-3 Outdoor Navigation - Why We Chose It, How We Tested It, What We Observed, and Why It Is Promising but Not Yet Ready

> Source: `semantic_segmentation_research_review.tex`
> Master Note: [[erc3_full_documentation]]

# Purpose of This Document

This document is a focused review of the semantic-segmentation research carried out for the ERC-3 outdoor rover stack. It is intentionally narrower than the larger outdoor perception review. The purpose here is to capture, in one place, the full reasoning process behind the semantic-segmentation investigation:
- why semantic segmentation became necessary,
- how the candidate model was chosen,
- how the offline experiment was designed,
- how the forward region was filtered,
- how labels were grouped into drivable, obstacle, and caution categories,
- what the overlays and summaries showed on real recorded frames,
- what worked,
- what failed,
- and what the correct next step is.

The intended use of this document is explanation. It should be possible to discuss this work confidently with a professor, reviewer, interviewer, or teammate without needing to rediscover the engineering reasoning later.

# Why We Started Semantic Segmentation Research

Semantic segmentation was not the first perception direction we tried. The earlier local-safety investigation focused on metric monocular depth. That line of work produced an important conclusion:
> Metric depth on this camera was too weak for hard safety decisions, especially for thin or semantically important obstacles.

More specifically, the depth work showed the following:
- broad vegetation or walls sometimes produced a weak signal,
- thin obstacles such as people and other narrow objects were often effectively invisible,
- threshold tuning did not solve the problem,
- increasing depth resolution did not solve the problem,
- changing depth aggregation percentile did not solve the problem.

That created a clear gap. The rover still needed some way to reason about objects that matter semantically, not just geometrically. A person, dog, tree trunk, pole, bush boundary, or non-drivable patch are not defined only by raw distance. They are defined by category and context.

This is exactly why semantic segmentation became the next research direction.

# The Core Hypothesis

The semantic-segmentation hypothesis was:
> A lightweight scene-understanding model may detect meaningful obstacle classes on our outdoor footage, even when metric depth fails to provide a reliable stop or slow signal.

The hypothesis was attractive for several reasons:
- the recorded outdoor data already existed,
- the next step could be tested entirely offline,
- semantics might detect classes such as `person`, `tree`, `plant`, `wall`, or `pole`,
- and even an imperfect semantic signal might be more useful than raw depth for off-road decisions.

The important point is that this was not meant to go into the runtime immediately. The first goal was only to answer:
> Does semantic segmentation show enough promise on our actual recorded data to justify deeper investment?

# What We Needed From a Candidate Model

Before selecting a model, we clarified what would make a semantic-segmentation model useful for this rover.

The model did *not* need to be perfect or custom-trained on our exact terrain. For an initial offline probe, it needed to satisfy these requirements:
1. It had to be lightweight enough to run as an experiment without turning the environment setup into a major project.
1. It had to have a label space broad enough to include outdoor scene elements that matter for navigation.
1. It had to be good enough to show category-level distinctions on the recorded frames.
1. It had to support easy offline inference on RGB images extracted from the `.h5` recordings.

The classes we especially cared about were:
- drivable-like classes such as `road`, `path`, `earth`, `sidewalk`,
- obstacle-like classes such as `person`, `animal`, `pole`, `wall`,
- caution-like vegetation classes such as `tree`, `plant`, `grass`,
- and context classes like `sky` so that the model did not confuse horizon content with ground content.

# Why We Chose SegFormer-B0 on ADE20K

The model selected for the first offline probe was:
- `nvidia/segformer-b0-finetuned-ade-512-512`

This choice was deliberate.

## Why SegFormer-B0 Specifically

SegFormer-B0 was chosen because it is one of the smallest practical segmentation backbones in the common Hugging Face ecosystem. That made it appropriate for a first-pass offline investigation where the main question was whether the labels were informative at all.

The benefits of this choice were:
- relatively lightweight model size,
- standard tooling through `transformers`,
- easy inference on still images,
- and a mature ADE20K label set.

## Why ADE20K Was Good Enough for a First Probe

ADE20K is not an off-road rover dataset. That limitation was known from the beginning. However, it still had one major advantage for exploratory work: the label inventory already includes many categories that are meaningful for this task.

We specifically verified that the label space includes categories such as:
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

That was enough to justify an offline probe. The goal was not to prove ADE20K was the final answer. The goal was to test whether a general semantic model could reveal obstacle categories that the depth-based approach had missed.

Relevant references:
- SegFormer ADE20K model card: [https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)
- RELLIS-3D off-road dataset reference: [https://github.com/unmannedlab/RELLIS-3D](https://github.com/unmannedlab/RELLIS-3D)

# Why We Did Not Put It Into the Runtime

The semantic work was intentionally done offline only. This was important for two reasons:
- the outdoor stack already had a partially working path through GPS plus LogoNav,
- and we had explicitly agreed not to introduce risky runtime changes without evidence.

So the semantic work followed the same philosophy as the calibration scripts:
> Research first, measure first, then decide whether runtime integration is justified.

# Implementation of the Offline Probe

The offline semantic prototype was implemented in:
- `scripts/probe_semantics.py`

This script performs the following steps:
1. load selected frames from recorded outdoor `.h5` files,
1. decode the frame as RGB,
1. run the SegFormer model,
1. upsample the segmentation map back to the frame resolution,
1. define a forward region of interest,
1. summarize semantic label fractions inside that region,
1. save color overlays for visual inspection.

## Why We Used Curated Frames

Instead of running blindly over thousands of frames first, we deliberately selected a small set of representative scenes. This was the right choice for early-stage analysis because it makes visual reasoning possible.

The curated frames were:
- `test_outdoor_4.h5 #1696`: person plus dog on the path,
- `test_outdoor_2.h5 #272`: person walking ahead on the trail,
- `test_outdoor_4.h5 #848`: bushes or vegetation encroaching from both sides,
- `test_outdoor_3.h5 #0`: tree trunk or vegetation close on the left,
- `test_outdoor_1.h5 #0`: open baseline frame.

This set was intentionally mixed:
- some frames contained obvious obstacles,
- one frame was intended as a baseline open scene,
- and together they gave us a fast way to judge whether the model was responding to meaningful categories.

# How We Filtered the Image: The Forward ROI

One of the most important design choices was the region of interest. We did not want to summarize the whole frame, because the rover does not need a single global semantic label. It needs to know what is happening *ahead of it*.

The first ROI used in the script was:
- rows from about $35%$ to $90%$ of image height,
- columns from about $25%$ to $75%$ of image width.

In code, this was:
- top = $0.35h$
- bottom = $0.90h$
- left = $0.25w$
- right = $0.75w$

## Why This ROI Initially Made Sense

This forward window was chosen because it tries to exclude the least useful parts of the frame:
- not too much sky,
- not too much far peripheral background,
- but still enough lower image area to capture the actual terrain corridor.

It was a sensible first choice for a low-mounted forward-facing camera.

## Why We Later Realized the ROI Was Also a Source of Error

After inspecting the overlays, an important issue became clear:
- the lower portion of the image contains a lot of rough ground texture,
- on unpaved off-road terrain, this ground does not always look like ADE20K's concept of `road` or `path`,
- and the wide forward ROI includes a lot of side vegetation and ground clutter.

This matters because the semantic score is only as good as the pixels being counted. If the ROI is too wide or too low, the model can be penalized for things that the rover would actually drive over or safely ignore.

This was one of the key lessons from the semantic experiment.

# How We Grouped Labels

The semantic experiment did not use raw ADE20K labels directly. Instead, we grouped labels into categories that were easier to reason about for navigation.

The initial grouping in the script was:

## Drivable Labels

- `road`
- `earth`
- `path`
- `sidewalk`

## Obstacle Labels

- `person`
- `animal`
- `pole`
- `wall`
- `tree`
- `plant`
- `grass`

## Caution Labels

- `grass`
- `plant`
- `tree`

## Why This Grouping Was Chosen

This taxonomy was intentionally conservative. At the start, the idea was to avoid underestimating risk. If vegetation was dominating the forward corridor, that was likely worth noticing.

This grouping also had another purpose: it gave us a quick way to compute three coarse fractions in the ROI:
- drivable fraction,
- obstacle fraction,
- caution fraction.

That let us ask the first practical question:
> Does the semantic model produce a visibly meaningful signal in the direction the rover cares about?

## Why This Grouping Was Later Critiqued

After inspecting the results, it became clear that this grouping was too harsh for off-road terrain.

The biggest issue was `grass`. In the recorded environment, the rover often drives over grass, dirt, or mixed rough ground. Treating `grass` as an obstacle class was therefore too strong.

Likewise, some `plant` or low vegetation pixels may describe scene texture rather than an actual no-go barrier.

This led to an important correction:
> The semantic label grouping must reflect *our terrain and vehicle behavior*, not just the label names in the benchmark dataset.

This was one of the main reasons the first semantic pass was judged promising but not runtime-ready.

# What the Overlays Showed

The saved overlays in `scripts/semantic_debug/` were extremely important. They allowed visual inspection instead of relying only on fractions.

## Person Plus Dog on the Path

This was one of the strongest positive results.

What we saw:
- the person and dog were visually segmented in the obstacle color,
- the placement was correct on the path,
- the surrounding path remained interpretable as drivable terrain,
- and this was exactly the kind of scene where metric depth had failed badly by reading the target as very far away.

Why this mattered:
- it proved that semantics could reveal a category-level hazard that depth had essentially missed,
- and it showed that even a lightweight model can add information the geometric stack does not have.

## Person on Trail

This was another encouraging frame.

What we saw:
- the person was smaller in the image,
- but the model still produced a visible obstacle response,
- and this was again better than the metric-depth path for thin objects.

Why this mattered:
- it suggested that people may be detectable even when they only occupy a small part of the forward corridor,
- which is valuable because person detection is one of the most safety-critical semantic capabilities we care about.

## Bushes or Vegetation on Both Sides

This frame showed another strength of semantics.

What we saw:
- vegetation boundaries were visually clear,
- the path-like central corridor was still identifiable,
- and the geometry of the scene made intuitive sense in the overlay.

Why this mattered:
- it suggested that semantics might be good at defining corridor boundaries,
- which is useful for staying centered and not drifting into bushes or tree lines.

## Tree Left Close

This frame was also encouraging.

What we saw:
- the tree and vegetation on the left appeared as caution-like content,
- the forward path still looked reasonably drivable,
- and the overlay aligned with common-sense scene interpretation.

Why this mattered:
- it supported the idea that semantics could be useful as a corridor-selection tool,
- not only as a binary obstacle detector.

# The Most Important Failure: The Open Baseline Frame

The open baseline frame was the critical negative example.

This frame was supposed to act as a sanity check: a scene that should look mostly safe or at least not obviously blocked.

Instead, the forward ROI was dominated by labels such as:
- `plant`,
- `tree`,
- `grass`,

with relatively little strong drivable labeling.

This was the main reason the first semantic pass was judged not runtime-ready.

## Why This Failure Happened

The failure was informative rather than random. The likely causes were:
- the terrain is unpaved and visually rough,
- ADE20K was trained on general scenes rather than our specific off-road driving distribution,
- the bottom and side portions of the ROI contain rough grass-dirt texture,
- and the label grouping treated too much vegetation-like content as obstacle content.

So the problem was not that semantic segmentation was useless. The problem was that the first mapping from labels to navigation meaning was too naive for this environment.

# Quantitative Summaries From the First Probe

The semantic probe printed coarse summary fractions for the forward ROI. Representative outputs included:

```text
\toprule
\textbf{Frame} & \textbf{Scene} & \textbf{Drivable} & \textbf{Obstacle} & \textbf{Caution} \\
\midrule
\texttt{outdoor\_4 \#1696} & person + dog on path & 0.61 & 0.30 & 0.28 \\
\texttt{outdoor\_2 \#272} & person on trail & 0.69 & 0.17 & 0.16 \\
\texttt{outdoor\_4 \#848} & bushes both sides & 0.54 & 0.41 & 0.41 \\
\texttt{outdoor\_3 \#0} & tree left close & 0.76 & 0.17 & 0.17 \\
\texttt{outdoor\_1 \#0} & open baseline & 0.11 & 0.83 & 0.83 \\
\bottomrule
```

The individual top labels were also informative. For example:
- `person + dog on path`: mostly `path`, `tree`, `sky`, `earth`, with `person` also present,
- `person on trail`: mostly `earth`, `tree`, `sky`, with `person` also present,
- `open baseline`: mostly `plant`, `tree`, `earth`, `grass`, which exposed the false-positive issue.

These outputs were enough to support both sides of the conclusion:
- the model is genuinely seeing meaningful classes,
- but the first-pass decision logic would overreact on off-road baseline terrain.

# Why This Was Still a Good Result

It is important to interpret the semantic results correctly. The first pass did *not* fail in the same way the early depth pass failed.

The depth pass failed because the modality did not support the required safety behavior on this camera. The semantic pass, by contrast, revealed useful object categories and corridor structure. Its problem was not lack of signal. Its problem was the mapping from raw labels to rover decisions.

That is a much better place to be.

In other words:
> Semantic segmentation looked promising enough to continue, but not clean enough to deploy without more offline task-specific filtering.

# What We Learned From Claude's Review of the Overlays

A second human-style inspection of the overlays produced several good observations that matched the raw outputs.

## Positive Observations

The following points were especially important:
- In the person-plus-dog frame, the obstacle blobs on the person and dog were clearly visible and properly located on the path.
- In the person-on-trail frame, the person was still picked up despite occupying a smaller part of the image.
- In the bushes frame, the vegetation boundaries were clean and the drivable corridor was visually obvious.
- In the tree-left-close frame, the tree and side vegetation were captured while the forward path remained interpretable.

These points matter because they reinforce the claim that semantics is doing something depth could not do reliably.

## Critical Negative Observation

The open baseline frame remained the decisive problem.

The interpretation was:
- on rough dirt and grass terrain, ADE20K often describes open drivable ground as vegetation-like content,
- which means a naive obstacle fraction would produce false positives,
- and therefore the first semantic rule set cannot be trusted directly.

This matched our own interpretation and strengthened confidence in the conclusion.

# What We Concluded About the ROI

One of the clearest next-step lessons was that ROI design matters a great deal.

The first ROI was a reasonable starting point, but likely too broad and too low. The bottom area of a low camera sees ground texture that is visually noisy and not always meaningfully related to imminent collision risk.

So the next iteration should likely test:
- a narrower forward corridor,
- a slightly higher bottom cutoff,
- and perhaps separate center-lane scoring from side-vegetation scoring.

This is a good example of why semantic integration should not be rushed. The model output alone is not enough. The spatial filtering policy matters just as much.

# What We Concluded About Label Grouping

The second major lesson was that label grouping must match the rover's real operating environment.

The initial grouping was useful as a conservative first pass, but not as a final design. In particular:
- `grass` should probably not be treated as a hard obstacle class in this environment,
- `earth` and rough off-road ground deserve stronger positive weight,
- `tree` and `plant` may be better treated as caution than direct obstacle,
- `person`, `animal`, `pole`, and `wall` should remain high-importance hazard classes.

This strongly suggests that the next output should not be a raw obstacle fraction. It should be a task-specific semantic risk score.

# Why a Semantic Risk Score Is the Right Next Offline Step

The first pass used coarse fractions. That was appropriate for exploration, but not ideal for action.

A better next-stage offline summary would likely be something like:
- high weight for `person`,
- high weight for `animal`, `pole`, and `wall`,
- medium weight for dense `tree`/`plant` occupancy in the center corridor,
- negative or offsetting weight for strongly drivable classes,
- separate handling for side vegetation versus center obstruction.

The key point is that this would still be an offline research tool first. It would allow the semantic output to be judged in rover terms, not just benchmark-label terms.

# Why This Research Was Worth Doing

This semantic investigation was valuable even though it did not immediately produce a runtime-ready module.

It gave us the following concrete gains:
1. It established that semantics can detect people and scene categories that metric depth missed.
1. It showed that broad corridor structure is visible in the segmentation overlays.
1. It exposed the mismatch between benchmark semantics and off-road drivable terrain.
1. It showed exactly where the next iteration should focus: ROI refinement, label regrouping, and risk scoring.
1. It kept all of this work offline, so the current runtime was not destabilized.

This is not a failed experiment. It is a useful narrowing of the problem.

# Final Position After the First Semantic Pass

The final conclusion after this semantic-segmentation research pass is:
- semantic segmentation is more promising than metric depth for category-level obstacle understanding on this rover,
- the chosen lightweight SegFormer model was a good first probe,
- the model output is visibly meaningful on several obstacle frames,
- but the first ROI and first label grouping are too naive for direct runtime use,
- and therefore the semantic module is promising but not runtime-ready.

The correct next step is not to drop semantics. The correct next step is to perform a second offline pass with:
- refined ROI design,
- better off-road label grouping,
- and a semantic risk score instead of raw fractions.

# Artifacts Produced

The main artifacts from this semantic research pass are:
- `scripts/probe_semantics.py`
- `scripts/semantic_debug/`
- the overlay images saved for the curated frames
- this document

These artifacts matter because they preserve the real evidence and the exact decision path, not just the final opinion.

# Code-Verified Addendum: What The First Probe Became In The Current Repo

The earlier sections of this report correctly explain the first offline semantic investigation. Since then, however, the repo has gone one step further: the offline ideas were distilled into a runtime-oriented semantic risk estimator. That later estimator is worth documenting explicitly because it also corrects a few first-pass intuitions.

## The Current Runtime Label Grouping Is More Refined Than The First Probe

The current `src/semantic_risk_estimator.py` no longer uses the original coarse "drivable / caution / obstacle" grouping exactly as the first probe described it. The current runtime grouping is:
```
DRIVABLE_LABELS = {"road", "earth", "path", "sidewalk", "dirt_track"}
NEUTRAL_LABELS  = {"grass", "field"}
HAZARD_LABELS   = {"person", "animal", "pole", "wall", "fence"}
CAUTION_LABELS  = {"tree", "plant"}
IGNORE_LABELS   = {"sky"}
```

The important correction here is that `grass` is no longer treated as a direct hazard. It is treated as neutral. This is exactly the kind of off-road adjustment that the first-pass report argued would be necessary.

## The Runtime ROI Is Raised And Tightened

The runtime estimator also uses a more conservative ROI than the first offline summary. The present fractions are:
```
roi_top_frac    = 0.40
roi_bottom_frac = 0.80
roi_left_frac   = 0.30
roi_right_frac  = 0.70
```

This is a narrower and slightly higher corridor than the first probe described. That change reflects the later realization that a low-mounted rover camera sees too much noisy ground texture if the ROI is too broad and too low.

The estimator also explicitly builds:
- a full ROI mask,
- a center mask,
- a left-half mask,
- and a right-half mask.

That spatial decomposition is what later made side-sensitive bias possible.

## The Actual Hard-Alert Thresholds Are Small But Explicit

The present code uses the following alert thresholds:
```
person_thresh  = 0.002
animal_thresh  = 0.002
pole_thresh    = 0.002
wall_thresh    = 0.010
drive_thresh   = 0.10
caution_thresh = 0.60
max_bias       = 0.50
```

These numbers are worth writing down because they reveal two important design choices.

First, person and animal detection are intentionally sensitive. The threshold is only 0.2% of the center ROI.

Second, wall-like evidence is treated more conservatively than person/animal/pole evidence, which makes sense for rough outdoor scenes where vegetation and large background structures can otherwise dominate too easily.

## The Current Risk Score Is Not A Raw Obstacle Fraction

One of the biggest later improvements is that the runtime estimator does not use a naive raw obstacle fraction. The relevant code is:
```
if person > self.person_thresh:
    score += 0.55 + 18.0 * person
    alerts.append("person")
if animal > self.animal_thresh:
    score += 0.55 + 18.0 * animal
    alerts.append("animal")
if pole > self.pole_thresh:
    score += 0.35 + 10.0 * pole
    alerts.append("pole")
if wall > self.wall_thresh:
    score += 0.30 + 8.0 * wall
    alerts.append("wall")

vegetation_blocked = drivable_center < self.drive_thresh \
                     and caution_center > self.caution_thresh
if vegetation_blocked:
    score += 0.45 + 0.50 * max(0.0, caution_center - self.caution_thresh)
```

This is exactly the kind of task-specific semantic risk score that the first version of this document said would eventually be needed.

## The Bias Computation Is Also More Structured Than The First Story Suggested

The side-bias logic is not arbitrary. The present estimator computes a side "free score" as:
```
free = drivable + 0.30 * neutral
free -= 0.60 * caution
if hard_mode:
    free -= 4.0 * (person + animal)
    free -= 3.0 * (pole + wall)
```

Then the left-right difference is normalized and clipped to `max_bias = 0.50`. This matters because it means the semantic estimator is not only saying "stop" or "go." It is also trying to answer a directional question: which side of the corridor appears more usable?

## The Runtime Now Uses Semantics In Three Distinct Ways

The outdoor runtime currently contains three separate semantic gates:
1. **Semantic hard stop**: stops when person or animal alerts persist strongly enough.
1. **Semantic yield**: caps linear speed for uncertain or socially sensitive scenes.
1. **Semantic sidewalk stop**: stops when the center corridor becomes too road-like with too little sidewalk/path evidence.

The actual runtime logic is explicit:
```
_semantic_stop_active = bool(_alerts & {"person", "animal"}) \
    and last_sem_result.risk_score >= args.semantic_stop_risk

_yield_active = (
    last_sem_result.risk_score >= args.semantic_yield_risk
    or last_sem_result.person_center > 0.0
    or last_sem_result.animal_center > 0.0
    or (last_sem_result.road_center >= 0.25
        and (last_sem_result.sidewalk_center + last_sem_result.path_center) <= 0.20)
)

_unsafe_sidewalk = _road_like >= args.semantic_road_dominance \
    and _sidewalk_like <= args.semantic_sidewalk_min
```

So the semantic layer has clearly moved beyond pure offline curiosity. At the same time, the runtime still labels some of these paths as experimental, which is the right level of honesty.

## What This Means For The Original Conclusions

The original conclusion of this document remains directionally correct, but the later code lets us say it more precisely.

The semantic investigation did not produce a perfect runtime-ready perception module immediately. What it produced was:
- a corrected ROI,
- a corrected label grouping,
- a task-specific risk score,
- a side-sensitive bias computation,
- and several runtime gates that can now be exercised conservatively.

That is a stronger result than the first-pass document could honestly claim at the time.

## The Remaining Weaknesses Are Still Real

Even after this later integration work, the original caution remains valid.

The semantic system still depends on:
- benchmark-trained labels that are not perfectly matched to off-road or rough-sidewalk rover terrain,
- ROI design choices that may still need more tuning under different camera angles,
- and operator trust in a model family that has not yet been validated across every real event condition.

That is why the current semantic runtime should be understood as a carefully gated support layer, not a proof that semantics is "solved."

# Related Documents

- [[erc3_full_documentation]] --- single master guide for the complete project story and current architecture.
- [[live_indoor_runtime_story]] --- indoor evolution, MBRA integration, and checkpoint-step runtime behavior.
- [[live_outdoor_ultra_marathon_story]] --- outdoor and marathon runtime evolution with safety-layer reasoning.
- [[outdoor_perception_review]] --- depth/semantic perception findings and their runtime implications.
