# ERC Indoor Literature Fact-Check Discoveries

> Source: `docs/discoveries.tex`
> Master Note: [[erc3_full_documentation]]

# What This Note Is

This is a direct fact-check of the current ERC indoor approach against related
robot navigation literature and the upstream MBRA materials.  The goal is not
to sound optimistic or pessimistic.  The goal is to answer one question:

\fbox{\parbox{0.9\textwidth}{
\textbf{Is our current approach fundamentally sensible, or are we doing
something structurally wrong?}
}}

# Short Answer

The current approach is **not fundamentally wrong**.  The main structure is
reasonable and is supported by the literature:

- visual place recognition / retrieval for localization,
- temporal filtering for consistency,
- a topological graph for planning,
- and a separate short-horizon local controller.

The part that still looks weak is **not the localization or graph idea**.
The weak part is the **local control and runtime recovery layer**.

# What Problem We Are Actually Solving

Before discussing papers, it helps to say the problem in very plain language.

We are **not** trying to build a robot that can walk into any random
building in the world and solve navigation from scratch.

We are trying to solve a much narrower and much more realistic problem:

- we know the corridor in advance,
- we can record it beforehand,
- we can store images from that corridor,
- and later the robot has to figure out where it is inside that same
remembered corridor and move to the correct place.

That is why our current backbone is based on **memory of the corridor**
rather than **building a new full map every time**.

# What Our Current Stack Means In Plain Language

The current stack can be explained as four simple questions.

## Question 1: "Where am I?"

This is the localization part.

The robot takes the current camera image and compares it against the stored
corridor images.  The system then asks:

*"Which stored image looks most like what I am seeing right now?"*

That is what visual place recognition does.

## Question 2: "Am I sure, or am I jumping around?"

This is the temporal filtering part.

Single-image matching can be noisy.  A corridor often contains repeated-looking
walls, doors, and floor tiles.  So one frame by itself can be misleading.

Temporal filtering fixes that by asking:

*"Does this new match make sense given where I was a moment ago?"*

So instead of trusting each frame independently, the system prefers location
estimates that are consistent over time.

## Question 3: "What small place should I go to next?"

This is the planning part.

Once the robot has a current place estimate, it does not try to jump mentally
all the way to the final goal in one step.  Instead, it uses the graph to find
a path and then chooses a **nearby subgoal**.

So the planner is not saying:

*"Drive magically to the final target."*

It is saying:

*"From where you are now, aim for this next small place."*

## Question 4: "How do I physically move toward that nearby place?"

This is the controller part.

This is also the weakest part right now.

Localization and planning can both be correct, but if the controller is crude,
the robot can still spin, hesitate, overshoot, or stop too often.  That is why
the controller keeps showing up as the current engineering bottleneck.

# Why This Approach Exists At All

It is easy to feel suspicious and think:

*"Why are we not just using SLAM or one smart learned model for everything?"*

The reason is that the current competition setting strongly favors a more
structured approach.

## Why not full monocular SLAM as the main backbone?

Because in a repetitive indoor corridor with a networked robot feed, monocular
SLAM is not automatically the cleanest or most reliable answer.  It can be very
sensitive to:

- low texture,
- repeated geometry,
- unstable exposure,
- motion blur,
- and compressed or delayed video.

Also, even if SLAM tracks local motion well, it still does not directly solve
the image-checkpoint or remembered-route problem as cleanly as a corridor memory
does.

## Why not let MBRA do everything?

Because MBRA is not meant to be the whole stack.

MBRA makes sense as a **short-horizon controller candidate**.  It does not
make sense as:

- the global localization system,
- the graph planner,
- the checkpoint manager,
- or the whole mission logic.

So the right role for MBRA is narrow and local, not global and everything-at-once.

# What We Already Know From Our Own Repo

Before even bringing in the papers, the repo behavior already told us
something important.

## What has looked strong

- offline retrieval against the corridor memory,
- temporal localization on held-out frames,
- live place estimates when the robot is in known corridor locations,
- and graph-based subgoal selection.

## What has looked weak

- the heuristic local controller,
- live motion execution,
- stop / relocalize / recover behavior,
- and full safety integration.

This distinction matters.  If localization were weak, then the whole approach
would be questionable.  But localization has been the strongest part, which is
why the papers mostly reinforced our current backbone rather than overturning it.

# What The Literature Says We Are Doing Right

## Known-route visual localization is a valid backbone

PlaceNav and related topological navigation work support the idea that a robot
can localize against a previously recorded traversal using visual place
recognition instead of forcing a full metric SLAM solution to be the whole
backbone.

This matches our current design:

- recorded corridor images become nodes,
- the live image is matched to that memory,
- temporal filtering prevents erratic jumps,
- planning operates over the resulting graph.

This is a good fit for a repeated indoor corridor where the route is known in
advance.

## Topological planning plus local execution is standard

RoboHop, PlaceNav, and the GNM / ViNT family all support the broader idea that
long-horizon navigation is easier to manage when broken into:

1. a global or topological planning layer, and
1. a separate short-horizon goal-reaching layer.

That means our decomposition

`localize -> plan path -> choose nearby subgoal -> execute`

is conceptually correct.

## IMU support is reasonable, but not as the main backbone

Visual-inertial teach-and-repeat work supports using inertial signals as a
supporting motion prior.  That means our recent addition of filtered heading,
gyro-based turn-rate hints, and weak RPM usage is sensible.

But the literature does *not* suggest that we should replace the visual
place backbone with IMU-only estimation indoors.  So the current
"vision-primary with IMU support" stance is reasonable.

# What Looks Weak Or Risky

## The current controller is much weaker than the localization stack

The main mismatch with recommended practice is the local controller.

Teach-and-repeat systems often use \textbf{image registration / heading
correction} against the stored route images, or a learned goal-image policy
that is explicitly trained for short-horizon execution.  Our current baseline
controller is still a hand-written heading-and-step heuristic.

That means:

- localization can be correct,
- planning can be correct,
- but execution can still look bad because the controller is crude.

This exactly matches what we observed in live tests: spinning, hesitation, and
unstable short-horizon behavior.

## MBRA should only be used locally

The upstream MBRA project page is actually very helpful here.  It says the
goal-image-conditioned MBRA policy is useful only for **short-horizon**
goal reaching, roughly up to a few meters, and then uses topological memory for
longer navigation.

That means our corrected interpretation is right:

- MBRA should not replace localization,
- MBRA should not replace graph planning,
- MBRA should only be a possible short-horizon controller.

So putting MBRA into the local-controller slot is the correct role.

## The map may currently be denser than it should be

The repeated-route literature typically assumes images indexed by traveled
distance or a deliberate teaching trajectory, not every near-duplicate frame as
an equally important node.  The MBRA project page also describes recording a
goal loop at 1 Hz for its topological memory.

That suggests a practical risk in our current pipeline:

- if the graph is built from too many near-duplicate frames,
- the controller sees tiny subgoal changes,
- and the planner/controller interface can become noisy.

This does not make the approach wrong.  It means we should probably be more
deliberate about keyframing and node spacing.

## Safety and recovery are mandatory, not optional

The literature consistently treats obstacle handling, relocalization, and
recovery as necessary companion systems.  A teach-and-repeat or topological
navigation method does not by itself solve dynamic obstacles or controller
failure.

So our separate safety and recovery layer is not "extra polish."  It is a
required part of a serious runtime.

# What This Means For Our Repo

## Good news

The following core decisions still look correct:

- use the corridor recording to build a place memory,
- localize by image retrieval plus temporal consistency,
- plan over a corridor graph,
- hand the controller a nearby subgoal instead of a far-away final target.

## Bad news

The main unresolved engineering task is still the local controller.

Right now the repo has:

- a strong localization/planning baseline,
- but only a baseline local controller.

That means we should stop worrying that the whole architecture is broken, and
instead focus on the real weak spot:

\fbox{\parbox{0.9\textwidth}{
**make short-horizon execution reliable**
}}

# The Best Corrected Stack

Based on the repo state and the literature review, the cleanest target stack is:

1. visual place-recognition localization over a pre-recorded corridor,
1. temporal filtering and motion-prior smoothing,
1. topological graph planning,
1. deliberate keyframed subgoals,
1. a short-horizon controller
- either improved image-based execution,
- or MBRA if properly installed and validated,

1. explicit stop / relocalize / recover logic,
1. explicit safety veto.

# Bottom Line

The architecture is broadly correct.

The main blunder would *not* be continuing with visual localization and
graph planning.  The main blunder would be pretending the current heuristic
controller is already good enough just because localization looks strong.

So the correct conclusion is:

- **keep** the localization + graph backbone,
- **tighten** keyframing / node spacing,
- **improve** the local controller,
- **integrate** safety and recovery properly,
- **treat MBRA only as a short-horizon controller candidate.**

# Reference Inventory

This section records both the sources used directly in the reasoning and the
related sources that were surfaced during the fact-check.

## Used Directly

1. **MBRA project page** \\
[https://model-base-reannotation.github.io/](https://model-base-reannotation.github.io/) \\
Used to confirm that MBRA is a short-horizon goal-image-conditioned policy and
that longer navigation uses topological memory.

1. **MBRA paper / OpenReview page** \\
[https://openreview.net/forum?id=9DyLaIHqrD](https://openreview.net/forum?id=9DyLaIHqrD) \\
Used to confirm the MBRA expert / LogoNav deployed-policy split.

1. **MBRA upstream GitHub README** \\
[https://github.com/NHirose/Learning-to-Drive-Anywhere-with-MBRA](https://github.com/NHirose/Learning-to-Drive-Anywhere-with-MBRA) \\
Used to confirm that upstream MBRA requires environment setup, dependencies,
and downloaded weights beyond just copying the repository tree.

1. **PlaceNav project page** \\
[https://lasuomela.github.io/placenav/](https://lasuomela.github.io/placenav/) \\
Used to support place-recognition-based subgoal selection and temporal
filtering for topological navigation.

1. **PlaceNav paper** \\
[https://arxiv.org/abs/2309.17260](https://arxiv.org/abs/2309.17260) \\
Used as the paper source behind the same design support as the project page.

1. **Teach-and-repeat feature matching paper** \\
[https://www.mdpi.com/1424-8220/22/8/2836](https://www.mdpi.com/1424-8220/22/8/2836) \\
Used to support repeated-route navigation, taught-route indexing by traveled
distance / dead-reckoning, and the claim that full global localization is not
always necessary in teach-and-repeat.

1. **Teach-and-repeat image registration paper** \\
[https://www.mdpi.com/1424-8220/22/8/2975](https://www.mdpi.com/1424-8220/22/8/2975) \\
Used to support the claim that local execution should ideally use image-based
alignment / heading correction, making our current heuristic controller look
weaker than the literature standard.

1. **Visual-inertial teach-and-repeat** \\
[https://www.sciencedirect.com/science/article/pii/S0921889020304176](https://www.sciencedirect.com/science/article/pii/S0921889020304176) \\
Used to support IMU / inertial signals as a sensible support layer for
repeated-route navigation.

1. **RoboHop project page** \\
[https://oravus.github.io/RoboHop/](https://oravus.github.io/RoboHop/) \\
Used to support topological planning as a legitimate long-horizon structure.

## Reviewed As Related Background

1. **General Navigation Models project page** \\
[https://general-navigation-models.github.io/](https://general-navigation-models.github.io/) \\
Reviewed to understand the broader Berkeley navigation model family and how
MBRA, ViNT, and NoMaD fit together conceptually.

1. **ViNT project page** \\
[https://general-navigation-models.github.io/vint/](https://general-navigation-models.github.io/vint/) \\
Reviewed as background on topological search plus local goal-reaching
policies.

1. **ViNT paper** \\
[https://proceedings.mlr.press/v229/shah23a.html](https://proceedings.mlr.press/v229/shah23a.html) \\
Reviewed as additional background on goal-conditioned local execution and
topological navigation decomposition.

## Surfaced During Search But Not Relied On

1. **SNAP** \\
[https://www.researchwithrutgers.com/en/publications/snapsuccessor-entropy-based-incremental-subgoal-discovery-for-ada/](https://www.researchwithrutgers.com/en/publications/snapsuccessor-entropy-based-incremental-subgoal-discovery-for-ada/) \\
Related to graph / subgoal decomposition, but less directly relevant to our
repeated-route corridor setting than PlaceNav or teach-and-repeat.

1. **Secondary index pages for PlaceNav and ViNT** \\
[https://huggingface.co/papers/2309.17260](https://huggingface.co/papers/2309.17260) \\
[https://trepo.tuni.fi/handle/10024/212114](https://trepo.tuni.fi/handle/10024/212114) \\
[https://huggingface.co/papers/2306.14846](https://huggingface.co/papers/2306.14846) \\
Useful search landing pages, but replaced by more direct project / paper
sources in the actual reasoning.

1. **Secondary index pages for MBRA** \\
[https://openreview.net/pdf/0e3e7ec127bc8d00f19c29cb08eb9dd4b6974527.pdf](https://openreview.net/pdf/0e3e7ec127bc8d00f19c29cb08eb9dd4b6974527.pdf) \\
[https://www.thejournal.club/c/paper/788055/](https://www.thejournal.club/c/paper/788055/) \\
[https://fugumt.com/fugumt/paper_check/2505.05592v2_enmode](https://fugumt.com/fugumt/paper_check/2505.05592v2_enmode) \\
Useful for discovery, but not used as the final authority in the note.

# Related Documents

- [[erc3_full_documentation]] --- single master guide for the complete project story and current architecture.
- [[live_indoor_runtime_story]] --- indoor evolution, MBRA integration, and checkpoint-step runtime behavior.
- [[live_outdoor_ultra_marathon_story]] --- outdoor and marathon runtime evolution with safety-layer reasoning.
- [[outdoor_perception_review]] --- depth/semantic perception findings and their runtime implications.
