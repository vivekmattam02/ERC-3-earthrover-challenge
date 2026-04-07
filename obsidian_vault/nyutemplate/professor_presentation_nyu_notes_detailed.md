# NYU Presentation Notes: Detailed Technical Version

> Source: `nyutemplate/professor_presentation_nyu_notes_detailed.tex`
> Master Note: [[erc3_full_documentation]]

\thispagestyle{empty}

# How To Use This Version

This is the deeper companion to the simpler note file.

Use this if:
- you want more technical confidence while presenting,
- your professor starts asking system-level questions,
- you want slightly richer wording than the lightweight notes.

Do not try to say all of this out loud. The slide deck is intentionally simple. This file is for:
- backup technical language,
- cleaner explanations,
- follow-up questions.

Main idea:
- the slides are simple,
- these notes are the technical layer underneath them.

# Slide 1: Title

**Short spoken version**

This is a short update on the ERC-3 EarthRover project. I will focus on indoor, outdoor, and marathon, and I will keep it to what we chose, why we chose it, what worked, and what failed.

**Technical framing**

The cleanest way to present the project is not as one monolithic autonomy stack. The indoor and outdoor parts evolved into different engineering problems with different failure modes, different assumptions, and different evaluation criteria.

If you want one concise technical sentence:

The most important systems-level insight was that indoor and outdoor needed different runtime mental models.

# Slide 2: Indoor

**Short spoken version**

For indoor, we chose MBRA as the main controller and used exact checkpoint-step navigation, because indoor turned out to be a known corridor problem rather than a generic exploration problem.

**Technical explanation**

Indoor became much clearer once we stopped framing it as generic navigation and instead treated it as:
- visual localization against a known corridor dataset,
- temporal stabilization of localization,
- graph progression over corridor steps,
- local short-horizon control toward the next subgoal.

That means the core structure is not:
- explore freely,
- detect arbitrary goals,
- do generic global planning.

It is:
- determine where we are in the corridor,
- determine which step or checkpoint comes next,
- generate a forward subgoal,
- use MBRA to execute the short local segment.

**Why MBRA made sense**

MBRA was useful once it was treated as the local controller rather than the whole intelligence of the stack. It worked better when:
- the target was clear,
- the runtime was simpler,
- reverse-heavy recovery was removed,
- stale context could be reset.

**Important technical points to remember**
- exact checkpoint-step targets were cleaner than vague image goals,
- MBRA is a short-horizon image-goal controller,
- indoor structure comes from the corridor graph, not from MBRA alone.

**One strong sentence if asked**

The indoor improvement was not mostly about inventing a better controller. It was about respecting the corridor structure and using MBRA in the right role.

# Slide 3: Indoor Results

**Short spoken version**

For indoor, we reached 8 out of 11 checkpoints. That was our strongest indoor result, and it also made the system much more understandable.

**Technical interpretation**

The number 8 / 11 matters, but the more important engineering result is that the indoor failure modes became legible. The big ones were:
- forward-only graph behavior,
- stale localization context,
- poor recovery logic for MBRA.

**Forward-only graph explanation**

One of the most important indoor realizations was that the corridor graph behaves like a forward-only graph in practice. That means:
- if the rover localizes past a target step,
- the planner may correctly report there is no path backward.

This looks like planner failure, but it is really a runtime interpretation problem.

That is why skip-past-checkpoint logic matters.

**Stale context explanation**

MBRA can become trapped if the visual context stays stale while the rover makes little or no real progress. That is why the runtime now benefits from:
- no reverse for MBRA,
- no-progress reset behavior,
- clearer controller-specific defaults.

**What improved**
- exact checkpoint-step mode,
- cleaner MBRA-first framing,
- more sensible recovery assumptions.

**What still remains**
- stronger repeatability,
- cleaner handling of edge cases,
- more confidence that good runs are not fragile.

**If asked why not 11 / 11**

The main remaining gap is repeatability and edge-case handling, not lack of a direction. We now understand the failure modes much better.

# Slide 4: Outdoor

**Short spoken version**

For outdoor, we kept LogoNav as the main controller, used OSM route expansion, and added safety layers around the runtime. We chose this because outdoor already had a usable runtime base. The bigger problem was not replacing the controller. The bigger problem was making the runtime safer and less brittle.

**Technical explanation**

Outdoor already had important pieces:
- mission checkpoints from the SDK,
- resume logic,
- LogoNav as the main learned controller,
- optional OSM routing,
- perception support hooks.

So the highest-value engineering work was not:
- throw everything away,
- replace the main controller,
- restart the stack from zero.

It was:
- harden the runtime,
- control route behavior,
- make waypoint progression saner,
- make field failures visible and recoverable.

**Why LogoNav remained the choice**

LogoNav remained the practical local controller because the main engineering bottleneck outdoors was not "there is no controller." It was:
- target transition instability,
- route-following discipline,
- safety around mission progression,
- local obstacle interactions.

**Important technical phrase**

Outdoor progress came more from runtime hardening than controller replacement.

# Slide 5: Outdoor Results

**Short spoken version**

For outdoor, one run reached all checkpoints, which was our strongest success. The other runs were around a 50 percent success rate overall, and three rounds failed.

**How to explain this technically**

This means the outdoor stack is clearly capable of end-to-end success, but it is not yet robust.

That is an important distinction:
- capability exists,
- stability is incomplete.

**What worked**
- one full checkpoint run proves the pipeline can complete,
- rerouting improvements helped,
- waypoint handling improved,
- logging became much more useful for debugging.

**What failed**

The outdoor failures were strongly tied to runtime behavior:
- spinning,
- poor waypoint transitions,
- inconsistent field behavior,
- sensitivity to real geometry and low obstacles.

**Why outdoor is harder than indoor**

Outdoor adds:
- live GPS uncertainty,
- route expansion,
- transition handling between mission and routed targets,
- real curb and step geometry,
- changing lighting and scene conditions.

**If asked what 50 percent success really means**

It means the outdoor system is partially successful and clearly improved, but I would still describe it as supervised and unstable under some conditions.

**If asked what the biggest outdoor weakness is**

Low-lying hazards and runtime stability around waypoint transitions.

# Slide 6: Marathon

**Short spoken version**

For the marathon, we kept the same outdoor runtime and made it stricter. We added stronger safety logic because the marathon problem is really about safe supervised completion, not just reaching checkpoints under ideal conditions.

**What we added technically**
- IMU safety,
- route corridor guard,
- rerouting from live GPS,
- waypoint handling improvements,
- clearer operator logs.

**Key point**

The marathon work was not about inventing a new algorithm. It was about wrapping the existing runtime in stronger discipline.

**If asked why this matters**

Because for a marathon-style task, the runtime envelope matters as much as the controller. A controller that works sometimes is not enough if the surrounding mission logic can still enter unstable states.

# Slide 7: Marathon Result And Failure Reason

**Short spoken version**

In the marathon run, the robot reached one checkpoint, but after that it started spinning and toppled.

**Best technical explanation**

The most useful explanation is:
- after checkpoint progress,
- the runtime likely became unstable around the next target transition,
- this created repeated turning or align behavior,
- repeated aggressive turning led to physical instability.

**Why this is the cleanest explanation**

It is specific enough to be meaningful, but not so speculative that it overclaims certainty.

**What not to say**

Avoid saying:
- it was definitely one exact line of code,
- it was definitely the model only,
- it was definitely the hardware only.

The more honest answer is:

This looks like a runtime transition and stability failure after checkpoint progress.

**Deeper technical wording if asked**

My current understanding is that the post-checkpoint target interpretation became poor enough to create repeated alignment behavior, and once turning became too persistent, the platform became physically unstable.

**Main marathon lesson**

The marathon exposed that:
- reaching checkpoints is not enough,
- the runtime must remain stable after checkpoint transitions.

That is why the marathon should be described as a runtime-discipline problem, not only a navigation problem.

# Slide 8: Final Summary

**Short spoken version**

So the short summary is this: for indoor, MBRA-first is our best direction and we reached 8 out of 11 checkpoints. For outdoor, LogoNav plus route and safety layers is still our best direction, with one full success and the others around 50 percent. For marathon, the main thing we still need to solve is stability after checkpoint and waypoint transitions.

**Strong closing sentence**

The project is better not because everything is solved, but because the remaining problems are now much clearer.

**Alternative closing sentence**

The main value of this work is that the system is now much better understood, and the failures are much less mysterious.

# Follow-Up Questions You May Get

**Why MBRA indoors?**

Because indoor has strong corridor structure already. MBRA works best as the short-horizon controller once exact checkpoint-step targets and graph progression define the overall motion structure.

**Why LogoNav outdoors?**

Because the outdoor stack already had a usable controller base, and the larger engineering problem was mission-runtime stability and safety, not absence of a controller.

**Why not unify indoor and outdoor?**

Because the assumptions are too different. Indoor depends on corridor localization and step progression. Outdoor depends on mission transitions, GPS, route expansion, and safety wrappers.

**What was the biggest indoor systems insight?**

That the graph is effectively forward-only in practice, so runtime logic must handle that instead of blaming the planner.

**What was the biggest outdoor systems insight?**

That runtime behavior around waypoint and checkpoint transitions matters as much as the controller itself.

**What is the biggest unresolved issue?**

Low-lying hazards outdoors and stable behavior after target transitions.

**What improved the project most?**

Not any one model. The biggest improvement was understanding the problem correctly and using real failures to harden the runtime.

# Numbers To Remember

- indoor: 8 / 11
- outdoor best run: all checkpoints
- outdoor overall: around 50%
- outdoor failed rounds: 3
- marathon: 1 checkpoint, then toppled

# Final Advice

- keep the main talk simple,
- use these notes only when you need more depth,
- do not try to prove too much,
- explain choices and failures clearly.

When in doubt, say the honest systems version:

We improved the runtime by understanding where it was getting confused.
