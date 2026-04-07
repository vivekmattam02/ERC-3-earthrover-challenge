# NYU Presentation Speaker Notes

> Source: `nyutemplate/professor_presentation_nyu_speaker_notes.tex`
> Master Note: [[erc3_full_documentation]]

\thispagestyle{empty}

# How To Use This

This is the short speaking version for a 5--7 minute presentation.

Use it like this:
- say the short line for each slide,
- add one technical sentence only if needed,
- answer questions using the backup lines below,
- do not drift into extra implementation detail.

Main structure:
- indoor
- outdoor
- marathon
- final summary

The overall story is simple:
- indoor is MBRA plus corridor steps,
- outdoor is LogoNav plus OSM-expanded waypoints and safety layers,
- marathon was where the runtime became too unstable after checkpoint transition.

# Slide 1: Title

**Say this**

This is a short update on the ERC-3 EarthRover project. I will keep it simple and focus on three parts: indoor, outdoor, and marathon. For each one, I will say what we chose, why we chose it, what worked, and what failed.

**If you want one more sentence**

The main point is that we learned the right role for each controller and where the runtime still breaks.

# Slide 2: Indoor

**Say this**

For indoor, we chose MBRA as the main controller and used exact checkpoint-step navigation. Indoor turned out to be a corridor problem, not a generic exploration problem, so a short-horizon controller on top of known steps made more sense.

**What that means technically**
- MBRA is the short-horizon local controller, not the planner.
- The corridor graph and localization decide which step to target.
- Exact checkpoint steps are cleaner because the indoor dataset already gives us real graph positions.

**If asked why this choice made sense**

The strongest part of indoor was already the corridor structure. The weaker part was local runtime behavior, so the right move was to keep the structure and make the controller/runtime relationship cleaner.

# Slide 3: Indoor Results

**Say this**

For indoor, we reached 8 out of 11 checkpoints. That was our strongest indoor result, and it made the system much easier to understand.

**What worked**
- exact checkpoint-step mode was cleaner than vague goals,
- MBRA-first was better than trying to make many controllers equally primary,
- removing bad recovery behavior helped.

**What failed**
- the graph behaves like a forward-only graph in practice,
- stale context could trap the robot,
- recovery still needed care.

**If asked why not 11/11**

The remaining gap is repeatability and edge cases, not lack of direction. We understand the failure modes much better now.

# Slide 4: Outdoor

**Say this**

For outdoor, we kept LogoNav as the main controller, used OSM-expanded waypoints, and added safety layers around the runtime. We chose this because outdoor already had a usable runtime base. The bigger problem was not replacing the controller. The bigger problem was making the runtime safer and less brittle.

**What that means technically**
- LogoNav is the outdoor local motion policy.
- OSM-expanded waypoints means one mission target is broken into smaller route targets from OpenStreetMap.
- The important work was runtime hardening, route discipline, and safer waypoint transitions.

**If asked why LogoNav stayed**

Outdoor progress came more from runtime hardening than from changing the learned controller.

# Slide 5: Outdoor Results

**Say this**

For outdoor, one run reached all checkpoints, which was our strongest success. The other runs were around a 50 percent success rate overall, and three rounds failed.

**What worked**
- one full checkpoint run proved the pipeline can complete,
- rerouting improvements helped,
- waypoint handling improved,
- logs became much more useful for debugging.

**What failed**
- spinning,
- poor waypoint transitions,
- sensitivity to real conditions,
- curbs and steps remain weak points.

**If asked what 50 percent means**

It means the outdoor system is partially successful and clearly improved, but still supervised and not yet robust.

# Slide 6: Marathon

**Say this**

For the marathon, we kept the same outdoor runtime and made it stricter. We added stronger safety logic because the marathon problem is really about safe supervised completion, not just reaching checkpoints under ideal conditions.

**What we added**
- IMU safety,
- route corridor guard,
- rerouting from live GPS,
- waypoint handling improvements,
- clearer logs,
- semantic and traversability hooks as safety layers.

**If asked what SegFormer was for**

SegFormer was part of the semantic-safety probe. It helped us test whether labels like road, sidewalk, grass, person, and plant could improve safety decisions, but that was still mostly an offline research step.

# Slide 7: Marathon Result And Failure Reason

**Say this**

In the marathon run, the robot reached one checkpoint, but after that it started spinning and toppled.

**Best simple explanation**
- after checkpoint progress, the runtime became unstable around the next target transition,
- that created repeated turning or align behavior,
- repeated aggressive turning led to physical instability.

**If asked why this happened**

We made the runtime more cautious, but we did not field-test every caution layer enough before this run. Too much caution can also be a problem if it makes the robot stop or turn too much instead of moving cleanly forward.

# Slide 8: Final Summary

**Say this**

So the short summary is this: for indoor, MBRA-first is our best direction and we reached 8 out of 11 checkpoints. For outdoor, LogoNav plus route and safety layers is still our best direction, with one full success and the others around 50 percent. For marathon, the main thing we still need to solve is stability after checkpoint and waypoint transitions.

**Optional closing line**

The project is better not because everything is solved, but because the remaining problems are now much clearer.

# Questions And Short Answers

**Q: Why MBRA indoors?**

Because indoor has strong corridor structure already. MBRA works best as the short-horizon controller once localization and graph progression define the motion structure.

**Q: What was MBRA intended to be?**

A short-horizon learned local controller, not a global planner. It steers toward a goal image using the current view and the target view.

**Q: Why LogoNav outdoors?**

Because the outdoor stack already had a usable controller base, and the bigger engineering problem was runtime stability and safety, not absence of a controller.

**Q: What is LogoNav?**

The outdoor local motion policy. It takes the current scene and a target waypoint and outputs motion commands, then the runtime wraps safety around it.

**Q: What is the biggest indoor insight?**

The graph is effectively forward-only in practice, so runtime logic has to handle that instead of blaming the planner.

**Q: What is the biggest outdoor insight?**

Runtime behavior around waypoint and checkpoint transitions matters as much as the controller itself.

**Q: What was the biggest marathon failure?**

Spinning and toppling after checkpoint transition.

**Q: What was SegFormer doing?**

It was an offline semantic-segmentation probe to see whether classes like road, sidewalk, grass, person, and plant could help with safety decisions.

**Q: What is the biggest unresolved issue?**

Low-lying outdoor hazards and stable behavior after target transitions.

# Numbers To Remember

- indoor: 8 / 11
- outdoor best run: all checkpoints
- outdoor overall: around 50%
- outdoor failed rounds: 3
- marathon: 1 checkpoint, then toppled

# Final Reminder

Keep the main talk simple. Say the numbers clearly. Do not oversell. Do not drift into documentation details. Keep coming back to:
- what we chose,
- why we chose it,
- what worked,
- what failed.
