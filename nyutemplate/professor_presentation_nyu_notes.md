# NYU Presentation Notes

Type: Presentation Notes  
Status: Current  
Audience: Presenter

Related deck:
- `professor_presentation_nyu.tex`

## Goal

These notes are for a short professor presentation.

The slides are intentionally simple.  
These notes are where the technical depth lives.

Your speaking style should be:
- calm
- simple
- direct
- technically honest
- conversational, not documentation-like

Do not dump implementation detail unless asked.
Use the slides for structure and these notes for confidence.

Main structure:
1. indoor
2. outdoor
3. marathon
4. final summary

Main speaking rule:
- explain the engineering choices
- explain the roadblocks
- explain the result
- explain the remaining weakness

One useful framing line:
"We kept the presentation simple, but the engineering behind it was about choosing the right controller, keeping the runtime stable, and understanding where the system still breaks."

---

## Slide 1: Title

### What to say
"This is a short update on the ERC-3 EarthRover project. I’ll keep it simple and focus on three parts: indoor, outdoor, and marathon. For each one, I’ll say what we chose, why we chose it, what worked, and what failed."

### Purpose
Set expectations that this is not a full technical deep dive.

---

## Slide 2: Indoor

### What to say
"For indoor, we chose MBRA as the main controller and used exact checkpoint-step navigation. Indoor turned out to be a known corridor problem, not a generic exploration problem, so a short-horizon controller on top of known steps made more sense."

### What you mean technically
- Indoor uses corridor localization plus graph progression.
- MBRA is the short-horizon image-goal controller, not the planner.
- The corridor graph and localization decide which step to target.
- Exact checkpoint steps are cleaner because the indoor dataset already gives us real graph positions.

### Why this choice made sense
- The strongest part of indoor was already the corridor structure.
- The weaker part was local runtime behavior.
- So the best move was not to add more complexity, but to make the controller/runtime relationship cleaner.

### If asked for one technical sentence
"The main indoor simplification was to treat MBRA as local control on top of known corridor structure, not as the thing that solves the whole problem by itself."

### If asked what MBRA actually does
"MBRA takes the current camera view and the target step image, then produces local driving commands. It is a learned local policy, not a global planner."

---

## Slide 3: Indoor Results

### What to say
"For indoor, we reached 8 out of 11 checkpoints. That is the strongest indoor result we had, and more importantly it gave us a much cleaner understanding of what the system needs."

### What worked technically
- Exact checkpoint-step mode was cleaner than vague goals.
- MBRA-first was better than trying to make many controllers equally primary.
- Removing bad recovery behavior helped.
- The runtime became much easier to reason about once we stopped treating reverse recovery as a normal option.

### What failed technically
- The graph behaves like a forward-only graph in practice.
- That means if the rover localizes past a checkpoint, the planner can correctly report no path backward.
- Stale context could also trap MBRA in no-progress situations.
- The old recovery path was too eager to reuse stale state instead of refreshing the target.

### Important indoor technical phrases
- "forward-only graph"
- "skip-past-checkpoint logic"
- "no reverse for MBRA"
- "stale context reset"

### If asked why 8 / 11 instead of 11 / 11
Say:
"The main remaining gap is repeatability and handling edge cases cleanly, not lack of a controller. We understand the failure modes much better now, but the system is not yet fully reliable."

---

## Slide 4: Outdoor

### What to say
"For outdoor, we kept LogoNav as the main controller, used OSM-expanded waypoints, and added safety layers around the runtime. We chose this because outdoor already had a usable runtime base. The bigger problem was not replacing the controller. The bigger problem was making the runtime safer and less brittle."

### What you mean technically
- Outdoor already had:
  - mission checkpoints
  - resume logic
  - LogoNav
  - route expansion
  - perception support hooks
- So the engineering focus shifted toward:
  - route discipline
  - rerouting behavior
  - waypoint transitions
  - safety intervention
- "OSM-expanded waypoints" just means the route is broken into smaller intermediate targets from OpenStreetMap instead of going straight to one far-away point.
- LogoNav is the outdoor local motion policy, meaning it handles the short-term steering toward those targets.

### Main technical point
"Outdoor progress came more from runtime hardening than from changing the learned controller."

---

## Slide 5: Outdoor Results

### What to say
"For outdoor, one run reached all checkpoints, which was our strongest success. The other runs were around a 50 percent success rate overall, and three rounds failed."

### What worked technically
- One full checkpoint run proves the stack can work end-to-end.
- Rerouting and waypoint logic improvements helped real behavior.
- Better logs made debugging much easier.
- The system was able to complete at least one full mission path, so the overall direction was valid.

### What failed technically
- Some runs spun.
- Some runs followed poor waypoint transitions.
- Outdoor behavior remained much more sensitive to real conditions than indoor.
- Curbs, steps, and low obstacles are still weak points.
- The robot was still too easy to confuse near transitions and near hazard edges.

### Best concise technical explanation
"The outdoor system can work, but it still has stability problems around route following and low-hazard handling."

### If asked what 50 percent means
Say:
"It means the outdoor system is partially successful and clearly improved, but not yet at the level where I would describe it as robust."

---

## Slide 6: Marathon

### What to say
"For the marathon, we kept the same outdoor runtime and made it stricter. We added stronger safety logic because the marathon problem is really about safe supervised completion, not just reaching checkpoints under ideal conditions."

### What we added technically
- IMU safety
- route corridor guard
- rerouting from live GPS
- waypoint handling improvements
- clearer operator logs
- semantic and traversability hooks were used as safety layers, not as the main controller

### What SegFormer was for
"SegFormer was part of the semantic-safety probe. It helped us test whether semantic labels like road, sidewalk, grass, person, and plant could improve outdoor safety decisions, but that was still mostly an offline research step."

### Main message
"The marathon work was a runtime-discipline effort, not a new-controller effort."

---

## Slide 7: Marathon Result And Failure Reason

### What to say
"In the marathon run, the robot reached one checkpoint, but after that it started spinning and toppled."

Then say:
"The likely reason is that after checkpoint progress, the runtime became unstable around the next target transition. That created repeated turning behavior, and once the turning became too aggressive or too repeated, the platform lost stability."

### Technical version of that same point
- The failure was likely tied to post-checkpoint target handling.
- Bad alignment behavior can emerge if the runtime chooses or interprets the next target poorly.
- Repeated or aggressive turning is dangerous on the real platform, especially outdoors.
- We made the runtime more cautious, but we did not field-test every caution layer enough before this run.

### Important presentation choice
Do not list ten speculative causes.
Give one clean explanation:
"unstable behavior after checkpoint transition."

### If asked whether this was a controller problem or runtime problem
Say:
"My current reading is that it was more a runtime transition and stability problem than a pure low-level controller problem."

### If asked why not just make it even more cautious
"Because too much caution can also hurt. If the robot keeps stopping or turning instead of moving cleanly forward, that can create instability too. The missing piece is balance and testing, not just more safety layers."

---

## Slide 8: Final Summary

### What to say
"So the short summary is this: for indoor, MBRA-first is our best direction and we reached 8 out of 11 checkpoints. For outdoor, LogoNav plus route and safety layers is still our best direction, with one full success and the others around 50 percent. For marathon, the main thing we still need to solve is stability after checkpoint and waypoint transitions."

### This is your closing sentence if you want one
"The project is much better understood now, but the remaining weaknesses are also much clearer."

### Optional final line
"So the core story is simple: we found the right split between indoor and outdoor, but we still need more stability at the transitions."

---

## Likely Questions And Good Answers

### Q: Why did you choose MBRA indoors?
Because indoor is a known corridor problem. MBRA works better as the short-horizon controller once localization and graph progression are doing the structural work.

### Q: What was MBRA intended to be?
It was intended to be a short-horizon learned local controller, not a global planner. It is meant to steer toward a goal image using the current view and the target view.

### Q: Why not use one controller for both indoor and outdoor?
Because the environments and failure modes are too different. Indoor depends on corridor structure and dataset localization. Outdoor depends much more on mission runtime behavior, GPS, route expansion, and safety wrappers.

### Q: What is LogoNav?
It is the outdoor local motion policy. In our setup it takes the current scene and a target waypoint and outputs motion commands, then the runtime wraps safety around it.

### Q: Why did outdoor fail more often than indoor?
Because outdoor is much more sensitive to live conditions, target transitions, route behavior, and physical hazards like curbs and steps.

### Q: What was the biggest marathon failure?
Spinning and toppling after checkpoint transition.

### Q: What was SegFormer doing?
It was an offline semantic-segmentation probe. We used it to test whether classes like road, sidewalk, grass, person, and plant could help with safety decisions.

### Q: What is the biggest unresolved issue right now?
Low-lying outdoor hazards and runtime stability after waypoint or checkpoint transitions.

### Q: What do you think is the biggest project improvement?
Not one model. The biggest improvement is understanding the project correctly and hardening the runtime based on real failures.

---

## Final Advice

- Keep it short.
- Say the numbers clearly.
- Do not oversell.
- Do not drift into documentation details.
- Keep coming back to:
  - what we chose
  - why we chose it
  - what worked
  - what failed

The four numbers to remember:
- indoor: `8 / 11`
- outdoor best run: `all checkpoints`
- outdoor overall: `around 50%`
- marathon: `1 checkpoint, then toppled`
