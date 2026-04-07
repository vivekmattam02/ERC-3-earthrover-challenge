# Ultra Marathon Story: What We Verified, What We Changed, and Why

> Source: `live_outdoor_ultra_marathon_story.tex`
> Master Note: [[erc3_full_documentation]]

# Purpose

This document explains the outdoor ultra marathon work in the same spirit as the indoor story file: not just a list of edits, but the actual reasoning path. The goal is to record what the codebase already had, what the organizers' marathon instructions changed about the risk profile, what we added, what we deliberately did not add, and what still needs real-world validation.

The target event is not a normal outdoor checkpoint run. It is a one-attempt, seven-leg marathon in a busy real-world environment. The organizers explicitly warned teams to stay on sidewalks, avoid highways, and intervene when things go wrong. That changes the engineering objective completely. The right objective is not maximum autonomy or maximum speed. The right objective is **safe completion without flipping, entering the road, or getting trapped in brittle failure modes**.

# The Starting Point

The first important step was to stop treating the marathon as a blank-slate problem. The codebase already had a real outdoor runtime in:
```
live_outdoor_runtime.py
```

That runtime was not just a toy. It already supported:
- mission mode,
- checkpoint reporting,
- mission resume,
- the GPS-conditioned LogoNav controller,
- optional OSM pedestrian routing,
- a traversability layer,
- a semantic bias layer,
- telemetry freeze detection,
- and stuck recovery.

That mattered because it meant the marathon was not a controller-invention task. The real task was to make the existing outdoor stack *safer and stricter* for a one-attempt long-distance run.

# What We Verified In The Actual Repo

The work began by checking the actual code and SDK, not by relying on second-hand summaries.

The main files that were read closely were:
- `live_outdoor_runtime.py`
- `src/earthrover_interface.py`
- `earth-rovers-sdk/main.py`
- `earth-rovers-sdk/README.md`
- [[live_outdoor_runtime_explained]]

From those files, several facts were confirmed.

First, mission mode was already real. The runtime can call the SDK `/start-mission` endpoint, fetch the official checkpoints, and later report progress through `/checkpoint-reached`. It also resumes by checking which checkpoints were already scanned.

Second, OSM routing was already present. The runtime can expand a mission leg into pedestrian waypoints rather than driving in a naive straight line to the next GPS point.

Third, traversability already existed as a soft guidance layer. It was not a full safety controller, but it could bias motion away from local obstacles using depth.

Fourth, the SDK exposed the data needed for stronger safety. Battery, GPS, orientation, accelerometer, gyroscope, and timestamp were all available through telemetry.

Fifth, intervention endpoints existed in the SDK. That was important because the organizers explicitly told teams to be ready to intervene.

# What The Marathon Instructions Changed

The marathon instructions changed the engineering priorities in a very sharp way.

The organizers said, in effect:
- this is a real busy area,
- stay on sidewalks,
- do not go to the highway,
- be ready to intervene,
- if the robot flips, it is over,
- and the event consists of seven runs in series.

That immediately implied several design constraints.

The controller should be conservative. The runtime should stop on danger rather than improvise. The route should respect pedestrian structure. The system should make operator intervention easy and acceptable. And above all, it should avoid the two catastrophic failure classes: **flipping** and **leaving the sidewalk corridor**.

# What Was Already Good Enough To Keep

Not everything needed to be changed.

Several parts of the outdoor system were already the right foundation:
- the LogoNav outdoor controller as the main controller,
- mission mode for checkpoint handling,
- OSM-based pedestrian routing,
- telemetry freeze detection,
- and the existing recovery logic, at least in moderate form.

This led to one of the key engineering decisions of the marathon work: **do not build a new controller**. The outdoor system already had a controller path that was plausible. The higher-value work was on runtime safety and mission discipline.

# What Was Missing

After verifying what already existed, the missing pieces became clearer.

The most important missing items were:
- no hard anti-flip logic integrated into the marathon profile,
- no hard route-corridor guard to stop if the robot drifted too far off the routed pedestrian path,
- no dedicated preflight validator script,
- no battery warning or battery stop behavior inside the runtime,
- and no clear documented safe-mode profile for the marathon itself.

This was the real gap. The repo already knew how to navigate. What it lacked was a strict marathon wrapper around that navigation.

# The First Important Principle: Do Not Overclaim

A lot of robotics failure comes from confusing "implemented" with "proven." Two parts of the codebase were especially important here.

The semantic layer existed, but that did not automatically mean it was race-proven for a one-attempt urban marathon. It was a soft bias layer, not a hard guarantee of pedestrian-safe behavior.

The depth safety path also existed, but the comments in the outdoor runtime already warned that it could produce false stops on this hardware and camera geometry.

Because of that, the marathon work intentionally did *not* promote every existing feature into the default race profile. The approach was more conservative: keep the reliable core, add high-value guardrails, and avoid pretending that every optional perception layer was already trustworthy.

# What Was Added

The implementation history was actually two-stage, and the story is clearer if it is told that way.

## Stage One: Claude's Marathon Pass

By the time this pass began, the outdoor runtime had already been pushed beyond the original plain mission loop by an earlier Claude marathon pass. That earlier pass mattered. It had already introduced the idea that the marathon should not just be "the normal outdoor runtime, but longer." It had started to turn the runtime into a stricter event-specific safety profile.

The main things already present in the repo from that earlier marathon pass were:
- a proper `--ultra-marathon` flag,
- ultra-marathon default tightening inside `live_outdoor_runtime.py`,
- an IMU safety path wired into the runtime,
- a health-gate function,
- optional per-leg pause behavior,
- a no-reverse marathon bias,
- camera-loss watchdog logic,
- operator-halt behavior after repeated recovery attempts,
- and hardened OSM highway costs for marathon routing.

In other words, Claude's pass had already changed the runtime from a generic outdoor controller loop into something that was visibly trying to respect the marathon's safety constraints.

This is important because the work described later in this document was not done on a pristine baseline. It was done on top of a runtime that already carried a first generation of marathon thinking.

## Stage Two: The Narrower Safety Pass Added Here

The work in this pass did not try to replace Claude's marathon layer. Instead, it focused on the most important pieces that still seemed missing after reviewing the actual runtime and SDK carefully.

### Modified File: `live_outdoor_runtime.py`

The outdoor runtime was extended rather than replaced.

The most important addition in this pass was a **route corridor guard**. OSM routing gives a sequence of pedestrian waypoints, but routing alone is not a hard guarantee that the robot will stay near that route. The route corridor guard computes how far the current GPS position is from the active routed corridor and stops the robot if that deviation exceeds a threshold for several consecutive ticks.

This matters because "do not go to the highway" is not just a routing preference. It is a safety rule. The route corridor guard turns part of that rule into executable runtime behavior.

The next addition was **battery monitoring**. The runtime now emits low-battery warnings and can optionally stop when battery falls below a critical threshold.

This pass also made the route-corridor state and battery-related marathon state more visible in runtime output so the operator can understand why the rover stopped instead of guessing from behavior.

### New File: `scripts/preflight_marathon.py`

A new validator script was added because marathon risk does not come only from algorithms. It also comes from forgetting to check the obvious.

This script validates:
- SDK connectivity,
- live telemetry,
- advancing telemetry timestamp,
- finite GPS,
- battery threshold,
- live camera availability,
- LogoNav weights and config presence,
- mission start and checkpoint retrieval,
- and optional OSM route expansion.

This was deliberately chosen as a separate script rather than hidden deep inside the runtime, because the team needs a reusable preflight ritual before the actual attempt.

# What Was Already There And Was Reused

It is important to say clearly what was *not* created from scratch in this pass.

The IMU safety module already existed in the repo state that was reviewed during this pass:
```
src/imu_safety.py
```

From the repo history as encountered, that module belonged to the earlier marathon-oriented work rather than to this narrower safety pass. It already implemented tilt and pitch/roll rate checks with a latch and reset behavior. It was verified and kept rather than rewritten.

Likewise, OSM routing and highway-cost shaping were already structurally possible because the OSM router already exposed highway cost multipliers. Claude's earlier marathon pass was already making use of that by pushing primary and secondary roads toward near-infinite cost in ultra-marathon mode. This pass kept that idea and focused on adding a runtime stop condition if the rover drifted away from the routed pedestrian corridor anyway.

The same pattern applied to the health gate and leg-pause ideas. Those were already present as part of the earlier marathon work. This pass treated them as existing marathon infrastructure and focused on filling in the specific missing guardrails rather than re-arguing those decisions from zero.

# What The Ultra-Marathon Mode Means In Practice

The marathon profile is intentionally conservative.

In practice, the ultra-marathon mode means:
- the outdoor controller remains LogoNav,
- pedestrian routing is expected,
- traversability remains the primary optional soft obstacle aid,
- IMU safety is expected,
- recovery is limited,
- reverse behavior is restricted in marathon mode,
- health-gate checks are encouraged,
- and leaving the routed corridor becomes a stop condition.

This is the right philosophy for a one-attempt event. The runtime should be more willing to stop and ask for human judgment than it is in a short, low-stakes test run.

# Why The Route Corridor Guard Was So Important

Of all the additions, the route corridor guard deserves special emphasis.

Before this addition, OSM routing helped choose where the rover ought to go, but there was no direct runtime check saying: *you are drifting too far away from the sidewalk route, stop now*. That gap mattered because the organizers' main safety concern was not abstract navigation quality. It was real-world path discipline.

A route corridor stop is not the same thing as a full geofence or road-segmentation system. It does not understand curbs, lanes, or traffic semantics. But it does provide a simple and high-value guarantee: if the rover's GPS solution wanders too far away from the pedestrian route, it will stop rather than continue blindly.

That is exactly the kind of modest but high-leverage safety improvement that makes sense late in the development cycle.

# Why We Added A Preflight Validator Instead Of More Fancy Perception

Another core judgment in this work was that a preflight validator was more valuable than adding yet another perception model.

A marathon can be lost because the SDK is stale, because telemetry timestamps are not advancing, because the camera never became live, because the mission start did not actually return checkpoints, or because the LogoNav weights are missing on the machine being used. None of those problems are glamorous, but all of them are realistic.

The validator script exists because the team does not have many options. When there is one attempt and failure is expensive, operational discipline is part of the autonomy system.

# What We Deliberately Did Not Add

Several tempting ideas were not turned into default marathon features.

We did not make semantics mandatory in the default marathon command. The semantic layer exists, but it was not treated as fully race-proven.

We did not make legacy depth-stop behavior mandatory either. The codebase itself already suggested caution there.

We did not build a new controller, a traffic-light detector, a lane detector, or a large new perception stack. Those may be research ideas, but they are not the right kind of change to trust late in a one-attempt competition cycle.

This was a deliberate engineering choice: **prefer smaller, better-justified safety additions over speculative autonomy expansion**.

# How To Use The New Pieces

The marathon flow now has a much clearer shape.

A basic preflight without touching mission mode is:
```
python scripts/preflight_marathon.py
```

A preflight that validates mission startup and OSM expansion is:
```
python scripts/preflight_marathon.py --mission --osm-route
```

A conservative marathon runtime command is:
```
python live_outdoor_runtime.py \
  --mission --send-control \
  --controller logonav \
  --osm-route \
  --traversability \
  --ultra-marathon
```

That is intentionally not overloaded with every optional flag. The default marathon profile should stay explainable.

# What Is Still Uncertain

Even after these changes, there are still important unknowns.

The first unknown is mission structure. It is still necessary to confirm whether the seven runs are represented as one mission or as seven separate mission slugs. That affects whether a dedicated multi-leg wrapper is needed.

The second unknown is field tuning. IMU thresholds, battery stop policy, recovery limits, and corridor-stop sensitivity all need to be validated on the real robot.

The third unknown is whether semantics should be enabled in the real marathon run. It remains a candidate, but not something that should be promoted to default without fresh validation.

The fourth unknown is the exact quality of OSM routing over the full Berkeley-to-Stanford route. The code now hardens highway costs, but route data quality still depends on OSM coverage.

# What I Would Tell The Team Clearly

If a teammate asked for the short truthful version, I would say this.

The outdoor system already had a real mission runtime. Claude's earlier marathon pass had already done the first serious transformation of that runtime into a marathon-aware system: ultra-marathon mode, IMU safety integration, health gating, leg pauses, and stricter recovery behavior were already there.

This later pass did not replace that work. It tightened the picture by adding the missing route-discipline and preflight pieces:
- a route-corridor stop to keep the rover near the routed sidewalk path,
- battery warnings and optional battery stop behavior,
- and a proper preflight validator.

So the real story is not "one model solved the marathon." The real story is that the runtime evolved in layers: first a broad marathon-safety wrapper, then a narrower pass that added the missing route and operational guardrails.

The biggest remaining risks are not abstract software bugs. They are real-world tuning, mission-format uncertainty, and the fact that any long run in a busy public environment still requires the operator to behave like a safety supervisor, not like a passive observer.

# Final Recommendation

The current recommendation is:
1. Run the preflight validator first.
1. Use the conservative marathon command built on mission mode, LogoNav, OSM routing, traversability, and ultra-marathon mode.
1. Do not enable extra optional layers by default unless they are revalidated immediately before the event.
1. Confirm with the organizers whether the seven runs arrive as one mission or several.
1. Treat operator intervention as part of the intended safety workflow, not as a failure of principle.

The most important idea behind all of this is simple: the marathon system should be judged first by whether it avoids catastrophic behavior. Everything else is secondary.

# Detailed Engineering History Of The Later Marathon Work

This final section records the longer and messier part of the marathon work, because the actual engineering story did not end with the first route-corridor and preflight additions. After the initial conservative marathon wrapper was in place, the runtime was exercised on the real robot and several new classes of failure appeared. Those later failures forced a second round of runtime surgery. That second round is important to document because it changed not only the code, but also the team's understanding of what the outdoor stack was and was not capable of.

The most important lesson from that later phase was that long-distance public-environment navigation does not fail in one way. It fails in layers: repository confusion, mission-mode integration mistakes, sensor-frame assumptions, control-law shaping, waypoint handoff behavior, and route recovery policy can all break independently. The rest of this section tells that story in the order it emerged.

## Repository And Path Confusion Had To Be Resolved First

One of the early practical issues was not algorithmic at all. There were two similarly named repository locations on disk, one under a desktop path and one under a rover path. At one point work was done in the populated outdoor code tree while another tree looked nearly empty. This created understandable confusion about whether files had been deleted, moved, or overwritten.

That confusion was resolved by explicitly verifying which directory contained the real outdoor runtime, the SDK integration, the indoor and outdoor entrypoints, and the supporting source files. The final source-of-truth repository became the one under the rover path. This mattered because every subsequent safety change and every run command needed to reference the same real working tree. The code itself did not require path-driven rewrites because the runtime mostly used repo-relative paths, but the team still needed a clear operational understanding of which checkout was real.

## The Model Audit Was Broader Than The First Story Section Suggests

Later in the work, the model inventory was revisited much more explicitly because the team needed to know whether the outdoor system really had the perception capacity to stay on sidewalks at night.

The audit confirmed the following.
- There was no YOLO-based object detector wired into the outdoor runtime.
- The semantic segmentation path existed and was originally based on SegFormer with an ADE20K-oriented model id.
- The monocular depth path existed through Depth Anything V2.
- The traversability layer used the depth output to create a local blocked-vs-open steering signal, but it was not a full semantic understanding layer.
- The main outdoor local controller remained LogoNav, not MBRA.

This was an important point of intellectual hygiene. The runtime did have meaningful perception components, but it did not have a full street-scene detector stack. That forced a careful distinction between what the stack could plausibly do and what would have been wishful thinking.

## External Research Was Used To Reframe The Night Problem

The later phase also included a web-informed review of relevant open-source and public research directions, especially for night-time outdoor behavior. The main point was not to chase novelty for its own sake. The point was to understand whether a night marathon should be treated as a trivial extension of daytime routing. It should not.

The review reinforced three points.
- Night-time pedestrian detection is materially harder than daytime detection.
- Monocular depth quality degrades in low light and glare conditions.
- Road-scene segmentation models trained on urban datasets such as Cityscapes are structurally more relevant than a generic ADE20K semantic model for sidewalk-versus-road interpretation.

That did not mean the correct response was to install an entirely new detector stack and hope for the best. It meant the runtime needed stronger operating-envelope checks and more appropriate semantic defaults.

## Several Tempting But Overconfident Ideas Were Explicitly Rejected

Part of the later work was not just adding code. It was backing away from brittle ideas when they turned out to be less defensible than they first appeared.

For example, there was a phase where stronger semantic stop logic, heavier city-scene semantics, stricter sidewalk-only routing, and more aggressive operator gating were all considered for the night baseline. Some of those ideas stayed as optional features. But others were deliberately removed from the baseline because they were too eager, too brittle, or too dependent on assumptions that were not yet field-validated.

This is worth recording clearly. The final recommended runtime did not emerge by monotonically adding features. It emerged through a process of trying to increase safety and then pruning away the pieces that were making the system less coherent.

## Night-Safe Mode Became A Real Runtime Profile

The later work added a much more explicit night-oriented runtime profile. This profile eventually included the following ideas:
- lamp-on behavior,
- a vision-quality safety gate for dark, glare-heavy, or low-detail frames,
- stronger GPS safety gating,
- explicit use of the IMU safety path,
- route-corridor guarding,
- and tighter runtime defaults for night operation.

A separate image-quality monitor was added so that night-time failure would not be treated only as an obstacle-avoidance problem. If the image itself is too dark, too washed out, or too textureless, the controller should not be trusted just because the camera technically returned a frame. This was an important conceptual shift: not every failure should be forced through the main controller. Sometimes the correct action is to recognize that the sensing regime is no longer acceptable.

At the same time, the night-safe profile was later softened in a few places after field feedback. In particular, semantic hard-stop behavior, semantic sidewalk-stop logic, forced no-reverse assumptions, and operator-confirm pauses were not left as silent mandatory defaults. They remained available, but they were treated more honestly as optional or experimental layers rather than as unquestionable baseline behavior.

## The IMU Safety Path Required Real Calibration Work

One of the most painful failures in the later phase was that the robot initially refused to move because the IMU safety path believed the rover was already tilted by nearly ninety degrees while sitting normally. That exposed a very concrete problem: the original tilt logic was tied too strongly to an assumed accelerometer rest orientation.

This led to a substantial correction. The IMU safety logic was changed to self-calibrate the gravity vector from live stationary samples at startup. Tilt was then measured relative to the learned rest gravity direction instead of a fixed assumed sensor axis. The gyro logic was also softened so that pure turning would not create emergency-stop behavior by itself. In practice, this meant gyro alone no longer triggered the same kind of hard stop unless there was corroborating evidence of a real instability event.

This episode was valuable because it demonstrated a broader lesson: safety code is only good safety code if its frame assumptions match the real robot. A mathematically clean detector tied to the wrong sensor frame is worse than useless because it repeatedly fires during normal behavior and trains the operator to distrust the stop signal.

## Mission Startup And SDK Behavior Needed Additional Discipline

The interaction with the SDK also became part of the engineering story. Mission mode required a proper mission slug and a successful call to the SDK's mission startup path before normal mission telemetry flow was valid. This was initially a source of confusion when the goal was only to test connectivity or to verify the robot indoors.

That confusion forced a clearer separation between three different operational cases:
- indoor connectivity checks without mission mode,
- outdoor preflight checks without actually committing to mission execution,
- and the full mission-run flow with the SDK's expected mission handshake.

Later, startup logic was also extended so that if the rover was already physically within the goal radius of the first remaining checkpoint at mission start, the runtime could auto-claim that checkpoint rather than attempting to navigate to it again. This was a very practical fix for stop-and-restart scenarios during testing.

## The OSM Story Became Much More Complex Than "Turn It On"

The early summary "use OSM routing" was directionally right but incomplete. Once the robot was tested on the real route, several deeper OSM-related issues became visible.

First, startup route expansion could make the runtime look frozen because it tried to expand too much of the mission before the main loop began. That was corrected by moving toward leg-wise route building rather than eagerly expanding everything at startup.

Second, strict sidewalk routing could fail immediately if the first tiny leg effectively degenerated into a straight-line fallback. That revealed a real tradeoff between ideological purity and operational pragmatism. For actual use, strict sidewalk-only rejection was too brittle as a universal default. It became something that could be enabled intentionally, not something that should break every run.

Third, route-corridor enforcement initially created deadlocks. The first version of the corridor stop could correctly detect that the robot had drifted away from the routed corridor, but after stopping it could get trapped in a loop where it kept evaluating the same bad route state and stopping again. The runtime had to be changed so that a corridor stop in OSM mode would re-route from the live GPS position instead of just reasserting the previous route.

Fourth, re-routing itself introduced a new class of failure: the first waypoint after a reroute could end up effectively behind the rover. This created long turning loops, repeated alignment behavior, and the feeling that the runtime was "arguing with itself." To mitigate that, waypoint-pruning logic was added. If an initial routed waypoint was close enough and sufficiently behind the rover heading, it could be dropped rather than treated as the new objective. Later, that same idea was extended not just to startup and reroute, but also to ordinary waypoint advances.

Fifth, the route-corridor threshold itself had to be tuned. A threshold that was too tight caused excessive rerouting and apparent indecision. A threshold that was too loose would defeat the whole purpose of the corridor guard. The eventual logic made the threshold depend not just on a static corridor-stop number but also on waypoint spacing, and later it was relaxed further in non-strict mode so that the robot would not constantly re-route while still making legitimate progress.

## Waypoint Handoff Turned Out To Be One Of The Hardest Practical Problems

The later logs made it very clear that the runtime's intermediate waypoint handoff policy was too blunt. Treating routed intermediate waypoints as "reached" at a fixed radius was causing the rover to jump to the next waypoint while it was still effectively in the middle of an awkward local maneuver. This then created large bearing changes, long realignment episodes, and the feeling that the robot kept changing its mind.

At first, the intermediate routed waypoint radius was tied to a five-meter threshold inspired by an older LogoNav deployment script. But on the real robot this was too permissive. The runtime was later changed so that routed intermediate waypoints used a smaller and more dynamic handoff radius, derived from the actual segment spacing and capped more conservatively. Mission checkpoints still kept their larger reach radius.

This distinction is central to understanding the later code. A mission checkpoint is an externally meaningful target that the robot must claim. An intermediate routed waypoint is only a local scaffold. Those two concepts should not share the same reach policy.

## Controller Diagnostics From Live Logs Forced Several Changes To LogoNav Shaping

A major part of the later phase consisted of reading the live outdoor logs carefully and using them as the main source of truth for controller behavior.

Those logs exposed several issues.
- The robot could crawl too slowly for too long, especially when the learned policy plus safety shaping reduced forward speed to an ineffective level.
- The stuck detector originally only considered relatively large forward commands as "real motion," so low-speed creeping could evade recovery logic.
- Turn-priority logic could be too eager, especially when a waypoint was not truly near but the bearing error happened to be large.
- After re-routing or waypoint changes, the controller could attempt to align to an obviously bad next point rather than maintaining monotonic progress.

To address those issues, several changes were made.

The effective minimum forward speed logic for LogoNav was adjusted so that the runtime would not spend too long pretending to move while only issuing weak unusable commands.

The stuck-detection thresholds were made more compatible with the actual low-speed regime being used outdoors.

Turn-priority alignment became more explicitly gated by distance and later by target type. In particular, far-away mission checkpoints were no longer allowed to trigger the same style of hard alignment loop that made sense only for nearby routed waypoints. This was a very important change because some of the worst "spin in place" behaviors were being driven by an alignment rule that was correct in principle but being applied to the wrong target regime.

## Traversability Had To Become More Than A Gentle Suggestion

Another major lesson from the live tests was that a soft obstacle bias was not enough when the rover wanted to push toward walls, barriers, or curb-like structures. The traversability layer originally behaved more like a steering hint than a hard local safety override.

That was changed so that traversability could now do more than gently nudge the command. When the local forward corridor looked blocked, the runtime could enter stronger traversability-driven behaviors such as slowing, turning, or stopping instead of letting the learned controller dominate. This made the obstacle logic more explicit in the runtime and also improved the operator's ability to understand what the robot believed it was doing.

However, this part of the story also carries an important caveat. The traversability module was designed around a middle image band, which is good for trunks, walls, and barriers at rover-eye level, but not ideal for everything. In particular, it is still not a complete curb, drop, or step detector. The lower image region and the exact geometry of step-like hazards remain a partially open issue. So traversability became stronger, but it did not become magic.

## The Terminal Output Had To Be Rewritten Because Debuggability Was Part Of Safety

One practical but important part of the later work was improving the runtime's output. The original log format was too opaque for fast operator judgment during a long real-world run. It was not enough to know only that the controller was in some generic mode. The operator needed to know which mission checkpoint was active, which routed waypoint was active, how far the robot was from the mission checkpoint, how far it was from the active local waypoint, what the forward clearance looked like, and whether the route corridor logic was about to trip.

Because of that, the runtime log line was rewritten to surface quantities such as progress through mission checkpoints, current leg, current waypoint, local goal distance, mission checkpoint distance, forward clearance, route-corridor deviation, active radius, and clearer mode labels. This was not cosmetic. For a marathon-style system, observability is part of the control system because the human operator is a legitimate component of the safety loop.

## The Saved-Log Calibration Idea Was Partly Possible And Partly Not

At one point, the idea arose to "use the old saved outdoor data" to tune the runtime more systematically. That was a sensible instinct. In practice, however, there was no clean saved outdoor run log in the repository that directly captured the exact failure regimes being seen on the current robot. As a result, the work had to rely on two imperfect but useful sources instead:
- the old outdoor deployment script and its design choices,
- and the live logs produced during the new outdoor tests.

That is why some thresholds were informed by historical deployment values while other fixes were driven directly by present-day live telemetry and command traces. This is worth documenting honestly because it explains why some later tuning decisions were iterative rather than theoretically clean.

## What The Final Runtime Had Actually Become By The End Of This Phase

By the end of the later work, the outdoor runtime had evolved well beyond its original plain checkpoint-following form. In narrative terms, it had become a layered supervised sidewalk marathon runtime with the following major characteristics.

At the top level, it still used mission mode and the LogoNav controller. It still relied on OSM-expanded waypoint structure rather than inventing an entirely new path planner. But wrapped around that were multiple newer layers: mission startup checks, checkpoint auto-claim at startup when appropriate, leg-wise route construction, route-corridor guarding, rerouting from live GPS, behind-waypoint pruning, dynamic intermediate waypoint handoff, IMU self-calibration, night-time vision gating, stronger traversability overrides, better stuck handling, and more informative logs.

In other words, the system had moved from "follow checkpoints with some optional extras" toward "follow checkpoints inside a restricted operating envelope, with multiple ways to stop or reset when the behavior no longer makes sense."

## What Still Remains Unresolved Even After All Of That

It is just as important to record the remaining weaknesses as the completed fixes.

First, the runtime is still not a true road-driving system. It is a supervised pedestrian-path system.

Second, curb, step, and drop detection remain incomplete. The traversability layer is much more useful than before, but it is still shaped around a middle image band and is therefore not the final answer to every low-lying hazard.

Third, semantic understanding remains a support layer, not a fully trustworthy night-time guardian. Different semantic model profiles were explored, including stronger road-scene-oriented options, but none of that turned the system into a guaranteed sidewalk-semantics engine.

Fourth, the route-corridor logic is still a compromise. If it is too tight, the runtime becomes indecisive and over-eager to reroute. If it is too loose, it stops serving its purpose. The non-strict mode relaxation was necessary for real use, but it is still a heuristic rather than a proof.

Fifth, the robot still needs more grounded tuning against real repeated outdoor runs. Some of the logic is now much more sensible than before, but a one-attempt marathon demands more than code elegance. It demands repeated closed-loop exposure to the same style of path geometry and hazard distribution that the event will actually present.

## What The Team Should Do Next

The next work should not be treated as open-ended feature chasing. It should be treated as a focused stabilization phase.

The immediate next steps are:
- collect more real outdoor logs from the current runtime,
- specifically review waypoint handoff behavior, reroute triggers, and low-clearance segments,
- improve curb and step handling rather than adding unrelated perception complexity,
- confirm the exact mission format and checkpoint semantics from the organizers,
- and keep the operator workflow explicit and disciplined.

The biggest thing to avoid from this point onward is pretending that the stack only needs "one more model." The history recorded here shows that most of the hard failures were about operating-envelope control, state transitions, and bad assumptions at interfaces. Those are the places that still deserve the most attention.

## The Honest Summary Of The Full Story

The full marathon work was not one clean idea. It was a long chain of increasingly concrete realizations.

First, the outdoor runtime already existed and did not need to be replaced.

Second, the marathon needed a safety wrapper rather than a new controller.

Third, that wrapper needed route discipline, preflight discipline, and mission discipline.

Fourth, night-time operation forced explicit image-quality and operating-envelope checks.

Fifth, the robot's live behavior exposed problems that could not be discovered by static code reading alone: IMU frame assumptions, overly eager alignment, overly permissive waypoint handoff, deadlocking corridor reroutes, and behind-the-rover target handoffs.

Sixth, the runtime had to be repeatedly simplified and restructured so that it would stop doing obviously dumb things.

That is the true story of this marathon work. It is not the story of a single brilliant autonomy algorithm. It is the story of taking a real but incomplete outdoor stack and repeatedly forcing it to become more honest, more observable, and less brittle under the specific pressures of a long supervised sidewalk marathon.

# Code-Verified Appendix: Exact Later Runtime Mechanics

The narrative sections above explain the later marathon work as a story. This appendix records several parts of that story in direct code-grounded form so the document can also serve as a technical handoff.

## Three Parser Defaults That Became Surprisingly Important

Several seemingly small parser defaults later turned out to control major pieces of behavior:
```
--intermediate-goal-radius-m = 3.0
--logonav-align-distance-m = 6.0
--osm-prune-behind-distance-m = 18.0
```

These correspond to three practical discoveries:
- routed intermediate waypoints needed a smaller handoff radius than mission checkpoints,
- hard align-turn behavior had to be restricted to nearby or clearly pathological cases,
- and OSM routes sometimes needed aggressive pruning of nearby behind-the-rover starter waypoints.

## What Ultra-Marathon And Night-Safe Really Do In Code

The present runtime does not just print that these modes are active. It mutates real safety parameters. Two representative snippets are worth preserving:
```
if args.ultra_marathon:
    args.camera_watchdog_ticks = min(args.camera_watchdog_ticks, 10)
    args.semantic_hard_stop = True
    args.semantic_yield = True
    args.semantic_sidewalk_stop = True
```

and:
```
if args.night_safe:
    args.trav_stop_m = max(args.trav_stop_m, 0.80)
    args.trav_slow_m = max(args.trav_slow_m, 1.50)
    args.camera_watchdog_ticks = min(args.camera_watchdog_ticks, 6)
    args.battery_warn_pct = max(args.battery_warn_pct, 35.0)
```

This is important because it shows that the marathon story is not only conceptual. The current code contains a real event-specific operating envelope.

## The Later Route-Corridor Logic Is Also More Nuanced Than The Early Story

The current route-corridor threshold is not only a fixed scalar. It depends on path geometry and strictness mode:
```
route_corridor_stop_threshold = max(route_corridor_stop_threshold,
                                    0.75 * args.osm_min_waypoint_spacing_m)
if not args.sidewalk_strict:
    route_corridor_stop_threshold = max(route_corridor_stop_threshold,
                                        args.goal_radius_m)
```

That later relaxation was essential for one reason: a corridor rule that is too eager does not make the system safer in practice. It makes it indecisive and reroute-happy.

## The Align Gate Was Ultimately Made Target-Aware

One of the most frustrating real failures was long turn-in-place behavior after a waypoint change or reroute. The present code now explicitly restricts when hard align-turn behavior is allowed:
```
_align_allowed = (
    math.isfinite(distance_to_goal)
    and (
        distance_to_goal <= args.logonav_align_distance_m
        or (
            not bool(target.get("mission_checkpoint", False))
            and _bearing_error_deg >= args.logonav_align_extreme_deg
        )
    )
)
```

This means the runtime distinguishes between a nearby routed waypoint and a far mission checkpoint. That distinction did not exist strongly enough in the earlier phases of the marathon work, and the logs clearly showed the cost of that omission.

## Intermediate Waypoint Handoff Is Now Dynamic

Another code path worth preserving is the dynamic handoff radius logic:
```
if bool(target.get("mission_checkpoint", False)):
    active_goal_radius_m = args.goal_radius_m
else:
    _dynamic_radius_m = float(args.intermediate_goal_radius_m)
    if math.isfinite(_segment_distance_m) and _segment_distance_m > 0.0:
        _dynamic_radius_m = min(_dynamic_radius_m,
                                max(2.5, 0.35 * _segment_distance_m))
    active_goal_radius_m = min(args.goal_radius_m, _dynamic_radius_m)
```

This snippet captures one of the most important conceptual shifts of the later work: mission checkpoints and routed intermediate waypoints are different kinds of targets and should not share the same reach semantics.

## Traversability Was Ultimately Promoted To Hard Local Override

The narrative above already explained that traversability had to become stronger. The current code makes that plain:
```
if _trav.all_blocked or _trav.linear_scale <= 0.0:
    command.linear = 0.0
    ...
    command.reason = f"trav_stop({_trav.forward_clearance:.2f}m)"
elif _trav.forward_blocked:
    command.linear *= _trav.linear_scale
    ...
    command.reason = f"trav_turn({_trav.forward_clearance:.2f}m)"
```

That promotion from "soft nudge" to "real local override" was one of the most concrete runtime-behavior changes in the whole outdoor effort.

## Representative Failure Logs That Drove The Later Fixes

Several later fixes are easiest to remember through the characteristic logs that exposed them:
- `[IMU EMERGENCY] imu_emergency(tilt=89...)` while the robot was physically upright: this revealed that the original tilt logic was assuming the wrong rest-frame orientation.
- `route_corridor_stop dist=... threshold=... ticks=3` repeated over and over: this exposed the stale-route deadlock and motivated reroute-from-current-pose.
- `intermediate waypoint reached ...` followed immediately by an `ALIGN` block to a bad next point: this exposed the handoff and behind-waypoint problem.
- `CP=... dist=... lin=+0.07...` with almost no net progress: this exposed the low-speed creeping regime and the mismatch with stuck detection.

These later failure signatures are part of the marathon story and should be remembered because they explain why the current runtime has so many apparently narrow guardrails.

## One Honest Technical Limitation Still Remains

Even after all of the later fixes, the code-grounded reality remains that the current stack is still best described as a supervised pedestrian-route rover runtime. It is not a general-purpose self-driving system, and it still lacks a complete solution for curbs, drops, and step-like hazards. The stronger runtime today is more disciplined and less self-contradictory than before, but it is still operating inside a carefully managed envelope rather than proving full autonomy.

# Related Documents

- [[erc3_full_documentation]] --- single master guide for the complete project story and current architecture.
- [[live_indoor_runtime_story]] --- indoor evolution, MBRA integration, and checkpoint-step runtime behavior.
- [[live_outdoor_ultra_marathon_story]] --- outdoor and marathon runtime evolution with safety-layer reasoning.
- [[outdoor_perception_review]] --- depth/semantic perception findings and their runtime implications.
