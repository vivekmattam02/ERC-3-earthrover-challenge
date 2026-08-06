# Architecture in My Words

This note is the version I should use when I need to explain the project without sounding like I am reciting documentation.

## Indoor

Indoor was a known-corridor memory problem.

The rover sees a frame, localizes itself against a stored corridor database, smooths that guess over time, chooses a nearby graph subgoal, and then lets MBRA handle the short-horizon movement toward that subgoal.

So the real indoor stack is:

- visual place recognition
- temporal stabilization
- graph progression
- MBRA local control
- depth veto as safety

## Current No-GPS Branch

The current active branch is different from the original indoor competition framing.

It is now a **teach-and-repeat rough-terrain problem**:

- manually record a route
- build a visual route package
- localize against that taught route
- run a differential-drive controller that can keep moving on uneven terrain

So the real current stack is:

- manual teach bag collection
- visual route extraction
- CosPlace localization + temporal stabilization
- graph subgoal progression
- rough-terrain route-follow controller
- relocalization search when progress stalls

The honest maturity boundary is important: the software loop and motion
commands work, but the current physical tests did not prove sustained route
progression. The active engineering problem is reference-route/start alignment
and recovery quality, not merely adding another controller label.

The most important practical difference is this:

- this branch has **no GPS**
- and it should not behave like a static align-only robot

The most important recent correction is this:

- a teach route is not just a visually clean clip
- it must preserve traversal evidence
- post-processing should therefore prefer
  `motion-rich, route-progressing episodes`
  over a single stable-looking window

## Outdoor

Outdoor was a mission-runtime problem.

The rover receives mission checkpoints, may expand them into OSM route waypoints, asks LogoNav for a local command, and then filters that command through safety and runtime logic.

So the real outdoor stack is:

- mission checkpoint semantics
- OSM waypoint semantics
- LogoNav local policy
- traversability / semantics / IMU / route corridor / health gating

## Marathon

Marathon was not a different controller. It was the outdoor stack under a stricter operating profile.

That means:

- slower
- more guarded
- more willing to stop
- still vulnerable if transitions become unstable

## The Most Important Insight

The strongest systems lesson in this project is that the failure boundary was almost never "the model predicted badly."

It was usually:

- target semantics became ambiguous
- transition behavior became unstable
- safety and control interacted badly
- hardware dynamics amplified the software mistake

## Notes To Pair With This

- [[03 Personal Notes/Current Truth]]
- [[02 Core Concepts/MBRA vs LogoNav]]
- [[04 Runs and Failures/Run Outcomes]]
- [[01 Source of Truth/No-GPS Field Trial - Findings]]
