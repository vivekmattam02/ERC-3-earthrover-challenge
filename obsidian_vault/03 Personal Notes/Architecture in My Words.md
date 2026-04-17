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
