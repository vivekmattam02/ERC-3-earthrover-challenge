# Transition Instability

This is one of the most important concepts in the whole project.

## What It Means

Transition instability is what happens when the robot does not settle cleanly after the active target changes.

That change may be:

- checkpoint to next checkpoint
- waypoint to next waypoint
- reroute to a new local target
- recovery back to nominal control

## Why It Matters

A lot of the ugliest failures were not because the controller had no answer.

They happened because the system changed targets and then got trapped in:

- over-alignment
- repeated turning
- stop-turn-stop-turn loops
- unstable command authority handoff

## Where It Showed Up

- outdoor waypoint handoff
- reroute events
- marathon post-checkpoint behavior

## Best Related Notes

- [[04 Runs and Failures/Marathon Failure - What Actually Happened]]
- [[live_outdoor_ultra_marathon_story]]
- [[live_outdoor_runtime_explained]]
