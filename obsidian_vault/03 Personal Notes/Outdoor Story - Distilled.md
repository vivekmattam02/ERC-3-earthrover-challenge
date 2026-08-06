# Outdoor Story - Distilled

This is the shortest honest version of the outdoor story.

## Current Status

This note is mostly about the older GPS outdoor mission branch.

The active off-road branch now is different:

- no GPS
- teach-and-repeat
- rough terrain
- differential-drive route repeat

So this note should be read as historical context for outdoor mission logic,
not as the full current off-road source of truth.

## What We Thought Outdoor Was At First

It was tempting to describe outdoor as "use LogoNav to drive to GPS checkpoints."

That was not enough.

## What Outdoor Actually Was

Outdoor was a mission-runtime problem.

The real system was:

- mission checkpoints from the SDK
- optional OSM-expanded intermediate waypoints
- LogoNav as the local controller
- runtime gates for route discipline, safety, and recovery

## What Was Already Good

- there was already a runnable outdoor base
- LogoNav already existed as a practical controller path
- OSM routing already gave useful local route structure

## What The Real Problem Became

The hardest part was not "have a controller."

The hardest part was:

- target semantics
- waypoint handoff
- reroute semantics
- stop/turn/realign behavior when the target changed

## What Worked

- one run reached all checkpoints
- the outdoor stack was real enough to finish a mission

## What Failed

- repeatability was mixed
- transition behavior around routed waypoints remained fragile
- safety and control could still interact in ways that slowed or destabilized progress

## What To Remember

Outdoor should be explained as a runtime orchestration problem, not as a single-model story.

## What Off-Road Actually Is Now

The current off-road track problem is:

- manually teach a route
- post-process the bag
- localize visually against that route
- repeat it with rough-terrain control

The key lesson so far is:

- the best teach route is not the cleanest-looking clip
- it is the route with the best traversal evidence
- route coverage matters more than image prettiness

## Read Next

- [[01 Source of Truth/Offroad Track - Full Story]]
- [[live_outdoor_runtime_explained]]
- [[live_outdoor_ultra_marathon_story]]
- [[02 Core Concepts/Route Corridor Guard]]
- [[02 Core Concepts/Transition Instability]]
