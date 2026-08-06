# Marathon Story - Distilled

This is the shortest honest version of the marathon story.

## What Marathon Was

Marathon was not a different autonomy stack.

It was the outdoor stack under a stricter safety profile and a much higher stability requirement.

## What We Added

- stricter operating profile
- route corridor discipline
- IMU safety
- preflight and health gating
- more runtime observability

## What We Were Trying To Prevent

- leaving the safe route
- tipping or flipping
- brittle failure loops

## What Actually Happened

- the robot reached the first checkpoint
- after target transition, behavior became unstable
- repeated turning and alignment persisted too long
- platform stability was not strong enough to absorb that loop
- the robot toppled

## Why This Matters

The marathon result is useful because it exposed the exact remaining weakness:

- transition stability under a stricter operating profile

## What To Remember

The marathon failure does not mean the safety work was meaningless.

It means the system still was not calm enough during and after target transitions.

## Read Next

- [[04 Runs and Failures/Marathon Failure - What Actually Happened]]
- [[live_outdoor_ultra_marathon_story]]
- [[02 Core Concepts/Transition Instability]]
