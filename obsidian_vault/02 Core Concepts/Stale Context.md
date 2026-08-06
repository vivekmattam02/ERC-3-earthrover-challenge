# Stale Context

This concept matters most indoors, but the idea is broader.

## What It Means

A controller can keep acting on short-horizon context that is no longer describing the real situation well.

That old context becomes stale.

## Why It Matters

If the rover is not making progress but the controller keeps trusting the same recent history, it can stay trapped in low-value behavior.

## Indoor Version

The MBRA path needed reset logic so that when progress stalled, the observation history was not trusted forever.

That is why no-progress reset mattered.

## What It Prevents

- solving the wrong short-horizon problem for too long
- repeating weak commands without reevaluating state
- making the controller look worse than the state-estimation pipeline really is

## Best Related Notes

- [[live_indoor_runtime_story]]
- [[03 Personal Notes/Indoor Story - Distilled]]
- [[03 Personal Notes/Things I Keep Forgetting]]
