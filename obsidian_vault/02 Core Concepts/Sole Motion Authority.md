# Sole Motion Authority

This note explains one of the most important indoor lessons.

## What It Means

When MBRA is active, backup logic should not keep fighting its motion decisions every few ticks.

## Why It Mattered

If a learned local controller is trying to solve a short-horizon visual problem, and the runtime keeps injecting competing steering or recovery behavior, the result can be worse than either system alone.

## Indoor Meaning

The strong lesson was:

- MBRA should be the local controller
- the runtime should protect it with guardrails
- the runtime should not constantly override it unless there is a real safety/state reason

## What This Prevented

- oscillation from competing control logic
- false diagnosis that MBRA itself was weak
- noisy recovery behavior that destroyed useful local visual context

## Best Related Notes

- [[docs/our_mbra_discoveries]]
- [[live_indoor_runtime_story]]
- [[03 Personal Notes/Indoor Story - Distilled]]
