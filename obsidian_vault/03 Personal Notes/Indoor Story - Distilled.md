# Indoor Story - Distilled

This is the shortest honest version of the indoor story.

## What We Thought Indoor Was At First

At first, it was easy to talk about indoor as if the main question was "which controller should drive the robot?"

That framing was too shallow.

## What Indoor Actually Was

Indoor was a known-corridor navigation problem with a recorded dataset and exact checkpoint steps.

That means the real backbone was:

- corridor localization
- temporal stabilization
- graph progression over known steps
- local control only after the state estimate was already clean

## What Was Strong

- CosPlace-based localization
- temporal filtering
- graph planning
- checkpoint-step formulation

## What Was Weak

- local control under edge cases
- recovery behavior
- handling of forward-only graph consequences

## What Changed The Indoor System Most

Three changes mattered more than the rest:

1. exact checkpoint-step mode
2. treating MBRA as the local controller instead of the whole system
3. removing runtime behaviors that fought MBRA or trusted bad state

## What Worked

- the indoor stack reached **8/11 checkpoints**
- the strongest version was MBRA-first, but only on top of the localization/planning backbone

## What Failed

- repetitive corridor structure still caused aliasing
- forward-only graph behavior could produce `no_path` cases
- stale context could keep the controller solving the wrong short-horizon problem

## What To Remember

Indoor improved when the team stopped blaming the controller for problems that were really about state semantics and progression.

## Read Next

- [[live_indoor_runtime_story]]
- [[docs/our_mbra_discoveries]]
- [[02 Core Concepts/Checkpoint-Step Logic]]
- [[02 Core Concepts/Stale Context]]
