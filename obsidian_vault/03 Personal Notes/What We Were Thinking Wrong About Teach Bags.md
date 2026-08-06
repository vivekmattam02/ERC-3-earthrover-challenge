# What We Were Thinking Wrong About Teach Bags

This note captures the main correction in the current no-GPS branch.

## Wrong Assumption

We were treating:

- the cleanest-looking clip
- the most stable-looking camera segment
- one contiguous "best window"

as if that automatically meant "best teach route."

That is wrong for this rover.

## Why It Was Wrong

For no-GPS off-road teach-and-repeat, the route reference is not just an image
quality problem. It is a **traversal evidence** problem.

A segment can look clean and still be bad if it has:

- too little route progression
- too little visual novelty
- too much parked / startup footage
- not enough relocalization support later

That is exactly what happened in the first post-processing pass.

## What The Data Showed

The original post-processing logic selected a visually clean startup span from
`run_01_1521.h5`, but the resulting route package was basically unusable:

- only `2` extracted route images
- `target_step=1`

That proved the selector was optimizing the wrong thing.

## Corrected View

For this branch, a good teach route should preserve:

- motion-rich coverage
- meaningful visual progression
- usable relocalization evidence
- enough continuity for graph progression

So the right selector should prefer:

- good traversal episodes
- short-gap bridging between motion bursts
- route coverage

not just "pretty" frames.

## Current Verdict

- `smoke_run01_c` is still the best coverage/reference candidate
- `run_01_1521_pp_v3` is the strongest post-processed comparison route so far
- `run_01_1521_pp_v3` is useful because it proves the post-processing logic is
  improving
- but it still does not replace `smoke_run01_c` as the main route because it
  is still shorter and sparser

## A Separate Limitation: No Recorded Teleop Commands

The current manual bags contain camera and telemetry evidence, but their
recorded `controls` stream is empty. We can learn a visual route and inspect
motion proxies, but we cannot faithfully replay the human driver's exact
linear/angular commands from these bags.

That keeps route-following and exact imitation as two different problems.

## Practical Rule

When evaluating a teach bag, ask:

- does this preserve route progression?
- does this preserve useful relocalization evidence?
- does this help a repeat controller move through the scene?

If the answer is no, then image cleanliness alone does not matter.

## Related Notes

- [[03 Personal Notes/Current Truth]]
- [[03 Personal Notes/Architecture in My Words]]
- [[90 Archive/CONTEXT - Indoor Source]]
- [[01 Source of Truth/No-GPS Field Trial - Findings]]
