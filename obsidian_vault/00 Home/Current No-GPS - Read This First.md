# Current No-GPS - Read This First

> **Use this note before making any no-GPS route-repeat decision.**
> It is the short version. The evidence is in [[01 Source of Truth/No-GPS Field Trial - Findings]].

## Status: Experimental, Not Cleared For Autonomous Competition Use

| Question | Current answer |
|---|---|
| Does the data pipeline work? | Yes. Manual H5 -> route package works. |
| Does the live runtime connect and send commands? | Yes. |
| Does the rover move? | Yes. |
| Has it completed a reliable autonomous taught-route repeat? | **No.** |
| Is `smoke_run01_c` a deployment route? | **No.** It is the best reference/coverage candidate. |
| Can existing bags replay human teleop exactly? | **No.** Their `controls` stream is empty. |

## Problem Contract

This is a **no-GPS, rough-terrain, differential-drive, visual teach-and-repeat**
problem. It is not the older GPS mission stack and it is not generic
exploration. The rover must localize against a manually taught front-camera
route, choose a nearby graph subgoal, and traverse rocks without losing the
route corridor.

## Current Reference Decision

- strongest original teach recording: `run_01_1521.h5`
- main coverage/reference candidate: `smoke_run01_c` (`126` frames,
  `target_step=125`)
- best post-processing comparison: `run_01_1521_pp_v3` (`15` frames,
  `target_step=14`)

`smoke_run01_c` remains the primary reference because it has useful route
coverage. The cleaner-looking 15-frame derivative is a comparison experiment,
not a replacement deployment route. For this rover, route quality means
motion-rich coverage, visual progression, and useful relocalization evidence;
it does not mean the prettiest video alone.

## What We Know

- GPS is unusable in this session: `(1000, 1000)` is a placeholder.
- The no-GPS runtime is front-camera-only. Rear camera is not fused.
- Rough-terrain prepared-route runs use the adaptive pursuit controller.
- Route heading is disabled by default in rough-terrain launches because it was unreliable in field tests.
- Rocks, tilt, weak visual matching, and recovery loops are the current limiting system, not missing CLI plumbing.

## What Failed In The Field

- Localization repeatedly remained around startup steps `0`, `1`, or `8`.
- Confidence was moderate, not decisive, and did not yield sustained graph-step progression.
- The runtime repeatedly entered scan/probe recovery loops.
- Stronger forward commands moved the rover but did not prove that it was following the taught route.

## The Only Valid Next Goal

Prove one **short, repeatable hardware-in-the-loop segment** on the exact same physical route:

1. Start position visually matches the reference start.
2. Localization remains within the expected route corridor.
3. Graph step advances meaningfully.
4. The rover does not cycle through repeated scan/probe recovery.
5. Repeat the segment successfully at least once.

Do not jump from “the rover moves” to “the rover can run the final route.”

## Read Next

- Evidence and exact logs: [[01 Source of Truth/No-GPS Field Trial - Findings]]
- Controller design and limits: [[01 Source of Truth/Offroad Controller - Full Story]]
- Full project narrative: [[erc3_full_documentation]]
