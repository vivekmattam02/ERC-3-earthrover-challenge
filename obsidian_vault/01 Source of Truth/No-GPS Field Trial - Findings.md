# No-GPS Field Trial - Findings

> **Purpose:** Evidence record for the rough-terrain no-GPS route-repeat tests.
> **Authority:** This note records what the rover actually did. It overrides optimistic wording elsewhere; active code remains the source for exact behavior.

## Bottom Line

The no-GPS branch has a usable data and software pipeline, but it has **not** demonstrated a reliable autonomous repeat of the taught route on the real terrain.

What is true:

- manual recording, route preparation, front-camera localization, and command sending work;
- the rover does move under `--send-control`;
- rough-terrain recovery and adaptive-pursuit code are implemented;
- current field runs repeatedly failed to make sustained graph-step progress.

This is therefore an experimental route-repeat system, not a proven competition-ready autonomy capability.

## Hardware And Sensor Reality

- GPS was unusable in the session: telemetry reported `(1000, 1000)`.
- The live no-GPS runtime uses **front camera only** through `/v2/front`.
- Rear-camera data is not part of the live localization or route-following path.
- The rover is differential drive and operates on uneven, rocky terrain; body tilt materially changes both image appearance and traction.

## What Was Tested

The tests used `data/manual_routes/smoke_run01_c` as the reference candidate, normally with:

- `--rough-terrain`
- `--send-control`
- front-camera-only live input
- startup-step hints near the expected start

We also tested route-heading guidance, no-route-heading operation, constrained startup localization, forward-biased recovery, adaptive pursuit, and more aggressive terrain settings.

## Observed Field Behavior

The important log signatures were consistent across attempts:

- localization commonly stayed near steps `0`, `1`, or `8` while confidence remained roughly `0.49-0.59`;
- `relocalize_scan`, `relocalize_probe_forward`, and recovery cycles repeated without sustained route progression;
- route-heading / compass guidance could increase heading error instead of reducing it, so it was disabled for the no-GPS branch;
- tight startup corridors prevented unsafe global jumps but did not create a visual match where the live start disagreed with the teach reference;
- at tilt around `14-22` degrees, cautious crawl commands were often too weak to cross rocks, while stronger probes still did not establish durable localization progress.

This means a lack of graph-step progress is not proof that the rover physically did nothing. It is evidence that the perception, reference route, and recovery state machine did not agree on where it was or how it should continue.

## Changes Made During The Tests

The current code incorporates these hardening changes:

- `src/adaptive_pursuit_controller.py` provides the rough-terrain controller family.
- `scripts/run_prepared_route.py --rough-terrain` selects that controller.
- The live runtime uses front camera only.
- rough-terrain prepared-route launches default to no route-heading guidance;
  `--use-route-heading` is an explicit opt-in because the compass/route-heading
  relation was unreliable in these tests.
- Startup localization can be constrained with a step hint, radius, and lock period.
- Corridor localization no longer silently falls through to a global match when a constrained window has no valid candidates.
- Rough-terrain recovery is forward-biased rather than repeatedly reversing on rocks.
- Tilt, RPM, confidence, and no-progress state affect command generation.

These are code changes, not field validation. The field evidence above still governs the maturity claim.

## Route And Recording Evidence

| Artifact | Role | Verdict |
|---|---|---|
| `run_01_1521.h5` | strongest original teach recording | best coverage candidate, not a proven autonomous route |
| `smoke_run01_c` | 126-frame reference, `target_step=125` | main comparison route; unproven in field repeat |
| `run_01_1521_pp_v3` | 15-frame post-processing experiment, `target_step=14` | useful selector experiment; too sparse for primary use |
| `run_02_1537.h5` | alternate teach recording | weaker: dark segment and poor tail |
| `run_03_1551.h5` | alternate teach recording | usable comparison only |
| `run_05_clean_teach.h5` | later recording | unreviewed; no route package or validation verdict yet |

The existing manual recordings have `controls: 0`. They contain video and telemetry but not the original teleoperation command stream, so exact command replay cannot be reconstructed from them.

## Decision Rule Before Another Autonomous Attempt

Do not treat a long autonomous no-GPS run as justified until a controlled hardware-in-the-loop test demonstrates all of the following on the same physical route:

1. startup localization stays in the expected corridor without a manual step hint being wrong;
2. graph step advances across a meaningful route segment;
3. the controller does not enter repeated scan/probe loops;
4. command strength is sufficient for the terrain without increasing unsafe tilt;
5. a short repeat can be reproduced, not merely observed once.

Until then, the honest operational position is that this branch is a promising prototype and its safest role is supervised evaluation, not unattended final-competition execution.

## Related Notes

- [[00 Home/Current No-GPS - Read This First]]
- [[01 Source of Truth/Offroad Track - Full Story]]
- [[01 Source of Truth/Offroad Controller - Full Story]]
- [[03 Personal Notes/Current Truth]]
- [[04 Runs and Failures/Run Outcomes]]
