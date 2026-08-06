# Offroad Track - Full Story

> **Purpose:** Full detailed story of the current off-road branch: how the problem changed, what we built, what failed, what we corrected, and what the actual current strategy is.
> **Read when:** I need the real no-GPS off-road story, not just a short status summary.
> **Authority:** Current source-of-truth narrative for the off-road branch. Trust active code for low-level behavior and [[01 Source of Truth/No-GPS Field Trial - Findings]] for current field maturity.

## Why This Note Exists

The vault already had long notes for:

- the original outdoor GPS mission branch
- the marathon hardening branch
- the outdoor perception investigation

Those notes still matter, but they do **not** fully describe the current off-road branch anymore.

The current branch became a different engineering problem:

- no GPS
- rough / uneven terrain
- differential-drive EarthRover
- limited time at the final event
- no room for exploratory wandering before the real attempt

So this note exists to record the actual branch we are on now.

## The Problem Changed

The original outdoor story was:

- mission checkpoints
- GPS coordinates
- optional OSM route expansion
- LogoNav as the local controller
- runtime safety around that mission flow

That is not the current situation.

The current situation became:

- the local session exposed placeholder GPS
- the rover still had a live camera, IMU, RPM, and telemetry stream
- the environment was rough and physically unforgiving
- we needed a solution that could start and run without relying on GPS

That changed the core problem from:

- “follow outdoor mission checkpoints with GPS support”

to:

- “teach a route visually, then repeat it autonomously on rough terrain”

That is the most important branch change in the entire recent project history.

## What We Learned Immediately

The first key reality check was that the current mission session was live, but GPS was not usable.

Operationally, that meant:

- the old `live_outdoor_runtime.py` branch was not the active answer
- the current active runtime had to become `live_indoor_runtime.py` repurposed for no-GPS route repeat

That was not just a convenience choice. It was the only branch in the repo that already matched:

- visual localization
- graph progression
- no-GPS operation
- teach-and-repeat logic

## Why We Did Not Jump Straight To Flag Detection

There was a tempting alternative:

- “just detect the flags directly and drive to them”

That sounds simpler than it really is.

At that moment in the repo, we did **not** have:

- a real trained flag detector
- a good enough label set
- enough field data to trust one quickly
- enough competition-time margin to build and prove it safely

So the practical decision was:

- do not pretend we already have robust free-space flag-seeking autonomy
- use a teach-and-repeat route strategy instead

That was a pragmatic engineering decision, not a theoretical claim that direct flag detection is impossible.

## The Actual Offroad Strategy

The current off-road branch is:

1. manually drive the route once
2. record the session into an H5 bag
3. turn that bag into a route-reference package
4. localize online against that visual route
5. follow the route with a conservative rough-terrain controller
6. relocalize and recover if progress stalls

This is not generic exploration.

It is:

- route-specific
- visually grounded
- repeat-oriented
- conservative by design

## What We Built For Data Collection

The first practical need was logging.

We added and used:

- `scripts/record_sdk_session.py`

This gave us:

- front camera frames
- telemetry
- IMU
- magnetometer
- RPMs

This mattered because without good bags, every later discussion about localization or control would have been speculation.

## The First Three Real Bags

The most important early recordings were:

- `run_01_1521.h5`
- `run_02_1537.h5`
- `run_03_1551.h5`

These are not equal.

### Run 1

`run_01_1521.h5` became the strongest bag.

Why:

- best overall route coverage
- most useful traversal evidence
- strongest candidate for a canonical teach route

### Run 2

`run_02_1537.h5` was not chosen as the main route.

Why:

- dark / near-black segment
- poor tail behavior
- weaker route quality than run 1

### Run 3

`run_03_1551.h5` was usable, but weaker than run 1.

Why:

- useful motion early
- poorer tail later
- not the strongest overall reference

The important lesson here is that “bag exists” is not the same thing as “bag is route-worthy.”

## How We Turned Bags Into Routes

We built a full route-preparation pipeline around:

- `tools/extract_h5_dataset.py`
- `baseline.py`
- `scripts/prepare_manual_route.py`
- `scripts/visualize_manual_route.py`
- `scripts/run_prepared_route.py`

This gave us a repeatable workflow:

- raw H5 bag
- extracted front-image dataset
- descriptor DB
- place graph
- navigation graph
- live launch command

That was the first point where the branch stopped being “data collection only” and became a real no-GPS autonomy branch.

## The Early Route Packages

The first prepared route packages were:

- `smoke_run01_c`
- `smoke_run02_c`
- `smoke_run03_c`

These were important because they gave us a concrete operational comparison between bags.

The early offline comparison verdict was:

- `smoke_run01_c` became the best coverage/reference candidate
- `smoke_run02_c` was weaker
- `smoke_run03_c` was usable but still weaker than run 1

That ranking remains useful for offline comparison. It is not a claim that
`smoke_run01_c` is field-proven for autonomous repeat.

## What We Changed In The Runtime

The route itself is not enough. The runtime had to adapt to rough terrain.

The main active runtime became:

- `live_indoor_runtime.py`

but used as a no-GPS off-road route-repeat runtime rather than a pure corridor runtime.

The important runtime changes included:

- rough-terrain mode
- startup relocalization probing
- no-progress relocalization search
- tilt-aware slowdown
- startup-step hinting and constrained startup localization

The important support changes included:

- `src/sensor_state.py` now estimating filtered roll, pitch, and tilt
- `src/local_controller.py` using continuous heading correction during forward drive

This is the beginning of a terrain-capable route-repeat stack, not the final controller.

## What The Controller Problem Actually Is

The main controller lesson was simple:

- align-only behavior is too brittle
- a rough-terrain differential-drive rover needs continuous steering while moving
- recovery must be active, not passive

So the current controller work moved toward:

- slower forward drive on bad terrain
- heading correction during motion
- relocalization search when progress stalls

It is still heuristic.

That matters.

We should not overclaim and say the final controller problem is solved.

## What The Field Trials Changed

The first live tests were necessary because an offline route package can look
credible while failing on the physical rover.

The result was mixed:

- the front-camera live path worked and commands reached the rover;
- the rover could physically move on the rocky terrain;
- localization often stayed near startup steps (`0`, `1`, or `8`) with flat,
  only moderate confidence;
- recovery then repeated scan/probe behavior rather than establishing route
  progression.

We also saw that compass/route-heading guidance could turn in the wrong useful
direction. It is therefore disabled for this no-GPS field branch rather than
being trusted as a control authority.

This is a key correction to the story: having a route, a controller, and live
motion is not the same as having a repeatable autonomous route follower. The
remaining issue is the agreement between the physical start, visual reference,
localization corridor, and recovery behavior.

The evidence is kept separately in [[01 Source of Truth/No-GPS Field Trial - Findings]].

## The Most Important Post-Processing Mistake

This was the biggest recent conceptual correction.

At first, the post-processing logic treated the teach bag like an image-quality problem.

That led to the wrong selector behavior:

- it chose visually clean, stable-looking clips
- it implicitly treated one contiguous “best window” as the ideal answer

That was wrong.

For this rover, the route reference is not just a visual cleanliness artifact.
It is a **traversal evidence artifact**.

The first post-processing pass proved this mistake very clearly:

- it selected a nice-looking segment from `run_01_1521.h5`
- but the resulting route package was effectively useless
- only `2` extracted route images
- `target_step=1`

That was the proof that the selector was optimizing the wrong thing.

## What We Corrected

The corrected post-processing logic moved toward:

- motion-supported traversal episodes
- short-gap bridging between motion bursts
- preservation of route progression
- preservation of relocalization evidence

That produced a much better comparison artifact:

- `run_01_1521_pp_v3`

This route matters because it proves the post-processing direction is now more correct.

But it still does **not** replace `smoke_run01_c`.

Why:

- `smoke_run01_c` still has much stronger route coverage
- `run_01_1521_pp_v3` is still shorter and sparser

Operationally:

- `smoke_run01_c`: `126` extracted frames, `target_step=125`
- `run_01_1521_pp_v3`: `15` extracted frames, `target_step=14`

That is the current route verdict.

## The Main Wrong Assumption In One Sentence

The best teach route is **not** the cleanest-looking clip.

The better teach route is the one that preserves:

- motion-rich coverage
- visual progression
- useful relocalization evidence

That is the core lesson.

## What The Current Offroad Branch Inherits From Older Work

Even though this branch is different, it inherited several important lessons from the older outdoor and marathon branches:

- transition stability matters more than adding more safety layers on paper
- weak perception signals should not be promoted into fake authority
- runtime discipline matters as much as controller cleverness

So the current branch is not disconnected from the older work. It is a new branch built on top of those lessons.

## What Is Actually Working Now

What is real right now:

- manual no-GPS bag collection works
- bag -> route package pipeline works
- prepared-route launch flow works
- the rover can receive and execute live route-following commands
- rough-terrain runtime hooks exist
- relocalization and no-progress logic exist

This is meaningful progress, but it is not proof of an autonomous repeat.

## What Is Not Solved Yet

The main unsolved items are:

- final off-road controller quality
- canonical teach-route selection
- long repeat stability on rough terrain
- startup/reference agreement and stable visual progression
- competition-grade direct-start reliability

That means we are not at “finished autonomy.”

We are at a real prototype branch with working infrastructure and motion, but
without a field-proven autonomous repeat.

## The Current Engineering Strategy

The correct engineering strategy is:

use a prebuilt visual route package, constrain localization only when the
physical start actually matches the reference, and validate short repeatable
segments before attempting a longer run.

It is not justified to call any current command a final-competition autonomous
deployment command until that validation exists.

## Current Route Verdict

- main coverage/reference candidate: `smoke_run01_c`
- strongest post-processed comparison route: `run_01_1521_pp_v3`
- neither route is a field-proven deployment route
- `run_01_1521_pp_v3` is the proof that our post-processing logic is getting closer to the right objective

## What To Read With This Note

- [[00 Home/Current No-GPS - Read This First]]
- [[01 Source of Truth/Offroad Controller - Full Story]]
- [[03 Personal Notes/Current Truth]]
- [[03 Personal Notes/What We Were Thinking Wrong About Teach Bags]]
- [[03 Personal Notes/Architecture in My Words]]
- [[04 Runs and Failures/Run Outcomes]]
- [[01 Source of Truth/No-GPS Field Trial - Findings]]
