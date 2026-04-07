# Indoor Runtime Story: - From Baseline, Through Failures, To The Current MBRA-First Path

> Source: `live_indoor_runtime_story.tex`
> Master Note: [[erc3_full_documentation]]

# Purpose

This document tells the full indoor story: what the indoor problem really was, what assumptions were correct, what we tried, what failed, what partially worked, what finally became the cleanest path, and why.

This is intentionally written like a narrative handoff, not just a changelog. The goal is that a new person can read this from top to bottom and understand not only *what* we changed, but *why* we changed it and what mistakes should not be repeated.

Two AI assistants contributed to this codebase --- Codex and Claude. This document is a joint account. Where their perspectives diverge or where one discovered something the other missed, it is noted explicitly.

# What Indoor Actually Was

Indoor was not a GPS problem. It was not an outdoor waypoint problem. It was not generic exploration. It was a **known-corridor visual navigation problem**.

The indoor stack was supposed to be:
- Visual localization against a recorded corridor dataset (1865 images, CosPlace ResNet18, 512-dim descriptors).
- Temporal stabilization of localization (continuity penalties, jump rejection, ambiguity handling).
- Graph planning over corridor steps (forward-only directed graph, topological subgoal selection).
- Short-horizon local control toward a nearby subgoal.

The most important repo-level observation was this: **localization and graph planning were the strong parts; local control was the weak part**. That single idea should have guided the whole indoor effort. When it did, things got better. When it didn't, things got worse.

# Why The Dataset Changed Everything

A crucial piece of information was that the competition would be run in the **same corridor as the dataset**. That changed the indoor problem from "generalize to a new building" into "recognize and traverse a corridor we have already mapped."

We also got exact checkpoint frames from the corridor database:
```
45 480 761 821 1094 1208 1345 1430 1544 1638 1764
```

This was extremely important because these were exact dataset steps, not guessed goals. That meant the cleanest competition formulation was step-based checkpoint navigation:
```
--checkpoint-steps 45 480 761 821 1094 1208 1345 1430 1544 1638 1764
```

That removed a whole class of ambiguity and operator error.

# What The Old Indoor Baseline Told Us

The old indoor runtime had one very important property: it could often recognize the corridor better than some of the later heavily modified versions. The old baseline was still weak in important ways --- turning was weak, short-horizon control was weak, obstacle handling was weak, getting stuck was not handled well, and operator mistakes like dry-run confusion were common.

Still, it served as a reference point. When the user said "*we are messing fundamentals, first gather foundations*," they were pointing at exactly this: the old code, for all its limitations, at least moved forward and localized. Some of the newer versions didn't even do that.

**Claude's observation:** When I was brought in, the robot was stuck at `runtime_no_path_stop` for 20+ iterations. The log showed `cur=161 target=45 path_error=no_path:161->45`. The robot had localized past checkpoint 1 in the directed graph. There was no backward path. The system was correctly reporting "no path" --- it wasn't a bug in the planner, it was a bug in the runtime's inability to handle this case. The old code didn't have this problem because it didn't use checkpoint-step mode in the same way.

# What We Observed In Practice

Several practical observations kept repeating.

## Dry-Run Confusion

If the command did not include `--send-control`, the runtime computed commands but the robot did not move. This caught the operator more than once.

## The Forward-Only Graph

The corridor graph behaves like a forward-only graph in practice. If the localizer said `cur=278 target=45 path_error=no_path:278->45`, the planner was not "being dumb." It was correctly saying there was no backward path to checkpoint 45.

**The fix:** Skip-past-checkpoint logic. If the robot is already past a checkpoint (current step > target step) with reasonable confidence ($\geq 0.45$), advance to the next checkpoint instead of stopping forever. This was one of the most important runtime fixes.

## Localization Failures Looked Like Controller Failures

A lot of runs that looked like "controller failures" were actually localization failures first. The usual pattern was:
1. Localization looked reasonable for a few ticks.
1. Then confidence collapsed.
1. Then low-confidence stop or jump-rejection logic took over.
1. Only after that did the controller appear useless.

## Compass Heading Was Poison Indoors

**Claude's observation:** The compass heading was being passed to the localizer's temporal filter, where it added a `heading_penalty` to match scoring. Indoors, the compass is unreliable (motor interference causes jumps of 50--150 degrees/tick). This meant the localizer was penalizing correct matches and rewarding wrong ones based on garbage heading data.

The fix was simple: pass `observation_heading_deg=None` to the localizer. Heading is still used by the simple controller for gyro-based drift correction (gyro Z is reliable), but it should never touch localization scoring indoors.

**Important:** The recovery file (`live_indoor_runtime_recovery.py`) still passes heading to the localizer. This is a known defect in that file.

## Repetitive Corridor Structure Caused Aliasing

The corridor has long stretches that look nearly identical. If the localizer was allowed to re-anchor too freely, it could jump to visually similar but wrong parts of the corridor. The temporal filter's continuity penalty (`jump_penalty=0.05` per step beyond `max_step_jump=20`) was the main defense against this. The runtime's jump rejection (reject jumps $>30$ steps unless confidence exceeds $0.35 + 0.005 \times \text{jump magnitude}$) was the secondary defense.

# Why MBRA Mattered

MBRA made sense only in one role: **short-horizon local control between nearby graph subgoals**. It was never the full indoor stack. It did not replace localization. It did not replace the graph planner. It was the better local controller candidate.

Here is what MBRA actually does:
- Takes a 6-frame observation history (96$\times$96 RGB images).
- Takes a single goal *image* (the subgoal from the graph).
- Takes fixed velocity past values (linear=0.5, angular=0.0 --- not model feedback).
- Outputs an 8-step trajectory. Index [0] is the immediate next command.
- Linear range: [0, 0.5] m/s. Angular range: [-1.0, 1.0] rad/s.

The key insight is that MBRA *steers toward the subgoal image*. The simple controller, when `subgoal_orientation=None` (which is always the case indoors because compass heading is unreliable), just drives forward with zero angular. It is literally blind to direction.

**Claude's observation:** The user's complaint "*this is walking to random things and not going to subgoal image*" was about the simple controller. With `subgoal_orientation=None`, the simple controller's heading-aware logic is entirely disabled. It becomes a forward-crawl-only controller. Switching to MBRA immediately produced directional behavior because MBRA uses the subgoal *image*, not the subgoal *orientation*.

# The Three Files: How We Got Here

The indoor effort produced three runtime files. Understanding how and why they diverged is important.

## `live_indoor_runtime.py` --- The Canonical File

This is the main file that both Claude and Codex edited. It carries the full history of changes. Key properties:
- Controller selection via `--controller simple|mbra` (default: simple).
- All MBRA-specific guards: recovery backup, RPM stall backup, and angular saturation override are **disabled** when `--controller mbra`.
- Compass heading disabled for localization (`observation_heading_deg=None`).
- Skip-past-checkpoint logic for the forward-only graph.
- Jump rejection with localizer state revert.
- MBRA subgoal hops: 8 (increased from 4 --- see Section~\ref{sec:subgoal_hops}).
- Depth thresholds: 0.25m stop / 0.6m slow for MBRA (tighter than simple's 0.4/0.8).
- No-progress context reset every 10 ticks (resets MBRA observation history).

## `live_indoor_runtime_mbra.py` --- Codex's MBRA-First Variant

This was created by Codex as a cleaner MBRA-first entry point. Its main difference is `default="mbra"` for the controller flag. In practice, this file is now functionally behind `live_indoor_runtime.py` because it is missing:
- The `use_mbra` flag and all MBRA-specific guard logic (backup disable, angular saturation disable, RPM stall disable).
- The updated subgoal hops (still 4, should be 8).
- The controller-aware depth thresholds (still hardcoded 0.4/0.8).

**Recommendation:** This file should be merged back into `live_indoor_runtime.py` or deprecated. The only difference that mattered --- defaulting to MBRA --- is a single line change. Having two nearly-identical files that drift apart is how bugs hide.

## `live_indoor_runtime_recovery.py` --- The Experimental Heavy Recovery File

This is the most complex variant (709 lines vs 593). It adds:
- **Same-frame detection:** Compares downsampled frame signatures across ticks. If frames are identical for 10+ ticks while commanding forward, triggers backup + spin.
- **IMU spin guard:** If gyro reports $>25$ deg/s while the robot is stuck and commanding both linear and angular, triggers backup + spin.
- **Saturation escalation:** Counts angular saturation events. After 3 events, triggers backup + spin.
- **Recovery cooldown:** 10-tick cooldown after any recovery to prevent chaining.
- **Stuck recovery stages:** First stuck triggers spin-only. Same area stuck again triggers backup + spin.
- **Confirmed skip:** Requires 2 consecutive ticks of skip-past-checkpoint before acting (vs immediate in main file).

This file also has a **known defect**: it passes compass heading to the localizer (`observation_heading_deg=heading_deg`), which adds noise to match scoring indoors.

**The lesson this file teaches:** Recovery logic can save a bad controller, but it can also drown a good controller. Once the runtime became too reactive, it spent more time escaping imagined failures than actually traversing the corridor. With MBRA as controller, the heavy recovery is actively harmful --- every backup destroys MBRA's 6-frame visual context.

# Experiments That Failed

## Aggressive Warmup And Localization Gating

This sounded good in theory: stabilize before moving, constrain startup behavior, reject early bad matches. In practice, it often locked the localizer into a wrong early belief and continuity logic kept dragging the state back there.

## Restricting Localization Around The Active Checkpoint

If the active checkpoint is step 45, why not localize near 45? But if the robot was not exactly where we expected, or if startup localization was already wrong, this became a trap. The localizer was forced to choose among the wrong candidates.

## Database-Orientation-Based Turning

This was one of the most attractive wrong ideas. The dataset had orientation metadata, so it was tempting to use database orientation as a turning reference. The problem: database orientation is static per matched frame, while the robot's live heading changes continuously during a real turn. In naive form, this produced brittle and misleading heading control.

**Claude's observation:** The simple controller had a full gyro-tracked turning system (align mode with start heading and target delta). But with `subgoal_orientation=None` forced indoors, this entire system was disabled. It was dead code that looked alive.

## Recovery Backup Fighting MBRA

**Claude's observation:** This was the single most destructive failure mode. Three mechanisms triggered reverse driving:
1. **Depth safety backup:** On `depth_stop`, set `recovery_backup_remaining = 5`. Robot reverses for 5 ticks.
1. **RPM stall backup:** If RPM $< 1.5$ while commanding forward for 8 ticks, trigger reverse.
1. **Jump rejection backup:** After 12 consecutive jump rejections, trigger reverse.

Each reverse cycle:
1. Destroyed MBRA's 6-frame observation context.
1. Reset the localizer.
1. MBRA restarted from scratch with a pre-filled context of identical frames.
1. MBRA drove forward again.
1. Hit the same trigger.
1. Reversed again.

This produced the back-and-forth oscillation the user described: "*this is just going back and front.*"

**The fix:** When `use_mbra=True`, all three backup triggers are disabled. Depth safety still stops the robot (linear=0, angular=0), but does not reverse. RPM stall detection is entirely disabled. Jump rejection resets the localizer without reversing. MBRA is the sole motion authority.

## "Already Past Means Reached" Logic

Because the graph was forward-only, it was tempting to say that if current step was greater than target step, the checkpoint should count as reached. This solved one problem but created another: if localization was wrong once, the runtime could skip many checkpoints, even all of them, before the robot had actually completed anything. The recovery file adds a 2-tick confirmation; the main file skips immediately but requires $\geq 0.45$ confidence.

## MBRA Warmup Stall

MBRA requires 6 frames (context_size + 1) before it can produce meaningful output. Early versions returned $(0, 0)$ for the first 2 seconds while the context filled.

**The fix:** Pre-fill the observation history with copies of the first frame. MBRA starts immediately with a "standing still, looking at this" context, which is better than no context. This is in `src/mbra_controller.py`:
```
if len(self._obs_history) < self.context_size + 1:
    while len(self._obs_history) < self.context_size + 1:
        self._obs_history.appendleft(obs_pil)
```

# What Worked

The things we should confidently say worked:
- Framing indoor as a known-corridor visual navigation problem.
- Using exact corridor-step checkpoints (not images, not GPS).
- Keeping CosPlace VPR + temporal filtering as the backbone.
- Using MBRA as the short-horizon controller (it steers toward *images*).
- Disabling compass heading for localization (gyro Z is fine for controller drift correction).
- Skip-past-checkpoint logic for the forward-only graph.
- Jump rejection with localizer state revert (prevents localization runaway).
- Letting MBRA be sole motion authority (no backup, no angular override, no RPM stall recovery).
- Pre-filling MBRA context on startup.
- Speed bumps: max_linear 0.24 $\to$ 0.40, making the robot actually move at a useful speed.

## MBRA Subgoal Hop Distance

MBRA navigates toward a goal *image*. If the subgoal is only 4 hops ahead in a straight corridor, the current frame and the subgoal frame look nearly identical. MBRA has no directional signal --- it doesn't know which way to steer. Bumping subgoal hops from 4 to 8 gives MBRA a clearer "where am I going" image that is visually distinct from "where I am now."

## Controller-Aware Depth Thresholds

MBRA handles obstacles visually --- it sees chairs and walls in its observation frames and steers around them (that's what its learned policy does). Hard-stopping at 0.4m was too aggressive; it prevented MBRA from doing its job. Reducing the stop threshold to 0.25m (only very close obstacles) and slow threshold to 0.6m lets MBRA navigate more naturally.

# The Concrete Edits Claude Made

In order, across the session:

1. **Skip-past-checkpoint logic** in `live_indoor_runtime.py`. When `cur_step > tgt_step` and `confidence >= 0.45`, advance to next checkpoint instead of stopping at `runtime_no_path_stop` forever.

1. **Speed bump** for both controllers: `max_linear 0.24 $\to$ 0.40`, `min_linear 0.12 $\to$ 0.20` (simple) / `0.10 $\to$ 0.18` (MBRA).

1. **MBRA warmup fix** in `src/mbra_controller.py`: Pre-fill observation history with copies of first frame instead of returning $(0, 0)$ for 6 ticks.

1. **Compass heading disabled for localizer**: Changed from `observation_heading_deg=heading_deg` to `observation_heading_deg=None` in both `step_to_active_checkpoint` and `step_to_target` calls.

1. **MBRA min_linear enforcement removed**: MBRA can now output zero linear (it may be stopping intentionally for obstacles). Previously, any output $< 0.10$ was clamped up.

1. **Recovery backup disabled for MBRA**: All three backup triggers (depth, RPM stall, jump rejection) gated by `if not use_mbra`.

1. **Angular saturation override disabled for MBRA**: Entire block wrapped in `if not use_mbra`.

1. **Jump rejection for MBRA**: Resets localizer + controller without reversing (vs backup + reverse for simple controller).

1. **Subgoal hops**: 4 $\to$ 8 for MBRA.

1. **Depth thresholds**: Controller-aware defaults. MBRA: 0.25m stop / 0.6m slow. Simple: 0.4m / 0.8m.

1. **Temporal localizer parameters**: `top_k: 5 $\to$ 10`, `max_step_jump: 15 $\to$ 20` (more candidates, wider acceptance for fast movement).

# Comparing The Three Files

```text
\hline
\textbf{Feature} & \textbf{runtime.py} & \textbf{\_mbra.py} & \textbf{\_recovery.py} \\
\hline
Default controller & simple & \textbf{mbra} & simple \\
MBRA guard logic & \textbf{yes} & no & no \\
Compass to localizer & \textbf{None} & \textbf{None} & heading\_deg (defect) \\
Subgoal hops (MBRA) & \textbf{8} & 4 & 4 \\
Depth thresholds (MBRA) & \textbf{0.25/0.6} & 0.4/0.8 & 0.4/0.8 \\
Skip-past-checkpoint & immediate & immediate & 2-tick confirm \\
No-progress reset & 10 ticks & 10 ticks & 20 ticks \\
Backup + spin recovery & no (MBRA) & no & yes \\
IMU spin guard & no & no & yes \\
Same-frame detection & no & no & yes \\
Saturation escalation & no & no & yes \\
Recovery cooldown & no & no & yes \\
DepthEstimator init & max\_depth=5.0 & indoor domain & indoor domain \\
Lines of code & 593 & 587 & 709 \\
\hline
```

\textbf{Winner: `live_indoor_runtime.py`} with `--controller mbra`. It has all the MBRA-specific fixes and none of the heavy recovery overhead.

The DepthEstimator initialization in `live_indoor_runtime.py` uses `max_depth=5.0` while the other files use `checkpoint_domain='indoor'`. This is a minor discrepancy that should be unified.

# Mistakes We Should Not Repeat

1. **Never reverse a visual context controller.** MBRA's 6-frame history is its memory. Reversing the robot destroys it. Every recovery mechanism that involves backward driving is incompatible with MBRA.

1. **Never pass compass heading to the indoor localizer.** The compass is garbage indoors. It adds noise to match scoring. Gyro Z is fine for drift correction in the controller, but heading should never touch localization.

1. **Never duplicate runtime files for minor differences.** The `_mbra.py` and `_recovery.py` files drifted from the main file. Fixes applied to one were not applied to the others. A single file with flags is always better than three files that look 90% identical.

1. **Never enforce minimum linear speed on a learned controller.** If MBRA outputs zero, it may be stopping intentionally. Clamping it up causes crashes into obstacles.

1. **Never add recovery logic without measuring whether it helps the actual controller.** Recovery backup was designed for the simple controller (which is truly blind without heading). It was catastrophic for MBRA (which has visual context). The same mechanism can be essential for one controller and fatal for another.

1. **Never restrict localization candidates to a narrow window.** If the robot isn't where you expect, restricting localization to "near the target" traps it. Let the full database speak and trust the temporal filter to maintain continuity.

1. **Never let localization jumps imply checkpoint completion.** A single bad localization that says "step 800" when the robot is at step 100 should not skip 5 checkpoints. Require confidence and ideally multi-tick confirmation.

1. **Never underestimate operator error.** Forgetting `--send-control`, typos in checkpoint lists, and shell formatting errors caused real debugging time. The startup banner should print every active setting clearly.

# Recommended Indoor Competition Procedure

## Before the Competition

1. Verify the SDK server is running and the robot is connected.
1. Verify the corridor database exists: `data/corrider_db/descriptors.npz` (1865 images, 512-dim).
1. Verify MBRA weights exist: `mbra_repo/deployment/model_weights/mbra.pth`.
1. Run a 20-iteration dry run to confirm localization works:
```
python live_indoor_runtime.py \
  --checkpoint-steps 45 480 761 821 1094 1208 1345 1430 1544 1638 1764 \
  --auto-advance-checkpoints --controller mbra --depth-safety --max-steps 20
```
1. Check that the first few iterations show reasonable `cur=` step values and `conf > 0.4`.

## Competition Run

```
# Terminal 1 — SDK
sudo fuser -k 8000/tcp 2>/dev/null
cd ~/Desktop/rover/ERC-3-earthrover-challenge/earth-rovers-sdk
conda activate erv
hypercorn main:app --reload

# Terminal 2 — Mission
curl -X POST http://127.0.0.1:8000/start-mission

# Terminal 3 — Navigation
python live_indoor_runtime.py \
  --checkpoint-steps 45 480 761 821 1094 1208 1345 1430 1544 1638 1764 \
  --auto-advance-checkpoints --send-control --controller mbra --depth-safety
```

## If Checkpoint Images Are Provided Instead of Steps

```
python live_indoor_runtime.py \
  --checkpoint-image-files /path/to/img1.jpg /path/to/img2.jpg ... \
  --auto-advance-checkpoints --send-control --controller mbra --depth-safety
```
The runtime will prelocalize each image to a database step at startup and print the equivalent `--checkpoint-steps` command.

## If the Robot Gets Stuck

1. Ctrl+C to stop the runtime.
1. Physically reposition the robot if needed.
1. Restart the runtime. Localization will re-anchor from the new position.
1. The temporal filter resets on startup, so there is no stale state.

## Emergency Fallback

If MBRA is not working, fall back to the simple controller:
```
python live_indoor_runtime.py \
  --checkpoint-steps 45 480 761 821 1094 1208 1345 1430 1544 1638 1764 \
  --auto-advance-checkpoints --send-control --controller simple --depth-safety
```
The simple controller drives forward with gyro drift correction but no directional steering. It may reach checkpoints in a straight corridor but will not navigate turns.

# What Is Still Uncertain

- **MBRA's obstacle avoidance quality.** MBRA was trained on outdoor paths. It may or may not steer well around indoor chairs and obstacles. The depth safety layer (stop at 0.25m, slow at 0.6m) is the backup, but it only looks forward.

- **Subgoal hop distance.** We changed from 4 to 8 based on reasoning ("MBRA needs visual difference"), but we have not run a controlled experiment to find the optimum. 6 or 10 might be better.

- **Startup localization in repetitive segments.** If the robot starts in a featureless corridor section, the first few localizations may be wrong. The temporal filter will correct over time, but the first few commands may be misguided.

- **Checkpoint reach detection tolerance.** Currently `checkpoint_reach_tolerance=3` steps and `min_confidence_to_advance=0.55`. If the robot passes near a checkpoint but localization jitters, it might not register as reached. The skip-past-checkpoint logic is the safety net.

- **The DepthEstimator init discrepancy.** The main file uses `max_depth=5.0` while the other files use `checkpoint_domain='indoor'`. The outdoor depth investigation found that `max_depth=5.0` compressed the depth range incorrectly for the vkitti checkpoint. Indoor should use the correct domain or a sensible max_depth.

# What We Learned

The biggest lesson was that indoor did not need a magical new architecture. It needed discipline. Localization and graph planning were already the backbone. MBRA was the better local controller. The runtime wrapper needed to stop interfering so much.

There was also an operator lesson: not every failure was deep robotics. Some were simply wrong flags, dry-run confusion, typoed checkpoint lists, or testing from a runtime whose logic had drifted too far from the original indoor assumptions.

There was a collaboration lesson: two AI assistants working on the same codebase can produce three divergent files where one would suffice. The recovery file taught us what not to do. The MBRA file duplicated what the main file already supported via a flag. Next time, one canonical file with clear flags and well-commented controller-specific branches.

Finally, there was a research lesson: plausible ideas are not the same thing as good runtime behavior. Indoor debugging repeatedly showed that sensible-looking fixes can make the system worse if they violate the structure of the actual problem. The most damaging interventions were the ones that sounded most reasonable.

# Final Summary

The indoor story had three phases.

**First**, we had to understand the problem correctly: known corridor, visual localization, graph planning, short-horizon local control. Not GPS. Not exploration. Not a new building.

**Second**, we had to fail in interesting ways: localization gating, candidate restriction, database-orientation turning, skip logic, heavy recovery, compass heading indoors, backup fighting MBRA. Each failure taught us where the brittle points really were.

**Third**, we moved toward a cleaner MBRA-first path: exact checkpoint steps, graph planning, MBRA as local controller, minimal runtime wrapper, controller-specific safety guards, no backup for MBRA. All in one file: `live_indoor_runtime.py --controller mbra`.

The recommended competition command:
```
python live_indoor_runtime.py \
  --checkpoint-steps 45 480 761 821 1094 1208 1345 1430 1544 1638 1764 \
  --auto-advance-checkpoints --send-control --controller mbra --depth-safety
```

That is the real point of the MBRA-first approach. It is not a claim that indoor is permanently solved. It is the clearest expression of what we learned after all the detours.

# Code-Verified Addendum: What The Current Indoor Runtime Actually Deploys

The story above records the indoor debugging arc correctly, but the repo has continued to evolve after some of those conclusions were first written down. This addendum states the current deployed indoor runtime behavior directly from code so the document reflects both the historical reasoning and the present implementation.

## Current Runtime Defaults For MBRA

In the current `live_indoor_runtime.py`, the MBRA path now defaults to:
```
args.max_subgoal_hops = 8 if args.controller == "mbra" else 15
args.tick_hz = 3.0 if args.controller == "mbra" else 2.0
args.depth_stop_m = 0.25 if args.controller == "mbra" else 0.4
args.depth_slow_m = 0.6 if args.controller == "mbra" else 0.8
```

That means the current indoor runtime is explicitly configured to:
- run MBRA at 3 Hz,
- use 8-hop graph subgoals by default,
- and keep a more aggressive depth safety envelope for MBRA than for the simple controller.

This is important because some earlier sections of this document preserve the intermediate conclusion that 4 hops was the right MBRA setting. That conclusion was part of the real debugging history, but it is no longer the current runtime default. The present deployed indoor runtime uses 8 hops.

## The Runtime Still Prints The Active Indoor Envelope Clearly

The startup banner in the current file explicitly prints the control regime:
```
print(f"Controller: {args.controller}")
print(f"Subgoal hops: {args.max_subgoal_hops}")
print(f"Depth safety: stop={args.depth_stop_m:.2f}m slow={args.depth_slow_m:.2f}m")
print(f"Tick rate: {args.tick_hz:.2f} Hz")
```

This is worth documenting because it directly answers one of the operator-error lessons discussed earlier: the runtime now surfaces the configuration clearly instead of making the user infer it.

## No-Reverse For MBRA Remains A Hard Runtime Principle

One of the strongest historical lessons in this document was that reversing is destructive for a visual-context controller. The current indoor runtime still encodes that principle explicitly:
```
if use_mbra:
    # MBRA: just reset localizer, no reverse (reverse kills context)
    runtime.reset()
    if hasattr(controller, 'reset'):
        controller.reset()
    command.linear = 0.0
    command.angular = 0.0
    command.reason = "jump_reject_reset"
```

This is a very important continuity point. The indoor story did not merely argue that reverse was harmful for MBRA; that lesson survived into the actual deployed runtime.

## The No-Progress Context Reset Also Survived

The other major MBRA-specific lesson that remains in the code is the stale-context reset:
```
NO_PROGRESS_RESET_TICKS = 10
...
if no_progress_count > 0 and no_progress_count % NO_PROGRESS_RESET_TICKS == 0:
    if hasattr(controller, 'reset'):
        controller.reset()
        print(f"[{iteration:04d}] no-progress reset after {no_progress_count} ticks at step {cur_step}")
```

So the current indoor runtime still assumes that if MBRA stops making graph progress for long enough, resetting visual context is a better intervention than forcing a reverse backup.

## Current MBRA Wrapper Defaults

The wrapper in `src/mbra_controller.py` currently uses:
```
max_linear = 0.40
min_linear = 0.18
max_angular = 0.34
min_confidence = 0.45
low_confidence_linear_scale = 0.7
robot_size = 0.30
delay_steps = 0.0
vel_past_linear = 0.5
vel_past_angular = 0.0
```

This confirms that the earlier reverse-engineering findings were not lost. Fixed `vel_past`, `robot_size=0.30`, and `delay=0.0` are still part of the present deployed MBRA wrapper.

## What This Means For The Indoor Story

The best way to read this document now is:
- the main body records the actual debugging and design-learning process,
- the addendum records the current deployed indoor state,
- and together they show both how the system was understood and how it is actually configured now.

That distinction matters. A good engineering story preserves the detours, but a good handoff also states the present runtime truth plainly.

# Related Documents

- [[erc3_full_documentation]] --- single master guide for the complete project story and current architecture.
- [[live_indoor_runtime_story]] --- indoor evolution, MBRA integration, and checkpoint-step runtime behavior.
- [[live_outdoor_ultra_marathon_story]] --- outdoor and marathon runtime evolution with safety-layer reasoning.
- [[outdoor_perception_review]] --- depth/semantic perception findings and their runtime implications.
