# MBRA Integration Discoveries: - Lessons from Deploying ExAug\_dist\_delay - on the ERC Indoor Corridor

> Source: `docs/our_mbra_discoveries.tex`
> Master Note: [[erc3_full_documentation]]

# Purpose of This Document

This document records everything we learned by reading the full MBRA model
architecture, training pipeline, and reference deployment code, and then
comparing that understanding against our own integration.  The goal is to
preserve these findings so that future development does not repeat the same
mistakes, and so that anyone joining the project can understand exactly how
MBRA works and how it should be used.

# What MBRA Actually Is

MBRA (Model-Based Re-Annotation) uses the `ExAug_dist_delay` model
class, defined in:

`mbra_repo/train/vint_train/models/exaug/exaug.py`

It is a vision-based, image-goal-conditioned local controller.  It is
**not** a localizer, not a planner, and not a full navigation stack.  Its
job is narrow:

\fbox{\parbox{0.85\textwidth}{
Given a short history of what the robot has seen, and an image of where it
should go, output the immediate velocity command (linear and angular) to
drive toward that goal.
}}

# Model Architecture

## Encoder

The backbone encoder is **EfficientNet-B0**.  Two separate EfficientNet
instances are used:

1. **Observation encoder**: standard EfficientNet-B0 with
`in_channels=3`.  Processes each observation frame independently
(batch-stacked for efficiency).
1. **Goal encoder**: EfficientNet-B0 with `in_channels=6`
(when `late_fusion=False`, which is the default).  Takes the
**concatenation of the most recent observation and the goal image**
as a single 6-channel input.

Both encoders produce 1024-dimensional feature vectors (after a linear
compression from EfficientNet's native 1280-dim output).

## Decoder

The decoder is a **Transformer** (`MultiLayerDecoder`) with:
- 4 attention heads
- 4 layers
- feed-forward expansion factor of 4
- embedding dimension 1024
- maximum sequence length of 21 tokens

## Token Composition

The transformer receives a sequence of **21 tokens**, each of dimension
1024:

```text
\toprule
\textbf{Token type} & \textbf{Count} & \textbf{Source} \\
\midrule
Observation encodings & 6 & EfficientNet on 6 RGB frames \\
Goal encoding & 1 & EfficientNet on (last\_obs $\|$ goal\_img) \\
Robot size & 1 & scalar broadcast to 1024 dims \\
Delay steps & 1 & scalar broadcast to 1024 dims \\
Velocity history & 12 & 6 linear + 6 angular, each broadcast \\
\midrule
\textbf{Total} & \textbf{21} & \\
\bottomrule
```

**Critical observation**: velocity history accounts for \textbf{12 of 21
tokens}---more than half the transformer's input sequence.  This means
`vel_past` has enormous influence on the model's output.

## Output Head

The transformer output is passed through a linear layer followed by a
**Sigmoid** activation, producing 16 values:

- Indices 0--7: linear velocity trajectory (8 timesteps)
- Indices 8--15: angular velocity trajectory (8 timesteps)

These raw sigmoid outputs are then scaled:

```
linear_vel = 0.5 * action_pred[:, 0:8]       # range [0, 0.5] m/s
angular_vel = 1.0 * 2.0 * (action_pred[:, 8:16] - 0.5)  # range [-1.0, 1.0] rad/s
```

A separate linear head predicts a scalar **distance estimate** to the
goal (trained with MSE loss).

For deployment, we use **index [0]** of both velocity arrays---the
immediate next command, not a future waypoint.

# Exact Input Specifications

## Observation Images

- Shape: `[batch, 3*(context_size+1), H, W]` = `[1, 18, 96, 96]`
- Format: 6 RGB images (5 past + 1 current), stacked along the channel
dimension
- Preprocessing: resize to 96$\times$96, then normalize with ImageNet
statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Use `transform_images_mbra()` for this---it applies only
normalization, no cropping

## Goal Image

- Shape: `[batch, 3, H, W]` = `[1, 3, 96, 96]`
- Format: single RGB image of the subgoal location
- This must be an **actual photograph** from the target location,
not an abstract representation
- The model concatenates it with the last observation internally
(line 90 of exaug.py):
```
obsgoal_img = torch.cat([obs_img[:, 3*context_size:, :, :], goal_img], dim=1)
```
- Use `transform_images()` for this---it handles
resize + normalization (center crop is off by default)

## Robot Size

- Shape: `[1, 1, 1]`, float
- Value: 0.30 (meters, representing the robot's collision radius)
- During training: randomized in [0, 1.0] for generalization
- For deployment: fixed at 0.30 (reference code, line 1813 of train_utils.py)

## Delay Steps

- Shape: `[1, 1, 1]`, float
- Value: 0.0 (no actuation delay compensation)
- During training: randomized in [0, 4] to teach latency robustness
- For deployment: fixed at 0.0 (reference code, line 1814 of train_utils.py)

## Velocity History (`vel_past`)

- Shape: `[1, 12, 1]`
- Layout: [linear_0, ..., linear_5, angular_0, ..., angular_5]
- Each value is broadcast to 1024 dimensions inside the model via
`.repeat(1, 1, encoding_size)`

**This is where our critical bug was.**  See Section~\ref{sec:velpast-bug}.

# The Two Transform Functions

The deployment code provides two distinct image transforms:

## `transform_images()`

- Used for: **goal images**
- Operations: optional center crop (off by default) $\to$ resize to
target size $\to$ ToTensor $\to$ ImageNet normalize
- Handles images of arbitrary input size

## `transform_images_mbra()`

- Used for: **observation images**
- Operations: ToTensor $\to$ ImageNet normalize (no crop, no resize)
- Expects images already at the correct 96$\times$96 size

**Important**: do not use `transform_images()` for observations
or `transform_images_mbra()` for goals.  They apply different
preprocessing pipelines.

# Training Data and Expected Use Case

From `MBRA.yaml`:

- Primary dataset: FrodoBot (11.4M training examples, 90% of data)
- Supplementary datasets: RECON, Go Stanford, Cory Hall, Tartan Drive,
SACson, Seattle, SCAND (10% of data)
- Goal distance range: 3--20 timesteps ahead
(`action.min_dist_cat: 3, max_dist_cat: 20`)
- Context type: temporal (5 past frames + 1 current = 6 total)
- 8-step velocity trajectory output at approximately 3~Hz
($\sim$0.333~s per step, $\sim$2.6~s lookahead)

The model was primarily trained on **outdoor, small-robot environments**
(sidewalks, campus paths, indoor lobbies).  It has not been specifically
fine-tuned for our indoor corridor.  This domain gap explains some of the
behavioral issues we observed.

# The `vel_past` Bug

## What We Did Wrong

Our initial MBRA controller maintained a rolling history of its own predicted
velocity commands and fed them back as `vel_past` on each tick:

```
# WRONG: self-reinforcing feedback loop
self._linear_history.append(linear_cmd)
self._angular_history.append(angular_cmd)
vel_past = build_tensor(self._linear_history, self._angular_history)
```

Since `vel_past` accounts for 12 of 21 transformer tokens, it has
massive influence on the model's output.  Once the model produced even a small
angular bias (natural for an outdoor-trained model seeing an indoor corridor),
that value was fed back as velocity history, causing the model to predict a
similar angular command on the next tick.  This created a
**self-reinforcing feedback loop**:

small angular bias $\to$ fed back as vel_past $\to$ model predicts similar
angular $\to$ fed back again $\to$ locked into persistent turning

## What The Authors Actually Do

In the MBRA training code (`train_utils.py`, lines 1810--1821), the
authors' own deployment-style inference uses **fixed constant values** for
`vel_past`:

```
# From train_utils.py, lines 1815-1817:
# "To simplify the implementation, we give the fixed rsize(0.3),
#  delay(0.0) and previous velocity (going straight) for MBRA model"
linear_vel_old = 0.5 * torch.ones(B, 6).float().to(device)
angular_vel_old = 0.0 * torch.ones(B, 6).float().to(device)
vel_past = torch.cat((linear_vel_old, angular_vel_old), axis=1).unsqueeze(2)
```

They do **not** feed the model's own predictions back.  The fixed values
tell the model: "the robot has been going straight at moderate speed."  All
steering decisions are then driven purely by the **visual context**
(observation frames and goal image).

## Why This Matters

During training, `vel_past` starts as random values and is updated
with the model's predictions, but this is across training batches where
gradient-based loss correction prevents runaway feedback.  During deployment,
there is no loss function to correct the feedback.  Self-feeding predictions
diverge freely.

The key insight: `vel_past` in MBRA is a **contextual hint**, not
an odometry signal.  It tells the model what kind of motion to expect, not what
the robot actually did.

## The Fix

We now pre-build a fixed `vel_past` tensor at initialization:

```
linear_past = [0.5] * 6   # "going forward at moderate speed"
angular_past = [0.0] * 6  # "going straight, no turning"
vel_np = np.array(linear_past + angular_past, dtype=np.float32).reshape(1, 12, 1)
self._vel_past_fixed = torch.from_numpy(vel_np).to(self.device)
```

This is reused on every inference call.  No feedback, no history tracking.

# The EMA Blending Mistake

## What We Did Wrong

We added exponential moving average (EMA) smoothing on top of MBRA's raw
velocity outputs:

```
# WRONG: fighting the model's own temporal smoothing
linear_cmd = 0.55 * linear_cmd + 0.45 * self._last_linear
angular_cmd = 0.45 * angular_cmd + 0.55 * self._last_angular
```

This was added to "stabilize" the angular output.  In practice, it made
things worse: the EMA preserved 55% of the previous angular value, so once
the model started turning (due to the vel_past bug), the EMA ensured the
angular command could never quickly return to zero.

## Why EMA Is Not Needed

MBRA already performs temporal smoothing internally through its
**6-frame observation context window**.  The transformer sees 6
consecutive frames and can infer the robot's recent motion from visual flow.
Adding another EMA layer on top creates double-smoothing that fights against
the model's learned dynamics.

The MBRA authors' reference deployment code applies \textbf{no EMA smoothing at
all}.  The raw velocity output (after clipping) is sent directly to the robot.

## The Fix

We removed all EMA blending.  The model's output is clipped to safe velocity
bounds and sent directly as the command.

# Symptoms We Observed

## Persistent Turning (Angular Lock)

- **Observed**: angular velocity stabilized at $\sim$0.15--0.18
rad/s and never returned to zero
- **Cause**: vel_past feedback loop + EMA preserving angular state
- **Effect**: robot veered into corridor walls; localizer saw no
step change for 65+ ticks

## MBRA Warmup Delay

- **Observed**: first 5 ticks returned `mbra_warmup` with
zero commands
- **Cause**: context buffer needs `context_size + 1 = 6`
frames before inference can start
- **Status**: this is expected and correct behavior, not a bug

## Missing Subgoal Images

- **Observed**: every tick returned `mbra_missing_images`
- **Cause**: graph node paths were hardcoded to
`/home/vivek/...` (original author's machine); images existed at
the correct relative path but under `/home/lunar/`
- **Fix**: `navigation_runtime.py` now resolves paths
relative to the repo root when absolute paths don't exist

## Corridor Aliasing Stall (Near-Zero Linear Deadlock)

This is the most insidious failure mode we observed, and it persisted even
**after** the vel_past and EMA fixes were applied.

- **Observed**: after making good forward progress (e.g.\ step
1266~$\rightarrow$~1288), linear velocity suddenly drops to
$\sim$0.001--0.005~m/s and stays there for 40--100+ ticks.  Angular output
remains small ($<$0.05), so the robot effectively stops in place.
- **Pattern**: the stall always occurs mid-corridor, never at
junctions.  Progress resumes if the robot is manually nudged forward.

\paragraph{Root cause: corridor visual aliasing.}
With only 4~hops of subgoal look-ahead, the subgoal image is $\sim$40--120~cm
ahead---in a featureless corridor, this looks almost identical to the current
camera view.  MBRA interprets high visual similarity between observation and
goal as "already at the goal" and outputs near-zero linear velocity.  Once
the robot stops:
1. All 6~observation context frames become identical (camera is static).
1. The goal image still looks the same as the observation.
1. The model keeps outputting near-zero linear $\rightarrow$ **deadlock**.

This is a fundamental limitation of image-goal controllers in visually
repetitive environments.  The model has no distance signal---only visual
similarity---and corridors provide little discriminative texture between
nearby positions.

\paragraph{Fix 1: Minimum linear velocity floor.}
We enforce `min_linear = 0.10`~m/s in `MBRALocalControllerConfig`.
After clipping and before returning the command:

```
linear_cmd = max(self.config.min_linear, linear_cmd)
```

This guarantees the robot always creeps forward, even when MBRA thinks it has
arrived.  The minimum is above the hardware friction threshold ($\sim$0.06~m/s)
so the wheels actually turn.  This mirrors the same principle used in our
simple controller.

\paragraph{Fix 2: No-progress context reset.}
In `live_indoor_runtime.py`, we track how many consecutive ticks the
localizer reports the same `current_step`.  If the robot is stuck at
the same step for 20~ticks, we call `controller.reset()` to clear
MBRA's observation history:

```
NO_PROGRESS_RESET_TICKS = 20
# ... inside main loop:
if cur_step == prev_cur_step:
    no_progress_count += 1
else:
    no_progress_count = 0
if no_progress_count % NO_PROGRESS_RESET_TICKS == 0:
    controller.reset()  # clears the 6-frame deque
```

Resetting forces MBRA back through its 5-tick warmup phase (outputting zero
during that time), after which it receives 6~fresh frames.  Combined with the
min_linear floor, those fresh frames will show slightly different views
(because the robot kept moving), breaking the identical-context deadlock.

\paragraph{Why not IMU or side cameras?}
We initially considered using IMU heading-rate data or side/rear cameras to
help MBRA break out of stalls.  However:
- MBRA's architecture accepts only front-facing RGB and a single goal
image.  Adding IMU or extra camera streams would require retraining.
- The indoor IMU compass is unreliable (already disabled for heading
alignment).
- The min_linear + context-reset approach solves the symptom at the
control layer without architectural changes.

# Correct Deployment Configuration

Based on the MBRA training code and reference deployment, the correct
deployment parameters are:

```text
\toprule
\textbf{Parameter} & \textbf{Value} & \textbf{Source} \\
\midrule
Image size & 96 $\times$ 96 & MBRA.yaml \\
Context size & 5 (6 frames total) & MBRA.yaml \\
Robot size & 0.30 & train\_utils.py line 1813 \\
Delay steps & 0.0 & train\_utils.py line 1814 \\
vel\_past linear & 0.5 (constant) & train\_utils.py line 1815 \\
vel\_past angular & 0.0 (constant) & train\_utils.py line 1816 \\
Tick rate & 3 Hz & LogoNav\_frodobot.py line 51 \\
Subgoal distance & 3--5 graph hops & action.min\_dist\_cat in MBRA.yaml \\
Max linear (our robot) & 0.24 m/s & hardware limit \\
Max angular (our robot) & 0.34 rad/s & hardware limit \\
EMA smoothing & \textbf{none} & reference uses raw outputs \\
\bottomrule
```

# Subgoal Distance Considerations

The MBRA training config specifies \texttt{action.min_dist_cat: 3,
max_dist_cat: 20}, meaning goal images during training were sampled from
3 to 20 timesteps ahead in the training trajectories.

For our corridor graph:
- 1865 nodes over the full corridor
- Each node is one frame from the recorded walkthrough
- Node spacing is small (estimated 10--30~cm per step)
- 4 hops ahead $\approx$ 40--120~cm (reasonable short-horizon goal)
- 15 hops ahead $\approx$ 1.5--4.5~m (too far; goal image looks
significantly different from current view)

Default subgoal hops for MBRA is now set to **4**, matching the
short-horizon nature of the model.  The simple controller retains its default
of 15 hops since it only uses the step gap for speed scaling, not the actual
subgoal image.

# Differences Between MBRA and LogoNav

The `mbra_repo/deployment/` directory contains
`LogoNav_frodobot.py`, which is the reference FrodoBot deployment
script.  It is important to understand that this script uses
**LogoNav**, not MBRA directly:

```text
\toprule
& \textbf{MBRA (ExAug\_dist\_delay)} & \textbf{LogoNav} \\
\midrule
Goal type & RGB image & GPS-relative pose (4D vector) \\
Output & 8-step (linear, angular) & waypoints (x, y, $h_x$, $h_y$) \\
Use case & Image-goal navigation & GPS-conditioned navigation \\
Indoor viable & Yes (with corridor images) & No (requires GPS) \\
\bottomrule
```

LogoNav converts waypoints to velocities through geometric calculations
(arctan-based heading extraction).  MBRA outputs velocities directly.  For our
indoor corridor (no GPS), MBRA is the correct model.

# What MBRA Cannot Do

1. **Global localization**: MBRA does not know where the robot is.
It only knows "current view" vs "goal view."  Localization must come
from CosPlace + temporal filtering.
1. **Long-horizon planning**: MBRA's trajectory output covers
$\sim$2.6 seconds ($8 \times 0.333$~s).  It cannot plan beyond that
window.  Graph planning must provide the subgoal sequence.
1. **Recovery from lost localization**: if the localizer is confused,
MBRA receives a wrong subgoal image and will drive in the wrong direction.
Recovery logic must be external.
1. **Obstacle avoidance**: while `robot_size` provides some
implicit collision awareness, MBRA has no explicit obstacle detection.
Safety veto must be external.

# Summary of Bugs Found and Fixed

1. **vel_past feedback loop** (critical): feeding MBRA's own angular
predictions back as velocity history caused self-reinforcing persistent
turning.  Fixed by using fixed constants matching the authors' deployment
reference.

1. **EMA blending** (harmful): exponential moving average on top of
MBRA's output fought against the model's internal temporal smoothing and
prevented angular recovery.  Removed entirely.

1. **Missing subgoal images**: graph paths hardcoded to wrong machine.
Fixed with relative path resolution.

1. **Subgoal too far ahead**: 15 hops for MBRA meant the goal image
was visually unrelated to the current view.  Reduced to 4 hops.

1. **Tick rate too slow**: running at 2~Hz meant the 6-frame context
window spanned 3 seconds with potentially stale/duplicate frames.
Increased to 3~Hz to match the reference deployment.

1. **Corridor aliasing stall** (critical): in featureless corridor
stretches, the subgoal image (4~hops ahead) looks nearly identical to the
current view.  MBRA interprets this as "at goal" and outputs near-zero
linear velocity.  Robot stops, context frames freeze, deadlock.  Fixed with
a min_linear floor (0.10~m/s) and a no-progress context reset every 20
stuck ticks.

# Lessons for Future Development

1. **Always read the reference deployment code**, not just the model
definition.  The model architecture alone does not reveal how inputs should
be constructed for inference.

1. \textbf{Neural network inputs that look like "history" may actually
be "hints."} The vel_past tensor is not odometry---it is a contextual
prior that shapes the model's behavior mode.

1. \textbf{Do not add smoothing on top of models that already have
temporal context.}  MBRA's 6-frame observation window already provides
temporal smoothing.  External EMA creates double-smoothing artifacts.

1. \textbf{When a learned controller produces persistent bias, check the
feedback path first.}  The natural instinct is to clamp or filter the
output, but the root cause is usually in the input pipeline.

1. **Domain gap is real but secondary.**  MBRA was trained primarily
on outdoor FrodoBot data, not indoor corridors.  This creates some
behavioral mismatch, but the vel_past feedback bug was a far larger
problem than domain gap.

1. \textbf{Image-goal controllers need a minimum velocity floor in
repetitive environments.}  Visual similarity between current and goal does
not mean the robot has arrived---it may just mean the environment lacks
discriminative texture.  A small constant forward speed prevents deadlock
and costs very little in terms of overshoot.

1. **Stale context is self-reinforcing.**  Once the robot stops, its
camera sees the same scene every frame.  Any model that uses a temporal
context window will lock into a fixed output.  Breaking this requires
either forced motion (min_linear) or context reset---ideally both.

# Code-Verified Addendum: How These Discoveries Appear In The Current Indoor Runtime

The earlier sections of this document preserve the original MBRA reverse-engineering and deployment lessons well. However, the repo has continued to evolve since those discoveries were written down, so it is useful to state explicitly how the present indoor runtime reflects them.

## Current Deployed MBRA Controller Defaults

The current `src/mbra_controller.py` config is:
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

So the core lessons documented earlier were not lost. Fixed `vel_past` remains in place, `robot_size=0.30` and `delay=0.0` remain explicit, and the wrapper still takes index `[0,0]` of the predicted trajectory as the immediate control command.

## Important Later Runtime Defaults

The present `live_indoor_runtime.py` chooses the following defaults when `--controller mbra` is used:
```
args.max_subgoal_hops = 8 if args.controller == "mbra" else 15
args.tick_hz = 3.0 if args.controller == "mbra" else 2.0
args.depth_stop_m = 0.25 if args.controller == "mbra" else 0.4
args.depth_slow_m = 0.6 if args.controller == "mbra" else 0.8
```

This is worth recording because it captures the deployed indoor state as it actually exists now, not only the earlier reasoning state.

## Historical Note: The 4-Hop Analysis Versus The Current 8-Hop Runtime Default

One important correction should be made explicitly rather than left implicit.

The earlier analysis in this document argues, with good reason, that MBRA is fundamentally a short-horizon image-goal controller and that very long subgoal spacing makes the goal view visually too different from the current view. That historical argument remains useful and should not be deleted.

However, the current deployed indoor runtime no longer defaults to 4 hops. It now defaults to 8 hops for MBRA. So the document should be read as follows:
- the earlier 4-hop discussion records a real design conclusion reached during the investigation,
- but the later indoor runtime evolved further and now deploys MBRA with 8-hop subgoals by default,
- which means the earlier section is historically accurate but no longer the final runtime constant.

That distinction matters because otherwise the document would silently disagree with the present code.

## No-Reverse For MBRA Is Still A Real Deployed Principle

The earlier writeup emphasized that reversing is especially harmful for MBRA because it destroys the image-context assumptions. The current indoor runtime still reflects that principle explicitly:
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

This is one of the most important bridges between the original MBRA investigation and the later runtime philosophy. The idea that "reverse kills context" later propagated far beyond indoor MBRA debugging and influenced how the outdoor marathon runtime was also shaped.

## The No-Progress Reset Also Survived Into The Current Code

Another original lesson that remained active is the belief that stale visual context is self-reinforcing. The current indoor runtime still contains the reset logic:
```
NO_PROGRESS_RESET_TICKS = 10
...
if no_progress_count > 0 and no_progress_count % NO_PROGRESS_RESET_TICKS == 0:
    if hasattr(controller, 'reset'):
        controller.reset()
        print(f"[{iteration:04d}] no-progress reset after {no_progress_count} ticks at step {cur_step}")
```

That means this document's original discussion of deadlock and stale context was not merely interpretive. It materially shaped the present deployed control loop.

## What This Document Now Represents

With these later corrections in mind, the best way to understand this file is:
- the first and middle sections record the original MBRA reverse-engineering and debugging discoveries,
- the addendum above records how those lessons were later absorbed, modified, or partially superseded by the deployed indoor runtime,
- and together they form a more honest engineering record than either one alone.

That is valuable because it preserves both the reasoning path and the later operational reality.

# Related Documents

- [[erc3_full_documentation]] --- single master guide for the complete project story and current architecture.
- [[live_indoor_runtime_story]] --- indoor evolution, MBRA integration, and checkpoint-step runtime behavior.
- [[live_outdoor_ultra_marathon_story]] --- outdoor and marathon runtime evolution with safety-layer reasoning.
- [[outdoor_perception_review]] --- depth/semantic perception findings and their runtime implications.
