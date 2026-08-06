# Understanding \texttt{live\_outdoor\_runtime.py

> Source: `live_outdoor_runtime_explained.tex`
> Master Note: [[erc3_full_documentation]]

> [!abstract] Note Header
> **Purpose:** Plain-language walkthrough of what the outdoor runtime is doing.
> **Read when:** I need an implementation-oriented explanation without reading the entire marathon story.
> **Authority:** Explanatory companion note. Trust active code if behavior differs.

> [!important] Current Status
> This note explains the historical GPS outdoor mission runtime.
> The current off-road branch is different: no-GPS teach-and-repeat on rough terrain using `live_indoor_runtime.py` plus prepared routes.
> So this note is still useful, but it is not the current off-road source of truth.

# Purpose of This Document

This document explains, in plain language, what the outdoor mission code is doing now.
It assumes you are new to the codebase and want to understand the full flow from startup to mission completion.

The main script discussed here is:
`live_outdoor_runtime.py`

This script runs the rover in outdoor mission mode, reads telemetry from the SDK, chooses motion commands, sends them to the robot, and handles mission progress, recovery, and stopping.

# Big Picture

At a high level, the runtime does the following:
1. Starts a mission and fetches the list of checkpoints.
1. Waits until live telemetry is available.
1. Builds a list of navigation targets.
1. Chooses a controller.
1. Repeats a control loop:
- read telemetry,
- read a camera frame,
- compute a command,
- apply safety or recovery logic,
- send the command,
- print a log line.

1. When the final checkpoint is reached and accepted by the SDK, it stops and ends.

# What the Main Command Means

The command you normally run is:
```
python live_outdoor_runtime.py --mission --send-control
```

Right now, this means:
- `--mission`: use SDK mission mode, so checkpoints come from the server.
- `--send-control`: actually send commands to the robot instead of dry-run.
- the default controller is now `logonav`, which is the visual learned controller.

So the plain command is intended to be the competition-style run path.

# Step 1: Parse Arguments

At startup, the script reads command-line arguments.
These arguments control:
- mission vs manual mode,
- which controller to use (GPS or LogoNav),
- loop rate,
- motion limits,
- goal radius,
- recovery thresholds,
- optional depth safety,
- optional OSM routing,
- LogoNav weights and config.

This gives the script a complete set of parameters before anything starts moving.

# Step 2: Connect to the SDK and Start the Mission

If `--mission` is enabled, the script does not use a manually provided goal list.
Instead it asks the SDK to start the mission and return the official checkpoint list.

The runtime also supports mission resume.
That means it asks the SDK which checkpoint was already confirmed, and if needed it skips earlier checkpoints so it continues from the correct place.

So the runtime's checkpoint list is synchronized with the server.

# Step 3: Wait for Live Telemetry

After mission start, the script does not immediately start driving.
It waits until telemetry looks real.

In practice, it checks for things like:
- latitude and longitude must be finite,
- they must not be zero,
- the telemetry timestamp must be advancing.

This matters because if the code starts navigating on stale or half-initialized data, the rover may move based on bad state.

If telemetry never becomes live within the timeout, the script exits instead of guessing.

# Step 4: Build Navigation Targets

The script works with a list called `navigation_targets`.

If OSM routing is *off*, the navigation targets are just the mission checkpoints.
If OSM routing is *on*, the script expands each mission leg into intermediate waypoints and still marks which targets are the true mission checkpoints.

This is useful because:
- the robot may need more than one point to follow a road-like path,
- but only the original mission checkpoints count for scoring.

So the code separates *navigation points* from *scored mission points*.

# Step 5: Choose the Controller

There are two controller families in the runtime:
1. **GPS controller**
This is a classical controller.
It uses the current position, the goal position, and optional depth-based obstacle information.

1. **LogoNav controller**
This is a learned visual controller.
It looks at recent camera frames plus a goal representation and predicts motion commands.

The current default for the main mission command is `logonav`.

# Step 6: Optional Depth Safety

The runtime can also create a depth estimator.
This is *not* the main controller.
It is an optional safety overlay.

If depth safety is enabled, the script estimates forward clearance from the camera image and can:
- slow the rover,
- stop the rover,
- provide obstacle clearance information to the GPS controller.

If depth safety is disabled, the main controller still runs, but without this extra clearance layer.

# Step 7: Wait for the Camera

If the chosen controller needs camera frames, the script waits for the camera to become live before entering the main loop.

This is important for LogoNav because it is a visual controller.
If frames are not arriving, the controller cannot do meaningful inference.

# Step 8: Enter the Main Control Loop

After startup, the script enters a loop that runs at a fixed frequency, for example 3 Hz.

Each iteration of the loop does the same broad sequence:
1. read a camera frame,
1. read telemetry,
1. convert GPS to UTM,
1. choose the active target,
1. run the controller,
1. apply mission logic,
1. apply recovery logic if needed,
1. smooth commands,
1. send commands,
1. print a status line.

# Step 9: Read and Process Telemetry

The runtime reads:
- latitude,
- longitude,
- orientation in degrees,
- timestamp.

Then it converts latitude and longitude into UTM coordinates.
UTM is useful because it turns positions into meters on an \((x, y)\) plane.
That makes distance and heading math much simpler.

The script also smooths GPS position with an exponential moving average.
This reduces random GPS jitter.

# Step 10: Compute the Current Goal

The script picks the currently active navigation target.
That target has a goal latitude and longitude, which are also converted into UTM.

For LogoNav, the runtime now recomputes the heading to that goal on every tick.
This is important.
A one-time heading computed only when a waypoint changes can become stale while the rover drifts.

So now the controller is told, every loop, where the goal is relative to the rover's current position.

# Step 11: What the GPS Controller Does

If the GPS controller is selected, it uses classical control logic.

Its basic idea is:
- compute the direction to the goal,
- compare that with the current heading,
- turn toward the goal,
- move forward when reasonably aligned,
- optionally use depth/VFH if obstacle data is available.

It also includes:
- turn-in-place hysteresis,
- approach slowdown near the goal,
- optional obstacle avoidance state.

This controller is more interpretable, but it is also more limited in cluttered scenes because it does not inherently "see" the world the way a learned visual controller does.

# Step 12: What the LogoNav Controller Does

If the LogoNav controller is selected, the controller uses recent RGB frames and a goal representation.

Internally, it keeps a short context queue of frames.
The model does not only look at the current image.
It looks at a short visual history.

It also builds a four-value goal description called `goal_pose`:
- local goal x,
- local goal y,
- \(\cos(\text{goal heading})\),
- \(\sin(\text{goal heading})\).

This goal representation is fed into the learned policy.
The model then predicts a short future trajectory, and the code converts one predicted waypoint into linear and angular velocity commands.

# Step 13: Why We Added a GPS Goal Bias to LogoNav

In the current code, LogoNav is no longer trusted blindly.
It is still the main visual controller, but it now gets an explicit GPS heading pull.

This was added because in field behavior the learned policy could drift, curve, or wander without reducing distance to the actual mission target.

So now the controller does two things at once:
- follow the learned visual policy,
- add a bounded angular correction toward the GPS goal.

The correction is bounded, which means it cannot completely overpower the learned visual behavior.
That is important, because the point is not to turn LogoNav into pure GPS steering.
The point is to keep it obstacle-aware *and* goal-seeking.

The controller also reduces forward speed when the heading-to-goal is poor.
This helps avoid charging ahead while pointed away from the target.

# Step 14: Goal Reaching Logic

After a command is computed, the runtime checks whether the rover is within the goal radius.

If the active target is only an intermediate routing waypoint, the script simply advances to the next one.

If the active target is a real mission checkpoint, the script calls the SDK endpoint that reports a checkpoint as reached.
Only if the SDK accepts that checkpoint does the local runtime advance.

This matters because it keeps local mission state aligned with the server.

# Step 15: Telemetry Freeze Detection

The code watches the telemetry timestamp.
If the timestamp stops advancing for too many loop iterations, the script assumes telemetry is frozen.

In that case it stops the robot for safety and skips command generation for that iteration.

This is a protection against driving on stale state.

# Step 16: Recovery Logic

The code has recovery logic for cases where the rover is not making progress.

Originally, the recovery logic was too aggressive for the visual controller.
It could falsely conclude that the robot was blocked and then force reverse-and-turn behavior too often.
That caused the "one step forward, several steps back" effect.

The current logic is more careful.

## For the GPS controller

The GPS controller still keeps the fast wall-hit logic and the more aggressive stuck recovery logic.
That is because the classical GPS path has fewer ways to infer local obstruction and benefits from stronger fallback behavior.

## For the LogoNav controller

The current code now does the following:
- disables the fast GPS-only wall-hit heuristic,
- requires clearer evidence that the rover is actually losing ground,
- uses turn-only recovery instead of reverse-first recovery.

This was done because the visual controller often makes progress in curved paths and with small GPS improvements.
A GPS-only "wall hit" rule was falsely firing and undoing real forward motion.

So the recovery system is now less likely to hijack normal visual navigation.

# Step 17: Command Smoothing

Before sending commands, the runtime smooths linear and angular commands with exponential smoothing.
This reduces command jitter.

However, recovery commands and goal-stop commands bypass or reset this smoothing so that safety or recovery actions take effect immediately.

This is a practical compromise:
- normal motion should be smooth,
- emergency or recovery motion should not be delayed by smoothing.

# Step 18: Send Commands

If the script is in dry-run mode, it only prints what it would have sent.
If `--send-control` is enabled, it sends the linear and angular command to the SDK.

So the runtime is the final decision-making layer between sensor input and robot motion.

# Step 19: Print the Log Line

Each loop prints a compact status line.
This line typically includes:
- checkpoint progress,
- active waypoint,
- distance to goal,
- bearing error,
- heading,
- mode,
- linear and angular command,
- whether the command was sent,
- whether telemetry was frozen.

This log is useful because it shows whether the rover is:
- getting closer or farther,
- trying to drive or turn,
- being overridden by recovery,
- fighting bad telemetry,
- behaving like the chosen controller expects.

# What "Good" Behavior Looks Like

For a healthy run, you want the following pattern:
- distance to goal trends downward over time,
- bearing error trends toward zero,
- the controller spends most of its time in normal drive mode,
- recovery mode is rare,
- checkpoint confirmations advance in order,
- the mission ends automatically after the final checkpoint is accepted.

# What "Bad" Behavior Looks Like

A bad run usually looks like one of these:
- distance stays flat or increases,
- the robot repeatedly spins in place,
- recovery triggers over and over,
- the rover moves but does not reduce goal distance,
- telemetry freezes frequently,
- the camera or controller is active but the motion is directionless.

These are exactly the types of behaviors the recent fixes were trying to reduce.

# How Mission Completion Works

When the final mission checkpoint is reached and accepted by the server, the script:
1. advances past the last target,
1. stops the robot,
1. prints mission completion,
1. exits.

So yes, the runtime automatically ends once the mission is fully completed.

# Summary in One Paragraph

The outdoor runtime is a loop that starts the mission, waits for live telemetry, builds a target list, reads sensor data, computes commands with either a classical GPS controller or a learned visual controller, checks mission progress, applies recovery when needed, sends commands to the robot, and stops automatically when all checkpoints are completed.

The current preferred path is the visual LogoNav controller, but it has been modified so it no longer just follows the learned policy blindly: it now gets a bounded GPS heading pull, fresh per-tick goal heading updates, and less aggressive recovery logic so it can keep making real progress toward the mission target.

# Code-Verified Addendum: What The Runtime Actually Looks Like Now

The earlier sections in this document still describe the broad outdoor loop correctly, but the current file has accumulated several layers that are important enough to document explicitly from code rather than from memory.

## Current Parser Defaults That Matter In Practice

Three parser defaults now shape later behavior strongly:
```
--intermediate-goal-radius-m = 3.0
--logonav-align-distance-m = 6.0
--osm-prune-behind-distance-m = 18.0
```

These three values correspond to three concrete late-stage problems seen on the live rover:
- routed intermediate waypoints were being handed off too early,
- LogoNav was entering long turn-priority episodes too far from the active waypoint,
- and OSM reroutes could create a first waypoint that was effectively behind the rover.

The current runtime therefore uses a tighter default intermediate handoff radius, a distance-aware align gate, and more aggressive pruning of nearby behind-the-rover OSM points.

## What Ultra-Marathon And Night-Safe Actually Change

The runtime no longer treats `--ultra-marathon` and `--night-safe` as cosmetic flags. They overwrite real safety-critical defaults. In the current code, the following logic is present:
```
if args.ultra_marathon:
    args.camera_watchdog_ticks = min(args.camera_watchdog_ticks, 10)
    args.semantic_hard_stop = True
    args.semantic_yield = True
    args.semantic_sidewalk_stop = True

if args.night_safe:
    args.trav_stop_m = max(args.trav_stop_m, 0.80)
    args.trav_slow_m = max(args.trav_slow_m, 1.50)
    args.camera_watchdog_ticks = min(args.camera_watchdog_ticks, 6)
    args.battery_warn_pct = max(args.battery_warn_pct, 35.0)
```

So the current outdoor runtime is not merely "the same controller, but at night." It is a stricter operating envelope with:
- tighter image-quality tolerance through the camera watchdog,
- earlier traversability braking,
- stronger battery conservatism,
- and semantic stop / yield / sidewalk-stop layers enabled in ultra-marathon mode.

## The Actual Authority Ladder In The Later Outdoor Loop

One of the most useful ways to understand the present outdoor runtime is to describe it as a sequence of authority checks rather than a single controller.

In the current code, the later loop behaves approximately like this:
1. startup mission state may auto-claim a checkpoint if the rover is already physically within the checkpoint radius;
1. battery and telemetry health can stop the loop before normal commanding;
1. route-corridor enforcement can stop and reroute before the controller is allowed to keep wandering;
1. semantic hard-stop and semantic sidewalk-stop can halt the rover entirely;
1. semantic yield can cap linear speed without fully taking over steering;
1. the chosen controller computes a nominal command;
1. LogoNav align-turn can temporarily promote turn-priority when the target is close enough or the routed waypoint geometry is extreme;
1. traversability can then hard-override the local command when the forward corridor is blocked;
1. and only after those steps does the runtime enforce minimum effective forward speed and send the command.

This is much more layered than the original outdoor runtime. The present system is best understood as "LogoNav inside a safety envelope" rather than "pure LogoNav."

## Representative Code Paths Worth Reading

The route-corridor logic was softened in non-strict mode so that the rover does not reroute constantly while still making legitimate progress:
```
route_corridor_stop_threshold = max(route_corridor_stop_threshold,
                                    0.75 * args.osm_min_waypoint_spacing_m)
if not args.sidewalk_strict:
    route_corridor_stop_threshold = max(route_corridor_stop_threshold,
                                        args.goal_radius_m)
```

That one change matters because it explains why later logs can show corridor distance growing above the old static stop threshold without immediately forcing a dead stop in every case.

The LogoNav align gate was also made target-aware:
```
_align_allowed = (
    math.isfinite(distance_to_goal)
    and (
        distance_to_goal <= args.logonav_align_distance_m
        or (
            not bool(target.get("mission_checkpoint", False))
            and _bearing_error_deg >= args.logonav_align_extreme_deg
        )
    )
)
```

This means a far-away mission checkpoint is no longer allowed to trigger the same kind of hard align-turn spin that makes sense only for a nearby routed waypoint.

The intermediate routed waypoint radius is also no longer a single blunt constant:
```
if bool(target.get("mission_checkpoint", False)):
    active_goal_radius_m = args.goal_radius_m
else:
    _dynamic_radius_m = float(args.intermediate_goal_radius_m)
    if math.isfinite(_segment_distance_m) and _segment_distance_m > 0.0:
        _dynamic_radius_m = min(_dynamic_radius_m,
                                max(2.5, 0.35 * _segment_distance_m))
    active_goal_radius_m = min(args.goal_radius_m, _dynamic_radius_m)
```

That distinction between mission checkpoints and intermediate routed waypoints is one of the most important later structural improvements in the file.

Finally, traversability is now explicitly a hard local safety layer rather than only a soft steering nudge:
```
if _trav.all_blocked or _trav.linear_scale <= 0.0:
    command.linear = 0.0
    ...
    command.reason = f"trav_stop({_trav.forward_clearance:.2f}m)"
elif _trav.forward_blocked:
    command.linear *= _trav.linear_scale
    ...
    command.reason = f"trav_turn({_trav.forward_clearance:.2f}m)"
elif _trav.linear_scale < 0.999:
    command.linear *= _trav.linear_scale
    command.reason = f"trav_slow({_trav.forward_clearance:.2f}m)"
```

## Later Failure Modes That Changed The Runtime

The later runtime changes were not arbitrary. They were driven by concrete field failures:
- **False IMU emergencies**: normal rest pose or aggressive turning could look like catastrophic tilt or dangerous gyro events if the IMU frame assumptions were wrong or too brittle.
- **Waypoint handoff instability**: the rover could effectively "reach" an intermediate routed point while still in a bad local orientation, causing the next target to appear behind it.
- **Corridor deadlocks**: an off-corridor stop could keep asserting the same stale route rather than rerouting from the live pose.
- **Long align spins**: the rover could sit inside `ALIGN` or `logonav_align_turn` even when the active target regime did not justify that behavior.
- **Ineffective creeping**: low but nonzero forward commands could prevent the rover from obviously stopping while also failing to produce meaningful progress.

These problems are part of the present runtime story and should be remembered as such. The current outdoor loop is the result of repeated corrections to those concrete failures, not a single clean-sheet design.

## What The Logs Mean In The Present Runtime

The later log format became more informative because the operator needed to understand which layer was currently in charge.

A few representative examples are worth preserving:
- `trav_stop(0.58m)` means traversability, not LogoNav, made the stop decision.
- `semantic_hard_stop` means semantics saw a person or animal class strongly enough to halt the rover.
- `route_corridor_stop dist=... threshold=... ticks=...` means the corridor guard tripped and the runtime is about to stop or reroute.
- `startup auto-claimed checkpoint seq=...` means the rover was already physically sitting within the first remaining mission checkpoint radius when mission logic began.
- `pruned N behind-waypoint(s)` means the route logic deliberately discarded an OSM point that would have caused backward-looking or circular behavior.

These log-line interpretations belong in the documentation because the human operator is part of the present outdoor safety loop.

# Related Documents

- [[erc3_full_documentation]] --- single master guide for the complete project story and current architecture.
- [[live_indoor_runtime_story]] --- indoor evolution, MBRA integration, and checkpoint-step runtime behavior.
- [[live_outdoor_ultra_marathon_story]] --- outdoor and marathon runtime evolution with safety-layer reasoning.
- [[outdoor_perception_review]] --- depth/semantic perception findings and their runtime implications.
