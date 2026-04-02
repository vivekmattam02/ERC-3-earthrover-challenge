#!/usr/bin/env python3
"""Live indoor runtime loop for the ERC known-corridor baseline.

This script ties together:
- EarthRover SDK input/output
- corridor localization
- graph planning
- local control

It defaults to dry-run mode. To actually send commands, pass --send-control.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from earthrover_interface import EarthRoverInterface  # type: ignore
from local_controller import SimpleLocalController, SimpleLocalControllerConfig  # type: ignore
from navigation_runtime import NavigationRuntime, NavigationRuntimeConfig  # type: ignore
from sensor_state import SensorStateFilter, SensorStateFilterConfig  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ERC indoor baseline live loop.")
    parser.add_argument(
        "--database",
        type=Path,
        default=REPO_ROOT / "data" / "corrider_db" / "descriptors.npz",
        help="Path to the built descriptor database.",
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=REPO_ROOT / "data" / "corrider_db" / "navigation_graph.json",
        help="Path to the runtime navigation graph JSON.",
    )
    parser.add_argument(
        "--data-info-json",
        type=Path,
        default=REPO_ROOT / "data" / "corrider_extracted" / "metadata" / "data_info.json",
        help="Path to corridor metadata JSON.",
    )
    parser.add_argument("--target-step", type=int, default=None, help="Single target step to navigate toward.")
    parser.add_argument("--target-image-name", type=str, default=None, help="Single target image name.")
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="*",
        default=None,
        help="Optional ordered checkpoint steps for sequential execution.",
    )
    parser.add_argument(
        "--checkpoint-images",
        type=str,
        nargs="*",
        default=None,
        help="Optional ordered checkpoint image names for sequential execution.",
    )
    parser.add_argument(
        "--checkpoint-image-files",
        type=Path,
        nargs="*",
        default=None,
        help="Arbitrary image file paths — prelocalized at startup to find matching DB steps.",
    )
    parser.add_argument("--tick-hz", type=float, default=None, help="Loop frequency in Hz. Default: 3 for mbra, 2 for simple.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional max loop iterations.")
    parser.add_argument("--max-subgoal-hops", type=int, default=None, help="Graph hops ahead for subgoal selection. Default: 4 for mbra, 15 for simple.")
    parser.add_argument("--sdk-url", default="http://localhost:8000", help="EarthRover SDK base URL.")
    parser.add_argument("--sdk-timeout", type=float, default=5.0, help="SDK request timeout in seconds.")
    parser.add_argument("--send-control", action="store_true", help="Actually send commands to the robot.")
    parser.add_argument(
        "--controller",
        choices=("simple", "mbra"),
        default="simple",
        help="Local controller implementation to use.",
    )
    parser.add_argument(
        "--mbra-weights",
        type=Path,
        default=REPO_ROOT / "mbra_repo" / "deployment" / "model_weights" / "mbra.pth",
        help="Path to MBRA weights when --controller mbra is selected.",
    )
    parser.add_argument("--auto-advance-checkpoints", action="store_true", help="Advance to next checkpoint automatically.")
    parser.add_argument("--stop-on-low-confidence", action="store_true", help="Stop when localization confidence is too low.")
    parser.add_argument("--min-confidence", type=float, default=0.35, help="Low-confidence stop threshold.")
    parser.add_argument("--print-json", action="store_true", help="Print each loop state as JSON.")
    parser.add_argument("--depth-safety", action="store_true", help="Enable monocular depth forward-clearance veto.")
    parser.add_argument("--depth-slow-m", type=float, default=0.8, help="Slow down when forward clearance below this (meters).")
    parser.add_argument("--depth-stop-m", type=float, default=0.4, help="Stop when forward clearance below this (meters).")
    return parser.parse_args()


def build_runtime(args: argparse.Namespace) -> NavigationRuntime:
    return NavigationRuntime(
        NavigationRuntimeConfig(
            database_npz=args.database,
            graph_json=args.graph,
            data_info_json=args.data_info_json,
            max_subgoal_hops=args.max_subgoal_hops,
        )
    )


def frame_signature(frame) -> np.ndarray:
    gray = frame.mean(axis=2) if frame.ndim == 3 else frame
    return gray[::24, ::24].astype(np.float32, copy=False)


def build_controller(args: argparse.Namespace):
    if args.controller == "mbra":
        try:
            from mbra_controller import MBRALocalController, MBRALocalControllerConfig  # type: ignore
        except ModuleNotFoundError as exc:
            raise SystemExit(
                f"MBRA controller dependencies are not installed: missing module '{exc.name}'. "
                "Install the mbra_repo environment/dependencies first."
            ) from exc

        return MBRALocalController(
            MBRALocalControllerConfig(
                weights_path=args.mbra_weights,
            )
        )
    return SimpleLocalController(SimpleLocalControllerConfig())


def main() -> int:
    args = parse_args()
    if args.max_subgoal_hops is None:
        args.max_subgoal_hops = 4 if args.controller == "mbra" else 15
    if args.tick_hz is None:
        args.tick_hz = 3.0 if args.controller == "mbra" else 2.0

    has_target = (
        args.target_step is not None
        or args.target_image_name is not None
        or args.checkpoint_steps
        or args.checkpoint_images
        or args.checkpoint_image_files
    )
    if not has_target:
        raise SystemExit("Provide --target-step, --target-image-name, --checkpoint-steps, --checkpoint-images, or --checkpoint-image-files.")
    if args.tick_hz <= 0:
        raise SystemExit("--tick-hz must be > 0")

    runtime = build_runtime(args)

    # --- Prelocalize arbitrary checkpoint image files at startup ---
    if args.checkpoint_image_files:
        print("Prelocalizing checkpoint images...")
        prelocalized_steps: list[int] = []
        for img_path in args.checkpoint_image_files:
            if not img_path.is_file():
                raise SystemExit(f"Checkpoint image not found: {img_path}")
            result = runtime.localizer.localize_image_path(img_path)
            runtime.localizer.reset()  # independent per image
            step = result.get("node_step")
            conf = result.get("confidence", 0)
            print(f"  {img_path.name} -> step={step}  conf={conf:.3f}")
            if step is None:
                raise SystemExit(f"Failed to localize {img_path}")
            prelocalized_steps.append(int(step))
        args.checkpoint_steps = prelocalized_steps
        print(f"Prelocalized steps: {prelocalized_steps}")
        print(f"Run command equivalent: --checkpoint-steps {' '.join(str(s) for s in prelocalized_steps)}")

    controller = build_controller(args)
    sensor_filter = SensorStateFilter(SensorStateFilterConfig())
    rover = EarthRoverInterface(base_url=args.sdk_url, timeout=args.sdk_timeout)

    if not rover.connect():
        raise SystemExit("Failed to connect to SDK.")

    # --- Optional depth safety ---
    depth_estimator = None
    if args.depth_safety:
        try:
            from depth_estimator import DepthEstimator  # type: ignore
            depth_estimator = DepthEstimator(model_size='small', checkpoint_domain='indoor')
            print("Depth safety: ENABLED")
        except Exception as exc:
            print(f"[warn] Depth safety disabled — could not load model: {exc}")

    if args.checkpoint_steps or args.checkpoint_images:
        runtime.set_checkpoints(
            checkpoint_steps=list(args.checkpoint_steps) if args.checkpoint_steps else None,
            checkpoint_images=list(args.checkpoint_images) if args.checkpoint_images else None,
        )

    dry_run = not args.send_control
    is_checkpoint_mode = bool(args.checkpoint_steps or args.checkpoint_images)

    print("Live indoor runtime")
    print("=" * 60)
    print(f"Mode: {'DRY RUN' if dry_run else 'SEND CONTROL'}")
    if dry_run:
        print("[warn] Dry run only: commands are computed but NOT sent to the robot.")
    print(f"Database: {args.database}")
    print(f"Graph: {args.graph}")
    print(f"Controller: {args.controller}")
    print(f"Subgoal hops: {args.max_subgoal_hops}")
    if args.target_step is not None:
        print(f"Target step: {args.target_step}")
    if args.target_image_name is not None:
        print(f"Target image: {args.target_image_name}")
    if args.checkpoint_steps:
        print(f"Checkpoint steps: {args.checkpoint_steps}")
    if args.checkpoint_images:
        print(f"Checkpoint images: {args.checkpoint_images}")
    print(f"Tick rate: {args.tick_hz:.2f} Hz")
    print("=" * 60)

    period = 1.0 / args.tick_hz
    iteration = 0
    prev_cur_step: Optional[int] = None
    target_reached_count = 0
    no_progress_count = 0
    TARGET_REACHED_MIN_CONF = 0.40
    TARGET_REACHED_CONFIRM_TICKS = 3
    JUMP_REJECT_THRESHOLD = 30
    PROXIMITY_SLOWDOWN_STEPS = 15
    NO_PROGRESS_RESET_TICKS = 20  # if stuck at same step for this many ticks, reset MBRA context

    # --- Angular saturation / wall-crash prevention ---
    ANGULAR_SAT_RATIO = 0.85           # fraction of max_angular that counts as "saturated"
    ANGULAR_SAT_MAX_TICKS = 4          # consecutive saturated ticks before safety action
    ANGULAR_OVERRIDE_TICKS = 4         # reused as pivot-hold ticks after saturation
    ANGULAR_PIVOT_ANGULAR = 0.16       # slow in-place pivot; never drive straight here
    angular_sat_count = 0
    angular_override_remaining = 0

    # --- Jump rejection recovery ---
    JUMP_REJECT_RECOVERY_TICKS = 12    # consecutive jump rejections before recovery
    RECOVERY_BACKUP_TICKS = 5          # ticks to reverse during recovery
    RECOVERY_BACKUP_LINEAR = -0.12     # gentle reverse
    jump_reject_consecutive = 0
    recovery_backup_remaining = 0

    # --- RPM stall detection ---
    RPM_STALL_THRESHOLD = 1.5          # RPM below this while commanding forward = stalled
    RPM_STALL_MAX_TICKS = 8            # ticks before triggering stall recovery
    rpm_stall_count = 0
    NO_PATH_SKIP_MIN_CONF = 0.50
    NO_PATH_SKIP_CONFIRM_TICKS = 2
    no_path_skip_count = 0
    NO_PROGRESS_RECOVERY_TICKS = 18
    RECOVERY_SPIN_TICKS = 8
    RECOVERY_SPIN_ANGULAR = 0.18
    recovery_spin_remaining = 0
    recovery_spin_direction = 1.0
    STUCK_STEP_TOLERANCE = 6
    stuck_recovery_stage = 0
    stuck_recovery_anchor: Optional[int] = None
    SATURATION_RECOVERY_EVENTS = 3
    saturation_event_count = 0
    last_turn_sign = 1.0
    IMU_SPIN_RATE_DPS = 25.0
    IMU_SPIN_CONFIRM_TICKS = 3
    imu_spin_count = 0
    SAME_FRAME_MAE_THRESHOLD = 2.0
    SAME_FRAME_CONFIRM_TICKS = 10
    same_frame_count = 0
    prev_frame_sig: Optional[np.ndarray] = None
    RECOVERY_COOLDOWN_TICKS = 10
    recovery_cooldown_remaining = 0

    try:
        while True:
            if args.max_steps is not None and iteration >= args.max_steps:
                break

            loop_start = time.time()
            frame = rover.get_camera_frame()
            data = rover.get_data()
            motion_state = sensor_filter.update(data)
            heading_deg = motion_state.get("heading_deg")

            if frame is None:
                if not dry_run:
                    rover.stop()
                print("[warn] No camera frame; stopping for this iteration.")
                time.sleep(period)
                iteration += 1
                continue

            # Log frame shape on first iteration for debugging
            if iteration == 0:
                print(f"[info] Live frame shape: {frame.shape}")

            cur_frame_sig = frame_signature(frame)

            if is_checkpoint_mode:
                step_output = runtime.step_to_active_checkpoint(
                    frame_rgb=frame,
                    observation_heading_deg=heading_deg,
                    auto_advance_checkpoint=args.auto_advance_checkpoints,
                )
            else:
                step_output = runtime.step_to_target(
                    frame_rgb=frame,
                    target_step=args.target_step,
                    target_image_name=args.target_image_name,
                    observation_heading_deg=heading_deg,
                )

            controller_input = step_output["controller_input"]
            controller_input["heading_rate_dps"] = motion_state.get("heading_rate_dps")
            controller_input["rpm_mean"] = motion_state.get("rpm_mean")
            controller_input["motion_state_stale"] = False if dry_run else motion_state.get("is_stale")
            # Indoor compass is unreliable — disable heading-based alignment
            controller_input["subgoal_orientation"] = None

            confidence = float(controller_input.get("confidence", 0.0))
            path_found = bool(controller_input.get("path_found", True))
            path_error = controller_input.get("path_error")
            cur_step = controller_input.get("current_step")
            tgt_step = controller_input.get("target_step")

            # --- Checkpoint completion / skip logic ---
            if is_checkpoint_mode and args.auto_advance_checkpoints:
                checkpoint_reached = bool(controller_input.get("checkpoint_reached", False))
                next_cp = controller_input.get("next_active_checkpoint")
                if checkpoint_reached and next_cp is None:
                    if not dry_run:
                        rover.stop()
                    print(
                        f"[{iteration:04d}] ALL CHECKPOINTS COMPLETED — "
                        f"cur={cur_step} conf={confidence:.3f}"
                    )
                    break

                skip_past_checkpoint = (
                    not path_found
                    and cur_step is not None
                    and tgt_step is not None
                    and int(cur_step) > int(tgt_step)
                    and confidence >= NO_PATH_SKIP_MIN_CONF
                )
                if skip_past_checkpoint:
                    no_path_skip_count += 1
                else:
                    no_path_skip_count = 0

                if no_path_skip_count >= NO_PATH_SKIP_CONFIRM_TICKS:
                    skipped_step = tgt_step
                    next_cp = runtime.planner.advance_checkpoint()
                    no_path_skip_count = 0
                    if next_cp is None:
                        if not dry_run:
                            rover.stop()
                        print(
                            f"[{iteration:04d}] ALL CHECKPOINTS COMPLETED — skipped final past checkpoint "
                            f"{skipped_step} at cur={cur_step} conf={confidence:.3f}"
                        )
                        break
                    next_step = runtime.planner.node_to_step.get(int(next_cp))
                    print(
                        f"[{iteration:04d}] skipping past checkpoint {skipped_step} — cur={cur_step} "
                        f"conf={confidence:.3f} next={next_step}"
                    )
                    prev_frame_sig = cur_frame_sig
                    elapsed = time.time() - loop_start
                    if elapsed < period:
                        time.sleep(period - elapsed)
                    iteration += 1
                    continue

            # --- Localization jump rejection ---
            # Bigger jumps require higher confidence to be trusted.
            # A 30-step jump needs ~0.50 conf; 100+ steps needs ~0.85.
            jump_rejected = False
            if prev_cur_step is not None and cur_step is not None:
                jump_mag = abs(int(cur_step) - prev_cur_step)
                if jump_mag > JUMP_REJECT_THRESHOLD:
                    required_conf = min(0.85, 0.35 + 0.005 * jump_mag)
                    if confidence < required_conf:
                        jump_rejected = True
                        # Revert localizer so it doesn't converge on wrong position
                        runtime.revert_localization()
                        controller_input["current_step"] = prev_cur_step
                        cur_step = prev_cur_step

            command = controller.compute_command(controller_input, observation_heading_deg=heading_deg)

            # --- Target reached detection (single-target mode only) ---
            # In checkpoint mode, advancement is handled by the runtime.
            if not is_checkpoint_mode and cur_step is not None and tgt_step is not None and int(cur_step) >= int(tgt_step):
                if confidence >= TARGET_REACHED_MIN_CONF:
                    target_reached_count += 1
                else:
                    target_reached_count = 0
                if target_reached_count >= TARGET_REACHED_CONFIRM_TICKS:
                    if not dry_run:
                        rover.stop()
                    print(
                        f"[{iteration:04d}] TARGET REACHED: cur={cur_step} >= target={tgt_step} "
                        f"conf={confidence:.3f} (confirmed {target_reached_count} ticks)"
                    )
                    break
            else:
                target_reached_count = 0

            # --- Proximity slowdown: reduce speed near target ---
            if cur_step is not None and tgt_step is not None and command.linear > 0:
                remaining = int(tgt_step) - int(cur_step)
                if 0 < remaining <= PROXIMITY_SLOWDOWN_STEPS:
                    scale = max(0.4, remaining / PROXIMITY_SLOWDOWN_STEPS)
                    command.linear *= scale
                    command.linear = max(0.06, command.linear)

            # --- Recovery: if backing up from a stuck state, override everything ---
            if recovery_backup_remaining > 0:
                command.linear = RECOVERY_BACKUP_LINEAR
                command.angular = 0.0
                command.reason = "recovery_backup"
                recovery_backup_remaining -= 1
            elif recovery_spin_remaining > 0:
                command.linear = 0.0
                command.angular = recovery_spin_direction * RECOVERY_SPIN_ANGULAR
                command.reason = "recovery_spin"
                recovery_spin_remaining -= 1
                if recovery_spin_remaining == 0:
                    runtime.reset()
                    if hasattr(controller, 'reset'):
                        controller.reset()
                    jump_reject_consecutive = 0
                    angular_sat_count = 0
                    angular_override_remaining = 0
                    saturation_event_count = 0
                    rpm_stall_count = 0
                    no_progress_count = 0
                    prev_cur_step = None
                    recovery_cooldown_remaining = RECOVERY_COOLDOWN_TICKS
                    print(f"[{iteration:04d}] recovery complete — relocalizing")
            elif jump_rejected:
                jump_reject_consecutive += 1
                if jump_reject_consecutive >= JUMP_REJECT_RECOVERY_TICKS:
                    # Stuck facing wall — initiate backup + spin recovery
                    recovery_backup_remaining = RECOVERY_BACKUP_TICKS
                    recovery_spin_remaining = RECOVERY_SPIN_TICKS
                    recovery_spin_direction = -1.0 if last_turn_sign >= 0.0 else 1.0
                    command.linear = RECOVERY_BACKUP_LINEAR
                    command.angular = 0.0
                    command.reason = "recovery_backup_start"
                    print(f"[{iteration:04d}] jump rejection stuck for {jump_reject_consecutive} ticks — backup + spin")
                else:
                    command.linear = 0.0
                    command.angular = 0.0
                    command.reason = "jump_rejected_stop"
            elif not path_found:
                command.linear = 0.0
                command.angular = 0.0
                command.reason = "runtime_no_path_stop"
                jump_reject_consecutive = 0
            elif args.stop_on_low_confidence and confidence < args.min_confidence:
                command.linear = 0.0
                command.angular = 0.0
                command.reason = "runtime_low_confidence_stop"
                jump_reject_consecutive = 0
            else:
                jump_reject_consecutive = 0

                # --- Angular saturation detection (prevents wall-crash) ---
                max_ang = getattr(controller, 'config', None)
                max_ang = max_ang.max_angular if max_ang and hasattr(max_ang, 'max_angular') else 0.34
                if abs(command.angular) > ANGULAR_SAT_RATIO * max_ang:
                    angular_sat_count += 1
                else:
                    angular_sat_count = 0

                if angular_sat_count >= ANGULAR_SAT_MAX_TICKS:
                    angular_override_remaining = ANGULAR_OVERRIDE_TICKS
                    angular_sat_count = 0
                    saturation_event_count += 1
                    print(f"[{iteration:04d}] angular saturation — pivot hold for {ANGULAR_OVERRIDE_TICKS} ticks")
                    if recovery_cooldown_remaining == 0 and saturation_event_count >= SATURATION_RECOVERY_EVENTS:
                        recovery_backup_remaining = RECOVERY_BACKUP_TICKS
                        recovery_spin_remaining = RECOVERY_SPIN_TICKS
                        recovery_spin_direction = -1.0 if last_turn_sign >= 0.0 else 1.0
                        command.linear = RECOVERY_BACKUP_LINEAR
                        command.angular = 0.0
                        command.reason = "recovery_backup_start"
                        saturation_event_count = 0
                        print(f"[{iteration:04d}] repeated angular saturation — backup + spin relocalize")

                if angular_override_remaining > 0:
                    command.linear = 0.0
                    command.angular = last_turn_sign * ANGULAR_PIVOT_ANGULAR
                    command.reason = "angular_pivot_hold"
                    angular_override_remaining -= 1
                if abs(command.angular) > 1e-4:
                    last_turn_sign = 1.0 if command.angular > 0.0 else -1.0

                heading_rate_dps = float(motion_state.get("heading_rate_dps") or 0.0)
                risky_spin = (
                    recovery_cooldown_remaining == 0
                    and cur_step is not None
                    and prev_cur_step is not None
                    and int(cur_step) == prev_cur_step
                    and no_progress_count >= 4
                    and command.linear > 0.08
                    and abs(command.angular) > 0.20
                    and abs(heading_rate_dps) >= IMU_SPIN_RATE_DPS
                )
                if risky_spin:
                    imu_spin_count += 1
                else:
                    imu_spin_count = 0
                if imu_spin_count >= IMU_SPIN_CONFIRM_TICKS:
                    recovery_backup_remaining = RECOVERY_BACKUP_TICKS
                    recovery_spin_remaining = RECOVERY_SPIN_TICKS
                    recovery_spin_direction = -1.0 if heading_rate_dps >= 0.0 else 1.0
                    command.linear = RECOVERY_BACKUP_LINEAR
                    command.angular = 0.0
                    command.reason = "imu_spin_recovery"
                    imu_spin_count = 0
                    print(f"[{iteration:04d}] IMU spin guard triggered at step {cur_step} rate={heading_rate_dps:.1f} dps")

                same_frame = False
                if prev_frame_sig is not None:
                    frame_mae = float(np.mean(np.abs(cur_frame_sig - prev_frame_sig)))
                    same_frame = frame_mae <= SAME_FRAME_MAE_THRESHOLD
                else:
                    frame_mae = None

                if (
                    recovery_cooldown_remaining == 0
                    and same_frame
                    and cur_step is not None
                    and prev_cur_step is not None
                    and int(cur_step) == prev_cur_step
                    and no_progress_count >= 4
                    and command.linear > 0.08
                ):
                    same_frame_count += 1
                else:
                    same_frame_count = 0

                if same_frame_count >= SAME_FRAME_CONFIRM_TICKS:
                    recovery_backup_remaining = RECOVERY_BACKUP_TICKS
                    recovery_spin_remaining = RECOVERY_SPIN_TICKS
                    recovery_spin_direction = -1.0 if last_turn_sign >= 0.0 else 1.0
                    command.linear = RECOVERY_BACKUP_LINEAR
                    command.angular = 0.0
                    command.reason = "same_frame_recovery"
                    same_frame_count = 0
                    if frame_mae is None:
                        print(f"[{iteration:04d}] same-frame guard triggered at step {cur_step}")
                    else:
                        print(f"[{iteration:04d}] same-frame guard triggered at step {cur_step} mae={frame_mae:.2f}")

                # --- Depth safety veto (monocular forward-clearance) ---
                if depth_estimator is not None and command.linear > 0.0 and frame is not None:
                    try:
                        # Run at half rate to save GPU time (~1.5Hz at 3Hz loop)
                        if iteration % 2 == 0:
                            depth_map = depth_estimator.estimate(frame, target_size=(120, 160))
                            clearance, _ = depth_estimator.get_polar_clearance(
                                depth_map, num_bins=16, fov_horizontal=90.0
                            )
                            # Forward clearance = center 4 of 16 bins (±22.5° arc)
                            fwd_clearance = float(min(clearance[6:10]))
                            if fwd_clearance < args.depth_stop_m:
                                command.linear = 0.0
                                command.angular = 0.0
                                command.reason = f"depth_stop({fwd_clearance:.2f}m)"
                                if recovery_backup_remaining == 0:
                                    recovery_backup_remaining = RECOVERY_BACKUP_TICKS
                            elif fwd_clearance < args.depth_slow_m:
                                scale = (fwd_clearance - args.depth_stop_m) / (args.depth_slow_m - args.depth_stop_m)
                                command.linear *= max(0.3, scale)
                                command.reason = f"depth_slow({fwd_clearance:.2f}m)"
                    except Exception:
                        pass  # depth inference failure is non-fatal

                # --- RPM stall detection (robot pushing against obstacle) ---
                # Guarded by not dry_run: RPM is near-zero in dry run and
                # would falsely trigger repeated backup cycles.
                rpm = motion_state.get("rpm_mean") or 0.0
                if not dry_run and command.linear > 0.05 and rpm < RPM_STALL_THRESHOLD:
                    rpm_stall_count += 1
                else:
                    rpm_stall_count = 0

                if recovery_cooldown_remaining == 0 and rpm_stall_count >= RPM_STALL_MAX_TICKS:
                    recovery_backup_remaining = RECOVERY_BACKUP_TICKS
                    recovery_spin_remaining = RECOVERY_SPIN_TICKS
                    recovery_spin_direction = -1.0 if last_turn_sign >= 0.0 else 1.0
                    command.linear = RECOVERY_BACKUP_LINEAR
                    command.angular = 0.0
                    command.reason = "rpm_stall_backup"
                    rpm_stall_count = 0
                    print(f"[{iteration:04d}] RPM stall detected — backup + spin")

            if dry_run:
                sent = False
            else:
                sent = rover.send_control(command.linear, command.angular)

            payload = {
                "iteration": iteration,
                "heading_deg": heading_deg,
                "heading_rate_dps": motion_state.get("heading_rate_dps"),
                "rpm_mean": motion_state.get("rpm_mean"),
                "motion_state_stale": motion_state.get("is_stale"),
                "current_step": controller_input.get("current_step"),
                "target_step": controller_input.get("target_step"),
                "subgoal_step": controller_input.get("subgoal_step"),
                "confidence": confidence,
                "held_previous": controller_input.get("held_previous"),
                "stable_steps": controller_input.get("stable_steps"),
                "path_found": path_found,
                "path_error": path_error,
                "linear": command.linear,
                "angular": command.angular,
                "reason": command.reason,
                "debug": command.debug or {},
                "sent": sent,
            }

            if args.print_json:
                print(json.dumps(payload))
            else:
                path_error_suffix = ""
                if payload["path_error"] is not None:
                    path_error_suffix = f" path_error={payload['path_error']}"
                print(
                    f"[{iteration:04d}] cur={payload['current_step']} "
                    f"target={payload['target_step']} subgoal={payload['subgoal_step']} "
                    f"conf={payload['confidence']:.3f} "
                    f"hdg={payload['heading_deg'] if payload['heading_deg'] is not None else 'NA'} "
                    f"cmd=({payload['linear']:.3f}, {payload['angular']:.3f}) "
                    f"reason={payload['reason']} sent={payload['sent']}"
                    f"{path_error_suffix}"
                    f"{'' if not payload['debug'] else ' dbg=' + json.dumps(payload['debug'], sort_keys=True)}"
                )

            if cur_step is not None and not jump_rejected:
                if prev_cur_step is not None and int(cur_step) == prev_cur_step:
                    no_progress_count += 1
                else:
                    no_progress_count = 0
                    saturation_event_count = 0
                    imu_spin_count = 0
                    same_frame_count = 0
                    stuck_recovery_stage = 0
                    stuck_recovery_anchor = None
                if (
                    recovery_cooldown_remaining == 0
                    and no_progress_count >= NO_PROGRESS_RECOVERY_TICKS
                    and recovery_backup_remaining == 0
                    and recovery_spin_remaining == 0
                    and command.linear > 0.08
                    and command.reason not in {"mbra_warmup", "mbra_low_confidence_stop", "low_confidence_stop"}
                ):
                    same_stuck_area = (
                        stuck_recovery_anchor is not None
                        and abs(int(cur_step) - int(stuck_recovery_anchor)) <= STUCK_STEP_TOLERANCE
                    )
                    recovery_spin_remaining = RECOVERY_SPIN_TICKS
                    recovery_spin_direction = -1.0 if last_turn_sign >= 0.0 else 1.0
                    no_progress_count = 0
                    if stuck_recovery_stage >= 1 and same_stuck_area:
                        recovery_backup_remaining = RECOVERY_BACKUP_TICKS
                        print(f"[{iteration:04d}] no progress at step {cur_step} — backup + spin relocalize")
                    else:
                        stuck_recovery_stage = 1
                        stuck_recovery_anchor = int(cur_step)
                        print(f"[{iteration:04d}] no progress at step {cur_step} — spin relocalize")
                elif no_progress_count > 0 and no_progress_count % NO_PROGRESS_RESET_TICKS == 0:
                    if hasattr(controller, 'reset'):
                        controller.reset()
                        print(f"[{iteration:04d}] no-progress reset after {no_progress_count} ticks at step {cur_step}")
                prev_cur_step = int(cur_step)

            if recovery_cooldown_remaining > 0:
                recovery_cooldown_remaining -= 1

            elapsed = time.time() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)
            iteration += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if not dry_run:
            rover.stop()
            print("Robot stopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
