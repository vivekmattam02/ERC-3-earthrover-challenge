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
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from earthrover_interface import EarthRoverInterface  # type: ignore
from local_controller import SimpleLocalController, SimpleLocalControllerConfig  # type: ignore
from navigation_runtime import NavigationRuntime, NavigationRuntimeConfig  # type: ignore


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
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="*",
        default=None,
        help="Optional ordered checkpoint steps for sequential execution.",
    )
    parser.add_argument("--tick-hz", type=float, default=2.0, help="Loop frequency in Hz.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional max loop iterations.")
    parser.add_argument("--max-subgoal-hops", type=int, default=3, help="Graph hops ahead for subgoal selection.")
    parser.add_argument("--sdk-url", default="http://localhost:8000", help="EarthRover SDK base URL.")
    parser.add_argument("--sdk-timeout", type=float, default=5.0, help="SDK request timeout in seconds.")
    parser.add_argument("--send-control", action="store_true", help="Actually send commands to the robot.")
    parser.add_argument("--auto-advance-checkpoints", action="store_true", help="Advance to next checkpoint automatically.")
    parser.add_argument("--stop-on-low-confidence", action="store_true", help="Stop when localization confidence is too low.")
    parser.add_argument("--min-confidence", type=float, default=0.35, help="Low-confidence stop threshold.")
    parser.add_argument("--print-json", action="store_true", help="Print each loop state as JSON.")
    parser.add_argument(
        "--controller",
        choices=("simple", "mbra", "logonav"),
        default="simple",
        help="Choose the local controller implementation.",
    )
    parser.add_argument(
        "--mbra-config",
        type=Path,
        default=None,
        help="Optional override for the MBRA/LogoNav YAML config.",
    )
    parser.add_argument(
        "--mbra-checkpoint",
        type=Path,
        default=None,
        help="Optional override for the MBRA/LogoNav checkpoint path.",
    )
    parser.add_argument(
        "--mbra-device",
        default=None,
        help="Optional torch device string for the MBRA/LogoNav controller, e.g. cpu or cuda:0.",
    )
    return parser.parse_args()


def get_orientation_deg(data: Optional[dict]) -> Optional[float]:
    if not data:
        return None
    orientation = data.get("orientation")
    if orientation is None:
        return None
    try:
        return float(orientation)
    except (TypeError, ValueError):
        return None


def build_runtime(args: argparse.Namespace) -> NavigationRuntime:
    return NavigationRuntime(
        NavigationRuntimeConfig(
            database_npz=args.database,
            graph_json=args.graph,
            data_info_json=args.data_info_json,
            max_subgoal_hops=args.max_subgoal_hops,
        )
    )


def build_controller(args: argparse.Namespace):
    if args.controller == "simple":
        return SimpleLocalController(SimpleLocalControllerConfig())

    from mbra_local_controller import MBRALocalController, MBRALocalControllerConfig  # type: ignore

    config_name = "MBRA.yaml" if args.controller == "mbra" else "LogoNav.yaml"
    checkpoint_name = "mbra.pth" if args.controller == "mbra" else "logonav.pth"
    return MBRALocalController(
        MBRALocalControllerConfig(
            repo_root=REPO_ROOT,
            model_config_path=args.mbra_config or REPO_ROOT / "mbra_repo_1" / "train" / "config" / config_name,
            checkpoint_path=args.mbra_checkpoint or REPO_ROOT / "mbra_repo_1" / "deployment" / "model_weights" / checkpoint_name,
            device=args.mbra_device,
        )
    )


def main() -> int:
    args = parse_args()
    if args.target_step is None and not args.checkpoint_steps:
        raise SystemExit("Provide --target-step or --checkpoint-steps.")
    if args.tick_hz <= 0:
        raise SystemExit("--tick-hz must be > 0")

    runtime = build_runtime(args)
    controller = build_controller(args)
    rover = EarthRoverInterface(base_url=args.sdk_url, timeout=args.sdk_timeout)

    if not rover.connect():
        raise SystemExit("Failed to connect to SDK.")

    if args.checkpoint_steps:
        runtime.set_checkpoints(checkpoint_steps=list(args.checkpoint_steps))

    dry_run = not args.send_control
    print("Live indoor runtime")
    print("=" * 60)
    print(f"Mode: {'DRY RUN' if dry_run else 'SEND CONTROL'}")
    print(f"Database: {args.database}")
    print(f"Graph: {args.graph}")
    if args.target_step is not None:
        print(f"Target step: {args.target_step}")
    if args.checkpoint_steps:
        print(f"Checkpoint steps: {args.checkpoint_steps}")
    print(f"Tick rate: {args.tick_hz:.2f} Hz")
    print("=" * 60)

    period = 1.0 / args.tick_hz
    iteration = 0

    try:
        while True:
            if args.max_steps is not None and iteration >= args.max_steps:
                break

            loop_start = time.time()
            frame = rover.get_camera_frame()
            data = rover.get_data()
            heading_deg = get_orientation_deg(data)

            if frame is None:
                if not dry_run:
                    rover.stop()
                print("[warn] No camera frame; stopping for this iteration.")
                time.sleep(period)
                iteration += 1
                continue

            if args.checkpoint_steps:
                step_output = runtime.step_to_active_checkpoint(
                    frame_rgb=frame,
                    observation_heading_deg=heading_deg,
                    auto_advance_checkpoint=args.auto_advance_checkpoints,
                )
            else:
                step_output = runtime.step_to_target(
                    frame_rgb=frame,
                    target_step=args.target_step,
                    observation_heading_deg=heading_deg,
                )

            controller_input = step_output["controller_input"]
            command = controller.compute_command(
                controller_input,
                observation_heading_deg=heading_deg,
                frame_rgb=frame,
            )

            confidence = float(controller_input.get("confidence", 0.0))
            if args.stop_on_low_confidence and confidence < args.min_confidence:
                command.linear = 0.0
                command.angular = 0.0
                command.reason = "runtime_low_confidence_stop"

            if dry_run:
                sent = False
            else:
                sent = rover.send_control(command.linear, command.angular)

            payload = {
                "iteration": iteration,
                "heading_deg": heading_deg,
                "current_step": controller_input.get("current_step"),
                "target_step": controller_input.get("target_step"),
                "subgoal_step": controller_input.get("subgoal_step"),
                "confidence": confidence,
                "held_previous": controller_input.get("held_previous"),
                "stable_steps": controller_input.get("stable_steps"),
                "linear": command.linear,
                "angular": command.angular,
                "reason": command.reason,
                "sent": sent,
            }

            if args.print_json:
                print(json.dumps(payload))
            else:
                print(
                    f"[{iteration:04d}] cur={payload['current_step']} "
                    f"target={payload['target_step']} subgoal={payload['subgoal_step']} "
                    f"conf={payload['confidence']:.3f} "
                    f"cmd=({payload['linear']:.3f}, {payload['angular']:.3f}) "
                    f"reason={payload['reason']} sent={payload['sent']}"
                )

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
