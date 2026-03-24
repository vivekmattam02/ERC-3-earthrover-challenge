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
from local_controller import ControlCommand, LocalControllerInput, SimpleLocalController, SimpleLocalControllerConfig  # type: ignore
from navigation_runtime import NavigationRuntime, NavigationRuntimeConfig  # type: ignore
from sensor_state import SensorStateFilter, SensorStateFilterConfig  # type: ignore


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the live indoor navigation loop.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run the ERC indoor baseline live loop.")
    parser.add_argument(
        "--database",
        type=Path,
        default=REPO_ROOT / "data" / "corridor_db" / "descriptors.npz",
        help="Path to the built descriptor database.",
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=REPO_ROOT / "data" / "corridor_db" / "navigation_graph.json",
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
    parser.add_argument("--max-subgoal-search-hops", type=int, default=3, help="Graph hops ahead for subgoal selection.")
    parser.add_argument("--max-subgoal-cost-threshold", type=int, default=10, help="Maximum cost threshold for subgoal selection.")
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
    return parser.parse_args()


def get_orientation_deg(data: Optional[dict]) -> Optional[float]:
    """Extracts the rover's orientation in degrees from the SDK data payload.

    Args:
        data (Optional[dict]): The data payload from the EarthRover SDK.

    Returns:
        Optional[float]: The orientation in degrees, or None if not available or invalid.
    """
    if not data:
        return None
    # The 'orientation' key holds the compass heading.
    orientation = data.get("orientation")
    if orientation is None:
        return None
    try:
        # Convert the orientation to a float.
        return float(orientation)
    except (TypeError, ValueError):
        # Return None if the conversion fails.
        return None


def build_runtime(args: argparse.Namespace) -> NavigationRuntime:
    """Initializes the NavigationRuntime with the specified configuration.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.

    Returns:
        NavigationRuntime: An instance of the NavigationRuntime.
    """
    return NavigationRuntime(
        NavigationRuntimeConfig(
            database_npz=args.database,
            graph_json=args.graph,
            data_info_json=args.data_info_json,
            max_subgoal_search_hops=args.max_subgoal_search_hops,
            max_subgoal_cost_threshold=args.max_subgoal_cost_threshold,
        )
    )


def build_controller(args: argparse.Namespace):
    """Initializes the SimpleLocalController with its default configuration.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.

    Returns:
        SimpleLocalController: An instance of the SimpleLocalController.
    """
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
    """Main function to run the live indoor navigation loop.

    This function orchestrates the entire process:
    1. Parses arguments.
    2. Initializes the navigation runtime, local controller, and rover interface.
    3. Runs a continuous loop to fetch sensor data, localize, plan, and control the robot.

    Returns:
        int: An exit code (0 for success).
    """
    args = parse_args()
    if args.target_step is None and not args.checkpoint_steps:
        raise SystemExit("Provide --target-step or --checkpoint-steps.")
    if args.tick_hz <= 0:
        raise SystemExit("--tick-hz must be > 0")

    # Initialize the core components for navigation, control, and rover communication.
    runtime = build_runtime(args)
    controller: SimpleLocalController = build_controller(args)
    sensor_filter = SensorStateFilter(SensorStateFilterConfig())
    rover = EarthRoverInterface(base_url=args.sdk_url, timeout=args.sdk_timeout)

    # Ensure connection to the EarthRover SDK is successful.
    if not rover.connect():
        raise SystemExit("Failed to connect to SDK.")

    # If checkpoint navigation is enabled, set the checkpoints in the runtime.
    if args.checkpoint_steps:
        runtime.set_checkpoints(checkpoint_steps=list(args.checkpoint_steps))

    # Determine if the script is in dry-run mode (no commands sent).
    dry_run = not args.send_control
    print("Live indoor runtime")
    print("=" * 60)
    print(f"Mode: {'DRY RUN' if dry_run else 'SEND CONTROL'}")
    print(f"Database: {args.database}")
    print(f"Graph: {args.graph}")
    print(f"Controller: {args.controller}")
    if args.target_step is not None:
        print(f"Target step: {args.target_step}")
    if args.checkpoint_steps:
        print(f"Checkpoint steps: {args.checkpoint_steps}")
    print(f"Tick rate: {args.tick_hz:.2f} Hz")
    print("=" * 60)

    # Calculate the time period for each loop iteration based on the desired frequency.
    period = 1.0 / args.tick_hz
    iteration = 0

    try:
        # Main control loop.
        while True:
            # Optional exit condition based on max iterations.
            if args.max_steps is not None and iteration >= args.max_steps:
                break

            loop_start = time.time()
            # Fetch the latest camera frame and sensor data from the rover.
            frame = rover.get_camera_frame()
            data = rover.get_data()
            motion_state = sensor_filter.update(data)
            heading_deg = motion_state.get("heading_deg")

            # If no camera frame is received, stop the robot and wait for the next cycle.
            if frame is None:
                if not dry_run:
                    rover.stop()
                print("[warn] No camera frame; stopping for this iteration.")
                time.sleep(period)
                iteration += 1
                continue

            # Based on the navigation mode, either step towards the active checkpoint or a single target.
            if args.checkpoint_steps:
                # Checkpoint navigation: progresses through a predefined sequence of points.
                step_output = runtime.step_to_active_checkpoint(
                    frame_rgb=frame,
                    observation_heading_deg=heading_deg,
                    auto_advance_checkpoint=args.auto_advance_checkpoints,
                )
            else:
                # Target navigation: moves towards a single specified goal.
                step_output = runtime.step_to_target(
                    frame_rgb=frame,
                    target_step=args.target_step,
                    observation_heading_deg=heading_deg,
                )

            controller_input: LocalControllerInput = step_output["controller_input"]
            controller_input.heading_rate_dps = motion_state.get("heading_rate_dps") # type: ignore
            controller_input.rpm_mean = motion_state.get("rpm_mean") # type: ignore
            controller_input.motion_state_stale = motion_state.get("is_stale") # type: ignore
            command = controller.compute_command(
                controller_input,
                observation_heading_deg=heading_deg,
                frame_rgb=frame,
            )

            confidence = float(controller_input.confidence)
            if args.stop_on_low_confidence and confidence < args.min_confidence:
                command.linear = 0.0
                command.angular = 0.0
                command.reason = "runtime_low_confidence_stop"

            # Send the command to the robot unless in dry-run mode.
            if dry_run:
                sent = False
            else:
                sent = rover.send_control(command.linear, command.angular)

            # Assemble a payload of the current state for logging/debugging.
            payload = {
                "iteration": iteration,
                "heading_deg": heading_deg,
                "heading_rate_dps": motion_state.get("heading_rate_dps"),
                "rpm_mean": motion_state.get("rpm_mean"),
                "current_step": controller_input.current_step,
                "target_step": controller_input.target_step,
                "subgoal_step": controller_input.subgoal_step,
                "confidence": confidence,
                "held_previous": controller_input.held_previous,
                "stable_steps": controller_input.stable_steps,
                "path_found": controller_input.path_found,
                "path_error": controller_input.path_error,
                "linear": command.linear,
                "angular": command.angular,
                "reason": command.reason,
                "debug": command.debug or {},
                "sent": sent,
            }

            # Print the state to the console, either as JSON or a formatted string.
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

            # Wait for the remainder of the period to maintain the loop frequency.
            elapsed = time.time() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)
            iteration += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        # Graceful shutdown: stop the robot upon exit.
        if not dry_run:
            rover.stop()
            print("Robot stopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
