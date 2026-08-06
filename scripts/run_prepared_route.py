#!/usr/bin/env python3
"""Launch live_indoor_runtime.py from a prepared manual route package."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a prepared manual route package.")
    parser.add_argument("--route-dir", type=Path, required=True, help="Route directory created by prepare_manual_route.py.")
    parser.add_argument("--sdk-url", default="http://localhost:8000", help="EarthRover SDK URL.")
    parser.add_argument("--controller", choices=("simple", "adaptive", "mbra"), default=None, help="Override controller from route info.")
    parser.add_argument("--tick-hz", type=float, default=None, help="Override tick rate.")
    parser.add_argument("--max-subgoal-hops", type=int, default=None, help="Override max subgoal hops.")
    parser.add_argument("--rough-terrain", action="store_true", help="Use conservative rough-terrain route-follow settings.")
    parser.add_argument("--use-route-heading", action="store_true", help="Use route heading metadata for steering.")
    parser.add_argument("--no-route-heading", action="store_true", help="Disable route heading metadata even in rough-terrain mode.")
    parser.add_argument("--startup-step-hint", type=int, default=None, help="Expected start step to constrain startup localization.")
    parser.add_argument("--startup-step-radius", type=int, default=18, help="Step radius around the startup hint.")
    parser.add_argument("--startup-step-lock-ticks", type=int, default=20, help="How long to keep startup localization constrained.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional bounded run length for dry runs or short live tests.")
    parser.add_argument("--strict-startup-health", action="store_true", help="Refuse to start if camera/telemetry/battery health looks invalid.")
    parser.add_argument("--startup-require-battery", action="store_true", help="Require a valid nonzero battery reading before starting.")
    parser.add_argument("--startup-min-battery", type=float, default=15.0, help="Minimum battery percentage for startup health checks.")
    parser.add_argument("--send-control", action="store_true", help="Actually send commands to the robot.")
    parser.add_argument("--print-only", action="store_true", help="Print the final command without running it.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    route_dir = args.route_dir.expanduser().resolve()
    route_info_path = route_dir / "route_info.json"
    if not route_info_path.is_file():
        raise SystemExit(f"route_info.json not found in {route_dir}")

    with route_info_path.open("r", encoding="utf-8") as handle:
        route_info = json.load(handle)

    database = Path(route_info["db_dir"]) / "descriptors.npz"
    graph = Path(route_info["db_dir"]) / "navigation_graph.json"
    data_info = Path(route_info["extracted_dir"]) / "metadata" / "data_info.json"
    target_step = int(route_info["target_step"])

    controller = args.controller or ("adaptive" if args.rough_terrain else "simple")
    tick_hz = args.tick_hz if args.tick_hz is not None else 2.0
    max_subgoal_hops = args.max_subgoal_hops if args.max_subgoal_hops is not None else (6 if args.rough_terrain else 12)
    startup_step_radius = args.startup_step_radius
    startup_step_lock_ticks = args.startup_step_lock_ticks
    if args.rough_terrain and args.startup_step_hint is not None:
        if startup_step_radius == 18:
            startup_step_radius = 8
        if startup_step_lock_ticks == 20:
            startup_step_lock_ticks = 200

    cmd = [
        sys.executable,
        "live_indoor_runtime.py",
        "--database",
        str(database),
        "--graph",
        str(graph),
        "--data-info-json",
        str(data_info),
        "--target-step",
        str(target_step),
        "--controller",
        controller,
        "--tick-hz",
        str(tick_hz),
        "--max-subgoal-hops",
        str(max_subgoal_hops),
        "--sdk-url",
        args.sdk_url,
    ]
    if args.rough_terrain:
        cmd.append("--rough-terrain")
    if args.use_route_heading and args.no_route_heading:
        raise SystemExit("Choose either --use-route-heading or --no-route-heading, not both")
    if args.no_route_heading or (args.rough_terrain and not args.use_route_heading):
        cmd.append("--no-route-heading")
    elif args.use_route_heading:
        cmd.append("--use-route-heading")
    if args.startup_step_hint is not None:
        cmd.extend(
            [
                "--startup-step-hint",
                str(args.startup_step_hint),
                "--startup-step-radius",
                str(startup_step_radius),
                "--startup-step-lock-ticks",
                str(startup_step_lock_ticks),
            ]
        )
    if args.strict_startup_health:
        cmd.append("--strict-startup-health")
    if args.startup_require_battery:
        cmd.extend(["--startup-require-battery", "--startup-min-battery", str(args.startup_min_battery)])
    if args.max_steps is not None:
        cmd.extend(["--max-steps", str(args.max_steps)])
    if args.send_control:
        cmd.append("--send-control")

    print(" ".join(cmd))
    if args.print_only:
        return 0

    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
