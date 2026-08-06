#!/usr/bin/env python3
"""Turn a recorded H5 manual run into a ready visual teach-and-repeat route.

This script automates the strongest no-GPS path available in this repo:
1. trim a recorded bag to its active motion window
2. extract front frames + metadata into a baseline-friendly dataset
3. build CosPlace descriptors and a navigation graph
4. print (or launch) the live visual route-follow command

It does not invent a flag detector. It reuses the existing known-route
localization stack on top of a manually collected reference traversal.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROUTE_ROOT = REPO_ROOT / "data" / "manual_routes"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a no-GPS visual route from a recorded H5 bag.")
    parser.add_argument("--input-h5", type=Path, required=True, help="Recorded H5 bag to use as the reference route.")
    parser.add_argument(
        "--route-name",
        type=str,
        default=None,
        help="Output route name. Default: derived from the H5 filename.",
    )
    parser.add_argument(
        "--route-root",
        type=Path,
        default=DEFAULT_ROUTE_ROOT,
        help="Directory under which extracted data and DB artifacts are written.",
    )
    parser.add_argument("--frame-step", type=int, default=3, help="Keep every Nth frame from the trimmed run.")
    parser.add_argument(
        "--auto-trim-motion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-trim to the active RPM motion window before extraction.",
    )
    parser.add_argument(
        "--motion-rpm-threshold",
        type=float,
        default=1.0,
        help="Summed RPM threshold used to detect motion windows.",
    )
    parser.add_argument(
        "--motion-pad-sec",
        type=float,
        default=2.0,
        help="Padding added before/after the detected motion window.",
    )
    parser.add_argument(
        "--cosplace-repo",
        type=Path,
        default=None,
        help="Optional local CosPlace repo. Default: cached torch hub checkout if present.",
    )
    parser.add_argument(
        "--controller",
        choices=("simple", "adaptive", "mbra"),
        default="adaptive",
        help="Controller to use when printing or launching the live command.",
    )
    parser.add_argument("--tick-hz", type=float, default=2.0, help="Live runtime tick rate.")
    parser.add_argument(
        "--max-subgoal-hops",
        type=int,
        default=12,
        help="Graph hops ahead for the live runtime subgoal choice.",
    )
    parser.add_argument(
        "--launch-live",
        action="store_true",
        help="Launch live_indoor_runtime.py after preparation completes.",
    )
    parser.add_argument(
        "--send-control",
        action="store_true",
        help="When used with --launch-live, actually send commands to the robot.",
    )
    parser.add_argument("--sdk-url", default="http://localhost:8000", help="SDK URL for the live runtime.")
    return parser.parse_args()


def sanitize_route_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "manual_route"


def detect_cosplace_repo(explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit.expanduser().resolve()
    cached = Path.home() / ".cache" / "torch" / "hub" / "gmberton_cosplace_main"
    if cached.is_dir():
        return cached.resolve()
    return None


def run_command(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def build_live_command(
    database: Path,
    graph: Path,
    data_info_json: Path,
    target_step: int,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        sys.executable,
        "live_indoor_runtime.py",
        "--database",
        str(database),
        "--graph",
        str(graph),
        "--data-info-json",
        str(data_info_json),
        "--target-step",
        str(target_step),
        "--controller",
        args.controller,
        "--tick-hz",
        str(args.tick_hz),
        "--max-subgoal-hops",
        str(args.max_subgoal_hops),
        "--sdk-url",
        args.sdk_url,
    ]
    if args.send_control:
        cmd.append("--send-control")
    return cmd


def main() -> int:
    args = parse_args()
    input_h5 = args.input_h5.expanduser().resolve()
    if not input_h5.is_file():
        raise SystemExit(f"H5 file not found: {input_h5}")
    if args.frame_step < 1:
        raise SystemExit("--frame-step must be >= 1")
    if args.tick_hz <= 0:
        raise SystemExit("--tick-hz must be > 0")

    route_name = sanitize_route_name(args.route_name or input_h5.stem)
    route_dir = args.route_root.expanduser().resolve() / route_name
    extracted_dir = route_dir / "extracted"
    db_dir = route_dir / "db"
    route_dir.mkdir(parents=True, exist_ok=True)

    extract_cmd = [
        sys.executable,
        "tools/extract_h5_dataset.py",
        "--input-h5",
        str(input_h5),
        "--output-dir",
        str(extracted_dir),
        "--frame-step",
        str(args.frame_step),
    ]
    if args.auto_trim_motion:
        extract_cmd.extend(
            [
                "--auto-trim-motion",
                "--motion-rpm-threshold",
                str(args.motion_rpm_threshold),
                "--motion-pad-sec",
                str(args.motion_pad_sec),
            ]
        )

    cosplace_repo = detect_cosplace_repo(args.cosplace_repo)
    build_cmd = [
        sys.executable,
        "baseline.py",
        "build-db",
        "--image-dir",
        str(extracted_dir / "front_images"),
        "--output-dir",
        str(db_dir),
        "--data-info-json",
        str(extracted_dir / "metadata" / "data_info.json"),
    ]
    if cosplace_repo is not None:
        build_cmd.extend(["--cosplace-repo", str(cosplace_repo)])

    print("Preparing visual route package")
    print("=" * 60)
    print(f"Input H5: {input_h5}")
    print(f"Route name: {route_name}")
    print(f"Route dir: {route_dir}")
    print(f"Frame step: {args.frame_step}")
    print(f"Auto trim motion: {args.auto_trim_motion}")
    if args.auto_trim_motion:
        print(f"Motion threshold: {args.motion_rpm_threshold}")
        print(f"Motion pad sec: {args.motion_pad_sec}")
    print(f"CosPlace repo: {cosplace_repo if cosplace_repo is not None else 'torch.hub default'}")
    print("=" * 60)

    run_command(extract_cmd)
    run_command(build_cmd)

    data_info_path = extracted_dir / "metadata" / "data_info.json"
    with data_info_path.open("r", encoding="utf-8") as handle:
        data_info = json.load(handle)
    if not data_info:
        raise SystemExit("Extracted data_info.json is empty; no route steps were produced.")
    target_step = int(data_info[-1]["step"])

    database = db_dir / "descriptors.npz"
    graph = db_dir / "navigation_graph.json"
    live_cmd = build_live_command(database, graph, data_info_path, target_step, args)

    route_info = {
        "input_h5": str(input_h5),
        "route_name": route_name,
        "route_dir": str(route_dir),
        "extracted_dir": str(extracted_dir),
        "db_dir": str(db_dir),
        "frame_step": args.frame_step,
        "auto_trim_motion": args.auto_trim_motion,
        "motion_rpm_threshold": args.motion_rpm_threshold,
        "motion_pad_sec": args.motion_pad_sec,
        "cosplace_repo": None if cosplace_repo is None else str(cosplace_repo),
        "target_step": target_step,
        "live_command": live_cmd,
    }
    with (route_dir / "route_info.json").open("w", encoding="utf-8") as handle:
        json.dump(route_info, handle, indent=2)

    print("\nRoute package ready.")
    print(f"Target step: {target_step}")
    print(f"Route info: {route_dir / 'route_info.json'}")
    print("Suggested live command:")
    print(" ".join(live_cmd))

    if args.launch_live:
        print("\nLaunching live runtime...")
        run_command(live_cmd)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
