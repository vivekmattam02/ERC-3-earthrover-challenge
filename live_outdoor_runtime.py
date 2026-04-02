#!/usr/bin/env python3
"""Live outdoor runtime loop for GPS-checkpoint ERC missions.

This script is the outdoor counterpart to ``live_indoor_runtime.py``.

Current status:
- supports a classical GPS + optional VFH controller
- supports the teammate's stabilized LogoNav-style learned controller
- handles ordered GPS checkpoints
- preserves the same dry-run vs send-control workflow as the indoor runner
- optionally expands mission legs through OSM pedestrian routing
- keeps mission scoring tied to original checkpoints, not internal OSM waypoints
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import utm


REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from earthrover_interface import EarthRoverInterface  # type: ignore
from osm_router import OSMRoutingConfig, get_pedestrian_route  # type: ignore
from outdoor_gps_controller import OutdoorGPSController, OutdoorGPSControllerConfig  # type: ignore


LatLon = tuple[float, float]
TargetItem = dict[str, Any]


def wrap_angle_rad(delta: float) -> float:
    return (delta + math.pi) % (2.0 * math.pi) - math.pi


def compass_deg_to_math_rad(compass_deg: float) -> float:
    """Convert SDK compass heading to standard math angle.

    SDK assumption:
    - 0 deg = North
    - 90 deg = East

    Output convention:
    - 0 rad = +x / East
    - +pi/2 = +y / North
    """
    return wrap_angle_rad(math.radians(90.0 - float(compass_deg)))


def math_rad_to_logonav_compass_rad(math_rad: float) -> float:
    """Convert a standard math heading into LogoNav's original compass-radian convention.

    LogoNav expects 0 rad at North and negative radians as heading rotates clockwise
    toward East, matching its original ``-orientation_deg / 180 * pi`` conversion.
    """
    return wrap_angle_rad(float(math_rad) - math.pi / 2.0)


def parse_checkpoint_pairs(values: list[float]) -> list[LatLon]:
    if len(values) % 2 != 0:
        raise SystemExit("--checkpoints requires an even number of values: lat1 lon1 lat2 lon2 ...")
    out: list[LatLon] = []
    for idx in range(0, len(values), 2):
        out.append((float(values[idx]), float(values[idx + 1])))
    return out


def _coerce_checkpoint_item(item: object) -> LatLon:
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return float(item[0]), float(item[1])
    if isinstance(item, dict):
        lat = item.get("lat", item.get("latitude"))
        lon = item.get("lon", item.get("longitude"))
        if lat is None or lon is None:
            raise ValueError(f"Invalid checkpoint dict: {item}")
        return float(lat), float(lon)
    raise ValueError(f"Unsupported checkpoint item: {item}")


def load_checkpoint_file(path: Path) -> list[LatLon]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        if "checkpoints" in payload:
            payload = payload["checkpoints"]
        elif "mission" in payload and isinstance(payload["mission"], dict) and "checkpoints" in payload["mission"]:
            payload = payload["mission"]["checkpoints"]
        else:
            raise SystemExit(f"Checkpoint file format not understood: {path}")
    if not isinstance(payload, list):
        raise SystemExit(f"Checkpoint file must contain a list or dict with checkpoints: {path}")
    return [_coerce_checkpoint_item(item) for item in payload]


def build_checkpoint_list(args: argparse.Namespace) -> list[LatLon]:
    checkpoints: list[LatLon] = []
    if args.checkpoint_file is not None:
        checkpoints.extend(load_checkpoint_file(args.checkpoint_file))
    if args.checkpoints:
        checkpoints.extend(parse_checkpoint_pairs(args.checkpoints))
    if args.goal_lat is not None or args.goal_lon is not None:
        if args.goal_lat is None or args.goal_lon is None:
            raise SystemExit("Provide both --goal-lat and --goal-lon together.")
        checkpoints.append((float(args.goal_lat), float(args.goal_lon)))
    if not checkpoints:
        raise SystemExit("Provide --goal-lat/--goal-lon, --checkpoints, --checkpoint-file, or --mission.")
    return checkpoints


def bearing_rad_from_latlon(a: LatLon, b: LatLon) -> float:
    a_utm = utm.from_latlon(a[0], a[1])
    b_utm = utm.from_latlon(b[0], b[1])
    dx = float(b_utm[0]) - float(a_utm[0])
    dy = float(b_utm[1]) - float(a_utm[1])
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0
    return math.atan2(dy, dx)


def build_osm_config(args: argparse.Namespace) -> OSMRoutingConfig:
    config = OSMRoutingConfig(
        query_timeout_s=args.osm_query_timeout_s,
        request_retries=args.osm_request_retries,
        buffer_m=args.osm_buffer_m,
        max_segment_m=args.osm_max_segment_m,
        min_waypoint_spacing_m=args.osm_min_waypoint_spacing_m,
        max_snap_distance_m=args.osm_max_snap_distance_m,
    )
    if getattr(args, "ultra_marathon", False):
        # Hard-block highways — near-infinite cost forces pedestrian paths only
        config.highway_cost_multipliers["primary"] = 1e9
        config.highway_cost_multipliers["secondary"] = 1e9
        config.highway_cost_multipliers["tertiary"] = 8.0
        # Longer timeouts for large multi-mile leg queries
        config.timeout_s = 60.0
        config.query_timeout_s = 60
        config.request_retries = 3
    if getattr(args, "sidewalk_strict", False):
        config.allowed_highways = ("footway", "path", "pedestrian", "living_street", "cycleway")
        config.highway_cost_multipliers["footway"] = 1.0
        config.highway_cost_multipliers["pedestrian"] = 1.0
        config.highway_cost_multipliers["path"] = 1.05
        config.highway_cost_multipliers["living_street"] = 1.10
        config.highway_cost_multipliers["cycleway"] = 1.30
        config.max_snap_distance_m = min(config.max_snap_distance_m, 35.0)
        config.timeout_s = max(config.timeout_s, 60.0)
        config.query_timeout_s = max(config.query_timeout_s, 60)
        config.request_retries = max(config.request_retries, 3)
    return config


def point_to_segment_distance_m(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    px, py = point
    ax, ay = start
    bx, by = end
    abx = bx - ax
    aby = by - ay
    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * abx + (py - ay) * aby) / ab_len_sq
    t = max(0.0, min(1.0, t))
    cx = ax + t * abx
    cy = ay + t * aby
    return math.hypot(px - cx, py - cy)


def route_corridor_distance_m(
    point: tuple[float, float],
    route_polyline_utms: list[tuple[float, float]] | None,
    active_idx: int,
    *,
    lookback_segments: int = 1,
    lookahead_segments: int = 2,
) -> float | None:
    if route_polyline_utms is None or len(route_polyline_utms) < 2:
        return None
    max_seg = len(route_polyline_utms) - 2
    start_seg = max(0, min(max_seg, active_idx - lookback_segments))
    end_seg = max(start_seg, min(max_seg, active_idx + lookahead_segments))
    best = float("inf")
    for seg_idx in range(start_seg, end_seg + 1):
        dist = point_to_segment_distance_m(
            point,
            route_polyline_utms[seg_idx],
            route_polyline_utms[seg_idx + 1],
        )
        if dist < best:
            best = dist
    return best


def build_navigation_targets(
    mission_checkpoints: list[LatLon],
    *,
    start_latlon: LatLon | None,
    use_osm_route: bool,
    osm_config: OSMRoutingConfig | None,
    require_osm_success: bool = False,
) -> tuple[list[TargetItem], list[dict[str, Any]]]:
    if not use_osm_route or start_latlon is None or osm_config is None:
        targets: list[TargetItem] = []
        previous = start_latlon
        for idx, checkpoint in enumerate(mission_checkpoints):
            goal_compass = bearing_rad_from_latlon(previous, checkpoint) if previous is not None else 0.0
            targets.append(
                {
                    "lat": checkpoint[0],
                    "lon": checkpoint[1],
                    "mission_checkpoint": True,
                    "mission_index": idx,
                    "goal_compass_rad": goal_compass,
                }
            )
            previous = checkpoint
        return targets, []

    targets: list[TargetItem] = []
    routing_debug: list[dict[str, Any]] = []
    leg_start = start_latlon

    for mission_idx, checkpoint in enumerate(mission_checkpoints):
        result = get_pedestrian_route(leg_start, checkpoint, config=osm_config)
        if require_osm_success and result.debug.get("routing") == "fallback_straight_line":
            raise SystemExit(
                f"Strict sidewalk routing failed for leg {mission_idx}: OSM fell back to straight line from {leg_start} to {checkpoint}"
            )
        leg_waypoints = result.waypoints[1:] if len(result.waypoints) > 1 else [checkpoint]
        if not leg_waypoints:
            leg_waypoints = [checkpoint]
        if leg_waypoints[-1] != checkpoint:
            leg_waypoints.append(checkpoint)

        prev_point = leg_start
        for waypoint_idx, waypoint in enumerate(leg_waypoints):
            is_mission_checkpoint = waypoint_idx == len(leg_waypoints) - 1
            goal_compass = bearing_rad_from_latlon(prev_point, waypoint)
            prev_utm = utm.from_latlon(float(prev_point[0]), float(prev_point[1]))
            waypoint_utm = utm.from_latlon(float(waypoint[0]), float(waypoint[1]))
            segment_distance_m = math.hypot(float(waypoint_utm[0]) - float(prev_utm[0]), float(waypoint_utm[1]) - float(prev_utm[1]))
            targets.append(
                {
                    "lat": float(waypoint[0]),
                    "lon": float(waypoint[1]),
                    "mission_checkpoint": is_mission_checkpoint,
                    "mission_index": mission_idx if is_mission_checkpoint else None,
                    "goal_compass_rad": goal_compass,
                    "segment_distance_m": segment_distance_m,
                }
            )
            prev_point = waypoint

        routing_debug.append(
            {
                "leg_index": mission_idx,
                "start": [leg_start[0], leg_start[1]],
                "goal": [checkpoint[0], checkpoint[1]],
                "routing": result.debug,
                "expanded_waypoints": len(leg_waypoints),
            }
        )
        leg_start = checkpoint

    return targets, routing_debug


def build_controller(args: argparse.Namespace):
    if args.controller == "gps":
        return OutdoorGPSController(
            OutdoorGPSControllerConfig(
                goal_reached_radius_m=args.goal_radius_m,
                max_linear=args.max_linear,
                min_linear=args.min_linear,
                max_angular=args.max_angular,
                angular_gain=args.angular_gain,
                nominal_linear=args.nominal_linear,
                in_place_turn_threshold_deg=args.in_place_turn_threshold_deg,
                in_place_turn_exit_deg=args.in_place_turn_exit_deg,
                vfh_num_bins=args.vfh_bins,
                vfh_fov_horizontal_deg=args.vfh_fov_deg,
                vfh_blocked_distance_m=args.vfh_blocked_distance_m,
                depth_slow_distance_m=args.depth_slow_m,
                depth_stop_distance_m=args.depth_stop_m,
            )
        )

    from outdoor_logonav_controller import (  # type: ignore
        OutdoorLogoNavController,
        OutdoorLogoNavControllerConfig,
    )

    cfg = OutdoorLogoNavControllerConfig(
        weights_path=args.logonav_weights,
        config_path=args.logonav_config,
        device=args.logonav_device,
        max_linear=args.logonav_max_linear,
        max_angular=args.logonav_max_angular,
    )
    if not cfg.config_path.exists():
        raise SystemExit(f"LogoNav config not found: {cfg.config_path}")
    if not cfg.weights_path.exists():
        raise SystemExit(f"LogoNav weights not found: {cfg.weights_path}")
    return OutdoorLogoNavController(cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ERC outdoor GPS-checkpoint live loop.")
    parser.add_argument("--goal-lat", type=float, default=None, help="Single goal latitude.")
    parser.add_argument("--goal-lon", type=float, default=None, help="Single goal longitude.")
    parser.add_argument(
        "--checkpoints",
        type=float,
        nargs="*",
        default=None,
        help="Flattened checkpoint list: lat1 lon1 lat2 lon2 ...",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=Path,
        default=None,
        help="JSON file containing mission checkpoints.",
    )
    parser.add_argument(
        "--mission",
        action="store_true",
        help="Fetch checkpoints from SDK /start-mission and auto-report /checkpoint-reached on arrival.",
    )
    parser.add_argument(
        "--controller",
        choices=("gps", "logonav"),
        default="logonav",
        help="Outdoor local controller. Defaults to the visual LogoNav policy for the plain competition command.",
    )
    parser.add_argument("--tick-hz", type=float, default=3.0, help="Loop frequency in Hz.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional max loop iterations.")
    parser.add_argument("--sdk-url", default="http://localhost:8000", help="EarthRover SDK base URL.")
    parser.add_argument("--sdk-timeout", type=float, default=10.0, help="SDK request timeout in seconds.")
    parser.add_argument("--send-control", action="store_true", help="Actually send commands to the robot.")
    parser.add_argument("--print-json", action="store_true", help="Print each loop state as JSON.")
    parser.add_argument("--goal-radius-m", type=float, default=8.0, help="Distance threshold for checkpoint reach.")
    parser.add_argument("--intermediate-goal-radius-m", type=float, default=3.0, help="Maximum distance threshold for routed intermediate waypoint updates. The runtime further tightens this dynamically from actual waypoint spacing so it does not skip ahead too early.")
    parser.add_argument("--checkpoint-confirm-ticks", type=int, default=1, help="How many consecutive ticks inside radius are required.")
    parser.add_argument("--max-linear", type=float, default=0.38, help="Controller max linear speed.")
    parser.add_argument("--min-linear", type=float, default=0.08, help="Controller min forward speed.")
    parser.add_argument("--nominal-linear", type=float, default=0.33, help="Nominal cruise speed.")
    parser.add_argument("--max-angular", type=float, default=0.45, help="Controller max angular speed.")
    parser.add_argument("--angular-gain", type=float, default=0.4, help="Proportional gain for steering.")
    parser.add_argument("--in-place-turn-threshold-deg", type=float, default=90.0, help="Stop forward motion and turn in place above this bearing error.")
    parser.add_argument("--in-place-turn-exit-deg", type=float, default=60.0, help="Exit turn-in-place mode when bearing error drops below this (hysteresis).")
    parser.add_argument("--heading-fusion", action="store_true", help="Blend compass heading with windowed course-over-ground after stable motion is detected.")
    parser.add_argument("--telemetry-freeze-ticks", type=int, default=4, help="Count telemetry as potentially frozen after this many repeated timestamps.")
    parser.add_argument("--telemetry-freeze-timeout-s", type=float, default=4.0, help="Actually stop only if telemetry has been frozen for at least this many wall-clock seconds.")
    parser.add_argument("--stuck-window-ticks", type=int, default=15, help="Window size for no-progress detection (counts only fresh telemetry ticks).")
    parser.add_argument("--stuck-min-displacement-m", type=float, default=0.30, help="Minimum displacement expected over the stuck window when commanding forward motion.")
    parser.add_argument("--stuck-progress-epsilon-m", type=float, default=0.20, help="Minimum reduction in distance-to-goal expected over the stuck window when commanding forward motion.")
    parser.add_argument("--logonav-stuck-progress-epsilon-m", type=float, default=-0.05, help="LogoNav-specific progress threshold for stuck recovery. Raise toward 0.0 to recover sooner from curb / wall traps.")
    parser.add_argument("--stuck-command-linear", type=float, default=0.18, help="Forward command threshold for no-progress detection.")
    parser.add_argument("--logonav-min-effective-linear", type=float, default=0.10, help="Minimum effective forward speed to maintain when LogoNav is trying to move and no safety layer is actively slowing it.")
    parser.add_argument("--logonav-align-turn-threshold-deg", type=float, default=35.0, help="LogoNav: above this bearing error, prioritize turning toward the waypoint before driving forward.")
    parser.add_argument("--logonav-align-turn-exit-deg", type=float, default=18.0, help="LogoNav: exit the turn-priority mode once bearing error drops below this.")
    parser.add_argument("--logonav-align-max-linear", type=float, default=0.04, help="LogoNav: maximum forward speed while turn-priority mode is active.")
    parser.add_argument("--logonav-align-min-angular", type=float, default=0.18, help="LogoNav: minimum angular command while turn-priority mode is active.")
    parser.add_argument("--logonav-align-distance-m", type=float, default=6.0, help="LogoNav: only force turn-priority below this waypoint distance unless bearing error is very large.")
    parser.add_argument("--logonav-align-extreme-deg", type=float, default=60.0, help="LogoNav: always allow turn-priority above this bearing error, even for farther waypoints.")
    parser.add_argument("--osm-prune-behind-waypoints", action="store_true", help="Drop initial routed waypoints that are very close and behind the rover after OSM re-routing.")
    parser.add_argument("--osm-prune-behind-distance-m", type=float, default=18.0, help="Maximum distance for pruning a behind-the-rover initial waypoint after OSM routing or waypoint handoff.")
    parser.add_argument("--osm-prune-behind-bearing-deg", type=float, default=100.0, help="Prune initial routed waypoints if they lie farther than this angle behind the rover heading.")
    parser.add_argument("--recovery-reverse-linear", type=float, default=-0.20, help="Reverse command during simple stuck recovery.")
    parser.add_argument("--recovery-turn-angular", type=float, default=0.85, help="Turn command during simple stuck recovery.")
    parser.add_argument("--recovery-reverse-ticks", type=int, default=5, help="How many ticks to reverse during stuck recovery.")
    parser.add_argument("--recovery-turn-ticks", type=int, default=8, help="How many ticks to turn during stuck recovery.")
    parser.add_argument("--depth-safety", action="store_true", help="Enable depth-based obstacle avoidance (disabled by default — camera geometry causes false stops on this platform).")
    parser.add_argument("--no-depth-safety", action="store_true", help=argparse.SUPPRESS)  # legacy alias
    parser.add_argument("--depth-model-size", choices=("small", "base", "large"), default="small", help="Depth model size.")
    parser.add_argument("--depth-every-n", type=int, default=2, help="Run depth inference every N ticks.")
    parser.add_argument("--depth-slow-m", type=float, default=0.8, help="Slow down below this forward clearance.")
    parser.add_argument("--depth-stop-m", type=float, default=0.4, help="Stop below this forward clearance.")
    parser.add_argument("--traversability", action="store_true", help="Enable middle-band traversability layer (obstacle detection at tree-trunk / wall height).")
    parser.add_argument("--trav-obstacle-m", type=float, default=1.5, help="Bins below this clearance are treated as blocked by the traversability layer.")
    parser.add_argument("--trav-stop-m", type=float, default=0.60, help="Traversability: stop below this forward clearance.")
    parser.add_argument("--trav-slow-m", type=float, default=1.20, help="Traversability: slow below this forward clearance.")
    parser.add_argument("--trav-memory-frames", type=int, default=4, help="Traversability obstacle memory: use min-pool over last N depth frames.")
    parser.add_argument("--semantics", action="store_true", help="Enable semantic scene-understanding soft bias (people / vegetation corridor checks).")
    parser.add_argument("--semantic-model-profile", choices=("ade20k", "cityscapes", "mapillary"), default=None, help="Optional semantic model preset. cityscapes is stronger for sidewalk/road structure; mapillary is the heaviest street-scene option.")
    parser.add_argument("--semantics-model-id", default="nvidia/segformer-b0-finetuned-ade-512-512", help="Semantic segmentation model id. Supports plug-and-play Hugging Face semantic/universal segmentation backends.")
    parser.add_argument("--semantics-device", choices=("cpu", "cuda", "auto"), default="cpu", help="Device for semantic segmentation inference.")
    parser.add_argument("--semantics-every-n", type=int, default=3, help="Run semantic inference every N ticks.")
    parser.add_argument("--vfh-bins", type=int, default=16, help="Number of polar clearance bins.")
    parser.add_argument("--vfh-fov-deg", type=float, default=90.0, help="Horizontal FOV for VFH clearance bins.")
    parser.add_argument("--vfh-blocked-distance-m", type=float, default=0.8, help="Bins below this clearance are treated as blocked.")
    parser.add_argument("--osm-route", action="store_true", help="Expand mission legs into pedestrian waypoints using OSM at startup.")
    parser.add_argument("--sidewalk-strict", action="store_true", help="Enforce strict sidewalk-first outdoor behavior: pedestrian-only OSM routing, no straight-line fallback, tighter route corridor, and stronger semantic sidewalk checks.")
    parser.add_argument("--osm-no-fallback", action="store_true", help="Abort startup if any OSM leg falls back to a straight-line route.")
    parser.add_argument("--osm-buffer-m", type=float, default=300.0, help="Bounding-box padding for OSM queries.")
    parser.add_argument("--osm-query-timeout-s", type=int, default=25, help="Overpass server-side timeout in seconds.")
    parser.add_argument("--osm-request-retries", type=int, default=2, help="Number of Overpass request attempts before fallback.")
    parser.add_argument("--osm-max-segment-m", type=float, default=20.0, help="Max gap between routed waypoints after densification.")
    parser.add_argument("--osm-min-waypoint-spacing-m", type=float, default=8.0, help="Minimum spacing between retained routed waypoints.")
    parser.add_argument("--osm-max-snap-distance-m", type=float, default=60.0, help="Max allowed snap distance to the OSM graph.")
    parser.add_argument("--logonav-weights", type=Path, default=REPO_ROOT / "mbra_repo" / "deployment" / "model_weights" / "logonav.pth", help="Path to LogoNav weights.")
    parser.add_argument("--logonav-config", type=Path, default=REPO_ROOT / "mbra_repo" / "train" / "config" / "LogoNav.yaml", help="Path to LogoNav config.")
    parser.add_argument("--logonav-device", choices=("cpu", "cuda", "auto"), default="cpu", help="Device for LogoNav inference.")
    parser.add_argument("--logonav-max-linear", type=float, default=0.30, help="LogoNav max linear speed cap.")
    parser.add_argument("--logonav-max-angular", type=float, default=0.30, help="LogoNav max angular speed cap.")
    # --- Ultra marathon safety ---
    parser.add_argument("--ultra-marathon", action="store_true", help="Enable all marathon safety features: IMU anti-flip, conservative recovery, speed caps, health gate, leg pauses.")
    parser.add_argument("--imu-safety", action="store_true", help="Enable IMU-based anti-flip monitoring.")
    parser.add_argument("--max-tilt-deg", type=float, default=40.0, help="IMU: emergency stop if tilt exceeds this angle.")
    parser.add_argument("--max-gyro-dps", type=float, default=150.0, help="IMU: emergency stop if roll/pitch rate exceeds this (deg/s).")
    parser.add_argument("--health-gate", action="store_true", help="Run health checks before starting navigation and between legs.")
    parser.add_argument("--leg-pause", action="store_true", help="Pause for operator confirmation after each mission checkpoint.")
    parser.add_argument("--camera-watchdog-ticks", type=int, default=10, help="Stop after this many consecutive ticks without a camera frame.")
    parser.add_argument("--no-reverse", action="store_true", help="Disable all reverse maneuvers in stuck recovery.")
    parser.add_argument("--max-recovery-attempts", type=int, default=0, help="Halt for operator after this many stuck recoveries per waypoint (0=unlimited).")
    parser.add_argument("--route-corridor-guard", action="store_true", help="Stop if GPS drifts too far away from the active routed path corridor.")
    parser.add_argument("--route-corridor-stop-m", type=float, default=10.0, help="Maximum lateral deviation from the active routed corridor before stopping.")
    parser.add_argument("--route-corridor-confirm-ticks", type=int, default=3, help="Require this many consecutive off-corridor ticks before stopping.")
    parser.add_argument("--battery-warn-pct", type=float, default=30.0, help="Warn when battery falls below this percentage during the run.")
    parser.add_argument("--battery-stop-pct", type=float, default=None, help="Emergency stop if battery falls to or below this percentage.")
    parser.add_argument("--night-safe", action="store_true", help="Enable a night-time outdoor safety profile: lamp on, tighter speed caps, vision safety gate, and semantic pedestrian stops.")
    parser.add_argument("--lamp-on", action="store_true", help="Keep the rover lamp on while this runtime is active.")
    parser.add_argument("--vision-safety", action="store_true", help="Enable an image-quality safety gate for dark / glare / low-detail night frames.")
    parser.add_argument("--vision-min-brightness", type=float, default=42.0, help="Vision safety: minimum mean grayscale brightness (0-255).")
    parser.add_argument("--vision-max-dark-fraction", type=float, default=0.65, help="Vision safety: maximum fraction of dark pixels before visibility is considered unsafe.")
    parser.add_argument("--vision-max-glare-fraction", type=float, default=0.12, help="Vision safety: maximum fraction of saturated bright pixels before glare is considered unsafe.")
    parser.add_argument("--vision-min-texture", type=float, default=8.0, help="Vision safety: minimum gradient-texture score before a frame is considered too low-detail.")
    parser.add_argument("--vision-confirm-ticks", type=int, default=3, help="Vision safety: require this many consecutive bad frames before stopping.")
    parser.add_argument("--semantic-hard-stop", action="store_true", help="Enable a hard stop when the semantic model sees a person or animal in the center corridor.")
    parser.add_argument("--semantic-yield", action="store_true", help="Slow aggressively and yield when semantic risk indicates people, animals, or ambiguous sidewalk occupancy ahead.")
    parser.add_argument("--semantic-yield-risk", type=float, default=0.25, help="Semantic yield activation risk threshold.")
    parser.add_argument("--semantic-yield-max-linear", type=float, default=0.08, help="Maximum linear speed while semantic yield mode is active.")
    parser.add_argument("--semantic-stop-risk", type=float, default=0.55, help="Semantic hard stop risk threshold.")
    parser.add_argument("--semantic-stop-confirm-ticks", type=int, default=2, help="Require this many consecutive semantic hazard ticks before stopping.")
    parser.add_argument("--semantic-sidewalk-stop", action="store_true", help="Stop when the center corridor looks road-dominant and not sidewalk-like for several ticks.")
    parser.add_argument("--semantic-road-dominance", type=float, default=0.55, help="Semantic sidewalk stop: road fraction threshold in the center corridor.")
    parser.add_argument("--semantic-sidewalk-min", type=float, default=0.05, help="Semantic sidewalk stop: minimum sidewalk+path fraction expected in the center corridor.")
    parser.add_argument("--gps-safety", action="store_true", help="Enable GPS signal/fix safety gating and implausible-jump stops.")
    parser.add_argument("--gps-min-signal", type=int, default=2, help="Minimum GPS signal level required before continuing.")
    parser.add_argument("--cell-min-signal", type=int, default=1, help="Minimum cellular signal level required before continuing.")
    parser.add_argument("--gps-safety-confirm-ticks", type=int, default=3, help="Require this many consecutive bad GPS signal/fix ticks before stopping.")
    parser.add_argument("--gps-jump-stop-m", type=float, default=12.0, help="Stop if raw GPS position jumps by more than this many meters in a single telemetry step.")
    parser.add_argument("--nav-ready-gate", action="store_true", help="Require several consecutive healthy ticks before allowing motion after startup, hard stops, or leg transitions.")
    parser.add_argument("--nav-ready-confirm-ticks", type=int, default=3, help="Navigation readiness gate: consecutive healthy ticks required before motion is allowed.")
    parser.add_argument("--report-interventions", action="store_true", help="Report hard safety stops to the SDK interventions endpoints.")
    parser.add_argument("--operator-confirm-hard-stop", action="store_true", help="On hard safety stops, wait for operator ENTER before resuming and optionally report intervention end.")
    return parser.parse_args()


def run_health_gate(rover: EarthRoverInterface, label: str = "navigation", gps_min_signal: int = 2) -> bool:
    """Pre-leg health check: battery, GPS quality, telemetry freshness, and camera."""
    print(f"\n{'=' * 60}")
    print(f"  HEALTH GATE — {label}")
    print(f"{'=' * 60}")
    issues: list[str] = []

    data = rover.get_data(use_cache=False)
    if data is None:
        print("  [FAIL] No telemetry data")
        return False

    battery = data.get("battery")
    if battery is not None and int(battery) >= 20:
        print(f"  Battery:  {battery}%")
    elif battery is not None:
        print(f"  Battery:  {battery}% — LOW")
        issues.append(f"battery={battery}%")
    else:
        print("  Battery:  unknown")
        issues.append("battery_unknown")

    try:
        lat = float(data.get("latitude", float("nan")))
        lon = float(data.get("longitude", float("nan")))
        if not (math.isfinite(lat) and math.isfinite(lon) and lat != 0.0):
            raise ValueError
        print(f"  GPS:      ({lat:.6f}, {lon:.6f})")
    except (TypeError, ValueError):
        print("  GPS:      NO FIX")
        issues.append("gps_no_fix")

    signal = data.get("signal_level")
    gps_sig = data.get("gps_signal")
    print(f"  Signal:   cell={signal}/5  gps={gps_sig}")
    try:
        if gps_sig is None or int(gps_sig) < int(gps_min_signal):
            issues.append(f"weak_gps_signal={gps_sig}")
    except Exception:
        issues.append("gps_signal_invalid")

    _ts_samples: list[float] = []
    for _ in range(2):
        _sample = rover.get_data(use_cache=False)
        if _sample is None:
            issues.append("telemetry_missing")
            break
        try:
            _ts_samples.append(float(_sample.get("timestamp", 0.0)))
        except Exception:
            issues.append("timestamp_invalid")
            break
        time.sleep(0.25)
    if len(_ts_samples) >= 2 and not any(_ts_samples[idx] > _ts_samples[idx - 1] for idx in range(1, len(_ts_samples))):
        issues.append("timestamp_not_advancing")

    frame = rover.get_camera_frame()
    if frame is not None:
        print(f"  Camera:   OK {frame.shape}")
    else:
        print("  Camera:   NO FRAME")
        issues.append("camera_fail")

    if issues:
        print(f"  RESULT:   FAILED — {', '.join(issues)}")
    else:
        print("  RESULT:   ALL CHECKS PASSED")
    print(f"{'=' * 60}\n")
    return len(issues) == 0


def main() -> int:
    args = parse_args()
    if args.tick_hz <= 0:
        raise SystemExit("--tick-hz must be > 0")
    if args.depth_every_n <= 0:
        raise SystemExit("--depth-every-n must be > 0")
    if args.checkpoint_confirm_ticks <= 0:
        raise SystemExit("--checkpoint-confirm-ticks must be > 0")
    if args.semantics_every_n <= 0:
        raise SystemExit("--semantics-every-n must be > 0")
    if args.telemetry_freeze_timeout_s <= 0:
        raise SystemExit("--telemetry-freeze-timeout-s must be > 0")
    if args.route_corridor_confirm_ticks <= 0:
        raise SystemExit("--route-corridor-confirm-ticks must be > 0")
    if args.route_corridor_stop_m <= 0:
        raise SystemExit("--route-corridor-stop-m must be > 0")
    if args.vision_confirm_ticks <= 0:
        raise SystemExit("--vision-confirm-ticks must be > 0")
    if args.semantic_stop_confirm_ticks <= 0:
        raise SystemExit("--semantic-stop-confirm-ticks must be > 0")
    if args.gps_safety_confirm_ticks <= 0:
        raise SystemExit("--gps-safety-confirm-ticks must be > 0")
    if args.gps_jump_stop_m <= 0:
        raise SystemExit("--gps-jump-stop-m must be > 0")
    if args.nav_ready_confirm_ticks <= 0:
        raise SystemExit("--nav-ready-confirm-ticks must be > 0")
    if args.sidewalk_strict and not args.osm_route:
        args.osm_route = True
    if args.semantic_model_profile == "ade20k":
        args.semantics_model_id = "nvidia/segformer-b0-finetuned-ade-512-512"
    elif args.semantic_model_profile == "cityscapes":
        args.semantics_model_id = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    elif args.semantic_model_profile == "mapillary":
        args.semantics_model_id = "facebook/mask2former-swin-large-mapillary-vistas-panoptic"

    # --- Ultra marathon safe defaults ---
    if args.ultra_marathon:
        args.imu_safety = True
        args.health_gate = True
        args.leg_pause = True
        args.no_reverse = True
        args.logonav_max_linear = min(args.logonav_max_linear, 0.24)
        args.logonav_max_angular = min(args.logonav_max_angular, 0.32)
        args.max_linear = min(args.max_linear, 0.24)
        args.max_angular = min(args.max_angular, 0.32)
        args.recovery_turn_angular = min(args.recovery_turn_angular, 0.30)
        args.recovery_turn_ticks = min(args.recovery_turn_ticks, 5)
        args.stuck_window_ticks = max(args.stuck_window_ticks, 25)
        if args.max_recovery_attempts == 0:
            args.max_recovery_attempts = 3
        args.camera_watchdog_ticks = min(args.camera_watchdog_ticks, 10)
        args.route_corridor_guard = True
        args.nav_ready_gate = True
        args.stuck_command_linear = min(args.stuck_command_linear, 0.08)
        args.osm_prune_behind_waypoints = True
        args.osm_prune_behind_distance_m = max(args.osm_prune_behind_distance_m, 18.0)
    if args.sidewalk_strict:
        args.osm_route = True
        args.osm_no_fallback = True
        args.route_corridor_guard = True
        args.route_corridor_stop_m = min(args.route_corridor_stop_m, 3.0)
        args.route_corridor_confirm_ticks = min(args.route_corridor_confirm_ticks, 2)
        args.gps_safety = True
        args.nav_ready_gate = True
        args.traversability = True
        args.semantics = True
        args.semantic_hard_stop = True
        args.semantic_yield = True
        args.semantic_sidewalk_stop = True
        args.semantic_stop_confirm_ticks = min(args.semantic_stop_confirm_ticks, 2)
        args.semantic_road_dominance = min(args.semantic_road_dominance, 0.45)
        args.semantic_sidewalk_min = max(args.semantic_sidewalk_min, 0.12)
        args.max_linear = min(args.max_linear, 0.18)
        args.logonav_max_linear = min(args.logonav_max_linear, 0.18)
        args.logonav_min_effective_linear = min(args.logonav_max_linear, max(args.logonav_min_effective_linear, 0.11))
        args.max_angular = min(args.max_angular, 0.20)
        args.logonav_max_angular = min(args.logonav_max_angular, 0.20)
        args.semantics_every_n = min(args.semantics_every_n, 1)
        args.logonav_device = "auto" if args.logonav_device == "cpu" else args.logonav_device
        args.semantics_device = "auto" if args.semantics_device == "cpu" else args.semantics_device
        if args.semantics_model_id == "nvidia/segformer-b0-finetuned-ade-512-512":
            args.semantics_model_id = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    if args.night_safe:
        args.lamp_on = True
        args.vision_safety = True
        args.gps_safety = True
        args.imu_safety = True
        args.route_corridor_guard = True
        args.nav_ready_gate = True
        args.traversability = True
        args.tick_hz = min(args.tick_hz, 2.0)
        args.depth_every_n = 1
        args.logonav_max_linear = min(args.logonav_max_linear, 0.22)
        args.logonav_max_angular = min(args.logonav_max_angular, 0.28)
        args.max_linear = min(args.max_linear, 0.24)
        args.max_angular = min(args.max_angular, 0.28)
        args.logonav_min_effective_linear = min(args.logonav_max_linear, max(args.logonav_min_effective_linear, 0.12))
        args.trav_obstacle_m = max(args.trav_obstacle_m, 1.8)
        args.trav_stop_m = max(args.trav_stop_m, 0.80)
        args.trav_slow_m = max(args.trav_slow_m, 1.50)
        args.route_corridor_stop_m = min(args.route_corridor_stop_m, 5.0)
        args.camera_watchdog_ticks = min(args.camera_watchdog_ticks, 6)
        args.battery_warn_pct = max(args.battery_warn_pct, 35.0)
        args.stuck_command_linear = min(args.stuck_command_linear, 0.08)
        if args.logonav_device == "cpu":
            args.logonav_device = "auto"
    if args.no_reverse:
        args.recovery_reverse_ticks = 0
        args.recovery_reverse_linear = 0.0

    rover = EarthRoverInterface(base_url=args.sdk_url, timeout=args.sdk_timeout)

    # When MISSION_SLUG is set the SDK gates /data on /start-mission having been called first
    # (main.py:need_start_mission).  So in --mission mode we must call start_mission() before
    # connect(), which calls /data internally.
    initial_reached_count = 0
    if args.mission:
        sdk_checkpoints = rover.start_mission()
        if sdk_checkpoints is None:
            raise SystemExit("Failed to start mission: /start-mission returned no data.")
        if not sdk_checkpoints:
            raise SystemExit("Mission started but /start-mission returned an empty checkpoint list.")
        total_mission_checkpoint_count = len(sdk_checkpoints)

        # Resume support: skip checkpoints the server already confirmed
        status = rover.get_checkpoints_list()
        latest_scanned = int((status or {}).get("latest_scanned_checkpoint", 0))
        initial_reached_count = max(0, min(latest_scanned, total_mission_checkpoint_count))
        if latest_scanned > 0:
            remaining = [cp for cp in sdk_checkpoints if cp["sequence"] > latest_scanned]
            print(f"[mission] resuming — skipping {latest_scanned} already-confirmed checkpoint(s)")
            sdk_checkpoints = remaining if remaining else sdk_checkpoints

        mission_checkpoints = [
            (float(cp["latitude"]), float(cp["longitude"])) for cp in sdk_checkpoints
        ]
        print(f"[mission] {len(mission_checkpoints)} checkpoint(s) remaining:")
        for cp in sdk_checkpoints:
            print(f"  seq={cp['sequence']}  ({cp['latitude']}, {cp['longitude']})")
        # Wait for telemetry to become live before starting autonomy.
        # /start-mission starts the browser session; it can take a few seconds before
        # /data returns a finite pose with an advancing timestamp.  Entering the nav
        # loop before that causes the first few ticks to navigate on stale/zero data.
        print("[mission] waiting for live telemetry (finite pose + advancing timestamp)...")
        rover.connected = True  # allows get_data() to proceed without connect()
        _wait_max_s = 20.0
        _wait_start = time.time()
        _last_ts: float | None = None
        _telemetry_ready = False
        while time.time() - _wait_start < _wait_max_s:
            _d = rover.get_data(use_cache=False)
            if _d is not None:
                try:
                    _lat = float(_d.get("latitude", float("nan")))
                    _lon = float(_d.get("longitude", float("nan")))
                    _ts = float(_d.get("timestamp", 0.0))
                    if (
                        math.isfinite(_lat) and math.isfinite(_lon)
                        and _lat != 0.0 and _lon != 0.0
                        and _last_ts is not None and _ts > _last_ts
                    ):
                        _telemetry_ready = True
                        print(f"[mission] telemetry live — GPS=({_lat:.6f},{_lon:.6f}) ts={_ts:.3f}")
                        break
                    _last_ts = _ts
                except Exception:
                    pass
            time.sleep(0.5)
        if not _telemetry_ready:
            raise SystemExit("Telemetry did not become live within 20 seconds after /start-mission.")

        # If the rover is already sitting on the first remaining checkpoint when the
        # mission starts, consume it immediately instead of trying to navigate back to it.
        _startup_latlon = (_lat, _lon) if _telemetry_ready else None
        if _startup_latlon is not None and sdk_checkpoints:
            while sdk_checkpoints:
                _cp0 = sdk_checkpoints[0]
                _cp_latlon = (float(_cp0["latitude"]), float(_cp0["longitude"]))
                _d0 = utm.from_latlon(_startup_latlon[0], _startup_latlon[1])
                _d1 = utm.from_latlon(_cp_latlon[0], _cp_latlon[1])
                _dist0 = math.hypot(float(_d1[0]) - float(_d0[0]), float(_d1[1]) - float(_d0[1]))
                if _dist0 > args.goal_radius_m:
                    break
                ok, info = rover.checkpoint_reached()
                if not ok:
                    print(f"[mission] startup checkpoint auto-claim rejected for seq={_cp0.get('sequence')}: {info}")
                    break
                initial_reached_count = min(initial_reached_count + 1, total_mission_checkpoint_count)
                print(f"[mission] startup auto-claimed checkpoint seq={_cp0.get('sequence')} dist={_dist0:.1f}m next_seq={info}")
                sdk_checkpoints = sdk_checkpoints[1:]

    else:
        if not rover.connect():
            raise SystemExit("Failed to connect to SDK.")
        mission_checkpoints = build_checkpoint_list(args)
        total_mission_checkpoint_count = len(mission_checkpoints)

    mission_checkpoint_count = total_mission_checkpoint_count
    legwise_osm_routing = bool(args.mission and args.osm_route)

    startup_latlon: LatLon | None = None
    startup_data = rover.get_data(use_cache=False)
    if startup_data is not None:
        try:
            start_lat = float(startup_data.get("latitude"))
            start_lon = float(startup_data.get("longitude"))
            if math.isfinite(start_lat) and math.isfinite(start_lon):
                startup_latlon = (start_lat, start_lon)
        except Exception:
            startup_latlon = None

    osm_config = build_osm_config(args) if args.osm_route else None

    def build_active_navigation(start_latlon_for_leg: LatLon | None, checkpoints_for_leg: list[LatLon], leg_label: str, *, current_heading_rad_for_prune: float | None = None) -> tuple[list[TargetItem], list[dict[str, Any]], list[tuple[float, float, int, str]], list[tuple[float, float]] | None]:
        if args.osm_route:
            print(f"[routing] building {leg_label} via OSM for {len(checkpoints_for_leg)} checkpoint(s)...")
        nav_targets, nav_debug = build_navigation_targets(
            checkpoints_for_leg,
            start_latlon=start_latlon_for_leg,
            use_osm_route=args.osm_route,
            osm_config=osm_config,
            require_osm_success=args.osm_no_fallback,
        )
        if (
            args.osm_prune_behind_waypoints
            and start_latlon_for_leg is not None
            and current_heading_rad_for_prune is not None
            and len(nav_targets) > 1
        ):
            start_utm_full = utm.from_latlon(start_latlon_for_leg[0], start_latlon_for_leg[1])
            start_xy = (float(start_utm_full[0]), float(start_utm_full[1]))
            _pruned = 0
            while len(nav_targets) > 1:
                item0 = nav_targets[0]
                wp0_full = utm.from_latlon(float(item0["lat"]), float(item0["lon"]))
                wp0 = (float(wp0_full[0]), float(wp0_full[1]))
                dx = wp0[0] - start_xy[0]
                dy = wp0[1] - start_xy[1]
                dist = math.hypot(dx, dy)
                bear = math.atan2(dy, dx)
                err_deg = abs(math.degrees(wrap_angle_rad(bear - current_heading_rad_for_prune)))
                if dist <= args.osm_prune_behind_distance_m and err_deg >= args.osm_prune_behind_bearing_deg and not bool(item0.get("mission_checkpoint", False)):
                    nav_targets.pop(0)
                    _pruned += 1
                    continue
                break
            if _pruned > 0:
                print(f"[routing] pruned {_pruned} behind-waypoint(s) at start of {leg_label}")
        nav_target_utms = [utm.from_latlon(item["lat"], item["lon"]) for item in nav_targets]
        nav_route_polyline_utms: list[tuple[float, float]] | None = None
        if start_latlon_for_leg is not None:
            start_utm_full = utm.from_latlon(start_latlon_for_leg[0], start_latlon_for_leg[1])
            nav_route_polyline_utms = [
                (float(start_utm_full[0]), float(start_utm_full[1])),
                *[(float(item[0]), float(item[1])) for item in nav_target_utms],
            ]
        if args.osm_route:
            print(f"[routing] {leg_label} ready: {len(nav_targets)} target(s)")
        return nav_targets, nav_debug, nav_target_utms, nav_route_polyline_utms

    if legwise_osm_routing:
        first_leg = mission_checkpoints[:1]
        navigation_targets, routing_debug, target_utms, route_polyline_utms = build_active_navigation(
            startup_latlon,
            first_leg,
            f"leg 1/{mission_checkpoint_count}",
            current_heading_rad_for_prune=(compass_deg_to_math_rad(float(startup_data.get("orientation"))) if startup_data is not None and startup_data.get("orientation") is not None else None),
        )
    else:
        navigation_targets, routing_debug, target_utms, route_polyline_utms = build_active_navigation(
            startup_latlon,
            mission_checkpoints,
            "full route",
            current_heading_rad_for_prune=(compass_deg_to_math_rad(float(startup_data.get("orientation"))) if startup_data is not None and startup_data.get("orientation") is not None else None),
        )

    controller = build_controller(args)

    depth_estimator = None
    last_clearance = None
    last_bin_centers = None
    last_trav_result = None
    last_sem_result = None
    last_vis_result = None
    if (args.depth_safety and not args.no_depth_safety) or args.traversability:
        try:
            from depth_estimator import DepthEstimator  # type: ignore
            depth_estimator = DepthEstimator(model_size=args.depth_model_size)
            print("Depth estimator: LOADED")
        except Exception as exc:
            raise SystemExit(f"Depth estimator failed to initialize: {exc}")

    traversability = None
    if args.traversability:
        try:
            from outdoor_traversability import OutdoorTraversability, OutdoorTraversabilityConfig  # type: ignore
            traversability = OutdoorTraversability(OutdoorTraversabilityConfig(
                num_bins=args.vfh_bins,
                fov_horizontal_deg=args.vfh_fov_deg,
                obstacle_distance_m=args.trav_obstacle_m,
                stop_distance_m=args.trav_stop_m,
                slow_distance_m=args.trav_slow_m,
                memory_frames=args.trav_memory_frames,
            ))
            print("Traversability layer: ENABLED")
        except Exception as exc:
            raise SystemExit(f"Traversability layer failed to initialize: {exc}")

    semantic_estimator = None
    if args.semantics:
        try:
            from semantic_risk_estimator import SemanticRiskEstimator  # type: ignore
            semantic_estimator = SemanticRiskEstimator(
                model_id=args.semantics_model_id,
                device=args.semantics_device,
            )
            print("Semantic risk layer: ENABLED")
        except Exception as exc:
            raise SystemExit(f"Semantic risk layer failed to initialize: {exc}")

    vision_monitor = None
    if args.vision_safety:
        from vision_safety_monitor import VisionSafetyMonitor, VisionSafetyConfig  # type: ignore
        vision_monitor = VisionSafetyMonitor(VisionSafetyConfig(
            min_brightness=args.vision_min_brightness,
            max_dark_fraction=args.vision_max_dark_fraction,
            max_glare_fraction=args.vision_max_glare_fraction,
            min_texture_score=args.vision_min_texture,
            consecutive_bad_ticks_to_stop=args.vision_confirm_ticks,
        ))
        print("Vision safety gate: ENABLED")

    imu_monitor = None
    if args.imu_safety:
        from imu_safety import IMUSafetyMonitor, IMUSafetyConfig  # type: ignore
        imu_monitor = IMUSafetyMonitor(IMUSafetyConfig(
            max_tilt_deg=args.max_tilt_deg,
            max_pitch_roll_rate_dps=args.max_gyro_dps,
        ))
        print("IMU safety monitor: ENABLED")

    lamp_value = 1 if args.lamp_on else 0
    if args.lamp_on and args.send_control:
        rover.send_control(0.0, 0.0, lamp=lamp_value)
        print("Lamp: ON")

    if args.controller == "logonav" or depth_estimator is not None or semantic_estimator is not None:
        print("[startup] waiting for camera frames...")
        _camera_deadline = time.time() + 15.0
        _camera_ready = False
        while time.time() < _camera_deadline:
            if rover.get_camera_frame() is not None:
                _camera_ready = True
                print("[startup] camera live")
                break
            time.sleep(0.5)
        if not _camera_ready:
            raise SystemExit("Camera did not become live within 15 seconds; aborting startup.")

    dry_run = not args.send_control

    intervention_active = False

    def safe_stop() -> bool:
        if dry_run:
            return False
        return rover.send_control(0.0, 0.0, lamp=lamp_value)

    def report_intervention_start(reason: str) -> None:
        nonlocal intervention_active
        if intervention_active or not args.report_interventions or not args.mission:
            return
        ok, info = rover.start_intervention()
        intervention_active = bool(ok)
        _msg = info if ok else f"failed:{info}"
        print(f"[intervention] start reason={reason} result={_msg}")

    def report_intervention_end(reason: str) -> None:
        nonlocal intervention_active
        if not intervention_active or not args.report_interventions or not args.mission:
            return
        ok, info = rover.end_intervention()
        _msg = info if ok else f"failed:{info}"
        print(f"[intervention] end reason={reason} result={_msg}")
        intervention_active = False

    def reset_after_hard_stop() -> None:
        nonlocal frozen_telemetry_ticks, route_corridor_violation_ticks, semantic_stop_ticks
        nonlocal sidewalk_stop_ticks, gps_safety_ticks, recovery_ticks_remaining, recovery_phase
        nonlocal recovery_attempt_count, _camera_fail_count, previous_raw_current_utm, previous_raw_timestamp
        nonlocal _smooth_linear, _smooth_angular, hard_stop_cooldown_ticks, last_vis_result, last_sem_result
        nonlocal nav_ready_ticks, nav_ready_passed, _camera_hard_stop_latched, logonav_align_mode
        frozen_telemetry_ticks = 0
        route_corridor_violation_ticks = 0
        semantic_stop_ticks = 0
        sidewalk_stop_ticks = 0
        gps_safety_ticks = 0
        recovery_ticks_remaining = 0
        recovery_phase = ""
        recovery_attempt_count = 0
        _camera_fail_count = 0
        previous_raw_current_utm = None
        previous_raw_timestamp = None
        _smooth_linear = 0.0
        _smooth_angular = 0.0
        recent_positions.clear()
        recent_distances.clear()
        recent_forward_flags.clear()
        fast_block_positions.clear()
        _utm_heading_window.clear()
        last_vis_result = None
        last_sem_result = None
        if imu_monitor is not None:
            imu_monitor.reset()
        if vision_monitor is not None and hasattr(vision_monitor, "reset"):
            vision_monitor.reset()
        if hasattr(controller, "reset"):
            controller.reset()
        nav_ready_ticks = 0
        nav_ready_passed = not args.nav_ready_gate
        logonav_align_mode = False
        _camera_hard_stop_latched = False
        hard_stop_cooldown_ticks = 2

    def operator_acknowledge(reason: str) -> None:
        if not args.operator_confirm_hard_stop:
            reset_after_hard_stop()
            return
        report_intervention_start(reason)
        input(f"  HARD STOP ({reason}). Press ENTER after operator has verified scene safety...")
        reset_after_hard_stop()
        report_intervention_end(reason)

    def prune_active_behind_waypoints(current_latlon_for_prune: LatLon, heading_rad_for_prune: float | None, *, context: str) -> int:
        nonlocal navigation_targets, target_utms, route_polyline_utms, active_idx
        if (
            not args.osm_prune_behind_waypoints
            or heading_rad_for_prune is None
            or active_idx >= len(navigation_targets)
        ):
            return 0
        start_utm_full = utm.from_latlon(current_latlon_for_prune[0], current_latlon_for_prune[1])
        start_xy = (float(start_utm_full[0]), float(start_utm_full[1]))
        pruned = 0
        while active_idx < len(navigation_targets) - 1:
            item0 = navigation_targets[active_idx]
            if bool(item0.get("mission_checkpoint", False)):
                break
            wp0_full = target_utms[active_idx]
            wp0 = (float(wp0_full[0]), float(wp0_full[1]))
            dx = wp0[0] - start_xy[0]
            dy = wp0[1] - start_xy[1]
            dist = math.hypot(dx, dy)
            bear = math.atan2(dy, dx)
            err_deg = abs(math.degrees(wrap_angle_rad(bear - heading_rad_for_prune)))
            if dist <= args.osm_prune_behind_distance_m and err_deg >= args.osm_prune_behind_bearing_deg:
                navigation_targets.pop(active_idx)
                target_utms.pop(active_idx)
                pruned += 1
                continue
            break
        if pruned > 0:
            if route_polyline_utms is not None:
                route_polyline_utms = [start_xy, *[(float(item[0]), float(item[1])) for item in target_utms[active_idx:]]]
            print(f"[routing] pruned {pruned} behind-waypoint(s) during {context}")
        return pruned

    def reroute_from_current_pose(start_latlon_for_leg: LatLon, *, reason: str) -> bool:
        nonlocal navigation_targets, routing_debug, target_utms, route_polyline_utms, active_idx
        if not args.osm_route:
            return False
        if legwise_osm_routing:
            if reached_count >= mission_checkpoint_count:
                return False
            checkpoints_for_leg = [mission_checkpoints[reached_count]]
            leg_label = f"leg {reached_count + 1}/{mission_checkpoint_count} reroute"
        else:
            checkpoints_for_leg = [(float(item["lat"]), float(item["lon"])) for item in navigation_targets[active_idx:]]
            if not checkpoints_for_leg:
                return False
            leg_label = "reroute remaining route"
        print(f"[routing] re-routing from current pose due to {reason}...")
        navigation_targets, routing_debug, target_utms, route_polyline_utms = build_active_navigation(
            start_latlon_for_leg,
            checkpoints_for_leg,
            leg_label,
            current_heading_rad_for_prune=current_heading_rad,
        )
        active_idx = 0
        prune_active_behind_waypoints(start_latlon_for_leg, current_heading_rad, context=reason)
        reset_after_hard_stop()
        return True

    print("Live outdoor runtime")
    print("=" * 60)
    print(f"Mode: {'DRY RUN' if dry_run else 'SEND CONTROL'}")
    print(f"Controller: {args.controller}")
    print(f"Mission checkpoints: {mission_checkpoint_count}")
    print(f"Navigation targets: {len(navigation_targets)}")
    if args.osm_route:
        fallback_legs = sum(
            1
            for item in routing_debug
            if isinstance(item.get("routing"), dict)
            and item["routing"].get("routing") == "fallback_straight_line"
        )
        print(f"OSM routing: ON (legs={len(routing_debug)}, fallback_legs={fallback_legs})")
    else:
        print("OSM routing: OFF")
    print(f"Goal radius: {args.goal_radius_m:.2f} m")
    print(f"Tick rate: {args.tick_hz:.2f} Hz")
    print(f"Depth safety: {'ON' if args.depth_safety and not args.no_depth_safety else 'OFF'}")
    print(f"Traversability: {'ON' if traversability is not None else 'OFF'}")
    print(f"Semantics: {'ON' if semantic_estimator is not None else 'OFF'}")
    if semantic_estimator is not None:
        print(f"Semantic model: {args.semantics_model_id}")
    print(f"Vision safety: {'ON' if vision_monitor is not None else 'OFF'}")
    print(f"GPS safety: {'ON' if args.gps_safety else 'OFF'}")
    print(f"Navigation readiness gate: {'ON' if args.nav_ready_gate else 'OFF'}")
    print(f"Semantic hard stop: {'ON' if args.semantic_hard_stop else 'OFF'} (experimental)")
    print(f"Semantic yield: {'ON' if args.semantic_yield else 'OFF'}")
    print(f"Semantic sidewalk stop: {'ON' if args.semantic_sidewalk_stop else 'OFF'} (experimental)")
    print(f"Sidewalk strict mode: {'ON' if args.sidewalk_strict else 'OFF'}")
    print(f"Lamp: {'ON' if args.lamp_on else 'OFF'}")
    if args.controller == "logonav":
        _policies = ["LogoNav visual navigation"]
        if traversability is not None:
            _policies.append("depth traversability soft bias")
        if semantic_estimator is not None:
            _policies.append("semantic soft bias")
        print(f"Obstacle policy: {' + '.join(_policies)}")
    elif semantic_estimator is not None and traversability is not None:
        print("Obstacle policy: GPS + traversability soft bias + semantic soft bias")
    elif traversability is not None:
        print("Obstacle policy: GPS + traversability layer (middle-band depth)")
    elif semantic_estimator is not None:
        print("Obstacle policy: GPS + semantic soft bias")
    elif depth_estimator is not None:
        print("Obstacle policy: GPS + depth overlay (bottom-band, legacy)")
    else:
        print("Obstacle policy: GPS only (blind to obstacles)")
    if args.night_safe:
        print("NIGHT SAFE MODE: ON")
    if args.sidewalk_strict:
        print("SIDEWALK STRICT MODE: ON")
    if args.ultra_marathon:
        _um_lin = args.logonav_max_linear if args.controller == "logonav" else args.max_linear
        _um_ang = args.logonav_max_angular if args.controller == "logonav" else args.max_angular
        print(f"ULTRA MARATHON MODE: ON")
        print(f"  Speed caps: linear={_um_lin:.2f} angular={_um_ang:.2f}")
        print(f"  IMU safety: {'ON' if imu_monitor else 'OFF'}")
        print(f"  Health gate: {'ON' if args.health_gate else 'OFF'}")
        print(f"  Leg pause: {'ON' if args.leg_pause else 'OFF'}")
        print(f"  No reverse: {'ON' if args.no_reverse else 'OFF'}")
        print(f"  Max recovery attempts: {args.max_recovery_attempts}")
        print(f"  Camera watchdog: {args.camera_watchdog_ticks} ticks")
        print(f"  Route corridor guard: {'ON' if args.route_corridor_guard and route_polyline_utms is not None else 'OFF'}")
    print("=" * 60)

    if args.controller == "gps" and depth_estimator is None and args.send_control:
        print("[warn] gps without --depth-safety will not avoid obstacles")

    period = 1.0 / args.tick_hz
    iteration = 0
    active_idx = 0
    reached_count = initial_reached_count
    confirm_count = 0
    last_goal_idx = None

    # EMA smoothing state — reduces GPS noise and command oscillation
    _GPS_ALPHA = 0.45       # lower = smoother position, slightly more lag
    _CMD_ALPHA = 0.55       # lower = smoother commands, slightly more lag
    _LINEAR_ALPHA = 0.60    # lower = smoother linear commands
    # Compass EMA disabled — the lag it introduces on steady rotation biases bearing_error
    # consistently in one direction, causing outward spirals. Raw compass works better.
    _smooth_lat: float | None = None
    _smooth_lon: float | None = None
    _smooth_angular: float = 0.0
    _smooth_linear: float = 0.0
    _smooth_heading_rad: float | None = None
    _prev_orientation_deg: float | None = None

    # Optional GPS-motion heading fusion (window-based, uses EMA-smoothed UTM).
    # Leave this off by default; it should only be enabled after field validation
    # on the actual robot and GPS noise profile.
    _MOTION_HEADING_WINDOW = 5        # ticks to accumulate over
    _MOTION_HEADING_WIN_MIN_M = 2.0   # only fuse when net displacement is >> GPS noise (~2m)
    _MOTION_HEADING_WIN_FULL_M = 4.0  # full-weight blend above this
    _utm_heading_window: deque[tuple[float, float]] = deque(maxlen=_MOTION_HEADING_WINDOW)

    last_telemetry_timestamp: float | None = None
    last_fresh_telemetry_wall_time: float | None = None
    frozen_telemetry_ticks = 0
    recent_positions: deque[tuple[float, float]] = deque(maxlen=max(2, args.stuck_window_ticks))
    recent_distances: deque[float] = deque(maxlen=max(2, args.stuck_window_ticks))
    recent_forward_flags: deque[bool] = deque(maxlen=max(2, args.stuck_window_ticks))
    logonav_align_mode = False
    fast_block_positions: deque[tuple[float, float]] = deque(maxlen=3)  # wall-hit fast check
    recovery_ticks_remaining = 0
    recovery_phase = ""
    recovery_turn_sign = 1.0
    recovery_attempt_count = 0
    _camera_fail_count = 0
    route_corridor_violation_ticks = 0
    semantic_stop_ticks = 0
    sidewalk_stop_ticks = 0
    gps_safety_ticks = 0
    hard_stop_cooldown_ticks = 0
    nav_ready_ticks = 0
    nav_ready_passed = not args.nav_ready_gate
    _camera_hard_stop_latched = False
    previous_raw_current_utm = None
    previous_raw_timestamp = None
    last_battery_warn_iteration = -99999

    if args.health_gate:
        if not run_health_gate(rover, "pre-start", gps_min_signal=args.gps_min_signal):
            print("[health-gate] FAILED — fix issues before proceeding.")
            input("  Press ENTER to continue anyway, or Ctrl+C to abort...")

    try:
        while True:
            if args.max_steps is not None and iteration >= args.max_steps:
                break
            if active_idx >= len(navigation_targets):
                break

            loop_start = time.time()
            frame = rover.get_camera_frame()
            if frame is None:
                _camera_fail_count += 1
            else:
                _camera_fail_count = 0
                _camera_hard_stop_latched = False
            if _camera_fail_count >= args.camera_watchdog_ticks and args.controller == "logonav":
                safe_stop()
                if not _camera_hard_stop_latched:
                    print(f"[{iteration:04d}] camera_watchdog_stop missing_frames={_camera_fail_count}")
                    operator_acknowledge("camera_watchdog_stop")
                    _camera_hard_stop_latched = True
                time.sleep(period)
                iteration += 1
                continue

            data = rover.get_data()
            sent = False

            if data is None:
                if not dry_run:
                    safe_stop()
                print(f"[{iteration:04d}] no telemetry; stopping for this iteration")
                time.sleep(period)
                iteration += 1
                continue

            try:
                lat = float(data.get("latitude"))
                lon = float(data.get("longitude"))
                orientation_deg = float(data.get("orientation", 0.0))
                telemetry_timestamp = float(data.get("timestamp", 0.0))
                if not (math.isfinite(lat) and math.isfinite(lon)):
                    raise ValueError("non-finite lat/lon")
                # EMA GPS smoothing — damps noise-driven bearing oscillation
                if _smooth_lat is None:
                    _smooth_lat, _smooth_lon = lat, lon
                else:
                    _smooth_lat = _GPS_ALPHA * lat + (1.0 - _GPS_ALPHA) * _smooth_lat
                    _smooth_lon = _GPS_ALPHA * lon + (1.0 - _GPS_ALPHA) * _smooth_lon
                raw_current_utm_full = utm.from_latlon(lat, lon)
                raw_current_utm = (float(raw_current_utm_full[0]), float(raw_current_utm_full[1]))
                current_utm_full = utm.from_latlon(_smooth_lat, _smooth_lon)
                current_utm = (float(current_utm_full[0]), float(current_utm_full[1]))
                current_heading_rad = compass_deg_to_math_rad(orientation_deg)

                # Use raw compass — EMA lag causes outward spirals on steady rotation.

                motion_heading_weight = 0.0
                if args.heading_fusion:
                    # GPS-motion heading fusion (window-based)
                    # Store EMA-smoothed UTM (not raw) so the window direction already
                    # has per-tick GPS noise suppressed before we accumulate over 5 ticks.
                    _utm_heading_window.append(current_utm)
                    if len(_utm_heading_window) >= 2:
                        _oldest = _utm_heading_window[0]
                        _newest = _utm_heading_window[-1]
                        _mdx = _newest[0] - _oldest[0]
                        _mdy = _newest[1] - _oldest[1]
                        _win_disp = math.hypot(_mdx, _mdy)
                        if _win_disp >= _MOTION_HEADING_WIN_MIN_M:
                            _motion_h = math.atan2(_mdy, _mdx)
                            motion_heading_weight = min(1.0, _win_disp / _MOTION_HEADING_WIN_FULL_M)
                            _hdiff = wrap_angle_rad(_motion_h - current_heading_rad)
                            current_heading_rad = current_heading_rad + motion_heading_weight * _hdiff
            except Exception as exc:
                if not dry_run:
                    safe_stop()
                print(f"[{iteration:04d}] invalid telemetry ({exc}); stopping for this iteration")
                time.sleep(period)
                iteration += 1
                continue

            _now_wall = time.time()
            if hard_stop_cooldown_ticks > 0:
                safe_stop()
                print(f"[{iteration:04d}] hard_stop_cooldown remaining={hard_stop_cooldown_ticks}")
                hard_stop_cooldown_ticks -= 1
                time.sleep(period)
                iteration += 1
                continue
            if last_telemetry_timestamp is not None and abs(telemetry_timestamp - last_telemetry_timestamp) < 1e-6:
                frozen_telemetry_ticks += 1
            else:
                frozen_telemetry_ticks = 0
                last_fresh_telemetry_wall_time = _now_wall
            last_telemetry_timestamp = telemetry_timestamp
            telemetry_stale_s = 0.0 if last_fresh_telemetry_wall_time is None else max(0.0, _now_wall - last_fresh_telemetry_wall_time)

            battery_pct = None
            try:
                _battery_raw = data.get("battery")
                battery_pct = None if _battery_raw is None else float(_battery_raw)
            except Exception:
                battery_pct = None

            if (
                battery_pct is not None
                and battery_pct <= args.battery_warn_pct
                and iteration - last_battery_warn_iteration >= 30
            ):
                print(f"[{iteration:04d}] [warn] low battery {battery_pct:.1f}%")
                last_battery_warn_iteration = iteration

            if args.gps_safety:
                _gps_signal = data.get("gps_signal")
                _cell_signal = data.get("signal_level")
                _gps_bad = False
                _gps_reason = []
                try:
                    if _gps_signal is None or int(_gps_signal) < int(args.gps_min_signal):
                        _gps_bad = True
                        _gps_reason.append(f"gps={_gps_signal}")
                except Exception:
                    _gps_bad = True
                    _gps_reason.append("gps=invalid")
                if previous_raw_current_utm is not None and previous_raw_timestamp is not None:
                    _dt = max(1e-3, telemetry_timestamp - previous_raw_timestamp)
                    _jump_m = math.hypot(raw_current_utm[0] - previous_raw_current_utm[0], raw_current_utm[1] - previous_raw_current_utm[1])
                    if _dt <= 3.0 and _jump_m > args.gps_jump_stop_m:
                        _gps_bad = True
                        _gps_reason.append(f"jump={_jump_m:.1f}m")
                if _gps_bad:
                    gps_safety_ticks += 1
                else:
                    gps_safety_ticks = 0
                if gps_safety_ticks >= args.gps_safety_confirm_ticks:
                    safe_stop()
                    print(
                        f"[{iteration:04d}] gps_safety_stop {' '.join(_gps_reason) if _gps_reason else ''} "
                        f"gps_signal={data.get('gps_signal')} cell={_cell_signal} ticks={gps_safety_ticks}"
                    )
                    operator_acknowledge("gps_safety_stop")
                    gps_safety_ticks = 0
                    previous_raw_current_utm = raw_current_utm
                    previous_raw_timestamp = telemetry_timestamp
                    time.sleep(period)
                    iteration += 1
                    continue
                previous_raw_current_utm = raw_current_utm
                previous_raw_timestamp = telemetry_timestamp

            if (
                args.battery_stop_pct is not None
                and battery_pct is not None
                and battery_pct <= float(args.battery_stop_pct)
            ):
                if not dry_run:
                    safe_stop()
                print(
                    f"[{iteration:04d}] battery_stop battery={battery_pct:.1f}% "
                    f"threshold={float(args.battery_stop_pct):.1f}%"
                )
                time.sleep(period)
                iteration += 1
                continue

            if (
                frozen_telemetry_ticks >= args.telemetry_freeze_ticks
                and telemetry_stale_s >= args.telemetry_freeze_timeout_s
            ):
                if not dry_run:
                    safe_stop()
                print(
                    f"[{iteration:04d}] telemetry_frozen_stop "
                    f"ts={telemetry_timestamp:.3f} frozen_ticks={frozen_telemetry_ticks} stale_s={telemetry_stale_s:.1f}"
                )
                time.sleep(period)
                iteration += 1
                continue

            # --- IMU anti-flip check ---
            if imu_monitor is not None:
                _imu = imu_monitor.update(data)
                if _imu.emergency_stop:
                    safe_stop()
                    print(
                        f"[{iteration:04d}] [IMU EMERGENCY] {_imu.reason} "
                        f"tilt={_imu.tilt_deg:.1f}deg gyro={_imu.pitch_roll_rate_dps:.1f}dps vib={_imu.vibration:.2f}"
                    )
                    if args.ultra_marathon or args.night_safe or args.operator_confirm_hard_stop:
                        print("  Robot may be tipping! Operator intervention required.")
                        operator_acknowledge("imu_emergency")
                    else:
                        reset_after_hard_stop()
                    time.sleep(period)
                    iteration += 1
                    continue

            if vision_monitor is not None and frame is not None:
                last_vis_result = vision_monitor.update(frame)
                if last_vis_result.emergency_stop:
                    safe_stop()
                    print(
                        f"[{iteration:04d}] vision_safety_stop {last_vis_result.reason} "
                        f"brightness={last_vis_result.mean_brightness:.1f} dark={last_vis_result.dark_fraction:.2f} "
                        f"glare={last_vis_result.glare_fraction:.2f} texture={last_vis_result.texture_score:.1f}"
                    )
                    operator_acknowledge("vision_safety_stop")
                    time.sleep(period)
                    iteration += 1
                    continue

            if args.nav_ready_gate and not nav_ready_passed:
                _ready_ok = True
                _ready_reasons: list[str] = []
                if args.controller == "logonav" and frame is None:
                    _ready_ok = False
                    _ready_reasons.append("camera")
                try:
                    _gps_signal = data.get("gps_signal")
                    if _gps_signal is None or int(_gps_signal) < int(args.gps_min_signal):
                        _ready_ok = False
                        _ready_reasons.append(f"gps={_gps_signal}")
                except Exception:
                    _ready_ok = False
                    _ready_reasons.append("gps=invalid")
                if telemetry_stale_s > max(1.0, period * 1.5):
                    _ready_ok = False
                    _ready_reasons.append(f"stale={telemetry_stale_s:.1f}s")
                if _ready_ok:
                    nav_ready_ticks += 1
                else:
                    nav_ready_ticks = 0
                if nav_ready_ticks >= args.nav_ready_confirm_ticks:
                    nav_ready_passed = True
                    print(f"[{iteration:04d}] navigation_ready ticks={nav_ready_ticks}")
                else:
                    safe_stop()
                    _reason_str = ",".join(_ready_reasons) if _ready_reasons else "warming"
                    print(
                        f"[{iteration:04d}] nav_ready_hold ticks={nav_ready_ticks}/{args.nav_ready_confirm_ticks} "
                        f"reason={_reason_str}"
                    )
                    time.sleep(period)
                    iteration += 1
                    continue

            target = navigation_targets[active_idx]
            goal_lat = float(target["lat"])
            goal_lon = float(target["lon"])
            target_utm_full = target_utms[active_idx]
            goal_utm = (float(target_utm_full[0]), float(target_utm_full[1]))
            goal_compass_rad = float(target.get("goal_compass_rad", 0.0))
            active_mission_idx = reached_count if reached_count < mission_checkpoint_count else max(0, mission_checkpoint_count - 1)
            mission_goal_lat = float(mission_checkpoints[active_mission_idx][0])
            mission_goal_lon = float(mission_checkpoints[active_mission_idx][1])
            mission_goal_utm_full = utm.from_latlon(mission_goal_lat, mission_goal_lon)
            mission_goal_utm = (float(mission_goal_utm_full[0]), float(mission_goal_utm_full[1]))
            mission_distance_m = math.hypot(mission_goal_utm[0] - current_utm[0], mission_goal_utm[1] - current_utm[1])
            route_corridor_distance = None
            route_corridor_stop_threshold = args.route_corridor_stop_m
            if args.osm_route:
                route_corridor_stop_threshold = max(route_corridor_stop_threshold, 0.75 * args.osm_min_waypoint_spacing_m)
                if not args.sidewalk_strict:
                    route_corridor_stop_threshold = max(route_corridor_stop_threshold, args.goal_radius_m)
            if args.route_corridor_guard and route_polyline_utms is not None:
                route_corridor_distance = route_corridor_distance_m(current_utm, route_polyline_utms, active_idx)
                if route_corridor_distance is not None and route_corridor_distance > route_corridor_stop_threshold:
                    route_corridor_violation_ticks += 1
                else:
                    route_corridor_violation_ticks = 0
                if route_corridor_violation_ticks >= args.route_corridor_confirm_ticks:
                    if not dry_run:
                        safe_stop()
                    print(
                        f"[{iteration:04d}] route_corridor_stop dist={route_corridor_distance:.1f}m "
                        f"threshold={route_corridor_stop_threshold:.1f}m ticks={route_corridor_violation_ticks}"
                    )
                    _rerouted = False
                    try:
                        _cur_lat = float(data.get("latitude", float("nan")))
                        _cur_lon = float(data.get("longitude", float("nan")))
                        if math.isfinite(_cur_lat) and math.isfinite(_cur_lon) and _cur_lat != 0.0 and _cur_lon != 0.0:
                            _rerouted = reroute_from_current_pose((_cur_lat, _cur_lon), reason="route_corridor_stop")
                    except Exception:
                        _rerouted = False
                    if not _rerouted:
                        operator_acknowledge("route_corridor_stop")
                    time.sleep(period)
                    iteration += 1
                    continue
            else:
                route_corridor_violation_ticks = 0

            if args.controller == "logonav":
                # LogoNav expects a live desired heading in its own compass-radian convention.
                # Updating only on waypoint changes leaves the heading stale as the rover drifts,
                # which can make the policy spin while trying to satisfy an outdated orientation.
                dynamic_goal_heading_math = math.atan2(
                    goal_utm[1] - current_utm[1],
                    goal_utm[0] - current_utm[0],
                )
                goal_compass_rad = math_rad_to_logonav_compass_rad(dynamic_goal_heading_math)
                controller.update_goal(goal_utm, goal_compass_rad)  # type: ignore[attr-defined]
                last_goal_idx = active_idx

            if depth_estimator is not None and frame is not None and iteration % args.depth_every_n == 0:
                try:
                    depth_map = depth_estimator.estimate(frame, target_size=(120, 160))
                    last_clearance, last_bin_centers = depth_estimator.get_polar_clearance(
                        depth_map,
                        num_bins=args.vfh_bins,
                        fov_horizontal=args.vfh_fov_deg,
                    )
                    if traversability is not None:
                        _dx = goal_utm[0] - current_utm[0]
                        _dy = goal_utm[1] - current_utm[1]
                        _goal_bearing_err = wrap_angle_rad(math.atan2(_dy, _dx) - current_heading_rad)
                        last_trav_result = traversability.compute(depth_map, _goal_bearing_err)
                except Exception as exc:
                    print(f"[{iteration:04d}] [warn] depth inference failed: {exc}")

            if semantic_estimator is not None and frame is not None and iteration % args.semantics_every_n == 0:
                try:
                    last_sem_result = semantic_estimator.estimate(frame)
                except Exception as exc:
                    print(f"[{iteration:04d}] [warn] semantic inference failed: {exc}")

            if args.semantic_hard_stop and last_sem_result is not None:
                _alerts = set(last_sem_result.hard_alerts)
                _semantic_stop_active = bool(_alerts & {"person", "animal"}) and last_sem_result.risk_score >= args.semantic_stop_risk
                if _semantic_stop_active:
                    semantic_stop_ticks += 1
                else:
                    semantic_stop_ticks = 0
                if semantic_stop_ticks >= args.semantic_stop_confirm_ticks:
                    safe_stop()
                    print(
                        f"[{iteration:04d}] semantic_hard_stop alerts={sorted(_alerts)} risk={last_sem_result.risk_score:.2f} "
                        f"drivable={last_sem_result.drivable_center:.2f} caution={last_sem_result.caution_center:.2f}"
                    )
                    operator_acknowledge("semantic_hard_stop")
                    time.sleep(period)
                    iteration += 1
                    continue
            else:
                semantic_stop_ticks = 0

            if args.semantic_yield and last_sem_result is not None and recovery_ticks_remaining == 0:
                _yield_active = (
                    last_sem_result.risk_score >= args.semantic_yield_risk
                    or last_sem_result.person_center > 0.0
                    or last_sem_result.animal_center > 0.0
                    or (last_sem_result.road_center >= 0.25 and (last_sem_result.sidewalk_center + last_sem_result.path_center) <= 0.20)
                )
                if _yield_active and command.linear > 0.0:
                    command.linear = min(command.linear, args.semantic_yield_max_linear)
                    if last_sem_result.person_center > 0.0 or last_sem_result.animal_center > 0.0:
                        command.linear = min(command.linear, 0.05)
                    if command.debug is None:
                        command.debug = {}
                    command.debug["semantic_yield"] = True
                    command.reason = "semantic_yield" if command.reason == "mbra_controller" or command.reason == "logonav" or command.reason.startswith("goal") else command.reason

            if args.semantic_sidewalk_stop and last_sem_result is not None:
                _sidewalk_like = last_sem_result.sidewalk_center + last_sem_result.path_center
                _road_like = last_sem_result.road_center
                _unsafe_sidewalk = _road_like >= args.semantic_road_dominance and _sidewalk_like <= args.semantic_sidewalk_min
                if _unsafe_sidewalk:
                    sidewalk_stop_ticks += 1
                else:
                    sidewalk_stop_ticks = 0
                if sidewalk_stop_ticks >= args.semantic_stop_confirm_ticks:
                    safe_stop()
                    print(
                        f"[{iteration:04d}] semantic_sidewalk_stop road={_road_like:.2f} sidewalk={last_sem_result.sidewalk_center:.2f} path={last_sem_result.path_center:.2f}"
                    )
                    operator_acknowledge("semantic_sidewalk_stop")
                    time.sleep(period)
                    iteration += 1
                    continue
            else:
                sidewalk_stop_ticks = 0

            # When traversability is the active safety layer, do not feed the old
            # bottom-band clearance into the GPS controller — traversability handles
            # obstacle steering separately after the command is computed.
            _gps_clearance = last_clearance if (traversability is None and args.depth_safety and not args.no_depth_safety) else None
            _gps_bin_centers = last_bin_centers if _gps_clearance is not None else None
            if args.controller == "gps":
                command = controller.compute_command(  # type: ignore[call-arg]
                    current_utm=current_utm,
                    goal_utm=goal_utm,
                    current_heading_rad=current_heading_rad,
                    clearance=_gps_clearance,
                    bin_centers=_gps_bin_centers,
                )
            else:
                if frame is None:
                    from outdoor_gps_controller import OutdoorControlCommand  # type: ignore
                    command = OutdoorControlCommand(0.0, 0.0, "logonav_missing_frame")
                else:
                    command = controller.compute_command(  # type: ignore[call-arg]
                        frame_rgb=frame,
                        current_utm=current_utm,
                        orientation_deg=orientation_deg,
                    )
                    # Old bottom-crop depth veto: only runs when --depth-safety is on
                    # and --traversability is NOT on (traversability replaces this).
                    if (depth_estimator is not None and last_clearance is not None
                            and command.linear > 0.0
                            and traversability is None and args.depth_safety and not args.no_depth_safety):
                        center = len(last_clearance) // 2
                        window = last_clearance[max(0, center - 1): min(len(last_clearance), center + 2)]
                        if len(window) > 0:
                            forward_clearance = float(min(window))
                            if forward_clearance < args.depth_stop_m:
                                command.linear = 0.0
                                command.angular = 0.0
                                command.reason = f"depth_stop({forward_clearance:.2f}m)"
                            elif forward_clearance < args.depth_slow_m:
                                scale = max(0.3, (forward_clearance - args.depth_stop_m) / max(1e-6, args.depth_slow_m - args.depth_stop_m))
                                command.linear *= scale
                                command.reason = f"depth_slow({forward_clearance:.2f}m)"

            distance_to_goal = float(command.debug.get("distance_to_goal_m", float("nan"))) if command.debug else float("nan")
            if args.controller == "logonav" and recovery_ticks_remaining == 0:
                _bearing_error_rad = float((command.debug or {}).get("bearing_error_rad", 0.0))
                _bearing_error_deg = abs(math.degrees(_bearing_error_rad))
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
                if logonav_align_mode:
                    if _bearing_error_deg <= args.logonav_align_turn_exit_deg or not _align_allowed:
                        logonav_align_mode = False
                elif _align_allowed and _bearing_error_deg >= args.logonav_align_turn_threshold_deg:
                    logonav_align_mode = True

                if logonav_align_mode:
                    _turn_sign = 1.0 if _bearing_error_rad >= 0.0 else -1.0
                    command.angular = _turn_sign * max(abs(command.angular), args.logonav_align_min_angular)
                    command.angular = float(max(-args.logonav_max_angular, min(args.logonav_max_angular, command.angular)))
                    command.linear = min(command.linear, args.logonav_align_max_linear)
                    if _bearing_error_deg >= max(args.logonav_align_turn_threshold_deg + 10.0, 45.0):
                        command.linear = 0.0
                    if command.debug is None:
                        command.debug = {}
                    command.debug["align_turn"] = round(_bearing_error_deg, 1)
                    command.reason = "logonav_align_turn"

            if bool(target.get("mission_checkpoint", False)):
                active_goal_radius_m = args.goal_radius_m
            else:
                _segment_distance_m = float(target.get("segment_distance_m", float("nan")))
                _dynamic_radius_m = float(args.intermediate_goal_radius_m)
                if math.isfinite(_segment_distance_m) and _segment_distance_m > 0.0:
                    _dynamic_radius_m = min(_dynamic_radius_m, max(2.5, 0.35 * _segment_distance_m))
                active_goal_radius_m = min(args.goal_radius_m, _dynamic_radius_m)
            reached_this_tick = math.isfinite(distance_to_goal) and distance_to_goal <= active_goal_radius_m

            if reached_this_tick:
                confirm_count += 1
            else:
                confirm_count = 0

            if confirm_count >= args.checkpoint_confirm_ticks:
                is_mission_cp = bool(target.get("mission_checkpoint", False))
                advance = True

                if is_mission_cp:
                    if args.mission:
                        ok, info = rover.checkpoint_reached()
                        if ok:
                            reached_count += 1
                            print(
                                f"[{iteration:04d}] MISSION CHECKPOINT REACHED "
                                f"{reached_count}/{mission_checkpoint_count} "
                                f"at ({goal_lat:.6f}, {goal_lon:.6f}) "
                                f"— /checkpoint-reached: accepted, next_seq={info}"
                            )
                        else:
                            # Server rejected: stay at this waypoint and retry next tick.
                            # Do NOT advance active_idx — server state must match local state.
                            print(
                                f"[{iteration:04d}] /checkpoint-reached REJECTED "
                                f"at ({goal_lat:.6f}, {goal_lon:.6f}) — {info}; retrying..."
                            )
                            confirm_count = args.checkpoint_confirm_ticks - 1
                            advance = False
                    else:
                        reached_count += 1
                        print(
                            f"[{iteration:04d}] MISSION CHECKPOINT REACHED "
                            f"{reached_count}/{mission_checkpoint_count} "
                            f"at ({goal_lat:.6f}, {goal_lon:.6f})"
                        )
                else:
                    print(
                        f"[{iteration:04d}] intermediate waypoint reached "
                        f"at ({goal_lat:.6f}, {goal_lon:.6f})"
                    )

                if advance:
                    # Marathon: pause between legs for operator confirmation
                    _has_more_after_current = (reached_count < mission_checkpoint_count) if (is_mission_cp and legwise_osm_routing) else (active_idx + 1 < len(navigation_targets))
                    if (is_mission_cp and args.leg_pause
                            and reached_count < mission_checkpoint_count
                            and _has_more_after_current):
                        if not dry_run:
                            safe_stop()
                        print(f"\n{'=' * 60}")
                        print(f"  LEG {reached_count}/{mission_checkpoint_count} COMPLETE")
                        print(f"{'=' * 60}")
                        if args.health_gate:
                            while not run_health_gate(rover, f"pre-leg-{reached_count + 1}", gps_min_signal=args.gps_min_signal):
                                input("  Health gate FAILED. Fix issues, press ENTER to re-check...")
                        input(f"  Press ENTER to start leg {reached_count + 1}...")
                        if hasattr(controller, "reset"):
                            controller.reset()

                    active_idx += 1
                    if active_idx < len(navigation_targets):
                        try:
                            prune_active_behind_waypoints((raw_lat, raw_lon), current_heading_rad, context="waypoint_advance")
                        except Exception:
                            pass
                    confirm_count = 0
                    _smooth_lat, _smooth_lon = None, None  # reset GPS filter for new target
                    _smooth_heading_rad = None              # reset compass EMA for new target
                    _smooth_angular = 0.0
                    _smooth_linear = 0.0
                    _utm_heading_window.clear()  # reset motion heading window
                    recent_positions.clear()
                    recent_distances.clear()
                    recent_forward_flags.clear()
                    fast_block_positions.clear()
                    recovery_ticks_remaining = 0
                    recovery_phase = ""
                    recovery_turn_sign = 1.0
                    recovery_attempt_count = 0
                    _camera_fail_count = 0
                    _camera_hard_stop_latched = False
                    route_corridor_violation_ticks = 0
                    nav_ready_ticks = 0
                    nav_ready_passed = not args.nav_ready_gate
                    logonav_align_mode = False
                    if active_idx >= len(navigation_targets):
                        if legwise_osm_routing and reached_count < mission_checkpoint_count:
                            next_checkpoint = mission_checkpoints[reached_count]
                            next_leg_index = reached_count + 1
                            leg_start_latlon = (goal_lat, goal_lon)
                            navigation_targets, routing_debug, target_utms, route_polyline_utms = build_active_navigation(
                                leg_start_latlon,
                                [next_checkpoint],
                                f"leg {next_leg_index}/{mission_checkpoint_count}",
                            )
                            active_idx = 0
                        else:
                            if not dry_run:
                                safe_stop()
                            print(f"Mission complete: reached all {mission_checkpoint_count} checkpoints.")
                            break
                    iteration += 1
                    elapsed = time.time() - loop_start
                    if elapsed < period:
                        time.sleep(period - elapsed)
                    continue

            if recovery_ticks_remaining > 0:
                if recovery_phase == "reverse":
                    command.linear = args.recovery_reverse_linear
                    command.angular = 0.0
                    command.reason = "recovery_reverse"
                else:
                    command.linear = 0.0
                    command.angular = recovery_turn_sign * abs(args.recovery_turn_angular)
                    command.reason = "recovery_turn"
                recovery_ticks_remaining -= 1
                if recovery_ticks_remaining == 0:
                    if recovery_phase == "reverse" and args.recovery_turn_ticks > 0:
                        recovery_phase = "turn"
                        recovery_ticks_remaining = args.recovery_turn_ticks
                    else:
                        recovery_phase = ""
            else:
                # Only update stuck window on fresh telemetry — frozen ticks have stale GPS
                # so appending them makes displacement look near-zero and fires stuck spuriously.
                if frozen_telemetry_ticks == 0:
                    recent_positions.append(raw_current_utm)
                    recent_distances.append(distance_to_goal)
                    recent_forward_flags.append(command.linear >= args.stuck_command_linear)
                    # Fast wall-hit check is only trustworthy for the classical GPS controller.
                    # On LogoNav it false-fires during slow curved progress and undoes real gains.
                    if args.controller == "gps" and command.linear >= 0.15:
                        fast_block_positions.append(raw_current_utm)
                    else:
                        fast_block_positions.clear()
                if args.controller == "gps" and len(fast_block_positions) == fast_block_positions.maxlen:
                    _fb_disp = math.hypot(
                        fast_block_positions[-1][0] - fast_block_positions[0][0],
                        fast_block_positions[-1][1] - fast_block_positions[0][1],
                    )
                    if _fb_disp < 0.10:
                        _be = float((command.debug or {}).get("bearing_error_rad", 0.0))
                        recovery_turn_sign = 1.0 if _be >= 0.0 else -1.0
                        recovery_phase = "reverse"
                        recovery_ticks_remaining = args.recovery_reverse_ticks
                        command.linear = args.recovery_reverse_linear
                        command.angular = 0.0
                        command.reason = "wall_hit_recovery"
                        fast_block_positions.clear()
                        recent_positions.clear()
                        recent_distances.clear()
                        recent_forward_flags.clear()
                if len(recent_positions) == recent_positions.maxlen:
                    start_pos = recent_positions[0]
                    end_pos = recent_positions[-1]
                    displacement = math.hypot(end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
                    distance_progress = recent_distances[0] - recent_distances[-1]
                    forward_ticks = sum(1 for flag in recent_forward_flags if flag)
                    if command.debug is None:
                        command.debug = {}
                    command.debug["stuck_window_displacement_m"] = displacement
                    command.debug["stuck_window_progress_m"] = distance_progress
                    command.debug["stuck_window_forward_ticks"] = forward_ticks
                    required_forward_ticks = max(2, math.ceil(recent_forward_flags.maxlen * 0.6))
                    progress_epsilon = args.stuck_progress_epsilon_m
                    if args.controller == "logonav":
                        # LogoNav often advances on shallow arcs with small GPS deltas.
                        # Only intervene when it is actually losing ground, not merely making
                        # less-than-ideal progress, and prefer turn-only recovery over reverse.
                        progress_epsilon = args.logonav_stuck_progress_epsilon_m
                    if (
                        forward_ticks >= required_forward_ticks
                        and displacement < args.stuck_min_displacement_m
                        and distance_progress < progress_epsilon
                    ):
                        recovery_attempt_count += 1
                        if args.max_recovery_attempts > 0 and recovery_attempt_count > args.max_recovery_attempts:
                            if not dry_run:
                                safe_stop()
                            print(
                                f"[{iteration:04d}] [MARATHON] {recovery_attempt_count} recovery attempts "
                                f"on waypoint {active_idx} — HALTED for operator"
                            )
                            input("  Press ENTER after repositioning robot to resume...")
                            recovery_attempt_count = 0
                            recent_positions.clear()
                            recent_distances.clear()
                            recent_forward_flags.clear()
                            fast_block_positions.clear()
                            if hasattr(controller, "reset"):
                                controller.reset()
                            time.sleep(period)
                            iteration += 1
                            continue
                        bearing_error = float((command.debug or {}).get("bearing_error_rad", 0.0))
                        if abs(command.angular) > 1e-3:
                            recovery_turn_sign = 1.0 if command.angular >= 0.0 else -1.0
                        else:
                            recovery_turn_sign = 1.0 if bearing_error >= 0.0 else -1.0
                        if args.controller == "logonav":
                            recovery_phase = "turn"
                            recovery_ticks_remaining = args.recovery_turn_ticks
                            command.linear = 0.0
                            command.angular = recovery_turn_sign * abs(args.recovery_turn_angular)
                        else:
                            recovery_phase = "reverse" if args.recovery_reverse_ticks > 0 else "turn"
                            recovery_ticks_remaining = args.recovery_reverse_ticks if recovery_phase == "reverse" else args.recovery_turn_ticks
                            command.linear = args.recovery_reverse_linear if recovery_phase == "reverse" else 0.0
                            command.angular = 0.0 if recovery_phase == "reverse" else recovery_turn_sign * abs(args.recovery_turn_angular)
                        command.reason = (
                            f"stuck_recovery_start(progress={distance_progress:.2f}m,disp={displacement:.2f}m)"
                        )
                        recent_positions.clear()
                        recent_distances.clear()
                        recent_forward_flags.clear()
                        fast_block_positions.clear()

            # LogoNav minimum effective forward floor.
            # If the learned controller is trying to move, but safety layers are not actively
            # asking for a stop/slow, avoid crawling forever at a few cm/s.
            if (
                args.controller == "logonav"
                and command.linear > 0.0
                and recovery_ticks_remaining == 0
                and command.reason not in {"semantic_yield", "logonav_align_turn"}
            ):
                _trav_blocked = bool(last_trav_result.forward_blocked) if last_trav_result is not None else False
                _legacy_depth_limited = bool(command.reason.startswith("depth_"))
                _semantic_limited = bool(command.reason.startswith("semantic_"))
                if not _trav_blocked and not _legacy_depth_limited and not _semantic_limited:
                    command.linear = max(command.linear, min(args.logonav_min_effective_linear, args.logonav_max_linear))

            # Traversability is a hard local safety layer for outdoor runs.
            # When the forward corridor is blocked, it overrides LogoNav instead of
            # merely nudging it. This prevents repeated wall contacts.
            if traversability is not None and last_trav_result is not None and recovery_ticks_remaining == 0:
                _trav = last_trav_result
                _trav_ang = None
                if _trav.angular_override is not None:
                    _trav_ang = float(max(-args.max_angular, min(args.max_angular, args.angular_gain * 2.5 * _trav.angular_override)))
                if _trav.all_blocked or _trav.linear_scale <= 0.0:
                    command.linear = 0.0
                    if _trav_ang is not None:
                        _turn_sign = 1.0 if _trav_ang >= 0.0 else -1.0
                        command.angular = _turn_sign * max(abs(_trav_ang), min(args.max_angular, 0.22))
                    command.reason = f"trav_stop({_trav.forward_clearance:.2f}m)"
                elif _trav.forward_blocked:
                    command.linear *= _trav.linear_scale
                    if _trav_ang is not None:
                        command.angular = _trav_ang
                    command.reason = f"trav_turn({_trav.forward_clearance:.2f}m)"
                elif _trav.linear_scale < 0.999:
                    command.linear *= _trav.linear_scale
                    command.reason = f"trav_slow({_trav.forward_clearance:.2f}m)"
                if command.debug is None:
                    command.debug = {}
                command.debug["trav_override"] = round(_trav.forward_clearance, 2)
                command.debug.update(_trav.debug)

            # Semantic soft bias — use only a weak angular nudge.
            # No stop, no slow, no controller takeover.
            _SEMANTIC_BIAS_WEIGHT = 0.15
            _SEMANTIC_HAZARD_BIAS_WEIGHT = 0.25
            if semantic_estimator is not None and last_sem_result is not None and recovery_ticks_remaining == 0:
                _sem = last_sem_result
                if command.debug is None:
                    command.debug = {}
                command.debug.update(_sem.debug)
                if command.linear > 0.0 and abs(_sem.bias_angular) > 1e-3:
                    _sem_ang = float(
                        max(-args.max_angular, min(args.max_angular,
                            args.angular_gain * 2.0 * _sem.bias_angular))
                    )
                    if _sem.hard_alerts:
                        _sem_w = _SEMANTIC_HAZARD_BIAS_WEIGHT
                    elif _sem.vegetation_blocked:
                        _sem_w = _SEMANTIC_BIAS_WEIGHT
                    else:
                        _sem_w = 0.0
                    if _sem_w > 0.0:
                        command.angular = (1.0 - _sem_w) * command.angular + _sem_w * _sem_ang
                        command.debug["sem_bias"] = _sem.debug.get("sem_event", "") or "active"
                        if args.semantic_yield and _sem.hard_alerts and command.linear > 0.0:
                            command.linear = min(command.linear, args.semantic_yield_max_linear)

            # EMA command smoothing — keep normal motion smooth, but let recovery commands apply immediately.
            if command.reason == "goal_reached":
                _smooth_linear = 0.0
                _smooth_angular = 0.0
            elif command.reason.startswith("stuck_recovery") or command.reason.startswith("recovery_") or command.reason == "wall_hit_recovery":
                _smooth_linear = command.linear
                _smooth_angular = command.angular
            else:
                _smooth_linear = _LINEAR_ALPHA * command.linear + (1.0 - _LINEAR_ALPHA) * _smooth_linear
                _smooth_angular = _CMD_ALPHA * command.angular + (1.0 - _CMD_ALPHA) * _smooth_angular
                command.linear = _smooth_linear
                command.angular = _smooth_angular

            if dry_run:
                sent = False
            else:
                sent = rover.send_control(command.linear, command.angular, lamp=lamp_value)

            payload = {
                "iteration": iteration,
                "target_index": active_idx,
                "targets_total": len(navigation_targets),
                "mission_checkpoints_total": mission_checkpoint_count,
                "mission_checkpoints_reached": reached_count,
                "active_mission_checkpoint_index": active_mission_idx + 1,
                "active_mission_checkpoint_lat": mission_goal_lat,
                "active_mission_checkpoint_lon": mission_goal_lon,
                "active_mission_checkpoint_distance_m": round(mission_distance_m, 2),
                "progress_fraction": reached_count / mission_checkpoint_count,
                "target_is_mission_checkpoint": bool(target.get("mission_checkpoint", False)),
                "current_lat": lat,
                "current_lon": lon,
                "goal_lat": goal_lat,
                "goal_lon": goal_lon,
                "distance_to_goal_m": distance_to_goal,
                "active_goal_radius_m": round(active_goal_radius_m, 2),
                "telemetry_timestamp": telemetry_timestamp,
                "frozen_telemetry_ticks": frozen_telemetry_ticks,
                "telemetry_stale_s": round(telemetry_stale_s, 2),
                "battery_pct": battery_pct,
                "vision_brightness": None if last_vis_result is None else round(last_vis_result.mean_brightness, 1),
                "vision_dark_fraction": None if last_vis_result is None else round(last_vis_result.dark_fraction, 2),
                "vision_glare_fraction": None if last_vis_result is None else round(last_vis_result.glare_fraction, 2),
                "vision_texture": None if last_vis_result is None else round(last_vis_result.texture_score, 1),
                "gps_signal": data.get("gps_signal"),
                "cell_signal": data.get("signal_level"),
                "route_corridor_distance_m": None if route_corridor_distance is None else round(route_corridor_distance, 2),
                "route_corridor_violation_ticks": route_corridor_violation_ticks,
                "recovery_phase": recovery_phase,
                "orientation_deg": orientation_deg,
                "heading_math_rad": current_heading_rad,
                "heading_fusion_weight": round(motion_heading_weight, 2),
                "linear": command.linear,
                "angular": command.angular,
                "reason": command.reason,
                "debug": command.debug or {},
                "sent": sent,
            }

            if args.print_json:
                print(json.dumps(payload))
            else:
                _dbg = command.debug or {}
                _bear_deg = math.degrees(_dbg.get("bearing_error_rad", float("nan")))
                _fwd = _dbg.get("trav_fwd_m", _dbg.get("forward_clearance_m", None))
                _fwd_str = f"{_fwd:.2f}m" if _fwd is not None else "---"
                _rsn = command.reason
                if _rsn.startswith("trav_stop"):
                    _mode = "TRV_STOP"
                elif _rsn.startswith("trav_turn"):
                    _mode = "TRV_TURN"
                elif _rsn.startswith("trav_slow"):
                    _mode = "TRV_SLOW"
                elif _rsn == "logonav_align_turn":
                    _mode = "ALIGN   "
                elif _rsn.startswith("semantic_"):
                    _mode = "SEM     "
                elif _rsn == "wall_hit_recovery":
                    _mode = "WALL!!! "
                elif _rsn.startswith("stuck_recovery") or _rsn.startswith("recovery_"):
                    _mode = "STUCK!! "
                else:
                    _mode = f"{_rsn[:8]:8s}"
                _sent_str = "SENT" if sent else "SKIP"
                _bear_str = f"{_bear_deg:+.0f}°" if math.isfinite(_bear_deg) else "  ---"
                if _prev_orientation_deg is not None:
                    _raw_ddeg = ((orientation_deg - _prev_orientation_deg + 180) % 360) - 180
                    _ddeg_str = f"{_raw_ddeg:+.0f}°"
                else:
                    _ddeg_str = "  +0°"
                _prev_orientation_deg = orientation_deg
                _frz_str = f" FRZ={frozen_telemetry_ticks}/{telemetry_stale_s:.1f}s" if frozen_telemetry_ticks > 0 else ""
                _corr_str = (
                    f" COR={route_corridor_distance:.1f}m/{route_corridor_violation_ticks}"
                    if route_corridor_distance is not None and args.route_corridor_guard
                    else ""
                )
                _vis_str = (
                    f" VIS={last_vis_result.mean_brightness:.0f}/{last_vis_result.texture_score:.1f}"
                    if last_vis_result is not None and vision_monitor is not None
                    else ""
                )
                _rad_str = f" RAD={active_goal_radius_m:.1f}m"
                _leg_str = f" LEG={active_mission_idx + 1}/{mission_checkpoint_count}"
                _wp_str = f" WP={active_idx + 1}/{len(navigation_targets)}"
                _cpdist_str = f" CPDIST={mission_distance_m:5.1f}m"
                print(
                    f"[{iteration:04d}]"
                    f" DONE={reached_count}/{mission_checkpoint_count}"
                    f"{_leg_str}{_wp_str}"
                    f" dist={distance_to_goal:5.1f}m{_cpdist_str}"
                    f" bear={_bear_str:>5s}"
                    f" hdg={orientation_deg:5.1f}°(Δ{_ddeg_str})"
                    f" clr={_fwd_str:>6s}"
                    f" mode={_mode}"
                    f" lin={command.linear:+.3f} ang={command.angular:+.3f}"
                    f" {_sent_str}{_frz_str}{_corr_str}{_rad_str}{_vis_str}"
                )

            elapsed = time.time() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)
            iteration += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if not dry_run:
            safe_stop()
            print("Robot stopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
