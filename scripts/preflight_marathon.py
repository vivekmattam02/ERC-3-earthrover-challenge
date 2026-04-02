#!/usr/bin/env python3
"""Preflight validator for the outdoor ultra-marathon runtime."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from earthrover_interface import EarthRoverInterface  # type: ignore
from live_outdoor_runtime import build_navigation_targets, build_osm_config  # type: ignore
from imu_safety import IMUSafetyMonitor  # type: ignore
from vision_safety_monitor import VisionSafetyConfig, VisionSafetyMonitor  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SDK, mission, sensors, and routing before an outdoor ultra-marathon run.")
    parser.add_argument("--sdk-url", default="http://localhost:8000", help="EarthRover SDK base URL.")
    parser.add_argument("--sdk-timeout", type=float, default=10.0, help="SDK request timeout in seconds.")
    parser.add_argument("--mission", action="store_true", help="Call /start-mission and validate the current mission checkpoints.")
    parser.add_argument("--osm-route", action="store_true", help="If mission mode is on, validate pedestrian route expansion too.")
    parser.add_argument("--sidewalk-strict", action="store_true", help="Require strict pedestrian OSM routing with no straight-line fallback.")
    parser.add_argument("--battery-min-pct", type=float, default=70.0, help="Fail if battery is below this percentage.")
    parser.add_argument("--camera-timeout-s", type=float, default=10.0, help="How long to wait for a live camera frame.")
    parser.add_argument("--telemetry-samples", type=int, default=4, help="How many telemetry polls to use for timestamp-advance validation.")
    parser.add_argument("--telemetry-gap-s", type=float, default=0.5, help="Delay between telemetry polls.")
    parser.add_argument("--gps-min-signal", type=int, default=2, help="Fail if GPS signal is below this level.")
    parser.add_argument("--osm-buffer-m", type=float, default=300.0, help="Bounding-box padding for OSM queries.")
    parser.add_argument("--osm-query-timeout-s", type=int, default=25, help="Overpass server-side timeout in seconds.")
    parser.add_argument("--osm-request-retries", type=int, default=2, help="Number of Overpass request attempts before fallback.")
    parser.add_argument("--osm-max-segment-m", type=float, default=20.0, help="Max gap between routed waypoints after densification.")
    parser.add_argument("--osm-min-waypoint-spacing-m", type=float, default=8.0, help="Minimum spacing between retained routed waypoints.")
    parser.add_argument("--osm-max-snap-distance-m", type=float, default=60.0, help="Max allowed snap distance to the OSM graph.")
    parser.add_argument("--night-safe", action="store_true", help="Run additional night-time preflight checks: image visibility and IMU tilt sanity.")
    parser.add_argument("--vision-min-brightness", type=float, default=42.0, help="Night preflight: minimum acceptable mean brightness.")
    parser.add_argument("--vision-max-dark-fraction", type=float, default=0.65, help="Night preflight: maximum acceptable dark-pixel fraction.")
    parser.add_argument("--vision-max-glare-fraction", type=float, default=0.12, help="Night preflight: maximum acceptable glare fraction.")
    parser.add_argument("--vision-min-texture", type=float, default=8.0, help="Night preflight: minimum acceptable texture score.")
    parser.add_argument("--vision-samples", type=int, default=3, help="Night preflight: number of recent frames to sample for visibility consistency.")
    return parser.parse_args()


def fail(message: str) -> int:
    print(f"[FAIL] {message}")
    return 1


def ok(message: str) -> None:
    print(f"[ OK ] {message}")


def main() -> int:
    args = parse_args()
    rover = EarthRoverInterface(base_url=args.sdk_url, timeout=args.sdk_timeout)

    checkpoints = None
    if args.mission:
        checkpoints = rover.start_mission()
        if checkpoints is None:
            return fail("/start-mission failed.")
        if not checkpoints:
            return fail("/start-mission succeeded but returned no checkpoints.")
        rover.connected = True
        ok(f"Mission started with {len(checkpoints)} checkpoint(s)")
    else:
        if not rover.connect():
            return fail("Could not connect to SDK.")

    data = rover.get_data(use_cache=False)
    if data is None:
        return fail("SDK connected but /data returned nothing.")

    battery = data.get("battery")
    try:
        battery_pct = float(battery)
    except Exception:
        return fail("Battery percentage is missing from telemetry.")
    ok(f"Battery telemetry present: {battery_pct:.1f}%")
    if battery_pct < args.battery_min_pct:
        return fail(f"Battery {battery_pct:.1f}% is below required minimum {args.battery_min_pct:.1f}%")

    try:
        lat = float(data.get("latitude"))
        lon = float(data.get("longitude"))
    except Exception:
        return fail("Latitude/longitude missing from telemetry.")
    if not (math.isfinite(lat) and math.isfinite(lon)) or lat == 0.0 or lon == 0.0:
        return fail(f"GPS is not live yet: ({lat}, {lon})")
    ok(f"GPS live: ({lat:.6f}, {lon:.6f})")

    timestamps: list[float] = []
    for _ in range(max(2, args.telemetry_samples)):
        sample = rover.get_data(use_cache=False)
        if sample is None:
            return fail("Telemetry became unavailable during validation.")
        try:
            timestamps.append(float(sample.get("timestamp", 0.0)))
        except Exception:
            return fail("Telemetry timestamp is missing or invalid.")
        time.sleep(args.telemetry_gap_s)
    if not any(timestamps[idx] > timestamps[idx - 1] for idx in range(1, len(timestamps))):
        return fail("Telemetry timestamp is not advancing.")
    ok("Telemetry timestamp is advancing")

    deadline = time.time() + args.camera_timeout_s
    frame = None
    while time.time() < deadline:
        frame = rover.get_camera_frame()
        if frame is not None:
            break
        time.sleep(0.5)
    if frame is None:
        return fail("Front camera did not become live.")
    ok(f"Front camera live: shape={tuple(frame.shape)}")

    signal = data.get("signal_level")
    gps_signal = data.get("gps_signal")
    print(f"[INFO] Telemetry signal: cell={signal} gps={gps_signal}")
    try:
        if gps_signal is None or int(gps_signal) < int(args.gps_min_signal):
            return fail(f"GPS signal {gps_signal} is below required minimum {args.gps_min_signal}")
    except Exception:
        return fail("GPS signal telemetry is missing or invalid.")
    ok(f"GPS signal is acceptable: {gps_signal}")

    if args.night_safe:
        vis_monitor = VisionSafetyMonitor(VisionSafetyConfig(
            min_brightness=args.vision_min_brightness,
            max_dark_fraction=args.vision_max_dark_fraction,
            max_glare_fraction=args.vision_max_glare_fraction,
            min_texture_score=args.vision_min_texture,
            consecutive_bad_ticks_to_stop=1,
        ))
        vis_results = []
        samples = max(1, int(args.vision_samples))
        for idx in range(samples):
            if idx > 0:
                _frame = rover.get_camera_frame()
                if _frame is not None:
                    frame = _frame
                time.sleep(0.2)
            vis_results.append(vis_monitor.update(frame))
        vis = vis_results[-1]
        good_frames = sum(1 for item in vis_results if not item.emergency_stop)
        print(
            f"[INFO] Night vision stats: brightness={vis.mean_brightness:.1f} dark={vis.dark_fraction:.2f} "
            f"glare={vis.glare_fraction:.2f} texture={vis.texture_score:.1f} good_frames={good_frames}/{samples}"
        )
        required_good = 1 if samples == 1 else max(2, samples - 1)
        if good_frames < required_good:
            return fail(f"Night visibility is inconsistent or unsafe ({good_frames}/{samples} good frames)")
        ok("Night visibility gate passed")

        imu = IMUSafetyMonitor().update(data)
        print(
            f"[INFO] IMU rest check: tilt={imu.tilt_deg:.1f}deg gyro={imu.pitch_roll_rate_dps:.1f}dps vib={imu.vibration:.2f}"
        )
        if imu.tilt_deg > 20.0:
            return fail(f"Robot appears tilted at rest ({imu.tilt_deg:.1f} deg)")
        ok("IMU rest pose looks sane")

    logonav_weights = REPO_ROOT / "mbra_repo" / "deployment" / "model_weights" / "logonav.pth"
    logonav_config = REPO_ROOT / "mbra_repo" / "train" / "config" / "LogoNav.yaml"
    if not logonav_weights.exists():
        return fail(f"LogoNav weights missing: {logonav_weights}")
    if not logonav_config.exists():
        return fail(f"LogoNav config missing: {logonav_config}")
    ok("LogoNav config and weights found")

    if not args.mission:
        print("[INFO] Mission validation skipped (pass --mission to validate /start-mission and checkpoint list).")
        return 0

    assert checkpoints is not None
    for cp in checkpoints:
        print(f"      seq={cp['sequence']} lat={cp['latitude']} lon={cp['longitude']}")

    status = rover.get_checkpoints_list()
    if status is None:
        return fail("/checkpoints-list failed after mission start.")
    ok(f"Mission checkpoint status available (latest_scanned_checkpoint={status.get('latest_scanned_checkpoint')})")

    if args.sidewalk_strict:
        args.osm_route = True
    if args.osm_route:
        mission_checkpoints = [(float(cp["latitude"]), float(cp["longitude"])) for cp in checkpoints]
        start_latlon = (lat, lon)
        targets, routing_debug = build_navigation_targets(
            mission_checkpoints,
            start_latlon=start_latlon,
            use_osm_route=True,
            osm_config=build_osm_config(args),
            require_osm_success=args.sidewalk_strict,
        )
        fallback_legs = sum(
            1
            for item in routing_debug
            if isinstance(item.get("routing"), dict)
            and item["routing"].get("routing") == "fallback_straight_line"
        )
        ok(f"OSM routing expanded mission into {len(targets)} navigation target(s)")
        print(f"[INFO] OSM fallback legs: {fallback_legs}/{len(routing_debug)}")
        if args.sidewalk_strict and fallback_legs > 0:
            return fail(f"Strict sidewalk routing saw {fallback_legs} straight-line fallback leg(s).")
        if routing_debug and fallback_legs == len(routing_debug):
            return fail("OSM routing fell back to straight-line for every leg.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
