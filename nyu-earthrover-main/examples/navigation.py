import argparse
import os
import sys
import time
from typing import List, Optional, Tuple

import h5py
import numpy as np
import requests


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Navigate by matching a target image in an HDF5 log and replaying controls up to that frame"
        ),
        epilog=(
            "Examples:\n"
            "  python examples/navigation.py logs/run.h5 --target assets/axis.jpg --url http://127.0.0.1:8000\n"
            "  python examples/navigation.py logs/run.h5 --group front_frames --idx 333 --url http://127.0.0.1:8000\n"
        ),
    )
    parser.add_argument("log", help="Path to HDF5 log")
    parser.add_argument("--target", help="Path to target image; if provided, best match idx is computed")
    parser.add_argument(
        "--method", choices=["orb", "sift"], default="orb", help="Feature method for matching"
    )
    parser.add_argument(
        "--group",
        choices=["front_frames", "rear_frames"],
        default="front_frames",
        help="Frame group to search/use",
    )
    parser.add_argument("--idx", type=int, help="Frame index to navigate to (skips matching if set)")
    parser.add_argument("--url", default=os.getenv("SDK_URL", "http://127.0.0.1:8000"), help="SDK base URL")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed factor (>1.0 faster)")
    parser.add_argument("--dry-run", action="store_true", help="Do not send control commands, just print")
    return parser.parse_args(argv)


def find_best_match_idx(log_path: str, group: str, target_path: str, method: str) -> Tuple[int, float]:
    """Return (best_idx, best_timestamp)."""
    try:
        from utils import image_match as im  # type: ignore
    except Exception as exc:
        raise SystemExit(f"Failed to import image_match: {exc}")

    target = im.load_target(target_path)
    kp_t, des_t = im.compute_keypoints_descriptors(target, method)
    if des_t is None or len(kp_t) == 0:
        raise SystemExit("No features found in target image")

    best_score = -1.0
    best_idx = -1
    best_ts = 0.0
    with h5py.File(log_path, "r") as f:
        if group not in f or f[group]["data"].shape[0] == 0:
            raise SystemExit(f"No frames in group {group}")
        grp = f[group]
        n = grp["data"].shape[0]
        for i in range(n):
            img = im.decode_h5_image(grp, i)
            if img is None:
                continue
            kp, des = im.compute_keypoints_descriptors(img, method)
            if des is None or len(kp) == 0:
                continue
            good = im.match_descriptors(des_t, des, method)
            inliers, _ = im.ransac_inliers(kp_t, kp, good)
            score = im.score_match(inliers, len(good))
            if score > best_score:
                best_score = score
                best_idx = i
                best_ts = float(grp["timestamps"][i]) if "timestamps" in grp else 0.0
    if best_idx < 0:
        raise SystemExit("No match found")
    return best_idx, best_ts


def load_controls_until(log_path: str, group: str, idx: int) -> np.ndarray:
    """Load controls proportional to the target frame index."""
    try:
        from utils import extract_controls as ec  # type: ignore
    except Exception as exc:
        raise SystemExit(f"Failed to import extract_controls: {exc}")
    return ec.extract_controls_until(log_path, group, idx)


def send_command(url: str, linear: float, angular: float, timeout_s: float = 2.0) -> bool:
    try:
        payload = {"command": {"linear": float(linear), "angular": float(angular)}}
        r = requests.post(url.rstrip("/") + "/control", json=payload, timeout=timeout_s)
        return bool(r.ok)
    except Exception:
        return False


def replay_controls(
    url: str,
    controls: np.ndarray,
    speed: float = 1.0,
    dry_run: bool = False,
) -> None:
    """Replay controls and stop at the end."""
    if controls.size == 0:
        print("No controls to replay")
        return
    
    # Replay all controls with timing
    t0 = float(controls["timestamp"][0])
    last_t = t0
    for i, row in enumerate(controls):
        ts = float(row["timestamp"])
        lin = float(row["linear"])
        ang = float(row["angular"])
        dt = (ts - last_t) / max(1e-6, speed)
        if dt > 0:
            time.sleep(min(dt, 0.2))  # cap to avoid very long delays
        last_t = ts
        
        if dry_run:
            is_last = i == len(controls) - 1
            marker = " [LAST]" if is_last else ""
            print(f"[{i+1}/{len(controls)}] send linear={lin:.3f} angular={ang:.3f}{marker}")
        else:
            ok = send_command(url, lin, ang)
            if not ok:
                print(f"warn: failed to send command {i+1}/{len(controls)}")
    
    # Stop the robot
    print("Reached target frame, stopping...")
    if dry_run:
        print("send linear=0.000 angular=0.000 (stop)")
    else:
        # Send stop commands at 20Hz for 2 seconds to ensure robot stops
        stop_duration = 2.0
        stop_interval = 0.05
        start_time = time.time()
        count = 0
        while time.time() - start_time < stop_duration:
            ok = send_command(url, 0.0, 0.0)
            count += 1
            time.sleep(stop_interval)
        print(f"Sent {count} stop commands")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.idx is not None and args.target is None:
        target_idx = int(args.idx)
    else:
        if not args.target:
            raise SystemExit("Provide --target or --idx")
        best_idx, _ = find_best_match_idx(args.log, args.group, args.target, args.method)
        print(f"Best match: idx={best_idx}")
        target_idx = best_idx

    # Get total frames for info
    with h5py.File(args.log, "r") as f:
        total_frames = f[args.group]["timestamps"].shape[0]
        total_controls = f["controls"].shape[0]

    # Load controls proportional to target frame
    controls = load_controls_until(args.log, args.group, target_idx)
    
    print(f"Target: frame {target_idx}/{total_frames}")
    print(f"Replaying: {controls.size}/{total_controls} controls at {args.url} (speed x{args.speed})")
    
    replay_controls(args.url, controls, speed=args.speed, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
