import argparse
import csv
import os
from typing import Optional

import h5py
import numpy as np


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract control velocities up to a given frame index based on the frame timestamp"
        ),
        epilog=(
            "Example:\n"
            "  python examples/utils/extract_controls.py logs/run.h5 --group front_frames --idx 333 --out controls_until_333.csv\n"
        ),
    )
    parser.add_argument("log", help="Path to HDF5 log")
    parser.add_argument("--group", choices=["front_frames", "rear_frames"], default="front_frames")
    parser.add_argument("--idx", type=int, required=True, help="Frame index in the chosen group")
    parser.add_argument("--out", default=None, help="Optional CSV output path; prints to stdout if omitted")
    return parser.parse_args(argv)


def action_label(linear: float, angular: float) -> str:
    t = 0.05
    fwd = linear > t
    back = linear < -t
    left = angular > t
    right = angular < -t
    if not (fwd or back or left or right):
        return "idle"
    if fwd and not (left or right):
        return "forward"
    if back and not (left or right):
        return "backward"
    if left and not (fwd or back):
        return "turn_left"
    if right and not (fwd or back):
        return "turn_right"
    if fwd and left:
        return "forward_left"
    if fwd and right:
        return "forward_right"
    if back and left:
        return "backward_left"
    if back and right:
        return "backward_right"
    return "move"


def get_frame_timestamp(log_path: str, group: str, idx: int) -> float:
    """Utility: return the timestamp of a given frame index in the specified group."""
    with h5py.File(log_path, "r") as f:
        if group not in f:
            raise SystemExit(f"Group not found in log: {group}")
        grp = f[group]
        if "timestamps" not in grp or grp["timestamps"].shape[0] == 0:
            raise SystemExit(f"No timestamps found in group {group}")
        n_frames = grp["timestamps"].shape[0]
        if idx < 0 or idx >= n_frames:
            raise SystemExit(f"idx out of range [0, {n_frames - 1}]")
        return float(grp["timestamps"][idx])


def extract_controls_until(log_path: str, group: str, idx: int) -> np.ndarray:
    """Return controls up to the frame index.
    
    Uses proportional index mapping
    """
    with h5py.File(log_path, "r") as f:
        if "controls" not in f:
            raise SystemExit("No controls dataset in log")
        controls = f["controls"][:]
        
        if group not in f or "timestamps" not in f[group]:
            raise SystemExit(f"No timestamps found in group {group}")
        total_frames = f[group]["timestamps"].shape[0]
        
    if controls.size == 0:
        return controls
    
    order = np.argsort(controls["timestamp"])  # chronological
    controls = controls[order]
    total_controls = len(controls)
    
    # Use proportional index mapping: frame idx/total_frames -> control idx/total_controls
    if idx >= total_frames:
        idx = total_frames - 1
    
    # Calculate proportional control index
    ratio = (idx + 1) / total_frames
    control_count = int(ratio * total_controls)
    control_count = max(1, min(control_count, total_controls))  # At least 1, at most all
    
    return controls[:control_count]


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if not os.path.exists(args.log):
        print(f"File not found: {args.log}")
        print("Example:")
        print("  python examples/extract_controls.py logs/run.h5 --group front_frames --idx 333 --out controls_until_333.csv")
        return 2

    segment = extract_controls_until(args.log, args.group, args.idx)

    if segment.size == 0:
        print("No control commands found up to the given frame timestamp.")
        return 0

    rows = []
    for r in segment:
        ts = float(r["timestamp"]) 
        lin = float(r["linear"]) 
        ang = float(r["angular"]) 
        rows.append((ts, lin, ang, action_label(lin, ang)))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["timestamp", "linear", "angular", "action"])
            writer.writerows(rows)
        print(f"Saved {len(rows)} control rows to {args.out}")
    else:
        print("timestamp,linear,angular,action")
        for ts, lin, ang, act in rows:
            print(f"{ts:.6f},{lin:.6f},{ang:.6f},{act}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


