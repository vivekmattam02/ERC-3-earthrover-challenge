#!/usr/bin/env python3
"""Offline traversability calibration on recorded outdoor .h5 data.

Runs DepthEstimator + OutdoorTraversability on sampled frames from
test_outdoor/*.h5 and reports aggregate statistics to help tune thresholds
before field testing.

No production code is changed.  This is analysis only.

Usage:
    python scripts/calibrate_traversability.py
    python scripts/calibrate_traversability.py --every-n 5 --obstacle-m 1.5 --stop-m 0.6 --slow-m 1.2
    python scripts/calibrate_traversability.py --save-interesting
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from depth_estimator import DepthEstimator  # type: ignore
from outdoor_traversability import (  # type: ignore
    OutdoorTraversability,
    OutdoorTraversabilityConfig,
)


def decode_frame(raw: np.ndarray) -> np.ndarray:
    """Decode a stored h5 JPEG/PNG blob to RGB uint8 array."""
    img = Image.open(BytesIO(bytes(raw)))
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    return arr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline traversability calibration.")
    p.add_argument(
        "--h5-dir",
        type=Path,
        default=REPO_ROOT / "test_outdoor",
        help="Directory containing .h5 recording files.",
    )
    p.add_argument("--every-n", type=int, default=10, help="Sample every Nth frame.")
    p.add_argument("--max-frames", type=int, default=500, help="Max frames to process total across all files.")
    p.add_argument("--depth-model-size", choices=("small", "base", "large"), default="small")

    # Traversability thresholds (match outdoor_traversability.py defaults)
    p.add_argument("--obstacle-m", type=float, default=1.5, help="Bins below this are blocked.")
    p.add_argument("--stop-m", type=float, default=0.60, help="Stop below this forward clearance.")
    p.add_argument("--slow-m", type=float, default=1.20, help="Slow below this forward clearance.")
    p.add_argument("--percentile", type=float, default=10, help="Depth percentile per bin (0=min, 50=median).")
    p.add_argument("--crop-top", type=float, default=0.15, help="Top crop fraction.")
    p.add_argument("--crop-bot", type=float, default=0.60, help="Bottom crop fraction.")
    p.add_argument("--memory-frames", type=int, default=4, help="Temporal min-pool window.")
    p.add_argument("--num-bins", type=int, default=16, help="Number of angular bins.")
    p.add_argument("--fov-deg", type=float, default=90.0, help="Horizontal FOV.")

    p.add_argument("--save-interesting", action="store_true", help="Save frames with lowest clearance / all-blocked / largest overrides.")
    p.add_argument("--save-dir", type=Path, default=REPO_ROOT / "scripts" / "trav_debug", help="Where to save interesting frames.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    h5_files = sorted(args.h5_dir.glob("*.h5"))
    if not h5_files:
        print(f"No .h5 files found in {args.h5_dir}")
        return 1
    print(f"Found {len(h5_files)} recording file(s)")

    # Load depth estimator
    print(f"Loading depth estimator ({args.depth_model_size})...")
    depth_est = DepthEstimator(model_size=args.depth_model_size)

    # Build traversability with CLI-specified thresholds
    trav = OutdoorTraversability(OutdoorTraversabilityConfig(
        num_bins=args.num_bins,
        fov_horizontal_deg=args.fov_deg,
        crop_top_frac=args.crop_top,
        crop_bot_frac=args.crop_bot,
        obstacle_distance_m=args.obstacle_m,
        stop_distance_m=args.stop_m,
        slow_distance_m=args.slow_m,
        memory_frames=args.memory_frames,
    ))

    # Override percentile if it differs from default (requires patching the module)
    # We'll just note this — the percentile is baked into the module at 10.
    if args.percentile != 10:
        print(f"[warn] --percentile {args.percentile} requested but module uses hardcoded 10th percentile.")
        print("       To test other values, edit outdoor_traversability.py:_clearance_from_depth().")

    # Stats accumulators
    all_fwd_clearances: list[float] = []
    stop_count = 0
    slow_count = 0
    blocked_count = 0
    all_blocked_count = 0
    total_frames = 0
    bin_block_counts = np.zeros(args.num_bins, dtype=int)
    safe_heading_hist: list[float] = []

    # Interesting frame tracking
    min_fwd_clearance = float("inf")
    min_fwd_frame_info: dict | None = None
    max_override_mag = 0.0
    max_override_frame_info: dict | None = None
    all_blocked_examples: list[dict] = []

    frames_processed = 0

    for h5_path in h5_files:
        print(f"\nProcessing {h5_path.name}...")
        f = h5py.File(h5_path, "r")
        front_data = f["front_frames/data"]
        n_frames = front_data.shape[0]

        # Reset traversability memory between files (different scenes)
        trav._history.clear()

        sampled_indices = list(range(0, n_frames, args.every_n))
        file_count = 0

        for idx in sampled_indices:
            if frames_processed >= args.max_frames:
                break

            try:
                raw = front_data[idx]
                frame = decode_frame(raw)
            except Exception:
                continue

            # Run depth estimation
            depth_map = depth_est.estimate(frame, target_size=(120, 160))

            # Run traversability (goal bearing = 0, straight ahead, for calibration)
            result = trav.compute(depth_map, goal_bearing_error_rad=0.0)

            fwd = result.forward_clearance
            all_fwd_clearances.append(fwd)

            if result.linear_scale == 0.0:
                stop_count += 1
            elif result.linear_scale < 1.0:
                slow_count += 1
            if result.forward_blocked:
                blocked_count += 1
            if result.all_blocked:
                all_blocked_count += 1

            # Per-bin block tracking
            bin_block_counts += (result.clearance < args.obstacle_m).astype(int)
            safe_heading_hist.append(math.degrees(result.safe_heading_rad))

            total_frames += 1
            frames_processed += 1
            file_count += 1

            # Track interesting frames
            frame_info = {
                "file": h5_path.name,
                "idx": idx,
                "fwd_clearance": fwd,
                "blocked": result.forward_blocked,
                "all_blocked": result.all_blocked,
                "safe_heading_deg": math.degrees(result.safe_heading_rad),
                "linear_scale": result.linear_scale,
                "angular_override": result.angular_override,
                "clearance_bins": [round(float(c), 2) for c in result.clearance],
            }

            if fwd < min_fwd_clearance:
                min_fwd_clearance = fwd
                min_fwd_frame_info = frame_info
                if args.save_interesting:
                    min_fwd_frame_info["frame"] = frame

            if result.angular_override is not None:
                mag = abs(result.angular_override)
                if mag > max_override_mag:
                    max_override_mag = mag
                    max_override_frame_info = frame_info
                    if args.save_interesting:
                        max_override_frame_info["frame"] = frame

            if result.all_blocked and len(all_blocked_examples) < 5:
                info = dict(frame_info)
                if args.save_interesting:
                    info["frame"] = frame
                all_blocked_examples.append(info)

        f.close()
        if frames_processed >= args.max_frames:
            print(f"  (hit --max-frames {args.max_frames}, stopping)")
            break
        print(f"  processed {file_count} frames")

    # --- Aggregate report ---
    print("\n" + "=" * 70)
    print("TRAVERSABILITY CALIBRATION REPORT")
    print("=" * 70)
    print(f"Files: {len(h5_files)}  |  Frames sampled: {total_frames}  |  every-{args.every_n}")
    print(f"\nThresholds:")
    print(f"  obstacle_distance_m = {args.obstacle_m}")
    print(f"  stop_distance_m     = {args.stop_m}")
    print(f"  slow_distance_m     = {args.slow_m}")
    print(f"  crop band           = {args.crop_top:.0%} – {args.crop_bot:.0%} from top")
    print(f"  percentile          = {args.percentile}th")
    print(f"  memory_frames       = {args.memory_frames}")

    if total_frames == 0:
        print("\nNo frames processed.")
        return 1

    arr = np.array(all_fwd_clearances)
    print(f"\nForward clearance distribution:")
    print(f"  min    = {arr.min():.3f} m")
    print(f"  p5     = {np.percentile(arr, 5):.3f} m")
    print(f"  p10    = {np.percentile(arr, 10):.3f} m")
    print(f"  p25    = {np.percentile(arr, 25):.3f} m")
    print(f"  median = {np.percentile(arr, 50):.3f} m")
    print(f"  p75    = {np.percentile(arr, 75):.3f} m")
    print(f"  p90    = {np.percentile(arr, 90):.3f} m")
    print(f"  max    = {arr.max():.3f} m")

    print(f"\nTrigger rates:")
    print(f"  STOP    (fwd < {args.stop_m}m)  : {stop_count:4d} / {total_frames}  ({100*stop_count/total_frames:.1f}%)")
    print(f"  SLOW    (fwd < {args.slow_m}m)  : {slow_count:4d} / {total_frames}  ({100*slow_count/total_frames:.1f}%)")
    print(f"  BLOCKED (fwd < {args.obstacle_m}m): {blocked_count:4d} / {total_frames}  ({100*blocked_count/total_frames:.1f}%)")
    print(f"  ALL_BLK (every bin)    : {all_blocked_count:4d} / {total_frames}  ({100*all_blocked_count/total_frames:.1f}%)")

    print(f"\nPer-bin block frequency (bin blocked in N frames):")
    bin_centers_deg = np.linspace(-args.fov_deg / 2, args.fov_deg / 2, args.num_bins)
    for b in range(args.num_bins):
        bar = "#" * int(40 * bin_block_counts[b] / max(1, total_frames))
        pct = 100 * bin_block_counts[b] / total_frames
        print(f"  bin {b:2d} ({bin_centers_deg[b]:+5.1f}°): {pct:5.1f}%  {bar}")

    print(f"\nSafe heading selection:")
    sh_arr = np.array(safe_heading_hist)
    print(f"  mean = {sh_arr.mean():+.1f}°  std = {sh_arr.std():.1f}°")
    print(f"  range = [{sh_arr.min():+.1f}°, {sh_arr.max():+.1f}°]")

    # Interesting frames
    print(f"\n--- Interesting frames ---")
    if min_fwd_frame_info:
        info = min_fwd_frame_info
        print(f"\nLowest forward clearance: {info['fwd_clearance']:.3f} m")
        print(f"  file={info['file']} frame={info['idx']}")
        print(f"  blocked={info['blocked']} all_blocked={info['all_blocked']}")
        print(f"  safe_heading={info['safe_heading_deg']:+.1f}° linear_scale={info['linear_scale']:.2f}")
        print(f"  bins={info['clearance_bins']}")

    if max_override_frame_info:
        info = max_override_frame_info
        print(f"\nLargest angular override: {math.degrees(info['angular_override']):.1f}°")
        print(f"  file={info['file']} frame={info['idx']}")
        print(f"  fwd={info['fwd_clearance']:.3f}m blocked={info['blocked']}")
        print(f"  bins={info['clearance_bins']}")

    if all_blocked_examples:
        print(f"\nAll-blocked examples ({len(all_blocked_examples)}):")
        for info in all_blocked_examples[:3]:
            print(f"  file={info['file']} frame={info['idx']} fwd={info['fwd_clearance']:.3f}m")
            print(f"    bins={info['clearance_bins']}")

    # Save interesting frames
    if args.save_interesting:
        save_dir = args.save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        for tag, info in [("min_fwd", min_fwd_frame_info), ("max_override", max_override_frame_info)]:
            if info and "frame" in info:
                path = save_dir / f"{tag}_{info['file']}_{info['idx']}.png"
                Image.fromarray(info["frame"]).save(path)
                saved += 1
        for i, info in enumerate(all_blocked_examples):
            if "frame" in info:
                path = save_dir / f"all_blocked_{i}_{info['file']}_{info['idx']}.png"
                Image.fromarray(info["frame"]).save(path)
                saved += 1
        if saved:
            print(f"\nSaved {saved} interesting frame(s) to {save_dir}/")

    # Recommendation
    print(f"\n--- Recommendation ---")
    if stop_count / total_frames > 0.15:
        print(f"  WARNING: STOP triggers on {100*stop_count/total_frames:.0f}% of frames.")
        print(f"  The stop threshold ({args.stop_m}m) may be too high for this camera.")
        print(f"  Consider lowering --stop-m to {arr.min() * 0.8:.2f} or less.")
    elif stop_count / total_frames > 0.05:
        print(f"  CAUTION: STOP triggers on {100*stop_count/total_frames:.0f}% of frames.")
        print(f"  May be slightly aggressive — monitor in field test.")
    else:
        print(f"  STOP rate ({100*stop_count/total_frames:.1f}%) looks reasonable.")

    if all_blocked_count > 0:
        print(f"  WARNING: {all_blocked_count} all-blocked frames — robot would freeze here.")
        print(f"  Check saved frames to see if these are real obstacles or noise.")

    if blocked_count / total_frames > 0.40:
        print(f"  WARNING: forward is blocked on {100*blocked_count/total_frames:.0f}% of frames.")
        print(f"  obstacle_distance_m={args.obstacle_m} may be too high, or crop band sees too much ground.")
    elif blocked_count / total_frames < 0.02:
        print(f"  NOTE: forward almost never blocked ({100*blocked_count/total_frames:.1f}%).")
        print(f"  Traversability layer may not activate at all — verify obstacles are present in data.")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
