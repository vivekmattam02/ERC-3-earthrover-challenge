#!/usr/bin/env python3
"""Diagnostic: check live VPR localization quality against the corridor database.

Run this BEFORE attempting autonomous navigation to verify:
1. Live camera frame resolution matches database expectations
2. Raw CosPlace descriptor distances are reasonable
3. Top matches are plausible corridor positions

Usage:
    python scripts/diagnose_localization.py [--sdk-url http://localhost:8000]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from earthrover_interface import EarthRoverInterface
from baseline import (
    DescriptorConfig,
    crop_image,
    descriptor_distance_search,
    get_device,
    load_cosplace_model,
    load_descriptor_archive,
    load_descriptor_config,
    make_cosplace_transform,
)
from PIL import Image
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdk-url", default="http://localhost:8000")
    parser.add_argument("--database", type=Path,
                        default=REPO_ROOT / "data" / "corrider_db" / "descriptors.npz")
    parser.add_argument("--num-frames", type=int, default=5,
                        help="Number of frames to capture and test")
    parser.add_argument("--offline-image", type=Path, default=None,
                        help="Test with an offline image instead of the SDK")
    args = parser.parse_args()

    # --- Load database ---
    print("=" * 60)
    print("LOCALIZATION DIAGNOSTIC")
    print("=" * 60)

    print(f"\nLoading database: {args.database}")
    descriptor_config = load_descriptor_config(args.database)
    print(f"  Descriptor config: backbone={descriptor_config.backbone}, "
          f"dim={descriptor_config.fc_output_dim}, "
          f"resize={descriptor_config.resize_height}x{descriptor_config.resize_width}, "
          f"crop_top={descriptor_config.crop_top_ratio}, crop_bot={descriptor_config.crop_bottom_ratio}")

    descriptors, image_names, image_paths = load_descriptor_archive(args.database)
    print(f"  Database: {len(image_names)} images, descriptors shape={descriptors.shape}")

    # Check a database image resolution
    db_sample_path = Path(image_paths[0])
    if not db_sample_path.is_file():
        # Try resolving relative to repo
        marker = "/data/"
        idx = str(db_sample_path).find(marker)
        if idx >= 0:
            db_sample_path = REPO_ROOT / str(db_sample_path)[idx + 1:]
    if db_sample_path.is_file():
        db_img = Image.open(db_sample_path)
        print(f"  Database image sample: {db_img.size} mode={db_img.mode}")
        db_aspect = db_img.size[0] / db_img.size[1]
        print(f"  Database aspect ratio: {db_aspect:.3f}")
    else:
        print(f"  [warn] Cannot find sample DB image at {image_paths[0]}")
        db_aspect = None

    # Typical distance stats
    norms = np.linalg.norm(descriptors, axis=1)
    adj_dists = [np.linalg.norm(descriptors[i] - descriptors[i + 1])
                 for i in range(min(20, len(descriptors) - 1))]
    print(f"  Descriptor norms: min={norms.min():.4f} max={norms.max():.4f}")
    print(f"  Adjacent frame distances (first 20): "
          f"mean={np.mean(adj_dists):.4f} max={max(adj_dists):.4f}")

    # --- Load model ---
    print("\nLoading CosPlace model...")
    device = get_device()
    print(f"  Device: {device}")
    model = load_cosplace_model(None, descriptor_config, device)
    transform = make_cosplace_transform(descriptor_config)
    print("  Model loaded.")

    def encode_frame(pil_image: Image.Image) -> np.ndarray:
        img = pil_image.convert("RGB")
        img = crop_image(img, descriptor_config.crop_top_ratio,
                         descriptor_config.crop_bottom_ratio)
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            desc = model(tensor)
            desc = torch.nn.functional.normalize(desc, p=2, dim=1)
        return desc.cpu().numpy()[0].astype(np.float32)

    # --- Self-test with database image ---
    print("\n--- Self-test: encode DB image and match against DB ---")
    if db_sample_path.is_file():
        db_test_img = Image.open(db_sample_path).convert("RGB")
        db_test_desc = encode_frame(db_test_img)
        candidates = descriptor_distance_search(descriptors, db_test_desc, top_k=5)
        print(f"  DB image 0 self-match results:")
        for rank, (idx, dist) in enumerate(candidates, 1):
            step = int(Path(image_names[idx]).stem) if image_names[idx].endswith('.png') else '?'
            conf = 1.0 / (1.0 + dist)
            print(f"    [{rank}] step={step} dist={dist:.4f} conf={conf:.3f}")
        if candidates[0][1] < 0.01:
            print("  PASS: Self-matching works correctly")
        else:
            print("  FAIL: Self-matching distance too high — possible model/config issue")
    else:
        print("  SKIP: Cannot find DB image on disk")

    # --- Get frames ---
    if args.offline_image:
        print(f"\n--- Testing offline image: {args.offline_image} ---")
        img = Image.open(args.offline_image).convert("RGB")
        frames = [(np.array(img, dtype=np.uint8), None)]
        args.num_frames = 1
    else:
        print(f"\n--- Connecting to SDK at {args.sdk_url} ---")
        rover = EarthRoverInterface(base_url=args.sdk_url, timeout=5.0)
        if not rover.connect():
            print("FATAL: Cannot connect to SDK. Is `hypercorn main:app --reload` running?")
            return 1

        print(f"\nCapturing {args.num_frames} frames...")
        frames = []
        for i in range(args.num_frames):
            frame = rover.get_camera_frame()
            data = rover.get_data()
            if frame is None:
                print(f"  Frame {i}: FAILED (no frame)")
                continue
            frames.append((frame, data))
            print(f"  Frame {i}: shape={frame.shape} dtype={frame.dtype}")
            if i < args.num_frames - 1:
                time.sleep(0.5)

    if not frames:
        print("FATAL: No frames captured.")
        return 1

    # --- Analyze frames ---
    print("\n" + "=" * 60)
    print("FRAME ANALYSIS")
    print("=" * 60)

    for i, (frame, data) in enumerate(frames):
        print(f"\n--- Frame {i} ---")
        h, w = frame.shape[:2]
        live_aspect = w / h
        print(f"  Resolution: {w}x{h} (aspect={live_aspect:.3f})")

        if db_aspect is not None:
            if abs(live_aspect - db_aspect) > 0.05:
                print(f"  WARNING: Aspect ratio mismatch! "
                      f"Live={live_aspect:.3f} vs DB={db_aspect:.3f}")
                print(f"  This WILL cause poor descriptor matching.")
                print(f"  The crop+resize to 320x320 will produce different "
                      f"images for different aspect ratios.")
            else:
                print(f"  OK: Aspect ratio matches database ({live_aspect:.3f} vs {db_aspect:.3f})")

        # Encode and match
        pil = Image.fromarray(frame.astype(np.uint8), mode="RGB")
        desc = encode_frame(pil)
        candidates = descriptor_distance_search(descriptors, desc, top_k=10)

        print(f"  Top-10 raw matches (NO temporal filtering):")
        for rank, (idx, dist) in enumerate(candidates, 1):
            step = int(Path(image_names[idx]).stem)
            conf = 1.0 / (1.0 + dist)
            print(f"    [{rank:2d}] step={step:4d} dist={dist:.4f} conf={conf:.3f}")

        best_dist = candidates[0][1]
        if best_dist < 0.4:
            quality = "EXCELLENT"
        elif best_dist < 0.7:
            quality = "GOOD"
        elif best_dist < 1.0:
            quality = "MEDIOCRE — may struggle with temporal filtering"
        else:
            quality = "POOR — robot is likely NOT in the known corridor"

        print(f"  Match quality: {quality} (best_dist={best_dist:.4f})")

        # Check if top matches are spatially coherent
        top5_steps = [int(Path(image_names[idx]).stem) for idx, _ in candidates[:5]]
        step_spread = max(top5_steps) - min(top5_steps)
        print(f"  Top-5 step spread: {step_spread} "
              f"(steps: {top5_steps})")
        if step_spread > 200:
            print(f"  WARNING: Top matches are spread across {step_spread} steps — "
                  f"localization is confused")

        if data:
            heading = data.get("orientation")
            print(f"  Compass heading: {heading}")

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    # Summarize
    all_best_dists = []
    for frame, data in frames:
        pil = Image.fromarray(frame.astype(np.uint8), mode="RGB")
        desc = encode_frame(pil)
        candidates = descriptor_distance_search(descriptors, desc, top_k=1)
        all_best_dists.append(candidates[0][1])

    avg_best = np.mean(all_best_dists)
    if avg_best < 0.4:
        print("Localization looks HEALTHY. If the runtime still shows low confidence,")
        print("the issue is likely in the temporal filtering or heading penalty.")
    elif avg_best < 0.7:
        print("Localization is OKAY but not great. Check:")
        print("  1. Lighting conditions vs when corridor was recorded")
        print("  2. Camera angle/position on robot")
    elif avg_best < 1.0:
        print("Localization is WEAK. Check:")
        print("  1. Is the robot actually in the recorded corridor?")
        print("  2. Is the camera resolution/aspect different from database?")
        print("  3. Are lighting conditions very different?")
    else:
        print("Localization is BROKEN. The live images do not match the database at all.")
        print("Most likely causes:")
        print("  1. Robot is NOT in the recorded corridor")
        print("  2. Camera aspect ratio mismatch (check warnings above)")
        print("  3. Wrong camera (rear instead of front?)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
