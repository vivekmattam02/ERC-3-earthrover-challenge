#!/usr/bin/env python3
"""Pre-localize indoor checkpoint images against the corridor database.

Given a folder of checkpoint images (the target photos for the competition),
this script:
1. Encodes each with CosPlace (same model as the runtime localizer)
2. Finds the best-matching corridor graph step for each
3. Saves a JSON mapping: checkpoint_image -> graph_step + descriptor

Use the output JSON with live_indoor_runtime.py to:
- Set exact --checkpoint-steps
- Enable direct visual checkpoint verification at runtime
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from corridor_localizer import CorridorLocalizer, CorridorLocalizerConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-localize checkpoint images against corridor DB.")
    p.add_argument(
        "checkpoint_dir",
        type=Path,
        help="Directory containing checkpoint images (PNG/JPG).",
    )
    p.add_argument(
        "--database",
        type=Path,
        default=REPO_ROOT / "data" / "corrider_db" / "descriptors.npz",
    )
    p.add_argument(
        "--data-info-json",
        type=Path,
        default=REPO_ROOT / "data" / "corrider_extracted" / "metadata" / "data_info.json",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "checkpoint_mapping.json",
        help="Output JSON with checkpoint -> step mapping + descriptors.",
    )
    p.add_argument(
        "--top-k", type=int, default=5,
        help="Top-K candidates to show per checkpoint.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.checkpoint_dir.is_dir():
        print(f"Error: {args.checkpoint_dir} is not a directory")
        return 1

    # Collect checkpoint images
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    checkpoint_images = sorted(
        p for p in args.checkpoint_dir.iterdir()
        if p.suffix.lower() in image_extensions
    )
    if not checkpoint_images:
        print(f"No images found in {args.checkpoint_dir}")
        return 1

    print(f"Found {len(checkpoint_images)} checkpoint images")

    # Initialize localizer
    print("Loading corridor localizer...")
    localizer = CorridorLocalizer(CorridorLocalizerConfig(
        database_npz=args.database,
        data_info_json=args.data_info_json,
        top_k=args.top_k,
    ))

    results = []
    descriptor_map = {}

    for img_path in checkpoint_images:
        print(f"\nProcessing: {img_path.name}")

        # Encode and localize
        result = localizer.localize_image_path(img_path)

        step = result["node_step"]
        conf = result["confidence"]
        candidates = result.get("candidates", [])

        print(f"  Best match: step={step}, confidence={conf:.3f}")
        print(f"  Top candidates:")
        for c in candidates[:args.top_k]:
            print(f"    step={c.get('step')} dist={c.get('distance', 0):.4f} image={c.get('image_name')}")

        # Also store the raw descriptor for runtime visual matching
        from PIL import Image
        pil_img = Image.open(img_path).convert("RGB")
        descriptor = localizer.encode_pil(pil_img)

        results.append({
            "checkpoint_image": img_path.name,
            "checkpoint_path": str(img_path),
            "matched_step": step,
            "confidence": float(conf),
            "top_candidates": [
                {"step": c.get("step"), "distance": float(c.get("distance", 0))}
                for c in candidates[:args.top_k]
            ],
        })

        # Store descriptor as list for JSON serialization
        descriptor_map[img_path.name] = descriptor.tolist()

        # Reset temporal state between checkpoints (each is independent)
        localizer.reset()

    # Sort by matched step to get sequential order
    results.sort(key=lambda r: r["matched_step"] or 0)

    # Build summary
    steps = [r["matched_step"] for r in results if r["matched_step"] is not None]
    print("\n" + "=" * 60)
    print("CHECKPOINT MAPPING SUMMARY")
    print("=" * 60)
    for i, r in enumerate(results):
        print(f"  CP {i+1}: {r['checkpoint_image']} -> step {r['matched_step']} (conf={r['confidence']:.3f})")
    print(f"\n  --checkpoint-steps {' '.join(str(s) for s in steps)}")
    print("=" * 60)

    # Save output
    output = {
        "checkpoints": results,
        "checkpoint_steps": steps,
        "descriptors": descriptor_map,
        "database": str(args.database),
        "num_db_images": len(localizer.image_names),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")

    # Also save descriptors as numpy for fast runtime loading
    npz_path = args.output.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        names=np.array(list(descriptor_map.keys())),
        descriptors=np.array(list(descriptor_map.values()), dtype=np.float32),
        steps=np.array(steps, dtype=np.int32),
    )
    print(f"Saved descriptors to {npz_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
