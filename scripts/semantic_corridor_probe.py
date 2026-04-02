#!/usr/bin/env python3
"""Offline semantic corridor / trap probe for ERC-3 outdoor data.

This script answers a different question from the earlier semantic risk work:
not just "is there traversable content?" but "is there a wide, centered,
forward-safe corridor, or is this a narrow trap / edge-hugging channel?"

This is analysis only. It does not modify runtime behavior.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable
import sys

import h5py
import numpy as np
from PIL import Image, ImageDraw
import torch
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from semantic_risk_estimator import (  # type: ignore
    DRIVABLE_LABELS,
    NEUTRAL_LABELS,
    HAZARD_LABELS,
    CAUTION_LABELS,
    IGNORE_LABELS,
)

DEFAULT_FRAMES = [
    ("test_outdoor_4.h5", 1696, "person_dog_path"),
    ("test_outdoor_2.h5", 272, "person_on_trail"),
    ("test_outdoor_4.h5", 848, "bushes_both_sides"),
    ("test_outdoor_3.h5", 0, "tree_left_close"),
    ("test_outdoor_1.h5", 0, "open_baseline"),
    ("test_outdoor_1.h5", 1500, "tree_cluster_1500"),
    ("test_outdoor_1.h5", 1800, "tree_cluster_1800"),
    ("test_outdoor_1.h5", 2100, "tree_cluster_2100"),
]

PALETTE = {
    "ignore": (120, 120, 160),
    "hazard": (220, 20, 60),
    "drivable": (70, 130, 180),
    "caution": (34, 139, 34),
    "neutral": (160, 120, 50),
    "other": (0, 0, 0),
}


@dataclass
class FrameSpec:
    file_name: str
    frame_idx: int
    tag: str


@dataclass
class CorridorMetrics:
    traversable_area_frac: float
    center_cross_frac: float
    lower_center_cross_frac: float
    median_width_frac: float
    median_offset_frac: float
    edge_touch_frac: float
    left_boundary_frac: float
    right_boundary_frac: float
    trap_score: float
    trap_reasons: list[str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline semantic corridor / trap probe.")
    p.add_argument("--h5-dir", type=Path, default=REPO_ROOT / "test_outdoor")
    p.add_argument("--model-id", default="nvidia/segformer-b0-finetuned-ade-512-512")
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    p.add_argument("--save-overlays", action="store_true")
    p.add_argument("--save-dir", type=Path, default=REPO_ROOT / "scripts" / "semantic_corridor_debug")
    p.add_argument(
        "--frames",
        nargs="*",
        default=None,
        help="Optional frame specs as file.h5:index:tag. Defaults to curated frames.",
    )
    return p.parse_args()


def parse_frame_specs(items: Iterable[str] | None) -> list[FrameSpec]:
    if not items:
        return [FrameSpec(*item) for item in DEFAULT_FRAMES]
    out: list[FrameSpec] = []
    for raw in items:
        parts = raw.split(":", 2)
        if len(parts) != 3:
            raise SystemExit(f"Invalid frame spec: {raw}. Expected file.h5:index:tag")
        out.append(FrameSpec(parts[0], int(parts[1]), parts[2]))
    return out


def decode_frame(raw: np.ndarray) -> np.ndarray:
    img = Image.open(BytesIO(bytes(raw)))
    return np.array(img.convert("RGB"), dtype=np.uint8)


def build_masks(h: int, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    top = int(h * 0.40)
    bottom = int(h * 0.80)
    left = int(w * 0.30)
    right = int(w * 0.70)

    roi = np.zeros((h, w), dtype=bool)
    roi[top:bottom, left:right] = True

    mid = left + (right - left) // 2
    center_left = left + int(0.25 * (right - left))
    center_right = right - int(0.25 * (right - left))

    center = np.zeros((h, w), dtype=bool)
    center[top:bottom, center_left:center_right] = True

    left_band = np.zeros((h, w), dtype=bool)
    right_band = np.zeros((h, w), dtype=bool)
    band_w = max(1, int(0.20 * (right - left)))
    left_band[top:bottom, left:left + band_w] = True
    right_band[top:bottom, right - band_w:right] = True
    return roi, center, left_band, right_band, (top, bottom, left, right)


def label_name_map(model) -> dict[int, str]:
    return {int(k): v.lower() for k, v in model.config.id2label.items()}


def color_for_label(label: str) -> tuple[int, int, int]:
    if label in HAZARD_LABELS:
        return PALETTE["hazard"]
    if label in DRIVABLE_LABELS:
        return PALETTE["drivable"]
    if label in CAUTION_LABELS:
        return PALETTE["caution"]
    if label in NEUTRAL_LABELS:
        return PALETTE["neutral"]
    if label in IGNORE_LABELS:
        return PALETTE["ignore"]
    return PALETTE["other"]


def infer_segmentation(rgb: np.ndarray, processor, model, device: str) -> np.ndarray:
    pil = Image.fromarray(rgb)
    inputs = processor(images=pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    up = torch.nn.functional.interpolate(logits, size=rgb.shape[:2], mode="bilinear", align_corners=False)
    return up.argmax(dim=1)[0].cpu().numpy().astype(np.int32)


def region_label_fracs(seg: np.ndarray, id2label: dict[int, str], mask: np.ndarray) -> dict[str, float]:
    vals = seg[mask]
    uniq, counts = np.unique(vals, return_counts=True)
    total = float(counts.sum()) if len(counts) else 1.0
    out: dict[str, float] = {}
    for idx, count in zip(uniq, counts):
        out[id2label.get(int(idx), str(int(idx)).lower())] = float(count) / total
    return out


def find_runs(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(mask_1d.tolist()):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask_1d) - 1))
    return runs


def corridor_metrics(seg: np.ndarray, id2label: dict[int, str]) -> CorridorMetrics:
    h, w = seg.shape
    roi, center, left_band, right_band, (top, bottom, left, right) = build_masks(h, w)
    roi_w = right - left

    traversable_ids = {idx for idx, name in id2label.items() if name in DRIVABLE_LABELS or name in NEUTRAL_LABELS}
    boundary_ids = {idx for idx, name in id2label.items() if name in HAZARD_LABELS or name in CAUTION_LABELS}

    rows_total = max(1, bottom - top)
    lower_start = top + int(0.50 * (bottom - top))
    lower_rows_total = max(1, bottom - lower_start)

    width_fracs: list[float] = []
    offset_fracs: list[float] = []
    edge_touch_count = 0
    center_cross_count = 0
    lower_center_cross_count = 0
    traversable_px = 0

    for y in range(top, bottom):
        row = seg[y, left:right]
        mask = np.isin(row, list(traversable_ids))
        traversable_px += int(mask.sum())
        runs = find_runs(mask)
        if not runs:
            continue

        roi_center_x = roi_w / 2.0
        crossing = [r for r in runs if r[0] <= roi_center_x <= r[1]]
        if crossing:
            best = max(crossing, key=lambda r: (r[1] - r[0] + 1))
            center_cross_count += 1
            if y >= lower_start:
                lower_center_cross_count += 1
        else:
            best = max(runs, key=lambda r: (r[1] - r[0] + 1))

        width = best[1] - best[0] + 1
        center_x = 0.5 * (best[0] + best[1])
        width_fracs.append(width / roi_w)
        offset_fracs.append(abs(center_x - roi_center_x) / max(1.0, roi_w / 2.0))
        if best[0] == 0 or best[1] == roi_w - 1:
            edge_touch_count += 1

    traversable_area_frac = traversable_px / max(1, rows_total * roi_w)
    center_cross_frac = center_cross_count / rows_total
    lower_center_cross_frac = lower_center_cross_count / lower_rows_total
    median_width_frac = float(np.median(width_fracs)) if width_fracs else 0.0
    median_offset_frac = float(np.median(offset_fracs)) if offset_fracs else 1.0
    edge_touch_frac = edge_touch_count / max(1, len(width_fracs))

    left_boundary = region_label_fracs(seg, id2label, left_band)
    right_boundary = region_label_fracs(seg, id2label, right_band)
    left_boundary_frac = sum(v for k, v in left_boundary.items() if k in CAUTION_LABELS or k in HAZARD_LABELS)
    right_boundary_frac = sum(v for k, v in right_boundary.items() if k in CAUTION_LABELS or k in HAZARD_LABELS)

    trap_score = 0.0
    reasons: list[str] = []
    if traversable_area_frac < 0.08:
        trap_score += 0.50
        reasons.append("no_traversable_area")
    if median_width_frac < 0.18:
        trap_score += 0.35
        reasons.append("too_narrow")
    if center_cross_frac < 0.25 and median_width_frac < 0.30:
        trap_score += 0.25
        reasons.append("no_center_corridor")
    if median_offset_frac > 0.45 and median_width_frac < 0.35:
        trap_score += 0.25
        reasons.append("side_channel")
    if edge_touch_frac > 0.50 and median_width_frac < 0.35:
        trap_score += 0.20
        reasons.append("edge_hugging")
    if left_boundary_frac > 0.45 and right_boundary_frac > 0.45 and median_width_frac < 0.30:
        trap_score += 0.25
        reasons.append("squeezed_between_boundaries")
    if lower_center_cross_frac < 0.20 and traversable_area_frac < 0.20:
        trap_score += 0.20
        reasons.append("no_lower_forward_opening")

    return CorridorMetrics(
        traversable_area_frac=traversable_area_frac,
        center_cross_frac=center_cross_frac,
        lower_center_cross_frac=lower_center_cross_frac,
        median_width_frac=median_width_frac,
        median_offset_frac=median_offset_frac,
        edge_touch_frac=edge_touch_frac,
        left_boundary_frac=left_boundary_frac,
        right_boundary_frac=right_boundary_frac,
        trap_score=min(1.0, trap_score),
        trap_reasons=reasons,
    )


def overlay(rgb: np.ndarray, seg: np.ndarray, id2label: dict[int, str], metrics: CorridorMetrics) -> Image.Image:
    h, w = seg.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, label in id2label.items():
        color[seg == idx] = color_for_label(label)
    blended = (0.55 * rgb + 0.45 * color).astype(np.uint8)
    img = Image.fromarray(blended)
    draw = ImageDraw.Draw(img)

    roi, center, left_band, right_band, (top, bottom, left, right) = build_masks(h, w)
    draw.rectangle([left, top, right, bottom], outline=(255, 255, 255), width=3)

    ys, xs = np.where(center)
    if len(xs):
        draw.rectangle([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], outline=(255, 215, 0), width=3)

    ys, xs = np.where(left_band)
    if len(xs):
        draw.rectangle([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], outline=(255, 120, 120), width=2)
    ys, xs = np.where(right_band)
    if len(xs):
        draw.rectangle([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], outline=(120, 255, 255), width=2)

    text = (
        f"trap={metrics.trap_score:.2f} width={metrics.median_width_frac:.2f} "
        f"offset={metrics.median_offset_frac:.2f} center={metrics.center_cross_frac:.2f}"
    )
    draw.rectangle([10, 10, 10 + 9 * len(text), 34], fill=(0, 0, 0))
    draw.text((14, 14), text, fill=(255, 255, 255))
    return img


def main() -> int:
    args = parse_args()
    frames = parse_frame_specs(args.frames)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Loading {args.model_id} on {device}...")
    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_id).to(device)
    model.eval()
    id2label = label_name_map(model)

    if args.save_overlays:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Corridor Trap Probe ===")
    for spec in frames:
        h5_path = args.h5_dir / spec.file_name
        if not h5_path.exists():
            print(f"[skip] missing {h5_path}")
            continue
        with h5py.File(h5_path, "r") as f:
            raw = f["front_frames"]["data"][spec.frame_idx]
        rgb = decode_frame(raw)
        seg = infer_segmentation(rgb, processor, model, device)
        metrics = corridor_metrics(seg, id2label)

        print(
            f"{spec.file_name} #{spec.frame_idx} ({spec.tag}) "
            f"trap={metrics.trap_score:.2f} width={metrics.median_width_frac:.2f} "
            f"offset={metrics.median_offset_frac:.2f} center={metrics.center_cross_frac:.2f} "
            f"lower_center={metrics.lower_center_cross_frac:.2f} edge={metrics.edge_touch_frac:.2f} "
            f"leftB={metrics.left_boundary_frac:.2f} rightB={metrics.right_boundary_frac:.2f} "
            f"reasons={','.join(metrics.trap_reasons) or '-'}"
        )

        if args.save_overlays:
            img = overlay(rgb, seg, id2label, metrics)
            out = args.save_dir / f"{Path(spec.file_name).stem}_{spec.frame_idx}_{spec.tag}.png"
            img.save(out)
            print(f"  saved -> {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
