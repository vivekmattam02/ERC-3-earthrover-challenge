#!/usr/bin/env python3
"""Second offline semantic pass for ERC-3 outdoor data.

This script keeps the same SegFormer model used in the first probe but adds:
- multiple ROI variants
- off-road-friendly label grouping
- center-vs-side weighting
- a semantic risk score
- broader replay sampling over recorded .h5 files

This is analysis only. It does not modify runtime behavior.
"""

from __future__ import annotations

import argparse
import csv
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

DEFAULT_CURATED = [
    ("test_outdoor_4.h5", 1696, "person_dog_path"),
    ("test_outdoor_2.h5", 272, "person_on_trail"),
    ("test_outdoor_4.h5", 848, "bushes_both_sides"),
    ("test_outdoor_3.h5", 0, "tree_left_close"),
    ("test_outdoor_1.h5", 0, "open_baseline"),
]

ROI_VARIANTS = {
    "baseline": (0.35, 0.90, 0.25, 0.75),
    "mid_corridor": (0.35, 0.75, 0.30, 0.70),
    "narrow_corridor": (0.30, 0.75, 0.35, 0.65),
    "raised_corridor": (0.40, 0.80, 0.30, 0.70),
}

DRIVABLE_LABELS = {"road", "earth", "path", "sidewalk", "dirt track"}
NEUTRAL_LABELS = {"grass", "field"}
HAZARD_LABELS = {"person", "animal", "pole", "wall", "fence"}
CAUTION_LABELS = {"tree", "plant"}
IGNORE_LABELS = {"sky"}

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
class RoiSpec:
    name: str
    top_frac: float
    bottom_frac: float
    left_frac: float
    right_frac: float


@dataclass
class AnalysisResult:
    file_name: str
    frame_idx: int
    tag: str
    roi_name: str
    drivable_frac: float
    neutral_frac: float
    hazard_frac: float
    caution_frac: float
    drivable_center_frac: float
    caution_center_frac: float
    risk_score: float
    obstructed: bool
    vegetation_blocked: bool
    top_labels: list[tuple[str, float]]
    hard_alerts: list[str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Second offline semantic pass on outdoor recordings.")
    p.add_argument("--h5-dir", type=Path, default=REPO_ROOT / "test_outdoor")
    p.add_argument("--model-id", default="nvidia/segformer-b0-finetuned-ade-512-512")
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    p.add_argument("--sample-every", type=int, default=250, help="Sample every Nth frame for dataset replay.")
    p.add_argument("--max-sampled-frames", type=int, default=40, help="Cap sampled frames across all files.")
    p.add_argument("--risk-threshold", type=float, default=0.35)
    p.add_argument("--save-overlays", action="store_true")
    p.add_argument("--save-dir", type=Path, default=REPO_ROOT / "scripts" / "semantic_debug_v2")
    p.add_argument("--csv-out", type=Path, default=REPO_ROOT / "scripts" / "semantic_debug_v2" / "semantic_second_pass.csv")
    p.add_argument(
        "--frames",
        nargs="*",
        default=None,
        help="Optional frame specs as file.h5:index:tag. Defaults to curated obstacle/open examples.",
    )
    p.add_argument(
        "--rois",
        nargs="*",
        choices=sorted(ROI_VARIANTS.keys()),
        default=["baseline", "mid_corridor", "narrow_corridor", "raised_corridor"],
        help="ROI variants to evaluate.",
    )
    p.add_argument("--skip-replay", action="store_true", help="Only analyze curated frames.")
    return p.parse_args()


def parse_frame_specs(items: Iterable[str] | None) -> list[FrameSpec]:
    if not items:
        return [FrameSpec(*item) for item in DEFAULT_CURATED]
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


def build_roi_mask(h: int, w: int, spec: RoiSpec) -> tuple[np.ndarray, np.ndarray]:
    top = int(h * spec.top_frac)
    bottom = int(h * spec.bottom_frac)
    left = int(w * spec.left_frac)
    right = int(w * spec.right_frac)
    roi = np.zeros((h, w), dtype=bool)
    roi[top:bottom, left:right] = True

    center_left = left + int(0.25 * (right - left))
    center_right = right - int(0.25 * (right - left))
    center = np.zeros((h, w), dtype=bool)
    center[top:bottom, center_left:center_right] = True
    return roi, center


def label_name_map(model) -> dict[int, str]:
    return {int(k): v.lower() for k, v in model.config.id2label.items()}


def region_label_fracs(seg: np.ndarray, id2label: dict[int, str], mask: np.ndarray) -> dict[str, float]:
    vals = seg[mask]
    uniq, counts = np.unique(vals, return_counts=True)
    total = float(counts.sum()) if len(counts) else 1.0
    out: dict[str, float] = {}
    for idx, count in zip(uniq, counts):
        out[id2label.get(int(idx), str(int(idx)).lower())] = float(count) / total
    return out


def group_fractions(fracs: dict[str, float]) -> dict[str, float]:
    drivable = sum(v for k, v in fracs.items() if k in DRIVABLE_LABELS)
    neutral = sum(v for k, v in fracs.items() if k in NEUTRAL_LABELS)
    hazard = sum(v for k, v in fracs.items() if k in HAZARD_LABELS)
    caution = sum(v for k, v in fracs.items() if k in CAUTION_LABELS)
    ignore = sum(v for k, v in fracs.items() if k in IGNORE_LABELS)
    other = max(0.0, 1.0 - drivable - neutral - hazard - caution - ignore)
    return {
        "drivable": drivable,
        "neutral": neutral,
        "hazard": hazard,
        "caution": caution,
        "ignore": ignore,
        "other": other,
    }


def semantic_risk(full_fracs: dict[str, float], center_fracs: dict[str, float]) -> tuple[float, list[str], bool, float, float]:
    person = center_fracs.get("person", 0.0)
    animal = center_fracs.get("animal", 0.0)
    pole = center_fracs.get("pole", 0.0)
    wall = center_fracs.get("wall", 0.0) + center_fracs.get("fence", 0.0)
    drivable_center = sum(center_fracs.get(lbl, 0.0) for lbl in DRIVABLE_LABELS)
    caution_center = sum(center_fracs.get(lbl, 0.0) for lbl in CAUTION_LABELS)

    score = 0.0
    hard_alerts: list[str] = []

    if person > 0.002:
        score += 0.55 + 18.0 * person
        hard_alerts.append("person")
    if animal > 0.002:
        score += 0.55 + 18.0 * animal
        hard_alerts.append("animal")
    if pole > 0.002:
        score += 0.35 + 10.0 * pole
        hard_alerts.append("pole")
    if wall > 0.01:
        score += 0.30 + 8.0 * wall
        hard_alerts.append("wall")

    vegetation_blocked = drivable_center < 0.10 and caution_center > 0.60
    if vegetation_blocked:
        score += 0.45 + 0.50 * max(0.0, caution_center - 0.60)

    return max(0.0, score), hard_alerts, vegetation_blocked, drivable_center, caution_center


def top_labels(fracs: dict[str, float], limit: int = 8) -> list[tuple[str, float]]:
    return sorted(fracs.items(), key=lambda kv: kv[1], reverse=True)[:limit]


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


def overlay(rgb: np.ndarray, seg: np.ndarray, id2label: dict[int, str], roi: np.ndarray, center: np.ndarray) -> Image.Image:
    h, w = seg.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, label in id2label.items():
        color[seg == idx] = color_for_label(label)
    blended = (0.55 * rgb + 0.45 * color).astype(np.uint8)
    img = Image.fromarray(blended)
    draw = ImageDraw.Draw(img)

    ys, xs = np.where(roi)
    if len(xs):
        draw.rectangle([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], outline=(255, 255, 255), width=3)
    ys, xs = np.where(center)
    if len(xs):
        draw.rectangle([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], outline=(255, 215, 0), width=3)
    return img


def infer_segmentation(rgb: np.ndarray, processor, model, device: str) -> np.ndarray:
    pil = Image.fromarray(rgb)
    inputs = processor(images=pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    up = torch.nn.functional.interpolate(logits, size=rgb.shape[:2], mode="bilinear", align_corners=False)
    return up.argmax(dim=1)[0].cpu().numpy().astype(np.int32)


def analyze_one(rgb: np.ndarray, seg: np.ndarray, id2label: dict[int, str], frame: FrameSpec, roi_spec: RoiSpec, threshold: float) -> tuple[AnalysisResult, np.ndarray, np.ndarray]:
    h, w = seg.shape
    roi, center = build_roi_mask(h, w, roi_spec)
    full_fracs = region_label_fracs(seg, id2label, roi)
    center_fracs = region_label_fracs(seg, id2label, center)
    groups = group_fractions(full_fracs)
    score, alerts, vegetation_blocked, drivable_center, caution_center = semantic_risk(full_fracs, center_fracs)
    obstructed = bool(alerts) or vegetation_blocked or score >= threshold
    result = AnalysisResult(
        file_name=frame.file_name,
        frame_idx=frame.frame_idx,
        tag=frame.tag,
        roi_name=roi_spec.name,
        drivable_frac=groups["drivable"],
        neutral_frac=groups["neutral"],
        hazard_frac=groups["hazard"],
        caution_frac=groups["caution"],
        drivable_center_frac=drivable_center,
        caution_center_frac=caution_center,
        risk_score=score,
        obstructed=obstructed,
        vegetation_blocked=vegetation_blocked,
        top_labels=top_labels(full_fracs),
        hard_alerts=alerts,
    )
    return result, roi, center


def iter_sampled_frames(h5_dir: Path, every: int, max_frames: int) -> list[FrameSpec]:
    out: list[FrameSpec] = []
    for path in sorted(h5_dir.glob("test_outdoor_*.h5")):
        with h5py.File(path, "r") as f:
            n = len(f["front_frames"]["data"])
        for idx in range(0, n, max(1, every)):
            out.append(FrameSpec(path.name, idx, f"sample_{path.stem}_{idx}"))
            if len(out) >= max_frames:
                return out
    return out


def save_csv(path: Path, rows: list[AnalysisResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file_name", "frame_idx", "tag", "roi_name",
            "drivable_frac", "neutral_frac", "hazard_frac", "caution_frac",
            "drivable_center_frac", "caution_center_frac",
            "risk_score", "obstructed", "vegetation_blocked", "hard_alerts", "top_labels",
        ])
        for r in rows:
            writer.writerow([
                r.file_name, r.frame_idx, r.tag, r.roi_name,
                f"{r.drivable_frac:.4f}", f"{r.neutral_frac:.4f}", f"{r.hazard_frac:.4f}", f"{r.caution_frac:.4f}",
                f"{r.drivable_center_frac:.4f}", f"{r.caution_center_frac:.4f}",
                f"{r.risk_score:.4f}", int(r.obstructed), int(r.vegetation_blocked), ",".join(r.hard_alerts),
                " | ".join(f"{name}:{frac:.2%}" for name, frac in r.top_labels),
            ])


def print_curated_summary(rows: list[AnalysisResult]) -> None:
    print("\n=== Curated Frames by ROI ===")
    for r in rows:
        print(
            f"{r.file_name} #{r.frame_idx} ({r.tag}) roi={r.roi_name} "
            f"drv={r.drivable_frac:.2f} neu={r.neutral_frac:.2f} haz={r.hazard_frac:.2f} cat={r.caution_frac:.2f} "
            f"drv_ctr={r.drivable_center_frac:.2f} cat_ctr={r.caution_center_frac:.2f} "
            f"risk={r.risk_score:.2f} obstructed={int(r.obstructed)} veg_blk={int(r.vegetation_blocked)} alerts={','.join(r.hard_alerts) or '-'}"
        )


def print_replay_summary(rows: list[AnalysisResult], selected_roi: str) -> None:
    subset = [r for r in rows if r.roi_name == selected_roi]
    if not subset:
        return
    print(f"\n=== Replay Summary ({selected_roi}) ===")
    obstructed_rate = sum(1 for r in subset if r.obstructed) / len(subset)
    mean_risk = sum(r.risk_score for r in subset) / len(subset)
    print(f"frames={len(subset)} obstructed_rate={obstructed_rate:.2%} mean_risk={mean_risk:.2f}")
    top = sorted(subset, key=lambda r: r.risk_score, reverse=True)[:5]
    low = sorted(subset, key=lambda r: r.risk_score)[:5]
    print("  highest-risk frames:")
    for r in top:
        print(f"    - {r.file_name} #{r.frame_idx} tag={r.tag} risk={r.risk_score:.2f} alerts={','.join(r.hard_alerts) or '-'}")
    print("  lowest-risk frames:")
    for r in low:
        print(f"    - {r.file_name} #{r.frame_idx} tag={r.tag} risk={r.risk_score:.2f} alerts={','.join(r.hard_alerts) or '-'}")


def main() -> int:
    args = parse_args()
    frame_specs = parse_frame_specs(args.frames)
    roi_specs = [RoiSpec(name, *ROI_VARIANTS[name]) for name in args.rois]

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

    all_rows: list[AnalysisResult] = []
    all_frames = list(frame_specs)
    if not args.skip_replay:
        all_frames.extend(iter_sampled_frames(args.h5_dir, args.sample_every, args.max_sampled_frames))

    for frame in all_frames:
        h5_path = args.h5_dir / frame.file_name
        if not h5_path.exists():
            print(f"[skip] missing {h5_path}")
            continue
        with h5py.File(h5_path, "r") as f:
            raw = f["front_frames"]["data"][frame.frame_idx]
        rgb = decode_frame(raw)
        seg = infer_segmentation(rgb, processor, model, device)

        for roi_spec in roi_specs:
            result, roi_mask, center_mask = analyze_one(rgb, seg, id2label, frame, roi_spec, args.risk_threshold)
            all_rows.append(result)
            if args.save_overlays and frame in frame_specs:
                img = overlay(rgb, seg, id2label, roi_mask, center_mask)
                out_name = f"{Path(frame.file_name).stem}_{frame.frame_idx}_{frame.tag}_{roi_spec.name}.png"
                img.save(args.save_dir / out_name)

    curated_rows = [r for r in all_rows if any(r.file_name == f.file_name and r.frame_idx == f.frame_idx and r.tag == f.tag for f in frame_specs)]
    print_curated_summary(curated_rows)

    if not args.skip_replay:
        selected = "raised_corridor" if "raised_corridor" in args.rois else args.rois[0]
        print_replay_summary(all_rows, selected)

    save_csv(args.csv_out, all_rows)
    print(f"\nSaved CSV -> {args.csv_out}")
    if args.save_overlays:
        print(f"Saved curated overlays -> {args.save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
