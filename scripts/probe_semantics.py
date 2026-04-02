#!/usr/bin/env python3
"""Offline semantic probing on recorded outdoor .h5 frames.

Runs a lightweight SegFormer ADE20K model on selected frames from test_outdoor/*.h5,
prints semantic summaries for the forward region, and optionally saves overlays.

This is analysis only. It does not modify runtime behavior.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import sys
from typing import Iterable

import h5py
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import torch


DEFAULT_FRAMES = [
    ('test_outdoor_4.h5', 1696, 'person_dog_path'),
    ('test_outdoor_2.h5', 272, 'person_on_trail'),
    ('test_outdoor_4.h5', 848, 'bushes_both_sides'),
    ('test_outdoor_3.h5', 0, 'tree_left_close'),
    ('test_outdoor_1.h5', 0, 'open_baseline'),
]

DRIVABLE_LABELS = {'road', 'earth', 'path', 'sidewalk'}
OBSTACLE_LABELS = {'person', 'animal', 'pole', 'wall', 'tree', 'plant', 'grass'}
CAUTION_LABELS = {'grass', 'plant', 'tree'}

PALETTE = np.array([
    [0, 0, 0],
    [220, 20, 60],
    [70, 130, 180],
    [34, 139, 34],
    [160, 82, 45],
], dtype=np.uint8)


@dataclass
class FrameSpec:
    file_name: str
    frame_idx: int
    tag: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Offline semantic probing on outdoor recordings.')
    p.add_argument('--h5-dir', type=Path, default=REPO_ROOT / 'test_outdoor')
    p.add_argument('--model-id', default='nvidia/segformer-b0-finetuned-ade-512-512')
    p.add_argument('--device', choices=('cpu', 'cuda', 'auto'), default='auto')
    p.add_argument('--save-overlays', action='store_true')
    p.add_argument('--save-dir', type=Path, default=REPO_ROOT / 'scripts' / 'semantic_debug')
    p.add_argument(
        '--frames', nargs='*', default=None,
        help='Optional frame specs as file.h5:index:tag ; defaults to curated obstacle/open examples.'
    )
    return p.parse_args()


def parse_frame_specs(items: Iterable[str] | None) -> list[FrameSpec]:
    if not items:
        return [FrameSpec(*item) for item in DEFAULT_FRAMES]
    out: list[FrameSpec] = []
    for raw in items:
        parts = raw.split(':', 2)
        if len(parts) != 3:
            raise SystemExit(f'Invalid frame spec: {raw}. Expected file.h5:index:tag')
        out.append(FrameSpec(parts[0], int(parts[1]), parts[2]))
    return out


def decode_frame(raw: np.ndarray) -> np.ndarray:
    img = Image.open(BytesIO(bytes(raw)))
    return np.array(img.convert('RGB'), dtype=np.uint8)


def make_roi_mask(h: int, w: int) -> np.ndarray:
    top = int(h * 0.35)
    bottom = int(h * 0.90)
    left = int(w * 0.25)
    right = int(w * 0.75)
    mask = np.zeros((h, w), dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def summarize_labels(seg: np.ndarray, labels: dict[int, str], roi_mask: np.ndarray) -> dict:
    roi = seg[roi_mask]
    vals, counts = np.unique(roi, return_counts=True)
    total = int(counts.sum())
    pairs = sorted(((int(v), int(c)) for v, c in zip(vals, counts)), key=lambda x: x[1], reverse=True)
    top = []
    drivable = 0.0
    obstacle = 0.0
    caution = 0.0
    for idx, c in pairs:
        name = labels.get(idx, str(idx)).lower()
        frac = c / max(1, total)
        top.append((name, frac))
        if name in DRIVABLE_LABELS:
            drivable += frac
        if name in OBSTACLE_LABELS:
            obstacle += frac
        if name in CAUTION_LABELS:
            caution += frac
    return {
        'total_px': total,
        'top': top[:8],
        'drivable_frac': drivable,
        'obstacle_frac': obstacle,
        'caution_frac': caution,
    }


def overlay_semantics(rgb: np.ndarray, seg: np.ndarray, labels: dict[int, str], roi_mask: np.ndarray) -> np.ndarray:
    h, w = seg.shape
    mapped = np.zeros((h, w), dtype=np.uint8)
    for idx, name in labels.items():
        lname = name.lower()
        if lname in DRIVABLE_LABELS:
            mapped[seg == idx] = 2
        elif lname in CAUTION_LABELS:
            mapped[seg == idx] = 3
        elif lname in {'person', 'animal', 'pole', 'wall'}:
            mapped[seg == idx] = 1
        elif lname == 'sky':
            mapped[seg == idx] = 4
    color = PALETTE[mapped]
    blended = (0.55 * rgb + 0.45 * color).astype(np.uint8)
    blended[roi_mask] = (0.7 * blended[roi_mask] + 0.3 * np.array([255, 255, 255])).astype(np.uint8)
    return blended


def main() -> int:
    args = parse_args()
    frame_specs = parse_frame_specs(args.frames)

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f'Loading {args.model_id} on {device}...')
    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_id).to(device)
    model.eval()
    labels = {int(k): v for k, v in model.config.id2label.items()}

    if args.save_overlays:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    for spec in frame_specs:
        h5_path = args.h5_dir / spec.file_name
        if not h5_path.exists():
            print(f'[skip] missing {h5_path}')
            continue
        with h5py.File(h5_path, 'r') as f:
            raw = f['front_frames']['data'][spec.frame_idx]
        rgb = decode_frame(raw)
        pil = Image.fromarray(rgb)
        inputs = processor(images=pil, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        up = torch.nn.functional.interpolate(
            logits,
            size=rgb.shape[:2],
            mode='bilinear',
            align_corners=False,
        )
        seg = up.argmax(dim=1)[0].cpu().numpy().astype(np.int32)
        roi_mask = make_roi_mask(seg.shape[0], seg.shape[1])
        summary = summarize_labels(seg, labels, roi_mask)

        print(f'\n{spec.file_name} #{spec.frame_idx} ({spec.tag})')
        print(f"  forward drivable={summary['drivable_frac']:.2f} obstacle={summary['obstacle_frac']:.2f} caution={summary['caution_frac']:.2f}")
        print('  top labels:')
        for name, frac in summary['top']:
            print(f'    - {name}: {frac:.2%}')

        if args.save_overlays:
            out = overlay_semantics(rgb, seg, labels, roi_mask)
            Image.fromarray(out).save(args.save_dir / f'{Path(spec.file_name).stem}_{spec.frame_idx}_{spec.tag}.png')
            print(f'  saved overlay -> {args.save_dir / (Path(spec.file_name).stem + "_" + str(spec.frame_idx) + "_" + spec.tag + ".png")}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
