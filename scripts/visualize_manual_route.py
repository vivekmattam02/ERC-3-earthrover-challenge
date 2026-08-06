#!/usr/bin/env python3
"""Generate quick visual summaries for a prepared manual route package."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a prepared manual route package.")
    parser.add_argument("--route-dir", type=Path, required=True, help="Route directory created by prepare_manual_route.py.")
    parser.add_argument("--num-samples", type=int, default=12, help="Number of evenly spaced thumbnails to show.")
    parser.add_argument("--thumb-width", type=int, default=320, help="Thumbnail width in pixels.")
    parser.add_argument("--columns", type=int, default=3, help="Number of columns in the contact sheet.")
    return parser.parse_args()


def load_json(path: Path) -> dict | list:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sample_indices(total: int, num_samples: int) -> list[int]:
    if total <= 0:
        return []
    if total <= num_samples:
        return list(range(total))
    return sorted({round(i * (total - 1) / (num_samples - 1)) for i in range(num_samples)})


def fit_thumbnail(image: Image.Image, target_width: int) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        return image
    scale = target_width / float(width)
    target_height = max(1, int(round(height * scale)))
    return image.resize((target_width, target_height))


def build_contact_sheet(route_dir: Path, num_samples: int, thumb_width: int, columns: int) -> Path:
    images_dir = route_dir / "extracted" / "front_images"
    data_info_path = route_dir / "extracted" / "metadata" / "data_info.json"
    summary_path = route_dir / "extracted" / "metadata" / "summary.json"
    route_info_path = route_dir / "route_info.json"

    image_paths = sorted(images_dir.glob("*"))
    data_info = load_json(data_info_path)
    summary = load_json(summary_path)
    route_info = load_json(route_info_path)
    indices = sample_indices(len(image_paths), num_samples)
    if not indices:
        raise SystemExit(f"No route images found in {images_dir}")

    font = ImageFont.load_default()
    thumb_pad = 18
    label_h = 42

    thumbs: list[tuple[Image.Image, str]] = []
    for idx in indices:
        image_path = image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = fit_thumbnail(image, thumb_width)
        entry = data_info[idx]
        label = f"step {entry['step']} | t={entry['relative_time_sec']:.1f}s"
        thumbs.append((image, label))

    max_h = max(img.height for img, _ in thumbs)
    rows = math.ceil(len(thumbs) / max(1, columns))
    sheet_w = columns * (thumb_width + thumb_pad) + thumb_pad
    header_h = 110
    row_h = max_h + label_h + thumb_pad
    sheet_h = header_h + rows * row_h + thumb_pad

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(245, 245, 245))
    draw = ImageDraw.Draw(sheet)

    route_name = route_info["route_name"]
    target_step = route_info["target_step"]
    trim_start = summary.get("trim_start_sec")
    trim_end = summary.get("trim_end_sec")
    stats = (
        f"{route_name} | target_step={target_step} | frames={summary['num_front_frames_kept']} | "
        f"trim={trim_start:.1f}s..{trim_end:.1f}s | frame_step={summary['frame_step']}"
    )
    draw.text((thumb_pad, 14), "Manual Route Overview", fill=(20, 20, 20), font=font)
    draw.text((thumb_pad, 38), stats, fill=(60, 60, 60), font=font)
    draw.text((thumb_pad, 62), f"Route dir: {route_dir}", fill=(90, 90, 90), font=font)

    for i, (image, label) in enumerate(thumbs):
        row = i // columns
        col = i % columns
        x = thumb_pad + col * (thumb_width + thumb_pad)
        y = header_h + row * row_h
        sheet.paste(image, (x, y))
        draw.rectangle((x, y, x + image.width - 1, y + image.height - 1), outline=(120, 120, 120), width=1)
        draw.text((x, y + image.height + 8), label, fill=(25, 25, 25), font=font)

    overview_dir = route_dir / "overview"
    overview_dir.mkdir(parents=True, exist_ok=True)
    output_path = overview_dir / "contact_sheet.jpg"
    sheet.save(output_path, quality=92)
    return output_path


def main() -> int:
    args = parse_args()
    route_dir = args.route_dir.expanduser().resolve()
    if not route_dir.is_dir():
        raise SystemExit(f"Route directory not found: {route_dir}")
    output = build_contact_sheet(route_dir, args.num_samples, args.thumb_width, args.columns)
    print(f"Saved contact sheet: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
