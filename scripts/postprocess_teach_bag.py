#!/usr/bin/env python3
"""Post-process a teach bag into a cleaner route-reference artifact.

This tool is meant for the current no-GPS teach-and-repeat workflow. It:
1. scores front-camera frames for route quality
2. detects the best contiguous teach window
3. exports metrics, summaries, and contact sheets
4. optionally writes a cleaned H5 bag for downstream route preparation

The goal is not to invent new autonomy. The goal is to produce a better
reference traversal so the existing route-localization stack sees cleaner,
more repeatable visual evidence.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process a no-GPS teach bag.")
    parser.add_argument("--input-h5", type=Path, required=True, help="Source H5 recording.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for metrics, summaries, and cleaned bag.")
    parser.add_argument("--brightness-min", type=float, default=28.0, help="Hard minimum grayscale brightness below which frames are rejected.")
    parser.add_argument("--tilt-max-deg", type=float, default=22.0, help="Hard maximum tilt angle for teach frames.")
    parser.add_argument("--rpm-motion-threshold", type=float, default=1.0, help="Motion threshold from summed wheel RPMs.")
    parser.add_argument("--score-threshold", type=float, default=0.48, help="Smoothed quality threshold for candidate teach windows.")
    parser.add_argument("--smooth-window", type=int, default=5, help="Smoothing window over frame quality scores.")
    parser.add_argument("--min-window-sec", type=float, default=20.0, help="Minimum duration for a selected contiguous teach window.")
    parser.add_argument(
        "--min-episode-sec",
        type=float,
        default=4.0,
        help="Minimum duration for an individual selected traversal episode.",
    )
    parser.add_argument(
        "--motion-support-sec",
        type=float,
        default=2.0,
        help="Seconds of dilation around moving frames when building traversal episodes.",
    )
    parser.add_argument(
        "--max-stationary-gap-sec",
        type=float,
        default=3.0,
        help="Fill stationary gaps shorter than this when selecting a traversal window.",
    )
    parser.add_argument(
        "--min-motion-fraction",
        type=float,
        default=0.35,
        help="Minimum moving-frame fraction for a preferred teach window.",
    )
    parser.add_argument("--pad-sec", type=float, default=1.5, help="Pad selected window before/after the chosen run.")
    parser.add_argument("--num-samples", type=int, default=12, help="Number of sample thumbnails for contact sheets.")
    parser.add_argument("--write-cleaned-h5", action="store_true", help="Write a cleaned H5 sliced to the selected teach window.")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_timestamps(timestamps: np.ndarray) -> np.ndarray:
    if len(timestamps) == 0:
        return np.array([], dtype=np.float64)
    base = float(timestamps[0])
    return timestamps.astype(np.float64) - base


def nearest_index(timestamps: np.ndarray, query_ts: float) -> int | None:
    if len(timestamps) == 0:
        return None
    idx = int(np.searchsorted(timestamps, query_ts, side="left"))
    if idx <= 0:
        return 0
    if idx >= len(timestamps):
        return len(timestamps) - 1
    left = idx - 1
    right = idx
    if abs(float(timestamps[left]) - query_ts) <= abs(float(timestamps[right]) - query_ts):
        return left
    return right


def slice_by_time_field(dataset: np.ndarray, field_name: str, start_sec: float | None, end_sec: float | None) -> np.ndarray:
    if len(dataset) == 0:
        return dataset
    values = normalize_timestamps(dataset[field_name].astype(np.float64))
    mask = np.ones(len(dataset), dtype=bool)
    if start_sec is not None:
        mask &= values >= float(start_sec)
    if end_sec is not None:
        mask &= values <= float(end_sec)
    return dataset[mask]


def slice_frames_by_time(
    frame_bytes: np.ndarray,
    frame_timestamps: np.ndarray,
    start_sec: float | None,
    end_sec: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    if len(frame_timestamps) == 0:
        return frame_bytes, frame_timestamps
    values = normalize_timestamps(frame_timestamps.astype(np.float64))
    mask = np.ones(len(frame_timestamps), dtype=bool)
    if start_sec is not None:
        mask &= values >= float(start_sec)
    if end_sec is not None:
        mask &= values <= float(end_sec)
    return frame_bytes[mask], frame_timestamps[mask]


def decode_image(frame_item: Any) -> Image.Image:
    payload = bytes(np.asarray(frame_item, dtype=np.uint8))
    return Image.open(io.BytesIO(payload)).convert("RGB")


def gradient_energy(gray: np.ndarray) -> float:
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    return float(np.mean(np.abs(gx)) + np.mean(np.abs(gy)))


def sharpness_score_raw(gray: np.ndarray) -> float:
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    return float(np.var(gx) + np.var(gy))


def frame_novelty_raw(previous_gray: np.ndarray | None, current_gray: np.ndarray) -> float:
    if previous_gray is None or previous_gray.shape != current_gray.shape:
        return 0.0
    return float(np.mean(np.abs(current_gray - previous_gray)))


def accel_to_tilt_deg(accel_row: np.void | None) -> tuple[float | None, float | None, float | None]:
    if accel_row is None:
        return None, None, None
    ax = float(accel_row["x"])
    ay = float(accel_row["y"])
    az = float(accel_row["z"])
    roll_deg = math.degrees(math.atan2(ay, az))
    pitch_deg = math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az)))
    tilt_deg = max(abs(roll_deg), abs(pitch_deg))
    return roll_deg, pitch_deg, tilt_deg


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def sample_indices(total: int, num_samples: int) -> list[int]:
    if total <= 0:
        return []
    if total <= num_samples:
        return list(range(total))
    return sorted({round(i * (total - 1) / (num_samples - 1)) for i in range(num_samples)})


def smooth_scores(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0 or window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="same")


def dilate_mask_by_time(mask: np.ndarray, rel_times: np.ndarray, radius_sec: float) -> np.ndarray:
    if len(mask) == 0 or radius_sec <= 0.0 or not np.any(mask):
        return mask.copy()
    true_times = rel_times[mask]
    result = np.zeros_like(mask, dtype=bool)
    left = 0
    right = 0
    for idx, current_time in enumerate(rel_times):
        while left < len(true_times) and float(true_times[left]) < float(current_time) - radius_sec:
            left += 1
        while right < len(true_times) and float(true_times[right]) <= float(current_time) + radius_sec:
            right += 1
        result[idx] = left < right
    return result


def bridge_short_false_gaps(mask: np.ndarray, rel_times: np.ndarray, max_gap_sec: float) -> np.ndarray:
    if len(mask) == 0 or max_gap_sec <= 0.0:
        return mask.copy()
    result = mask.copy()
    idx = 0
    while idx < len(mask):
        if result[idx]:
            idx += 1
            continue
        gap_start = idx
        while idx < len(mask) and not result[idx]:
            idx += 1
        gap_end = idx - 1
        prev_idx = gap_start - 1
        next_idx = idx
        if prev_idx < 0 or next_idx >= len(mask):
            continue
        gap_duration = float(rel_times[next_idx] - rel_times[prev_idx])
        if gap_duration <= max_gap_sec:
            result[gap_start : gap_end + 1] = True
    return result


def summarize_runs(mask: np.ndarray, rel_times: np.ndarray) -> list[dict[str, float | int]]:
    runs: list[dict[str, float | int]] = []
    start_idx: int | None = None
    for idx, keep in enumerate(mask):
        if keep and start_idx is None:
            start_idx = idx
        elif not keep and start_idx is not None:
            end_idx = idx - 1
            runs.append(
                {
                    "start_index": start_idx,
                    "end_index": end_idx,
                    "start_sec": float(rel_times[start_idx]),
                    "end_sec": float(rel_times[end_idx]),
                    "duration_sec": float(rel_times[end_idx] - rel_times[start_idx]),
                }
            )
            start_idx = None
    if start_idx is not None:
        end_idx = len(mask) - 1
        runs.append(
            {
                "start_index": start_idx,
                "end_index": end_idx,
                "start_sec": float(rel_times[start_idx]),
                "end_sec": float(rel_times[end_idx]),
                "duration_sec": float(rel_times[end_idx] - rel_times[start_idx]),
            }
        )
    return runs


def pick_best_run(
    keep_mask: np.ndarray,
    rel_times: np.ndarray,
    smoothed_scores: np.ndarray,
    motion_mask: np.ndarray,
    novelty_scores: np.ndarray,
    min_window_sec: float,
    min_motion_fraction: float,
) -> dict[str, float | int] | None:
    runs = summarize_runs(keep_mask, rel_times)
    if not runs:
        return None

    best_run: dict[str, float | int] | None = None
    best_value = -1.0
    for run in runs:
        start_idx = int(run["start_index"])
        end_idx = int(run["end_index"])
        duration = float(run["duration_sec"])
        mean_score = float(np.mean(smoothed_scores[start_idx : end_idx + 1]))
        motion_fraction = float(np.mean(motion_mask[start_idx : end_idx + 1]))
        novelty_mean = float(np.mean(novelty_scores[start_idx : end_idx + 1]))
        run["mean_score"] = mean_score
        run["motion_fraction"] = motion_fraction
        run["novelty_mean"] = novelty_mean
        if duration < min_window_sec:
            continue
        objective = duration * (0.25 + 0.75 * mean_score) * (0.15 + 0.85 * motion_fraction) * (0.20 + 0.80 * novelty_mean)
        run["objective"] = objective
        if motion_fraction < min_motion_fraction:
            continue
        if objective > best_value:
            best_value = objective
            best_run = run

    if best_run is not None:
        return best_run

    # Fallback: prefer motion-rich runs even if they miss the preferred minimums.
    for run in runs:
        start_idx = int(run["start_index"])
        end_idx = int(run["end_index"])
        duration = float(run["duration_sec"])
        mean_score = float(np.mean(smoothed_scores[start_idx : end_idx + 1]))
        motion_fraction = float(np.mean(motion_mask[start_idx : end_idx + 1]))
        novelty_mean = float(np.mean(novelty_scores[start_idx : end_idx + 1]))
        run["mean_score"] = mean_score
        run["motion_fraction"] = motion_fraction
        run["novelty_mean"] = novelty_mean
        run["objective"] = duration * (0.20 + 0.80 * mean_score) * (0.10 + 0.90 * motion_fraction) * (0.15 + 0.85 * novelty_mean)
    runs.sort(key=lambda run: (float(run.get("objective", 0.0)), float(run["duration_sec"])), reverse=True)
    return runs[0]


def score_candidate_runs(
    keep_mask: np.ndarray,
    rel_times: np.ndarray,
    smoothed_scores: np.ndarray,
    motion_mask: np.ndarray,
    novelty_scores: np.ndarray,
) -> list[dict[str, float | int]]:
    runs = summarize_runs(keep_mask, rel_times)
    scored: list[dict[str, float | int]] = []
    for run in runs:
        start_idx = int(run["start_index"])
        end_idx = int(run["end_index"])
        duration = float(run["duration_sec"])
        mean_score = float(np.mean(smoothed_scores[start_idx : end_idx + 1]))
        motion_fraction = float(np.mean(motion_mask[start_idx : end_idx + 1]))
        novelty_mean = float(np.mean(novelty_scores[start_idx : end_idx + 1]))
        objective = duration * (0.25 + 0.75 * mean_score) * (0.15 + 0.85 * motion_fraction) * (0.20 + 0.80 * novelty_mean)
        run["mean_score"] = mean_score
        run["motion_fraction"] = motion_fraction
        run["novelty_mean"] = novelty_mean
        run["objective"] = objective
        scored.append(run)
    scored.sort(key=lambda run: float(run["objective"]), reverse=True)
    return scored


def write_metrics_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_contact_sheet(
    title: str,
    frames: list[tuple[Image.Image, str]],
    output_path: Path,
    columns: int = 3,
    thumb_width: int = 320,
) -> None:
    if not frames:
        return
    font = ImageFont.load_default()
    pad = 18
    label_h = 42

    thumbs: list[tuple[Image.Image, str]] = []
    for image, label in frames:
        width, height = image.size
        scale = thumb_width / float(width)
        resized = image.resize((thumb_width, max(1, int(round(height * scale)))))
        thumbs.append((resized, label))

    rows = math.ceil(len(thumbs) / columns)
    max_h = max(img.height for img, _ in thumbs)
    sheet_w = columns * (thumb_width + pad) + pad
    header_h = 60
    row_h = max_h + label_h + pad
    sheet_h = header_h + rows * row_h + pad

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 14), title, fill=(20, 20, 20), font=font)

    for i, (image, label) in enumerate(thumbs):
        row = i // columns
        col = i % columns
        x = pad + col * (thumb_width + pad)
        y = header_h + row * row_h
        sheet.paste(image, (x, y))
        draw.rectangle((x, y, x + image.width - 1, y + image.height - 1), outline=(120, 120, 120), width=1)
        draw.text((x, y + image.height + 8), label, fill=(25, 25, 25), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def to_vlen_uint8_array(frames: np.ndarray) -> np.ndarray:
    result = np.empty(len(frames), dtype=object)
    for idx, item in enumerate(frames):
        result[idx] = np.frombuffer(bytes(np.asarray(item, dtype=np.uint8)), dtype=np.uint8)
    return result


def write_cleaned_h5(
    input_h5: Path,
    output_h5: Path,
    start_sec: float,
    end_sec: float,
    front_keep_mask_within_window: np.ndarray,
) -> None:
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_h5, "r") as src:
        accels = slice_by_time_field(src["accels"][:], "t", start_sec, end_sec)
        controls = slice_by_time_field(src["controls"][:], "timestamp", start_sec, end_sec)
        gyros = slice_by_time_field(src["gyros"][:], "t", start_sec, end_sec)
        mags = slice_by_time_field(src["mags"][:], "t", start_sec, end_sec)
        rpms = slice_by_time_field(src["rpms"][:], "t", start_sec, end_sec)
        telemetry = slice_by_time_field(src["telemetry"][:], "timestamp", start_sec, end_sec)
        front_data, front_ts = slice_frames_by_time(src["front_frames/data"][:], src["front_frames/timestamps"][:], start_sec, end_sec)

        if len(front_keep_mask_within_window) != len(front_ts):
            raise ValueError("Front-frame keep mask does not match trimmed front-frame count")

        front_data = front_data[front_keep_mask_within_window]
        front_ts = front_ts[front_keep_mask_within_window]

        rear_data = np.array([], dtype=object)
        rear_ts = np.array([], dtype=np.float64)
        if "rear_frames" in src:
            rear_data, rear_ts = slice_frames_by_time(src["rear_frames/data"][:], src["rear_frames/timestamps"][:], start_sec, end_sec)

        attrs = {key: src.attrs[key] for key in src.attrs.keys()}

    with h5py.File(output_h5, "w") as dst:
        dst.create_dataset("telemetry", data=np.array(telemetry))
        dst.create_dataset("controls", data=np.array(controls))
        dst.create_dataset("accels", data=np.array(accels))
        dst.create_dataset("gyros", data=np.array(gyros))
        dst.create_dataset("mags", data=np.array(mags))
        dst.create_dataset("rpms", data=np.array(rpms))

        vlen_bytes = h5py.vlen_dtype(np.dtype("uint8"))
        front_group = dst.create_group("front_frames")
        front_group.create_dataset(
            "data",
            data=to_vlen_uint8_array(front_data),
            dtype=vlen_bytes,
        )
        front_group.create_dataset("timestamps", data=np.array(front_ts, dtype=np.float64))

        rear_group = dst.create_group("rear_frames")
        rear_group.create_dataset(
            "data",
            data=to_vlen_uint8_array(rear_data),
            dtype=vlen_bytes,
        )
        rear_group.create_dataset("timestamps", data=np.array(rear_ts, dtype=np.float64))

        for key, value in attrs.items():
            dst.attrs[key] = value
        dst.attrs["postprocessed"] = True
        dst.attrs["postprocess_start_sec"] = float(start_sec)
        dst.attrs["postprocess_end_sec"] = float(end_sec)


def main() -> int:
    args = parse_args()
    input_h5 = args.input_h5.expanduser().resolve()
    if not input_h5.is_file():
        raise SystemExit(f"H5 file not found: {input_h5}")
    if args.smooth_window < 1:
        raise SystemExit("--smooth-window must be >= 1")
    if args.min_window_sec <= 0:
        raise SystemExit("--min-window-sec must be > 0")

    output_dir = args.output_dir.expanduser().resolve()
    ensure_dir(output_dir)
    metrics_dir = output_dir / "metrics"
    preview_dir = output_dir / "preview"
    ensure_dir(metrics_dir)
    ensure_dir(preview_dir)

    with h5py.File(input_h5, "r") as handle:
        front_data = handle["front_frames/data"][:]
        front_ts = handle["front_frames/timestamps"][:]
        accels = handle["accels"][:]
        rpms = handle["rpms"][:]

    if len(front_ts) == 0:
        raise SystemExit("No front frames found in source H5")

    front_rel = normalize_timestamps(front_ts.astype(np.float64))
    accel_rel = normalize_timestamps(accels["t"].astype(np.float64)) if len(accels) else np.array([], dtype=np.float64)
    rpm_rel = normalize_timestamps(rpms["t"].astype(np.float64)) if len(rpms) else np.array([], dtype=np.float64)

    raw_metrics: list[dict[str, Any]] = []
    sharpness_values: list[float] = []
    terrain_focus_values: list[float] = []
    novelty_values: list[float] = []
    previous_gray: np.ndarray | None = None

    for idx, frame_item in enumerate(front_data):
        image = decode_image(frame_item)
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        brightness = float(np.mean(gray))
        sharpness = sharpness_score_raw(gray)
        novelty = frame_novelty_raw(previous_gray, gray)
        previous_gray = gray

        top_half = gray[: max(1, gray.shape[0] // 2), :]
        bottom_half = gray[max(1, gray.shape[0] // 2) :, :]
        top_texture = gradient_energy(top_half)
        bottom_texture = gradient_energy(bottom_half)
        terrain_focus = float(bottom_texture / max(1e-6, top_texture))

        accel_idx = nearest_index(accel_rel, float(front_rel[idx]))
        accel_row = accels[accel_idx] if accel_idx is not None and len(accels) else None
        roll_deg, pitch_deg, tilt_deg = accel_to_tilt_deg(accel_row)

        rpm_idx = nearest_index(rpm_rel, float(front_rel[idx]))
        rpm_sum = 0.0
        if rpm_idx is not None and len(rpms):
            rpm_row = rpms[rpm_idx]
            rpm_sum = float(
                abs(float(rpm_row["front_left"]))
                + abs(float(rpm_row["front_right"]))
                + abs(float(rpm_row["rear_left"]))
                + abs(float(rpm_row["rear_right"]))
            )

        raw_metrics.append(
            {
                "frame_index": idx,
                "timestamp": float(front_ts[idx]),
                "relative_time_sec": float(front_rel[idx]),
                "brightness": brightness,
                "sharpness_raw": sharpness,
                "novelty_raw": novelty,
                "top_texture": top_texture,
                "bottom_texture": bottom_texture,
                "terrain_focus_raw": terrain_focus,
                "roll_deg": roll_deg,
                "pitch_deg": pitch_deg,
                "tilt_deg": tilt_deg,
                "rpm_sum": rpm_sum,
            }
        )
        sharpness_values.append(sharpness)
        terrain_focus_values.append(terrain_focus)
        novelty_values.append(novelty)

    sharp_arr = np.array(sharpness_values, dtype=np.float64)
    terrain_arr = np.array(terrain_focus_values, dtype=np.float64)
    novelty_arr = np.array(novelty_values, dtype=np.float64)
    sharp_lo = float(np.percentile(sharp_arr, 15))
    sharp_hi = float(np.percentile(sharp_arr, 85))
    terrain_lo = float(np.percentile(terrain_arr, 10))
    terrain_hi = float(np.percentile(terrain_arr, 90))
    novelty_lo = float(np.percentile(novelty_arr, 15))
    novelty_hi = float(np.percentile(novelty_arr, 85))

    frame_scores: list[float] = []
    keep_flags: list[bool] = []
    motion_flags: list[bool] = []
    for row in raw_metrics:
        brightness = float(row["brightness"])
        sharpness = float(row["sharpness_raw"])
        terrain_focus = float(row["terrain_focus_raw"])
        tilt_deg = float(row["tilt_deg"]) if row["tilt_deg"] is not None else args.tilt_max_deg + 10.0
        rpm_sum = float(row["rpm_sum"])

        brightness_score = clamp01((brightness - args.brightness_min) / 60.0)
        sharpness_score = clamp01((sharpness - sharp_lo) / max(1e-6, sharp_hi - sharp_lo))
        terrain_score = clamp01((terrain_focus - terrain_lo) / max(1e-6, terrain_hi - terrain_lo))
        novelty_score = clamp01((float(row["novelty_raw"]) - novelty_lo) / max(1e-6, novelty_hi - novelty_lo))
        moving = rpm_sum >= args.rpm_motion_threshold
        motion_score = 1.0 if moving else 0.0
        tilt_score = 1.0 - clamp01((tilt_deg - 4.0) / max(1e-6, args.tilt_max_deg - 4.0))

        score = (
            0.25 * sharpness_score
            + 0.20 * brightness_score
            + 0.15 * terrain_score
            + 0.15 * novelty_score
            + 0.20 * motion_score
            + 0.05 * tilt_score
        )

        hard_reject = (
            brightness < args.brightness_min
            or tilt_deg > args.tilt_max_deg
            or sharpness_score < 0.05
        )
        keep = (not hard_reject)

        row["brightness_score"] = brightness_score
        row["sharpness_score"] = sharpness_score
        row["terrain_score"] = terrain_score
        row["novelty_score"] = novelty_score
        row["motion_score"] = motion_score
        row["tilt_score"] = tilt_score
        row["quality_score"] = score
        row["hard_reject"] = hard_reject
        raw_metrics_value = keep
        keep_flags.append(raw_metrics_value)
        motion_flags.append(moving)
        frame_scores.append(score)

    frame_scores_arr = np.array(frame_scores, dtype=np.float64)
    keep_flags_arr = np.array(keep_flags, dtype=bool)
    motion_flags_arr = np.array(motion_flags, dtype=bool)
    smoothed_scores = smooth_scores(frame_scores_arr, args.smooth_window)
    traversal_support_mask = dilate_mask_by_time(motion_flags_arr, front_rel, args.motion_support_sec)
    traversal_support_mask = bridge_short_false_gaps(traversal_support_mask, front_rel, args.max_stationary_gap_sec)
    candidate_mask = keep_flags_arr & traversal_support_mask

    novelty_score_arr = np.array([float(row["novelty_score"]) for row in raw_metrics], dtype=np.float64)
    scored_runs = score_candidate_runs(
        candidate_mask,
        front_rel,
        smoothed_scores,
        motion_flags_arr,
        novelty_score_arr,
    )
    best_run = pick_best_run(
        candidate_mask,
        front_rel,
        smoothed_scores,
        motion_flags_arr,
        novelty_score_arr,
        args.min_window_sec,
        args.min_motion_fraction,
    )
    if best_run is None:
        raise SystemExit("No teach window could be selected from this bag")

    best_objective = float(best_run.get("objective", 0.0))
    selected_runs = [
        run
        for run in scored_runs
        if float(run["duration_sec"]) >= args.min_episode_sec
        and float(run["motion_fraction"]) >= max(0.20, args.min_motion_fraction * 0.65)
        and float(run["objective"]) >= max(0.35, best_objective * 0.18)
    ]
    if not selected_runs:
        selected_runs = [best_run]

    start_sec = max(0.0, min(float(run["start_sec"]) for run in selected_runs) - args.pad_sec)
    end_sec = min(float(front_rel[-1]), max(float(run["end_sec"]) for run in selected_runs) + args.pad_sec)
    window_mask = (front_rel >= start_sec) & (front_rel <= end_sec)
    selected_mask = np.zeros(len(front_rel), dtype=bool)
    for run in selected_runs:
        start_idx = int(run["start_index"])
        end_idx = int(run["end_index"])
        selected_mask[start_idx : end_idx + 1] = True
    selected_mask &= keep_flags_arr

    for idx, row in enumerate(raw_metrics):
        row["smoothed_quality_score"] = float(smoothed_scores[idx])
        row["candidate_keep"] = bool(candidate_mask[idx])
        row["motion_candidate_keep"] = bool(traversal_support_mask[idx])
        row["selected_window_keep"] = bool(selected_mask[idx])

    metrics_csv = metrics_dir / "frame_metrics.csv"
    write_metrics_csv(
        metrics_csv,
        raw_metrics,
        [
            "frame_index",
            "timestamp",
            "relative_time_sec",
            "brightness",
            "sharpness_raw",
            "novelty_raw",
            "top_texture",
            "bottom_texture",
            "terrain_focus_raw",
            "roll_deg",
            "pitch_deg",
            "tilt_deg",
            "rpm_sum",
            "brightness_score",
            "sharpness_score",
            "terrain_score",
            "novelty_score",
            "motion_score",
            "tilt_score",
            "quality_score",
            "smoothed_quality_score",
            "hard_reject",
            "motion_candidate_keep",
            "candidate_keep",
            "selected_window_keep",
        ],
    )

    selected_indices = [idx for idx, keep in enumerate(selected_mask) if keep]
    rejected_indices = [idx for idx, keep in enumerate(candidate_mask) if not keep]

    selected_frames: list[tuple[Image.Image, str]] = []
    for idx in sample_indices(len(selected_indices), min(args.num_samples, len(selected_indices))):
        source_idx = selected_indices[idx]
        image = decode_image(front_data[source_idx])
        row = raw_metrics[source_idx]
        label = (
            f"f{source_idx} t={row['relative_time_sec']:.1f}s "
            f"q={row['smoothed_quality_score']:.2f} tilt={row['tilt_deg']:.1f}"
        )
        selected_frames.append((image, label))

    rejected_frames: list[tuple[Image.Image, str]] = []
    if rejected_indices:
        rejected_sorted = sorted(rejected_indices, key=lambda i: raw_metrics[i]["smoothed_quality_score"])
        sample_bad = sample_indices(len(rejected_sorted), min(args.num_samples, len(rejected_sorted)))
        for bad_idx in sample_bad:
            source_idx = rejected_sorted[bad_idx]
            image = decode_image(front_data[source_idx])
            row = raw_metrics[source_idx]
            label = (
                f"f{source_idx} t={row['relative_time_sec']:.1f}s "
                f"q={row['smoothed_quality_score']:.2f} tilt={row['tilt_deg']:.1f}"
            )
            rejected_frames.append((image, label))

    build_contact_sheet(
        title=f"Selected teach window | {input_h5.name} | {start_sec:.1f}s..{end_sec:.1f}s",
        frames=selected_frames,
        output_path=preview_dir / "selected_contact_sheet.jpg",
    )
    if rejected_frames:
        build_contact_sheet(
            title=f"Rejected / low-quality examples | {input_h5.name}",
            frames=rejected_frames,
            output_path=preview_dir / "rejected_contact_sheet.jpg",
        )

    cleaned_h5 = output_dir / f"{input_h5.stem}_cleaned.h5"
    if args.write_cleaned_h5:
        trimmed_front_data, trimmed_front_ts = slice_frames_by_time(front_data, front_ts, start_sec, end_sec)
        trimmed_front_rel = normalize_timestamps(trimmed_front_ts.astype(np.float64))
        trimmed_selected_mask = np.array(
            [bool(keep) for keep in selected_mask[(front_rel >= start_sec) & (front_rel <= end_sec)]],
            dtype=bool,
        )
        if len(trimmed_selected_mask) != len(trimmed_front_rel):
            raise ValueError("Selected mask / trimmed front timestamps length mismatch")
        write_cleaned_h5(
            input_h5=input_h5,
            output_h5=cleaned_h5,
            start_sec=start_sec,
            end_sec=end_sec,
            front_keep_mask_within_window=trimmed_selected_mask,
        )

    summary = {
        "input_h5": str(input_h5),
        "output_dir": str(output_dir),
        "num_front_frames": int(len(front_data)),
        "selected_frame_count": int(np.sum(selected_mask)),
        "candidate_frame_count": int(np.sum(candidate_mask)),
        "hard_reject_count": int(np.sum([bool(row["hard_reject"]) for row in raw_metrics])),
        "selected_start_sec": start_sec,
        "selected_end_sec": end_sec,
        "selected_duration_sec": float(end_sec - start_sec),
        "selected_mean_score": float(np.mean(smoothed_scores[selected_mask])) if np.any(selected_mask) else 0.0,
        "selected_window_frame_count": int(np.sum(window_mask)),
        "selected_motion_fraction": float(np.mean(motion_flags_arr[window_mask])) if np.any(window_mask) else 0.0,
        "selected_novelty_mean": float(np.mean(novelty_score_arr[selected_mask])) if np.any(selected_mask) else 0.0,
        "selected_episode_count": int(len(selected_runs)),
        "selected_episode_total_duration_sec": float(sum(float(run["duration_sec"]) for run in selected_runs)),
        "primary_episode_start_sec": float(best_run["start_sec"]),
        "primary_episode_end_sec": float(best_run["end_sec"]),
        "primary_episode_duration_sec": float(best_run["duration_sec"]),
        "selected_max_tilt_deg": float(max(float(raw_metrics[idx]["tilt_deg"]) for idx in selected_indices)) if selected_indices else None,
        "selected_min_brightness": float(min(float(raw_metrics[idx]["brightness"]) for idx in selected_indices)) if selected_indices else None,
        "selected_contact_sheet": str(preview_dir / "selected_contact_sheet.jpg"),
        "rejected_contact_sheet": str(preview_dir / "rejected_contact_sheet.jpg") if rejected_frames else None,
        "metrics_csv": str(metrics_csv),
        "cleaned_h5": str(cleaned_h5) if args.write_cleaned_h5 else None,
        "suggested_prepare_command": (
            f"python scripts/prepare_manual_route.py --input-h5 {cleaned_h5 if args.write_cleaned_h5 else input_h5} "
            f"--route-name {input_h5.stem}_pp --frame-step 12 --controller adaptive --tick-hz 2.0 --max-subgoal-hops 12"
        ),
    }
    summary_path = output_dir / "postprocess_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Post-processed teach bag: {input_h5}")
    print(f"Selected window: {start_sec:.1f}s .. {end_sec:.1f}s")
    print(f"Selected frames: {summary['selected_frame_count']} / {summary['num_front_frames']}")
    print(f"Summary: {summary_path}")
    if args.write_cleaned_h5:
        print(f"Cleaned H5: {cleaned_h5}")
    print(f"Suggested prepare command:\n  {summary['suggested_prepare_command']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
