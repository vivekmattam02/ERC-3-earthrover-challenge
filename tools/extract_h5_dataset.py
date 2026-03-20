#!/usr/bin/env python3
"""Extract a teleoperation H5 recording into a reusable corridor dataset.

This script converts the recorded EarthRover H5 log into:
- an ordered front-image directory
- CSV exports for telemetry, controls, IMU, and RPM streams
- a baseline-friendly data_info.json aligned to frame timestamps
- a small summary JSON for quick inspection

The goal is to make one H5 recording immediately usable by baseline.py and
other runtime modules without hand-written one-off notebooks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract an EarthRover teleop H5 into images and metadata.")
    parser.add_argument("--input-h5", type=Path, required=True, help="Path to the source H5 recording.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write the extracted dataset.")
    parser.add_argument("--frame-step", type=int, default=1, help="Keep every Nth front frame.")
    parser.add_argument(
        "--image-ext",
        choices=["png", "jpg", "jpeg"],
        default="png",
        help="File extension for exported frames. PNG bytes are written directly when possible.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def dataset_to_rows(dataset: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in dataset:
        row: dict[str, Any] = {}
        for name in dataset.dtype.names or []:
            row[name] = sanitize_value(item[name])
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def nearest_index_before_or_equal(timestamps: np.ndarray, query_ts: float) -> int | None:
    if len(timestamps) == 0:
        return None
    idx = int(np.searchsorted(timestamps, query_ts, side="right") - 1)
    if idx < 0:
        return 0
    return idx


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


def normalize_timestamps(timestamps: np.ndarray) -> np.ndarray:
    if len(timestamps) == 0:
        return np.array([], dtype=np.float64)
    base = float(timestamps[0])
    return timestamps.astype(np.float64) - base


def build_action_strings(control: np.void | None) -> list[str]:
    if control is None:
        return []
    linear = float(control["linear"])
    angular = float(control["angular"])
    return [f"linear={linear:.6f}", f"angular={angular:.6f}"]


def export_front_frames(
    frame_bytes: np.ndarray,
    frame_timestamps: np.ndarray,
    controls: np.ndarray,
    telemetry: np.ndarray,
    output_dir: Path,
    frame_step: int,
    image_ext: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    images_dir = output_dir / "front_images"
    ensure_dir(images_dir)

    frame_ts_norm = normalize_timestamps(frame_timestamps)
    control_ts_norm = normalize_timestamps(controls["timestamp"]) if len(controls) else np.array([])
    telemetry_ts_norm = normalize_timestamps(telemetry["timestamp"]) if len(telemetry) else np.array([])

    frame_rows: list[dict[str, Any]] = []
    data_info: list[dict[str, Any]] = []

    kept_indices = range(0, len(frame_bytes), frame_step)
    for step, source_index in enumerate(kept_indices):
        timestamp = float(frame_timestamps[source_index])
        timestamp_norm = float(frame_ts_norm[source_index])
        image_name = f"{step:06d}.{image_ext}"
        image_path = images_dir / image_name
        payload = bytes(frame_bytes[source_index])
        image_path.write_bytes(payload)

        control_idx = nearest_index(control_ts_norm, timestamp_norm)
        control = controls[control_idx] if control_idx is not None and len(controls) else None

        telemetry_idx = nearest_index(telemetry_ts_norm, timestamp_norm)
        telemetry_row = telemetry[telemetry_idx] if telemetry_idx is not None and len(telemetry) else None

        frame_row = {
            "step": step,
            "source_index": int(source_index),
            "timestamp": timestamp,
            "relative_time_sec": timestamp_norm,
            "image": image_name,
            "control_index": int(control_idx) if control_idx is not None else None,
            "telemetry_index": int(telemetry_idx) if telemetry_idx is not None else None,
        }

        if control is not None:
            frame_row["control_timestamp"] = sanitize_value(control["timestamp"])
            frame_row["control_relative_time_sec"] = sanitize_value(control_ts_norm[control_idx])
            frame_row["linear"] = sanitize_value(control["linear"])
            frame_row["angular"] = sanitize_value(control["angular"])

        if telemetry_row is not None:
            frame_row["telemetry_timestamp"] = sanitize_value(telemetry_row["timestamp"])
            frame_row["telemetry_relative_time_sec"] = sanitize_value(telemetry_ts_norm[telemetry_idx])
            frame_row["orientation"] = sanitize_value(telemetry_row["orientation"])
            frame_row["speed"] = sanitize_value(telemetry_row["speed"])
            frame_row["battery"] = sanitize_value(telemetry_row["battery"])
            frame_row["signal_level"] = sanitize_value(telemetry_row["signal_level"])
            frame_row["gps_signal"] = sanitize_value(telemetry_row["gps_signal"])
            frame_row["latitude"] = sanitize_value(telemetry_row["latitude"])
            frame_row["longitude"] = sanitize_value(telemetry_row["longitude"])
            frame_row["vibration"] = sanitize_value(telemetry_row["vibration"])

        frame_rows.append(frame_row)

        info_entry = {
            "step": step,
            "image": image_name,
            "timestamp": timestamp,
            "relative_time_sec": timestamp_norm,
            "action": build_action_strings(control),
            "linear": sanitize_value(control["linear"]) if control is not None else None,
            "angular": sanitize_value(control["angular"]) if control is not None else None,
            "orientation": sanitize_value(telemetry_row["orientation"]) if telemetry_row is not None else None,
            "speed": sanitize_value(telemetry_row["speed"]) if telemetry_row is not None else None,
            "control_index": int(control_idx) if control_idx is not None else None,
            "telemetry_index": int(telemetry_idx) if telemetry_idx is not None else None,
        }
        data_info.append(info_entry)

    return frame_rows, data_info


def build_summary(
    input_h5: Path,
    frame_rows: list[dict[str, Any]],
    controls: np.ndarray,
    telemetry: np.ndarray,
    accels: np.ndarray,
    gyros: np.ndarray,
    mags: np.ndarray,
    rpms: np.ndarray,
    frame_step: int,
) -> dict[str, Any]:
    frame_rate = None
    if len(frame_rows) > 1:
        duration = float(frame_rows[-1]["timestamp"] - frame_rows[0]["timestamp"])
        if duration > 0:
            frame_rate = (len(frame_rows) - 1) / duration

    rpm_nonzero = None
    if len(rpms):
        rpm_nonzero = int(
            np.sum(
                (
                    np.abs(rpms["front_left"])
                    + np.abs(rpms["front_right"])
                    + np.abs(rpms["rear_left"])
                    + np.abs(rpms["rear_right"])
                )
                > 1e-6
            )
        )

    orientation_nan_count = 0
    speed_nan_count = 0
    if len(telemetry):
        orientation_nan_count = int(np.isnan(telemetry["orientation"]).sum())
        speed_nan_count = int(np.isnan(telemetry["speed"]).sum())

    return {
        "input_h5": str(input_h5.resolve()),
        "frame_step": frame_step,
        "num_front_frames_kept": len(frame_rows),
        "num_controls": int(len(controls)),
        "num_telemetry": int(len(telemetry)),
        "num_accels": int(len(accels)),
        "num_gyros": int(len(gyros)),
        "num_mags": int(len(mags)),
        "num_rpms": int(len(rpms)),
        "estimated_frame_rate_hz": frame_rate,
        "rpm_nonzero_count": rpm_nonzero,
        "orientation_nan_count": orientation_nan_count,
        "speed_nan_count": speed_nan_count,
    }


def main() -> None:
    args = parse_args()
    if args.frame_step < 1:
        raise ValueError("--frame-step must be >= 1")
    if not args.input_h5.is_file():
        raise FileNotFoundError(f"H5 file not found: {args.input_h5}")

    output_dir = args.output_dir.resolve()
    metadata_dir = output_dir / "metadata"
    ensure_dir(output_dir)
    ensure_dir(metadata_dir)

    with h5py.File(args.input_h5, "r") as handle:
        accels = handle["accels"][:]
        controls = handle["controls"][:]
        gyros = handle["gyros"][:]
        mags = handle["mags"][:]
        rpms = handle["rpms"][:]
        telemetry = handle["telemetry"][:]
        front_frame_data = handle["front_frames/data"][:]
        front_frame_timestamps = handle["front_frames/timestamps"][:]

    frame_rows, data_info = export_front_frames(
        frame_bytes=front_frame_data,
        frame_timestamps=front_frame_timestamps,
        controls=controls,
        telemetry=telemetry,
        output_dir=output_dir,
        frame_step=args.frame_step,
        image_ext=args.image_ext,
    )

    controls_rows = dataset_to_rows(controls)
    telemetry_rows = dataset_to_rows(telemetry)
    accels_rows = dataset_to_rows(accels)
    gyros_rows = dataset_to_rows(gyros)
    mags_rows = dataset_to_rows(mags)
    rpms_rows = dataset_to_rows(rpms)

    write_csv(
        metadata_dir / "front_frames.csv",
        frame_rows,
        [
            "step",
            "source_index",
            "timestamp",
            "relative_time_sec",
            "image",
            "control_index",
            "telemetry_index",
            "control_timestamp",
            "control_relative_time_sec",
            "telemetry_timestamp",
            "telemetry_relative_time_sec",
            "linear",
            "angular",
            "orientation",
            "speed",
            "battery",
            "signal_level",
            "gps_signal",
            "latitude",
            "longitude",
            "vibration",
        ],
    )
    write_csv(metadata_dir / "controls.csv", controls_rows, ["timestamp", "linear", "angular"])
    write_csv(
        metadata_dir / "telemetry.csv",
        telemetry_rows,
        ["timestamp", "battery", "signal_level", "orientation", "lamp", "speed", "gps_signal", "latitude", "longitude", "vibration"],
    )
    write_csv(metadata_dir / "accels.csv", accels_rows, ["x", "y", "z", "t"])
    write_csv(metadata_dir / "gyros.csv", gyros_rows, ["x", "y", "z", "t"])
    write_csv(metadata_dir / "mags.csv", mags_rows, ["x", "y", "z", "t"])
    write_csv(metadata_dir / "rpms.csv", rpms_rows, ["front_left", "front_right", "rear_left", "rear_right", "t"])

    with (metadata_dir / "data_info.json").open("w", encoding="utf-8") as handle:
        json.dump(data_info, handle, indent=2)

    summary = build_summary(
        input_h5=args.input_h5,
        frame_rows=frame_rows,
        controls=controls,
        telemetry=telemetry,
        accels=accels,
        gyros=gyros,
        mags=mags,
        rpms=rpms,
        frame_step=args.frame_step,
    )
    with (metadata_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Extracted dataset to: {output_dir}")
    print(f"Front images: {len(frame_rows)}")
    print(f"Metadata: {metadata_dir}")
    print("Use with baseline.py build-db:")
    print(
        f"  python baseline.py build-db --image-dir {output_dir / 'front_images'} "
        f"--output-dir <artifacts_dir> --data-info-json {metadata_dir / 'data_info.json'}"
    )


if __name__ == "__main__":
    main()
