#!/usr/bin/env python3
"""Record a manual EarthRover session into an H5 file.

This script polls the local SDK while an operator manually drives the robot
and saves:
- front camera frames
- telemetry
- IMU streams
- magnetometer
- RPMs

The H5 layout matches the existing outdoor recordings used elsewhere in this
repo so downstream scripts can reuse the captured session immediately.

Front-only capture is the default because the no-GPS teach-repeat stack uses
the forward camera only. Rear capture is opt-in.
"""

from __future__ import annotations

import argparse
import base64
import json
import signal
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import requests


TELEMETRY_DTYPE = np.dtype([
    ("timestamp", np.float64),
    ("battery", np.float32),
    ("signal_level", np.float32),
    ("orientation", np.float32),
    ("lamp", np.int8),
    ("speed", np.float32),
    ("gps_signal", np.float32),
    ("latitude", np.float64),
    ("longitude", np.float64),
    ("vibration", np.float32),
])

CONTROL_DTYPE = np.dtype([
    ("timestamp", np.float64),
    ("linear", np.float32),
    ("angular", np.float32),
])

XYZT_DTYPE = np.dtype([
    ("x", np.float32),
    ("y", np.float32),
    ("z", np.float32),
    ("t", np.float64),
])

RPM_DTYPE = np.dtype([
    ("front_left", np.float32),
    ("front_right", np.float32),
    ("rear_left", np.float32),
    ("rear_right", np.float32),
    ("t", np.float64),
])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record SDK telemetry + frames into an H5 session log.")
    p.add_argument("--sdk-url", default="http://localhost:8000", help="Local SDK base URL.")
    p.add_argument("--output-h5", type=Path, required=True, help="Path to output H5 file.")
    p.add_argument("--poll-hz", type=float, default=2.0, help="Polling rate for telemetry/frame capture.")
    p.add_argument("--duration-s", type=float, default=None, help="Optional fixed recording duration.")
    rear_mode = p.add_mutually_exclusive_group()
    rear_mode.add_argument(
        "--with-rear",
        action="store_true",
        help="Also capture rear camera frames. Default is front-only.",
    )
    rear_mode.add_argument(
        "--no-rear",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument("--session-name", default="manual_flag_collection", help="Human-readable session label saved as metadata.")
    return p.parse_args()


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _decode_frame_bytes(payload: dict[str, Any], key: str) -> bytes | None:
    encoded = payload.get(key)
    if not encoded:
        return None
    try:
        return base64.b64decode(encoded)
    except Exception:
        return None


def _get_json(session: requests.Session, url: str) -> dict[str, Any] | None:
    try:
        resp = session.get(url, timeout=5.0)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None


def _telemetry_row(data: dict[str, Any]) -> np.void:
    return np.array([(
        _safe_float(data.get("timestamp")),
        _safe_float(data.get("battery")),
        _safe_float(data.get("signal_level")),
        _safe_float(data.get("orientation")),
        _safe_int(data.get("lamp")),
        _safe_float(data.get("speed")),
        _safe_float(data.get("gps_signal")),
        _safe_float(data.get("latitude")),
        _safe_float(data.get("longitude")),
        _safe_float(data.get("vibration")),
    )], dtype=TELEMETRY_DTYPE)[0]


def _append_xyz_stream(
    target: list[np.void],
    samples: Any,
    seen: set[float],
) -> None:
    if not isinstance(samples, list):
        return
    for item in samples:
        if not isinstance(item, (list, tuple)) or len(item) < 4:
            continue
        ts = _safe_float(item[3])
        if ts in seen:
            continue
        seen.add(ts)
        row = np.array([(
            _safe_float(item[0]),
            _safe_float(item[1]),
            _safe_float(item[2]),
            ts,
        )], dtype=XYZT_DTYPE)[0]
        target.append(row)


def _append_rpm_stream(
    target: list[np.void],
    samples: Any,
    seen: set[float],
) -> None:
    if not isinstance(samples, list):
        return
    for item in samples:
        if not isinstance(item, (list, tuple)) or len(item) < 5:
            continue
        ts = _safe_float(item[4])
        if ts in seen:
            continue
        seen.add(ts)
        row = np.array([(
            _safe_float(item[0]),
            _safe_float(item[1]),
            _safe_float(item[2]),
            _safe_float(item[3]),
            ts,
        )], dtype=RPM_DTYPE)[0]
        target.append(row)


def normalize_output_h5(path: Path) -> Path:
    path = path.expanduser()
    if path.suffix.lower() != ".h5":
        path = path.with_suffix(".h5")
    return path


def main() -> int:
    args = parse_args()
    if args.poll_hz <= 0:
        raise SystemExit("--poll-hz must be > 0")

    output_h5 = normalize_output_h5(args.output_h5).resolve()
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    base = args.sdk_url.rstrip("/")
    period = 1.0 / float(args.poll_hz)

    telemetry_rows: list[np.void] = []
    control_rows: list[np.void] = []
    accel_rows: list[np.void] = []
    gyro_rows: list[np.void] = []
    mag_rows: list[np.void] = []
    rpm_rows: list[np.void] = []
    front_frame_bytes: list[bytes] = []
    front_frame_timestamps: list[float] = []
    rear_frame_bytes: list[bytes] = []
    rear_frame_timestamps: list[float] = []

    seen_accel_ts: set[float] = set()
    seen_gyro_ts: set[float] = set()
    seen_mag_ts: set[float] = set()
    seen_rpm_ts: set[float] = set()
    seen_front_ts: set[float] = set()
    seen_rear_ts: set[float] = set()

    stop_requested = False

    def _handle_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    started = time.time()
    print(f"Recording session -> {output_h5}")
    print("Press Ctrl+C to stop.")

    while not stop_requested:
        if args.duration_s is not None and (time.time() - started) >= float(args.duration_s):
            break

        loop_start = time.time()

        data = _get_json(session, f"{base}/data")
        if data is not None:
            telemetry_rows.append(_telemetry_row(data))
            _append_xyz_stream(accel_rows, data.get("accels"), seen_accel_ts)
            _append_xyz_stream(gyro_rows, data.get("gyros"), seen_gyro_ts)
            _append_xyz_stream(mag_rows, data.get("mags"), seen_mag_ts)
            _append_rpm_stream(rpm_rows, data.get("rpms"), seen_rpm_ts)

        front = _get_json(session, f"{base}/v2/front")
        if front is not None:
            ts = _safe_float(front.get("timestamp"))
            payload = _decode_frame_bytes(front, "front_frame")
            if payload is not None and ts not in seen_front_ts:
                seen_front_ts.add(ts)
                front_frame_timestamps.append(ts)
                front_frame_bytes.append(payload)

        if args.with_rear:
            rear = _get_json(session, f"{base}/v2/rear")
            if rear is not None:
                ts = _safe_float(rear.get("timestamp"))
                payload = _decode_frame_bytes(rear, "rear_frame")
                if payload is not None and ts not in seen_rear_ts:
                    seen_rear_ts.add(ts)
                    rear_frame_timestamps.append(ts)
                    rear_frame_bytes.append(payload)

        elapsed = time.time() - loop_start
        if elapsed < period:
            time.sleep(period - elapsed)

    with h5py.File(output_h5, "w") as f:
        f.create_dataset("telemetry", data=np.array(telemetry_rows, dtype=TELEMETRY_DTYPE))
        f.create_dataset("controls", data=np.array(control_rows, dtype=CONTROL_DTYPE))
        f.create_dataset("accels", data=np.array(accel_rows, dtype=XYZT_DTYPE))
        f.create_dataset("gyros", data=np.array(gyro_rows, dtype=XYZT_DTYPE))
        f.create_dataset("mags", data=np.array(mag_rows, dtype=XYZT_DTYPE))
        f.create_dataset("rpms", data=np.array(rpm_rows, dtype=RPM_DTYPE))

        vlen_bytes = h5py.vlen_dtype(np.dtype("uint8"))
        front_group = f.create_group("front_frames")
        front_group.create_dataset(
            "data",
            data=np.array([np.frombuffer(x, dtype=np.uint8) for x in front_frame_bytes], dtype=object),
            dtype=vlen_bytes,
        )
        front_group.create_dataset("timestamps", data=np.array(front_frame_timestamps, dtype=np.float64))

        rear_group = f.create_group("rear_frames")
        rear_group.create_dataset(
            "data",
            data=np.array([np.frombuffer(x, dtype=np.uint8) for x in rear_frame_bytes], dtype=object),
            dtype=vlen_bytes,
        )
        rear_group.create_dataset("timestamps", data=np.array(rear_frame_timestamps, dtype=np.float64))

        f.attrs["session_name"] = args.session_name
        f.attrs["sdk_url"] = base
        f.attrs["capture_rear"] = bool(args.with_rear)
        f.attrs["created_at_epoch_s"] = time.time()
        f.attrs["duration_s"] = float(time.time() - started)

    summary = {
        "output_h5": str(output_h5),
        "session_name": args.session_name,
        "capture_rear": bool(args.with_rear),
        "telemetry_rows": len(telemetry_rows),
        "front_frames": len(front_frame_bytes),
        "rear_frames": len(rear_frame_bytes),
        "accels": len(accel_rows),
        "gyros": len(gyro_rows),
        "mags": len(mag_rows),
        "rpms": len(rpm_rows),
        "controls": len(control_rows),
    }
    summary_path = output_h5.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
