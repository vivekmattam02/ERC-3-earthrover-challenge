import argparse
import os
import signal
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import h5py
import base64


DEFAULT_BASE_URL = os.getenv("SDK_URL", "http://127.0.0.1:8000")
DATA_URL = f"{DEFAULT_BASE_URL}/data"
SDK_URL = f"{DEFAULT_BASE_URL}/sdk"


def initialize_sdk_session() -> None:
    try:
        requests.get(SDK_URL, timeout=5)
    except Exception:
        pass


class H5DataLogger:
    def __init__(
        self,
        output_path: str,
        compression: str = "gzip",
        compression_level: int = 4,
        mode: str = "w",
    ) -> None:
        self.output_path = output_path
        self.compression = compression
        self.compression_level = compression_level
        self.file: Optional[h5py.File] = None
        self.mode = mode  # "w" to overwrite, "a" to append

        parent_dir = os.path.dirname(self.output_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        self._open_file()
        self._ensure_datasets()

    def _open_file(self) -> None:
        self.file = h5py.File(self.output_path, self.mode)

    def close(self) -> None:
        if self.file is not None:
            self.file.flush()
            self.file.close()
            self.file = None

    def _ensure_datasets(self) -> None:
        assert self.file is not None

        telemetry_dtype = np.dtype([
            ("timestamp", "<f8"),
            ("battery", "<f4"),
            ("signal_level", "<f4"),
            ("orientation", "<f4"),
            ("lamp", "<i1"),
            ("speed", "<f4"),
            ("gps_signal", "<f4"),
            ("latitude", "<f8"),
            ("longitude", "<f8"),
            ("vibration", "<f4"),
        ])

        sensors_xyz_dtype = np.dtype([
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("t", "<f8"),
        ])

        rpms_dtype = np.dtype([
            ("front_left", "<f4"),
            ("front_right", "<f4"),
            ("rear_left", "<f4"),
            ("rear_right", "<f4"),
            ("t", "<f8"),
        ])

        def ensure(name: str, dtype: np.dtype) -> None:
            if name not in self.file:
                self.file.create_dataset(
                    name,
                    shape=(0,),
                    maxshape=(None,),
                    dtype=dtype,
                    chunks=True,
                    compression=self.compression,
                    compression_opts=self.compression_level,
                )

        ensure("telemetry", telemetry_dtype)
        ensure("accels", sensors_xyz_dtype)
        ensure("gyros", sensors_xyz_dtype)
        ensure("mags", sensors_xyz_dtype)
        ensure("rpms", rpms_dtype)

        # Images: front/rear as groups with timestamps and variable-length bytes
        def ensure_image_group(group_name: str) -> None:
            if group_name not in self.file:
                grp = self.file.create_group(group_name)
            else:
                grp = self.file[group_name]
            if "timestamps" not in grp:
                grp.create_dataset(
                    "timestamps",
                    shape=(0,),
                    maxshape=(None,),
                    dtype="<f8",
                    chunks=True,
                    compression=self.compression,
                    compression_opts=self.compression_level,
                )
            if "data" not in grp:
                vlen_uint8 = h5py.vlen_dtype(np.dtype("uint8"))
                grp.create_dataset(
                    "data",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=vlen_uint8,
                    chunks=True,
                    compression=self.compression,
                    compression_opts=self.compression_level,
                )

        ensure_image_group("front_frames")
        ensure_image_group("rear_frames")

        # Control commands (commanded velocities)
        controls_dtype = np.dtype([
            ("timestamp", "<f8"),
            ("linear", "<f4"),
            ("angular", "<f4"),
        ])
        ensure("controls", controls_dtype)

    def _append_rows(self, dataset_name: str, rows: np.ndarray) -> None:
        if rows.size == 0:
            return
        assert self.file is not None
        ds = self.file[dataset_name]
        new_size = ds.shape[0] + rows.shape[0]
        ds.resize((new_size,))
        ds[-rows.shape[0] :] = rows

    def log_payload(self, payload: Dict[str, Any]) -> None:
        assert self.file is not None

        t = float(payload.get("timestamp", time.time()))

        def fget(name: str, default: float = np.nan) -> float:
            value = payload.get(name, default)
            try:
                return float(value)
            except Exception:
                return float(default)

        # Sanitize lamp to {0,1} to avoid out-of-range int8 casts
        lamp_raw = payload.get("lamp", 0)
        try:
            lamp_val = 1 if int(lamp_raw) != 0 else 0
        except Exception:
            lamp_val = 0

        telemetry_row = np.array(
            (
                t,
                fget("battery"),
                fget("signal_level"),
                fget("orientation"),
                lamp_val,
                fget("speed"),
                fget("gps_signal"),
                fget("latitude"),
                fget("longitude"),
                fget("vibration"),
            ),
            dtype=self.file["telemetry"].dtype,
        )

        self._append_rows("telemetry", telemetry_row.reshape(1,))

        def to_xyz_rows(key: str) -> np.ndarray:
            samples = payload.get(key) or []
            cleaned: List[Tuple[float, float, float, float]] = []
            for item in samples:
                if not isinstance(item, (list, tuple)) or len(item) < 4:
                    continue
                try:
                    x, y, z, ts = float(item[0]), float(item[1]), float(item[2]), float(item[3])
                    cleaned.append((x, y, z, ts))
                except Exception:
                    continue
            if not cleaned:
                return np.empty((0,), dtype=self.file[key].dtype)
            arr = np.array(cleaned, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("t", "<f8")])
            return arr

        def to_rpm_rows(key: str) -> np.ndarray:
            samples = payload.get(key) or []
            cleaned: List[Tuple[float, float, float, float, float]] = []
            for item in samples:
                if not isinstance(item, (list, tuple)) or len(item) < 5:
                    continue
                try:
                    fl, fr, rl, rr, ts = (
                        float(item[0]),
                        float(item[1]),
                        float(item[2]),
                        float(item[3]),
                        float(item[4]),
                    )
                    cleaned.append((fl, fr, rl, rr, ts))
                except Exception:
                    continue
            if not cleaned:
                return np.empty((0,), dtype=self.file[key].dtype)
            arr = np.array(
                cleaned,
                dtype=[
                    ("front_left", "<f4"),
                    ("front_right", "<f4"),
                    ("rear_left", "<f4"),
                    ("rear_right", "<f4"),
                    ("t", "<f8"),
                ],
            )
            return arr

        for name in ("accels", "gyros", "mags"):
            rows = to_xyz_rows(name)
            self._append_rows(name, rows)

        rows = to_rpm_rows("rpms")
        self._append_rows("rpms", rows)

        # Keep file in a consistent state
        self.file.flush()

    def _append_image(self, group_name: str, image_bytes: bytes, timestamp: float) -> None:
        assert self.file is not None
        grp = self.file[group_name]
        ts_ds = grp["timestamps"]
        data_ds = grp["data"]
        # Append timestamp
        ts_ds.resize((ts_ds.shape[0] + 1,))
        ts_ds[-1] = float(timestamp)
        # Append bytes as vlen uint8
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        data_ds.resize((data_ds.shape[0] + 1,))
        data_ds[-1] = arr
        self.file.flush()

    def log_front_frame_b64(self, b64_frame: str, timestamp: float) -> None:
        try:
            img_bytes = base64.b64decode(b64_frame, validate=False)
        except Exception:
            return
        self._append_image("front_frames", img_bytes, timestamp)

    def log_rear_frame_b64(self, b64_frame: str, timestamp: float) -> None:
        try:
            img_bytes = base64.b64decode(b64_frame, validate=False)
        except Exception:
            return
        self._append_image("rear_frames", img_bytes, timestamp)

    def log_control(self, linear: float, angular: float, timestamp: float) -> None:
        assert self.file is not None
        row = np.array((float(timestamp), float(linear), float(angular)), dtype=self.file["controls"].dtype)
        self._append_rows("controls", row.reshape(1,))


def build_default_output_path(prefix: str = "logs") -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return os.path.join(prefix, f"rover_log_{ts}.h5")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Earth Rovers SDK Data Logger (HDF5)")
    parser.add_argument("--url", default=DEFAULT_BASE_URL, help="SDK base URL (default: %(default)s)")
    parser.add_argument("--rate", type=float, default=5.0, help="Polling rate in Hz (default: %(default)s)")
    parser.add_argument("--out", default=build_default_output_path(), help="Output HDF5 file path (default: logs/rover_log_YYYYmmdd_HHMMSS.h5)")
    parser.add_argument("--gzip", type=int, default=4, help="Gzip compression level 0-9 (default: %(default)s)")
    parser.add_argument("--no-frames", action="store_true", help="Disable logging front/rear frames")
    parser.add_argument("--append", action="store_true", help="Append to an existing file instead of overwriting")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    global DEFAULT_BASE_URL, DATA_URL, SDK_URL
    DEFAULT_BASE_URL = args.url.rstrip("/")
    DATA_URL = f"{DEFAULT_BASE_URL}/data"
    SDK_URL = f"{DEFAULT_BASE_URL}/sdk"
    V2_URL = f"{DEFAULT_BASE_URL}/v2/screenshot"

    initialize_sdk_session()

    file_mode = "a" if args.append else "w"
    logger = H5DataLogger(args.out, compression="gzip", compression_level=args.gzip, mode=file_mode)

    stop = False

    def handle_sigint(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    interval = 1.0 / max(0.1, args.rate)
    last_print = 0.0
    samples = 0

    try:
        while not stop:
            loop_start = time.time()
            try:
                resp = requests.get(DATA_URL, timeout=5)
                if resp.ok:
                    payload = resp.json()
                    logger.log_payload(payload)
                    samples += 1
                else:
                    # If service not ready, keep trying
                    pass
            except Exception:
                # Continue trying; transient network/browser errors are expected during startup
                pass

            # Optionally fetch camera frames at the same cadence
            if not args.no_frames:
                try:
                    v2 = requests.get(V2_URL, timeout=5)
                    if v2.ok:
                        js = v2.json()
                        ts = float(js.get("timestamp", time.time()))
                        if "front_frame" in js:
                            logger.log_front_frame_b64(js["front_frame"], ts)
                        if "rear_frame" in js:
                            logger.log_rear_frame_b64(js["rear_frame"], ts)
                except Exception:
                    pass

            now = time.time()
            if now - last_print > 5.0:
                print(f"Logged samples: {samples}  -> {logger.output_path}")
                last_print = now

            elapsed = time.time() - loop_start
            sleep_time = max(0.0, interval - elapsed)
            time.sleep(sleep_time)
    finally:
        logger.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())


