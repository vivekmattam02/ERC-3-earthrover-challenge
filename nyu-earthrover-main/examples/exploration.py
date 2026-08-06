import argparse
import os
import sys
import time
from threading import Event, Thread
from typing import Optional

import requests


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exploration mode: drive with keyboard and log data",
        epilog=(
            "Example:\n"
            "  python examples/exploration.py --url http://127.0.0.1:8000 --rate 10 --out logs/run.h5\n"
        ),
    )
    parser.add_argument("--url", default=os.getenv("SDK_URL", "http://127.0.0.1:8000"), help="SDK base URL")
    parser.add_argument("--rate", type=float, default=5.0, help="Logging rate in Hz (default: %(default)s)")
    parser.add_argument("--out", default=None, help="Output HDF5 path (default: auto timestamp)")
    parser.add_argument("--gzip", type=int, default=4, help="Gzip compression level 0-9 (default: %(default)s)")
    return parser.parse_args(argv)


def start_logging_thread(base_url: str, rate_hz: float, out_path: Optional[str], gzip_level: int) -> tuple[Thread, Event, str]:
    # Lazy import from local directory; this file should be run from project root
    try:
        from utils.data_logger import H5DataLogger, build_default_output_path  # type: ignore
    except Exception as exc:
        print(f"Failed to import data_logger: {exc}")
        raise

    stop_event: Event = Event()
    output_path = out_path or build_default_output_path()
    interval = 1.0 / max(0.1, rate_hz)
    data_url = base_url.rstrip("/") + "/data"
    sdk_url = base_url.rstrip("/") + "/sdk"
    v2_url = base_url.rstrip("/") + "/v2/screenshot"

    def _runner() -> None:
        # Initialize SDK page once (non-fatal)
        try:
            requests.get(sdk_url, timeout=5)
        except Exception:
            pass

        # Overwrite file for each exploration session
        logger = H5DataLogger(output_path, compression="gzip", compression_level=gzip_level, mode="w")
        try:
            while not stop_event.is_set():
                start = time.time()
                try:
                    resp = requests.get(data_url, timeout=5)
                    if resp.ok:
                        payload = resp.json()
                        logger.log_payload(payload)
                except Exception:
                    # Swallow transient errors to avoid interfering with UI
                    pass

                # Opportunistically fetch frames at the same cadence
                try:
                    v2 = requests.get(v2_url, timeout=5)
                    if v2.ok:
                        js = v2.json()
                        ts = float(js.get("timestamp", time.time()))
                        if "front_frame" in js:
                            logger.log_front_frame_b64(js["front_frame"], ts)
                        if "rear_frame" in js:
                            logger.log_rear_frame_b64(js["rear_frame"], ts)
                except Exception:
                    pass
                # Maintain approximate rate
                elapsed = time.time() - start
                sleep_s = max(0.0, interval - elapsed)
                stop_event.wait(sleep_s)
        finally:
            logger.close()

    th = Thread(target=_runner, daemon=True)
    th.start()
    return th, stop_event, output_path


def wrap_keyboard_send_logging(log_path: str) -> None:
    """Monkey-patch keyboard_control.send_command to also log commanded velocities."""
    try:
        from utils.data_logger import H5DataLogger  # type: ignore
        from utils import keyboard_control  # type: ignore
        import time as _time
    except Exception:
        return

    # Append to the same session file for command taps
    logger = H5DataLogger(log_path, mode="a")

    original_send = keyboard_control.send_command

    def _wrapped(linear: float, angular: float):
        ts = _time.time()
        try:
            logger.log_control(linear, angular, ts)
        except Exception:
            pass
        return original_send(linear, angular)

    keyboard_control.send_command = _wrapped  # type: ignore


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    # Ensure keyboard controller and logger share the same base URL
    os.environ["SDK_URL"] = args.url

    # Start logging in the background
    log_thread, stop_event, log_path = start_logging_thread(args.url, args.rate, args.out, args.gzip)

    # Patch keyboard send to log commanded velocities into the same HDF5
    wrap_keyboard_send_logging(log_path)

    # Run keyboard controller (foreground, curses UI)
    try:
        import curses  # local import to avoid issues in non-TTY contexts
        from utils import keyboard_control  # type: ignore
    except Exception as exc:
        # Stop logger before exiting
        stop_event.set()
        log_thread.join(timeout=2.0)
        print(f"Failed to start keyboard control: {exc}")
        return 1

    try:
        curses.wrapper(keyboard_control.main)
    finally:
        # Graceful shutdown of logging
        stop_event.set()
        log_thread.join(timeout=5.0)
        print(f"Exploration ended. Log saved to: {log_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())


