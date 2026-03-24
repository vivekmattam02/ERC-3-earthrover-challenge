#!/usr/bin/env python3
"""Simple line-based manual control for EarthRover via /control-legacy.

Examples:
  w        -> forward for default duration
  a        -> left turn for default duration
  d 1.2    -> right turn for 1.2 seconds
  wa 0.5   -> forward-left for 0.5 seconds
  x        -> stop
  q        -> quit
"""

from __future__ import annotations

import argparse
import time

import requests


COMMANDS = {
    "w": (0.35, 0.0),
    "s": (-0.30, 0.0),
    "a": (0.0, 0.55),
    "d": (0.0, -0.55),
    "wa": (0.25, 0.35),
    "wd": (0.25, -0.35),
    "sa": (-0.20, 0.35),
    "sd": (-0.20, -0.35),
    "x": (0.0, 0.0),
    "stop": (0.0, 0.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple EarthRover manual control.")
    parser.add_argument("--sdk-url", default="http://localhost:8000", help="SDK base URL.")
    parser.add_argument(
        "--default-duration",
        type=float,
        default=0.35,
        help="Seconds to send a motion command when duration is omitted.",
    )
    parser.add_argument("--hz", type=float, default=6.0, help="Command resend rate while moving.")
    return parser.parse_args()


def send_command(base_url: str, linear: float, angular: float) -> None:
    response = requests.post(
        f"{base_url}/control-legacy",
        json={"command": {"linear": linear, "angular": angular, "lamp": 0}},
        timeout=2.0,
    )
    response.raise_for_status()


def send_for_duration(base_url: str, linear: float, angular: float, duration: float, hz: float) -> None:
    duration = max(0.0, float(duration))
    if duration == 0.0:
        send_command(base_url, linear, angular)
        return

    period = 1.0 / max(hz, 1e-6)
    deadline = time.time() + duration
    while True:
        send_command(base_url, linear, angular)
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        time.sleep(min(period, remaining))


def main() -> int:
    args = parse_args()

    print("Simple EarthRover Control")
    print("Commands: w s a d wa wd sa sd x(stop) q(quit)")
    print("Optional duration: e.g. 'w 1.0' or 'a 0.5'")

    while True:
        try:
            raw = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            raw = "q"

        if not raw:
            continue
        if raw in {"q", "quit", "exit"}:
            try:
                send_command(args.sdk_url, 0.0, 0.0)
            except Exception:
                pass
            print("Stopped and exiting.")
            return 0

        parts = raw.split()
        command_name = parts[0]
        if command_name not in COMMANDS:
            print("Unknown command.")
            continue

        duration = args.default_duration
        if len(parts) >= 2:
            try:
                duration = float(parts[1])
            except ValueError:
                print("Invalid duration.")
                continue

        linear, angular = COMMANDS[command_name]
        try:
            send_for_duration(args.sdk_url, linear, angular, duration, args.hz)
            if (linear, angular) != (0.0, 0.0):
                send_command(args.sdk_url, 0.0, 0.0)
            print(
                f"sent linear={linear:+.2f} angular={angular:+.2f} duration={duration:.2f}s"
            )
        except Exception as exc:
            print(f"error: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
