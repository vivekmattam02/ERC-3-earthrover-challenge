#!/usr/bin/env python3
"""Terminal teleop for EarthRover using /control-legacy.

Controls:
- w: forward
- s: reverse
- a: turn left
- d: turn right
- space: stop
- q: quit
"""

from __future__ import annotations

import argparse
import curses
import time

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Terminal teleop for EarthRover.")
    parser.add_argument("--sdk-url", default="http://localhost:8000", help="SDK base URL.")
    parser.add_argument("--linear", type=float, default=0.35, help="Linear magnitude to send.")
    parser.add_argument("--angular", type=float, default=0.6, help="Angular magnitude to send.")
    parser.add_argument("--hz", type=float, default=5.0, help="Command resend rate.")
    return parser.parse_args()


def post_command(base_url: str, linear: float, angular: float) -> str:
    response = requests.post(
        f"{base_url}/control-legacy",
        json={"command": {"linear": linear, "angular": angular, "lamp": 0}},
        timeout=2.0,
    )
    response.raise_for_status()
    return response.json().get("message", "ok")


def command_from_key(key: int, linear_mag: float, angular_mag: float) -> tuple[float, float] | None:
    if key in (ord("w"), ord("W")):
        return (linear_mag, 0.0)
    if key in (ord("s"), ord("S")):
        return (-linear_mag, 0.0)
    if key in (ord("a"), ord("A")):
        return (0.0, angular_mag)
    if key in (ord("d"), ord("D")):
        return (0.0, -angular_mag)
    if key == ord(" "):
        return (0.0, 0.0)
    return None


def run(stdscr: curses.window, args: argparse.Namespace) -> int:
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(50)

    linear = 0.0
    angular = 0.0
    last_send = 0.0
    send_period = 1.0 / max(args.hz, 1e-6)
    last_status = "Idle"

    while True:
        stdscr.erase()
        stdscr.addstr(0, 0, "EarthRover Legacy Teleop")
        stdscr.addstr(2, 0, "w forward | s reverse | a left | d right | space stop | q quit")
        stdscr.addstr(4, 0, f"linear={linear:+.2f} angular={angular:+.2f}")
        stdscr.addstr(5, 0, f"status={last_status}")
        stdscr.refresh()

        key = stdscr.getch()
        if key in (ord("q"), ord("Q")):
            break

        next_command = command_from_key(key, args.linear, args.angular)
        if next_command is not None:
            linear, angular = next_command

        now = time.time()
        if now - last_send >= send_period:
            try:
                last_status = post_command(args.sdk_url, linear, angular)
            except Exception as exc:  # pragma: no cover - runtime only
                last_status = f"error: {exc}"
            last_send = now

    try:
        post_command(args.sdk_url, 0.0, 0.0)
    except Exception:
        pass
    return 0


def main() -> int:
    args = parse_args()
    return curses.wrapper(lambda stdscr: run(stdscr, args))


if __name__ == "__main__":
    raise SystemExit(main())
