import os
import time
import curses
import requests
from threading import Lock


# Global keyboard listener for simultaneous keys
from pynput import keyboard as _pynput_kb  # type: ignore

DEFAULT_BASE_URL = os.getenv("SDK_URL", "http://127.0.0.1:8000")
CONTROL_URL = f"{DEFAULT_BASE_URL}/control"
SDK_URL = f"{DEFAULT_BASE_URL}/sdk"


def clamp(value: float, min_value: float = -1.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


def initialize_sdk_session() -> None:
    try:
        # Trigger the SDK page once to ensure backend RTM session is ready
        requests.get(SDK_URL, timeout=5)
    except Exception:
        # Non-fatal if this fails; /control path will also lazy-init
        pass


def calculate_target_from_keys(pressed_keys: set[str]) -> tuple[float, float]:
    linear = 0.0
    angular = 0.0
    if "w" in pressed_keys:
        linear += 1.0
    if "s" in pressed_keys:
        linear -= 1.0
    if "a" in pressed_keys:
        angular += 1.0
    if "d" in pressed_keys:
        angular -= 1.0

    if linear != 0.0 and angular != 0.0:
        factor = 2 ** -0.5
        linear *= factor
        angular *= factor

    return linear, angular


def lerp(start: float, end: float, t: float) -> float:
    return start + (end - start) * t


def send_command(linear: float, angular: float) -> tuple[bool, str]:
    try:
        payload = {"command": {"linear": float(linear), "angular": float(angular)}}
        response = requests.post(CONTROL_URL, json=payload, timeout=2)
        if response.ok:
            return True, "ok"
        return False, f"http {response.status_code}"
    except Exception as exc:
        return False, str(exc)


def draw_ui(stdscr, status: dict) -> None:
    stdscr.erase()
    stdscr.addstr(0, 0, "Earth Rovers SDK - Keyboard Control (W/A/S/D to drive, Space to stop, +/- speed, Q to quit)")
    stdscr.addstr(2, 0, f"Base URL: {DEFAULT_BASE_URL}")
    stdscr.addstr(3, 0, f"Tick: {status['tick_ms']} ms, Smoothing: {int(status['smoothing']*100)}%  MaxSpeed: {status['max_speed']:.2f}  Input: {status['input_mode']}")
    stdscr.addstr(5, 0, f"Target linear: {status['target_linear']:+.2f}   Target angular: {status['target_angular']:+.2f}")
    stdscr.addstr(6, 0, f"Actual linear: {status['current_linear']:+.2f}   Actual angular: {status['current_angular']:+.2f}")
    stdscr.addstr(8, 0, f"Last send: {status['last_send_result']} at {status['last_send_time']:.2f}s")
    stdscr.addstr(10, 0, f"Active keys: {' '.join(sorted(status['pressed'])) or '-'}")
    stdscr.refresh()


def main(stdscr) -> None:
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(0)

    initialize_sdk_session()

    update_interval_s = 0.05  # 20 Hz
    smoothing = 0.5  # 0..1
    max_speed = 0.5  # scales target output
    # input driven entirely by pynput listener

    pressed_keys: set[str] = set()

    # Shared state for pynput listener
    held_keys: set[str] = set()
    held_lock: Lock = Lock()
    quit_requested = False
    listener = None

    def _normalize_key(k) -> str | None:
        try:
            if isinstance(k, _pynput_kb.KeyCode) and k.char:
                return k.char.lower()
            if k == _pynput_kb.Key.esc:
                return "esc"
            if k == _pynput_kb.Key.space:
                return "space"
        except Exception:
            return None
        return None

    def _on_press(k):
        nonlocal quit_requested
        name = _normalize_key(k)
        if name is None:
            return
        if name in ("q", "esc"):
            quit_requested = True
            return
        with held_lock:
            held_keys.add(name)

    def _on_release(k):
        name = _normalize_key(k)
        if name is None:
            return
        with held_lock:
            if name in held_keys:
                held_keys.remove(name)

    listener = _pynput_kb.Listener(on_press=_on_press, on_release=_on_release)
    listener.daemon = True
    listener.start()
    current_linear = 0.0
    current_angular = 0.0
    target_linear = 0.0
    target_angular = 0.0
    last_send_time = 0.0
    last_send_result = "-"
    start_time = time.time()

    while True:
        # Read held keys snapshot from listener
        with held_lock:
            snapshot = set(held_keys)

        if quit_requested:
            try:
                listener.stop()
            except Exception:
                pass
            return

        pressed_keys = {k for k in ("w", "a", "s", "d") if k in snapshot}

        # Space = immediate stop while held
        if "space" in snapshot:
            pressed_keys.clear()
            target_linear = 0.0
            target_angular = 0.0

        # Drain any pending curses input for speed/backspace
        key = stdscr.getch()
        while key != -1:
            try:
                if key in (ord("+"), ord("=")):
                    max_speed = clamp(max_speed + 0.05, 0.05, 1.0)
                elif key in (ord("-"), ord("_")):
                    max_speed = clamp(max_speed - 0.05, 0.05, 1.0)
                elif key == curses.KEY_BACKSPACE:
                    pressed_keys.clear()
            finally:
                key = stdscr.getch()

        # Derive targets from active keys
        t_linear, t_angular = calculate_target_from_keys(pressed_keys)
        target_linear = t_linear * max_speed
        target_angular = t_angular * max_speed

        # Smooth towards targets
        current_linear = lerp(current_linear, target_linear, smoothing)
        current_angular = lerp(current_angular, target_angular, smoothing)

        # Send only if significant or periodically
        should_send = abs(current_linear) > 0.01 or abs(current_angular) > 0.01
        now = time.time() - start_time
        if should_send or (now - last_send_time) > 1.0:
            ok, msg = send_command(clamp(current_linear), clamp(current_angular))
            last_send_result = "ok" if ok else f"err: {msg}"
            last_send_time = now

        status = {
            "tick_ms": int(update_interval_s * 1000),
            "smoothing": smoothing,
            "max_speed": max_speed,
            "target_linear": target_linear,
            "target_angular": target_angular,
            "current_linear": current_linear,
            "current_angular": current_angular,
            "last_send_result": last_send_result,
            "last_send_time": last_send_time,
            "pressed": pressed_keys,
            "input_mode": "pynput",
        }

        draw_ui(stdscr, status)
        time.sleep(update_interval_s)


if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        pass


