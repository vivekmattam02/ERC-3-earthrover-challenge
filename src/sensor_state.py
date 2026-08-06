"""Lightweight motion-state filtering for ERC indoor runtime.

This is intentionally not a full EKF. It provides a simple filtered heading and
short-term motion prior from orientation, gyros, and RPMs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import time


def wrap_angle_deg(delta: float) -> float:
    return ((delta + 180.0) % 360.0) - 180.0


def blend_angle_deg(previous: float, current: float, alpha: float) -> float:
    delta = wrap_angle_deg(current - previous)
    return (previous + alpha * delta) % 360.0


@dataclass
class SensorStateFilterConfig:
    heading_alpha: float = 0.35
    gyro_alpha: float = 0.30
    rpm_alpha: float = 0.30
    tilt_alpha: float = 0.25
    stale_timeout_s: float = 5.0


class SensorStateFilter:
    """Small motion prior layer for heading/gyro/RPM smoothing."""

    def __init__(self, config: SensorStateFilterConfig = SensorStateFilterConfig()):
        self.config = config
        self._filtered_heading_deg: Optional[float] = None
        self._filtered_heading_rate_dps: float = 0.0
        self._filtered_rpm_mean: float = 0.0
        self._filtered_roll_deg: float = 0.0
        self._filtered_pitch_deg: float = 0.0
        self._last_update_ts: Optional[float] = None
        self._last_sensor_timestamp: Optional[float] = None
        self._last_fresh_data_wall_ts: Optional[float] = None

    def reset(self) -> None:
        self._filtered_heading_deg = None
        self._filtered_heading_rate_dps = 0.0
        self._filtered_rpm_mean = 0.0
        self._filtered_roll_deg = 0.0
        self._filtered_pitch_deg = 0.0
        self._last_update_ts = None
        self._last_sensor_timestamp = None
        self._last_fresh_data_wall_ts = None

    def _latest_gyro_z(self, data: dict) -> Optional[float]:
        gyros = data.get("gyros") or []
        if not gyros:
            return None
        latest = gyros[-1]
        if not isinstance(latest, (list, tuple)) or len(latest) < 3:
            return None
        try:
            return float(latest[2])
        except (TypeError, ValueError):
            return None

    def _rpm_mean(self, data: dict) -> Optional[float]:
        rpms = data.get("rpms") or []
        if not rpms:
            return None
        latest = rpms[-1]
        if not isinstance(latest, (list, tuple)) or len(latest) < 4:
            return None
        values = []
        for item in latest[:4]:
            try:
                values.append(abs(float(item)))
            except (TypeError, ValueError):
                continue
        if not values:
            return None
        return sum(values) / len(values)

    def _latest_accel(self, data: dict) -> Optional[tuple[float, float, float]]:
        accels = data.get("accels") or []
        if not accels:
            return None
        latest = accels[-1]
        if not isinstance(latest, (list, tuple)) or len(latest) < 3:
            return None
        try:
            return float(latest[0]), float(latest[1]), float(latest[2])
        except (TypeError, ValueError):
            return None

    def update(self, data: Optional[dict]) -> dict:
        now = time.time()
        if not data:
            return {
                "heading_deg": self._filtered_heading_deg,
                "heading_rate_dps": self._filtered_heading_rate_dps,
                "rpm_mean": self._filtered_rpm_mean,
                "roll_deg": self._filtered_roll_deg,
                "pitch_deg": self._filtered_pitch_deg,
                "tilt_deg": max(abs(self._filtered_roll_deg), abs(self._filtered_pitch_deg)),
                "sensor_timestamp": self._last_sensor_timestamp,
                "is_stale": True,
            }

        sensor_timestamp = data.get("timestamp")
        try:
            sensor_timestamp = None if sensor_timestamp is None else float(sensor_timestamp)
        except (TypeError, ValueError):
            sensor_timestamp = None

        if sensor_timestamp is None:
            self._last_fresh_data_wall_ts = now
        elif self._last_sensor_timestamp is None or sensor_timestamp > self._last_sensor_timestamp:
            self._last_sensor_timestamp = sensor_timestamp
            self._last_fresh_data_wall_ts = now

        raw_orientation = data.get("orientation")
        try:
            raw_heading_deg = None if raw_orientation is None else float(raw_orientation) % 360.0
        except (TypeError, ValueError):
            raw_heading_deg = None

        if raw_heading_deg is not None:
            if self._filtered_heading_deg is None:
                self._filtered_heading_deg = raw_heading_deg
            else:
                self._filtered_heading_deg = blend_angle_deg(
                    self._filtered_heading_deg,
                    raw_heading_deg,
                    self.config.heading_alpha,
                )

        gyro_z = self._latest_gyro_z(data)
        if gyro_z is not None:
            # Assume SDK gyro z is rad/s and convert to deg/s.
            heading_rate_dps = math.degrees(gyro_z)
            self._filtered_heading_rate_dps = (
                self.config.gyro_alpha * heading_rate_dps
                + (1.0 - self.config.gyro_alpha) * self._filtered_heading_rate_dps
            )

        rpm_mean = self._rpm_mean(data)
        if rpm_mean is not None:
            self._filtered_rpm_mean = (
                self.config.rpm_alpha * rpm_mean
                + (1.0 - self.config.rpm_alpha) * self._filtered_rpm_mean
            )

        accel = self._latest_accel(data)
        if accel is not None:
            ax, ay, az = accel
            roll_deg = math.degrees(math.atan2(ay, az))
            pitch_deg = math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az)))
            self._filtered_roll_deg = (
                self.config.tilt_alpha * roll_deg
                + (1.0 - self.config.tilt_alpha) * self._filtered_roll_deg
            )
            self._filtered_pitch_deg = (
                self.config.tilt_alpha * pitch_deg
                + (1.0 - self.config.tilt_alpha) * self._filtered_pitch_deg
            )

        self._last_update_ts = now
        is_stale = True
        if self._last_fresh_data_wall_ts is not None:
            is_stale = (now - self._last_fresh_data_wall_ts) > self.config.stale_timeout_s

        return {
            "heading_deg": self._filtered_heading_deg,
            "heading_rate_dps": self._filtered_heading_rate_dps,
            "rpm_mean": self._filtered_rpm_mean,
            "roll_deg": self._filtered_roll_deg,
            "pitch_deg": self._filtered_pitch_deg,
            "tilt_deg": max(abs(self._filtered_roll_deg), abs(self._filtered_pitch_deg)),
            "raw_heading_deg": raw_heading_deg,
            "raw_gyro_z": gyro_z,
            "sensor_timestamp": self._last_sensor_timestamp,
            "is_stale": is_stale,
        }
