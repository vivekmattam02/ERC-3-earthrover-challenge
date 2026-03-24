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
    stale_timeout_s: float = 1.0


class SensorStateFilter:
    """Small motion prior layer for heading/gyro/RPM smoothing."""

    def __init__(self, config: SensorStateFilterConfig = SensorStateFilterConfig()):
        self.config = config
        self._filtered_heading_deg: Optional[float] = None
        self._filtered_heading_rate_dps: float = 0.0
        self._filtered_rpm_mean: float = 0.0
        self._last_update_ts: Optional[float] = None

    def reset(self) -> None:
        self._filtered_heading_deg = None
        self._filtered_heading_rate_dps = 0.0
        self._filtered_rpm_mean = 0.0
        self._last_update_ts = None

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

    def update(self, data: Optional[dict]) -> dict:
        now = time.time()
        if not data:
            return {
                "heading_deg": self._filtered_heading_deg,
                "heading_rate_dps": self._filtered_heading_rate_dps,
                "rpm_mean": self._filtered_rpm_mean,
                "is_stale": True,
            }

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

        self._last_update_ts = now
        is_stale = False
        if self._last_update_ts is not None:
            is_stale = (now - self._last_update_ts) > self.config.stale_timeout_s

        return {
            "heading_deg": self._filtered_heading_deg,
            "heading_rate_dps": self._filtered_heading_rate_dps,
            "rpm_mean": self._filtered_rpm_mean,
            "raw_heading_deg": raw_heading_deg,
            "raw_gyro_z": gyro_z,
            "is_stale": is_stale,
        }
