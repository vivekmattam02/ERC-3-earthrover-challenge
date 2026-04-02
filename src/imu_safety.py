"""IMU-based safety monitor for anti-flip protection.

Reads accelerometer and gyroscope data from the SDK telemetry to detect
dangerous tilt angles and angular velocities that indicate the robot is
about to tip over.

EarthRover platform at rest on flat ground:
  accels ≈ [1.0, 0.0, 0.0, timestamp]   (gravity along first axis)
  gyros  ≈ [gx, gy, gz, timestamp]        (rad/s)

Usage:
    monitor = IMUSafetyMonitor()
    result = monitor.update(telemetry_data)
    if result.emergency_stop:
        rover.stop()
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class IMUSafetyConfig:
    """Configuration for IMU safety thresholds.

    Defaults are deliberately generous to avoid false positives while
    still catching genuine tipping events.
    """

    # Tilt: emergency if the robot tilts beyond this from its normal upright pose.
    # At rest, accel ≈ [1g, 0, 0].  Tilt angle = acos(ax / |a|).
    max_tilt_deg: float = 40.0

    # Angular rate: emergency if roll/pitch rate exceeds this.
    # Gyro values gx, gy are nominally roll and pitch in rad/s, but on the
    # real robot the axis mapping is noisy enough that turn-in-place motion can
    # leak into these channels.  So high gyro alone is not treated as a tip.
    max_pitch_roll_rate_dps: float = 150.0

    # Require some corroborating evidence before gyro can trigger a stop.  This
    # keeps flat-ground turning from latching IMU emergencies.
    gyro_min_tilt_deg: float = 12.0
    gyro_min_vibration: float = 1.0

    # Vibration: emergency on extreme vibration (SDK 'vibration' field).
    vibration_limit: float = 3.0

    # Require this many consecutive bad readings before triggering.
    # Prevents single noisy samples from causing a false stop.
    consecutive_trips_to_stop: int = 2

    # Once emergency is latched, require this many consecutive clear
    # readings before auto-clearing.  Set to 0 to require manual reset.
    consecutive_clear_to_resume: int = 0  # 0 = manual reset required

    # Startup self-calibration: learn the gravity direction from stationary samples
    # instead of assuming a fixed sensor axis.
    calibration_samples: int = 8
    calibration_max_pitch_roll_rate_dps: float = 45.0
    accel_norm_tolerance_g: float = 0.35


@dataclass
class IMUSafetyResult:
    """Result of one IMU safety check tick."""

    tilt_deg: float
    pitch_roll_rate_dps: float
    vibration: float
    emergency_stop: bool
    reason: str


class IMUSafetyMonitor:
    """Monitors IMU data for dangerous tilt and angular rates.

    When a dangerous condition is detected for ``consecutive_trips_to_stop``
    ticks in a row, ``emergency_stop`` latches True.  By default it stays
    latched until ``reset()`` is called (operator confirmation).
    """

    def __init__(self, config: Optional[IMUSafetyConfig] = None) -> None:
        self.config = config or IMUSafetyConfig()
        self._consecutive_tilt_trips: int = 0
        self._consecutive_gyro_trips: int = 0
        self._consecutive_clear: int = 0
        self._emergency_latched: bool = False
        self._gravity_ref: Optional[np.ndarray] = None
        self._calibration_samples: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Raw data extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _latest_accel(data: dict) -> Optional[tuple[float, float, float]]:
        """Extract the most recent [ax, ay, az] from telemetry."""
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

    @staticmethod
    def _latest_gyro(data: dict) -> Optional[tuple[float, float, float]]:
        """Extract the most recent [gx, gy, gz] from telemetry."""
        gyros = data.get("gyros") or []
        if not gyros:
            return None
        latest = gyros[-1]
        if not isinstance(latest, (list, tuple)) or len(latest) < 3:
            return None
        try:
            return float(latest[0]), float(latest[1]), float(latest[2])
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Derived safety metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_accel(ax: float, ay: float, az: float) -> Optional[np.ndarray]:
        vec = np.array([ax, ay, az], dtype=np.float64)
        g_mag = float(np.linalg.norm(vec))
        if g_mag < 0.1:
            return None
        return vec / g_mag

    @staticmethod
    def _compute_tilt_deg(accel_unit: np.ndarray, gravity_ref: np.ndarray) -> float:
        """Tilt angle from the learned gravity-aligned rest pose."""
        dot = float(np.clip(np.dot(accel_unit, gravity_ref), -1.0, 1.0))
        return math.degrees(math.acos(dot))

    @staticmethod
    def _compute_pitch_roll_rate_dps(gx: float, gy: float) -> float:
        """Combined pitch/roll angular rate in deg/s.

        gz (yaw) is intentionally excluded — yaw rotation is normal
        turning, not a tipping hazard.
        """
        rate_rad_s = math.sqrt(gx * gx + gy * gy)
        return math.degrees(rate_rad_s)

    def _update_calibration(self, accel: Optional[tuple[float, float, float]], pitch_roll_dps: float) -> None:
        if accel is None or self._gravity_ref is not None:
            return
        accel_unit = self._normalize_accel(*accel)
        if accel_unit is None:
            return
        accel_norm = math.sqrt(accel[0] * accel[0] + accel[1] * accel[1] + accel[2] * accel[2])
        if abs(accel_norm - 1.0) > self.config.accel_norm_tolerance_g:
            return
        if pitch_roll_dps > self.config.calibration_max_pitch_roll_rate_dps:
            return
        self._calibration_samples.append(accel_unit)
        if len(self._calibration_samples) >= self.config.calibration_samples:
            mean_vec = np.mean(self._calibration_samples, axis=0)
            mean_norm = float(np.linalg.norm(mean_vec))
            if mean_norm > 1e-6:
                self._gravity_ref = mean_vec / mean_norm
                self._calibration_samples.clear()

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(self, data: Optional[dict]) -> IMUSafetyResult:
        """Process one telemetry tick and return the safety verdict.

        Args:
            data: Raw telemetry dict from ``rover.get_data()``, or None.

        Returns:
            IMUSafetyResult with current measurements and stop decision.
        """
        if data is None:
            return IMUSafetyResult(
                tilt_deg=0.0,
                pitch_roll_rate_dps=0.0,
                vibration=0.0,
                emergency_stop=self._emergency_latched,
                reason="imu_no_data" if self._emergency_latched else "",
            )

        # --- Measure ---
        tilt_deg = 0.0
        pitch_roll_dps = 0.0

        accel = self._latest_accel(data)

        gyro = self._latest_gyro(data)
        if gyro is not None:
            pitch_roll_dps = self._compute_pitch_roll_rate_dps(gyro[0], gyro[1])

        self._update_calibration(accel, pitch_roll_dps)
        if accel is not None and self._gravity_ref is not None:
            accel_unit = self._normalize_accel(*accel)
            if accel_unit is not None:
                tilt_deg = self._compute_tilt_deg(accel_unit, self._gravity_ref)

        try:
            vibration = float(data.get("vibration", 0.0) or 0.0)
        except (TypeError, ValueError):
            vibration = 0.0

        # --- Trip detection ---
        tilt_trip = self._gravity_ref is not None and tilt_deg > self.config.max_tilt_deg
        gyro_supported = (
            self._gravity_ref is not None
            and (tilt_deg >= self.config.gyro_min_tilt_deg or vibration >= self.config.gyro_min_vibration)
        )
        gyro_trip = pitch_roll_dps > self.config.max_pitch_roll_rate_dps and gyro_supported
        vibration_trip = vibration > self.config.vibration_limit

        any_trip = tilt_trip or gyro_trip or vibration_trip

        if any_trip:
            self._consecutive_clear = 0
            if tilt_trip:
                self._consecutive_tilt_trips += 1
            else:
                self._consecutive_tilt_trips = 0
            if gyro_trip:
                self._consecutive_gyro_trips += 1
            else:
                self._consecutive_gyro_trips = 0
        else:
            self._consecutive_tilt_trips = 0
            self._consecutive_gyro_trips = 0
            self._consecutive_clear += 1

        # --- Latch logic ---
        should_latch = (
            self._consecutive_tilt_trips >= self.config.consecutive_trips_to_stop
            or self._consecutive_gyro_trips >= self.config.consecutive_trips_to_stop
            or vibration_trip  # vibration is instantaneous — no consecutive requirement
        )

        if should_latch:
            self._emergency_latched = True

        # Auto-resume only if configured (consecutive_clear_to_resume > 0).
        if (
            self._emergency_latched
            and self.config.consecutive_clear_to_resume > 0
            and self._consecutive_clear >= self.config.consecutive_clear_to_resume
        ):
            self._emergency_latched = False

        # --- Build reason string ---
        reason = ""
        if self._emergency_latched:
            parts = []
            if tilt_trip:
                parts.append(f"tilt={tilt_deg:.1f}deg")
            if gyro_trip:
                parts.append(f"gyro={pitch_roll_dps:.1f}dps")
            if vibration_trip:
                parts.append(f"vib={vibration:.2f}")
            if not parts:
                parts.append("latched")
            reason = "imu_emergency(" + ",".join(parts) + ")"
        elif self._gravity_ref is None:
            reason = f"imu_calibrating({len(self._calibration_samples)}/{self.config.calibration_samples})"

        return IMUSafetyResult(
            tilt_deg=tilt_deg,
            pitch_roll_rate_dps=pitch_roll_dps,
            vibration=vibration,
            emergency_stop=self._emergency_latched,
            reason=reason,
        )

    def reset(self, recalibrate: bool = False) -> None:
        """Clear the emergency latch. Set recalibrate=True to discard learned gravity reference."""
        self._consecutive_tilt_trips = 0
        self._consecutive_gyro_trips = 0
        self._consecutive_clear = 0
        self._emergency_latched = False
        if recalibrate:
            self._gravity_ref = None
            self._calibration_samples.clear()
