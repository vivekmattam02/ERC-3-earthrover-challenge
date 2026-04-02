"""Classical outdoor GPS controller with optional VFH obstacle avoidance.

This module is the non-ML fallback controller for ERC outdoor missions.

Its core policy is:
- drive toward the next GPS / UTM waypoint using a proportional heading controller
- if polar depth clearance is available, steer toward the closest open sector
- stop or slow down when forward clearance is too small
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


def wrap_angle_rad(delta: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return (delta + math.pi) % (2.0 * math.pi) - math.pi


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class OutdoorControlCommand:
    linear: float
    angular: float
    reason: str
    debug: Optional[dict[str, Any]] = None


@dataclass
class OutdoorGPSControllerConfig:
    goal_reached_radius_m: float = 3.0
    max_linear: float = 0.30
    min_linear: float = 0.08
    max_angular: float = 1.00
    angular_gain: float = 1.20
    forward_heading_deadband_deg: float = 8.0
    in_place_turn_threshold_deg: float = 80.0
    in_place_turn_exit_deg: float = 60.0    # hysteresis: stay in turn-in-place until error drops below this (must be < threshold)
    nominal_linear: float = 0.25
    approach_slowdown_m: float = 12.0       # begin proportional speed reduction this far from goal
    vfh_engage_distance_m: float = 0.80     # obstacle avoidance engages below this forward clearance
    vfh_clear_distance_m: float = 1.10      # obstacle avoidance releases above this forward clearance
    vfh_forward_bin_window: int = 1         # center +/- N bins used for forward-clearance checks
    avoidance_linear_scale: float = 0.55    # cap forward speed while actively avoiding obstacles
    avoidance_heading_alpha: float = 0.45   # smooth VFH steering changes while avoiding
    vfh_num_bins: int = 16
    vfh_fov_horizontal_deg: float = 90.0
    vfh_blocked_distance_m: float = 0.80
    depth_slow_distance_m: float = 0.80
    depth_stop_distance_m: float = 0.40
    depth_crop_bottom: float = 0.60
    fallback_clearance_margin_m: float = 0.50


class OutdoorGPSController:
    """GPS waypoint follower with optional VFH steering override."""

    def __init__(
        self,
        config: OutdoorGPSControllerConfig,
        depth_estimator: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.depth_estimator = depth_estimator
        self._turn_in_place: bool = False  # hysteresis state for in-place turning
        self._avoidance_active: bool = False
        self._avoidance_heading_rad: Optional[float] = None

    def _compute_goal_geometry(
        self,
        current_utm: tuple[float, float],
        goal_utm: tuple[float, float],
        current_heading_rad: float,
    ) -> tuple[float, float, float]:
        dx = float(goal_utm[0]) - float(current_utm[0])
        dy = float(goal_utm[1]) - float(current_utm[1])
        distance = math.hypot(dx, dy)
        goal_bearing = math.atan2(dy, dx)
        bearing_error = wrap_angle_rad(goal_bearing - float(current_heading_rad))
        return distance, goal_bearing, bearing_error

    def _default_command(
        self, bearing_error: float, distance_to_goal: float = float("inf")
    ) -> tuple[float, float, str]:
        heading_error_deg = math.degrees(abs(bearing_error))
        angular = clamp(
            self.config.angular_gain * bearing_error,
            -self.config.max_angular,
            self.config.max_angular,
        )

        # Hysteresis: enter turn-in-place when error exceeds threshold, exit when it
        # drops below the (lower) exit threshold — prevents threshold-edge flickering.
        if heading_error_deg >= self.config.in_place_turn_threshold_deg:
            self._turn_in_place = True
        elif heading_error_deg < self.config.in_place_turn_exit_deg:
            self._turn_in_place = False

        if self._turn_in_place:
            return 0.0, angular, "turn_in_place_to_goal"

        if heading_error_deg <= self.config.forward_heading_deadband_deg:
            angular = 0.0

        # Bearing-based speed scaling
        linear_scale = max(0.0, math.cos(abs(bearing_error) / 2.0))

        # Distance-based approach slowdown: proportionally reduce speed near goal
        # so we don't overshoot the checkpoint radius at high speed.
        if math.isfinite(distance_to_goal) and distance_to_goal < self.config.approach_slowdown_m:
            dist_scale = max(0.35, distance_to_goal / self.config.approach_slowdown_m)
            linear_scale *= dist_scale

        linear = self.config.nominal_linear * linear_scale
        if linear > 0.0:
            linear = clamp(linear, self.config.min_linear, self.config.max_linear)

        return linear, angular, "gps_bearing_follow"

    def _compute_clearance(
        self,
        depth_map: Optional[np.ndarray],
        clearance: Optional[np.ndarray],
        bin_centers: Optional[np.ndarray],
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if clearance is not None and bin_centers is not None:
            return clearance, bin_centers

        if depth_map is None or self.depth_estimator is None:
            return None, None

        return self.depth_estimator.get_polar_clearance(
            depth_map,
            num_bins=self.config.vfh_num_bins,
            crop_bottom=self.config.depth_crop_bottom,
            fov_horizontal=self.config.vfh_fov_horizontal_deg,
        )

    def _select_vfh_heading(
        self,
        clearance: np.ndarray,
        bin_centers: np.ndarray,
        goal_error_rad: float,
    ) -> tuple[float, float, bool]:
        if clearance.shape[0] != bin_centers.shape[0]:
            raise ValueError("clearance and bin_centers must have the same length")

        open_mask = clearance >= self.config.vfh_blocked_distance_m

        if np.any(open_mask):
            open_angles = bin_centers[open_mask]
            costs = np.abs([wrap_angle_rad(float(a) - goal_error_rad) for a in open_angles])
            best_open_idx = int(np.argmin(costs))
            selected_heading = float(open_angles[best_open_idx])
            selected_clearance = float(clearance[open_mask][best_open_idx])
            return selected_heading, selected_clearance, False

        safest_heading = float(bin_centers[int(np.argmax(clearance))])
        safest_clearance = float(np.max(clearance))
        return safest_heading, safest_clearance, True

    def _forward_clearance(self, clearance: np.ndarray, center_idx: int) -> float:
        half_window = max(0, int(self.config.vfh_forward_bin_window))
        lo = max(0, center_idx - half_window)
        hi = min(clearance.shape[0], center_idx + half_window + 1)
        return float(np.min(clearance[lo:hi]))

    def _smooth_avoidance_heading(self, selected_heading: float) -> float:
        if self._avoidance_heading_rad is None:
            self._avoidance_heading_rad = selected_heading
        else:
            diff = wrap_angle_rad(selected_heading - self._avoidance_heading_rad)
            self._avoidance_heading_rad = wrap_angle_rad(
                self._avoidance_heading_rad + self.config.avoidance_heading_alpha * diff
            )
        return self._avoidance_heading_rad

    def compute_command(
        self,
        current_utm: tuple[float, float],
        goal_utm: tuple[float, float],
        current_heading_rad: float,
        *,
        depth_map: Optional[np.ndarray] = None,
        clearance: Optional[np.ndarray] = None,
        bin_centers: Optional[np.ndarray] = None,
    ) -> OutdoorControlCommand:
        distance_to_goal, goal_bearing, bearing_error = self._compute_goal_geometry(
            current_utm=current_utm,
            goal_utm=goal_utm,
            current_heading_rad=current_heading_rad,
        )

        debug: dict[str, Any] = {
            "current_utm": [float(current_utm[0]), float(current_utm[1])],
            "goal_utm": [float(goal_utm[0]), float(goal_utm[1])],
            "distance_to_goal_m": distance_to_goal,
            "goal_bearing_rad": goal_bearing,
            "bearing_error_rad": bearing_error,
        }

        if distance_to_goal <= self.config.goal_reached_radius_m:
            return OutdoorControlCommand(0.0, 0.0, "goal_reached", debug=debug)

        linear, angular, reason = self._default_command(bearing_error, distance_to_goal)

        clearance_vec, bin_centers_vec = self._compute_clearance(depth_map, clearance, bin_centers)
        if clearance_vec is None or bin_centers_vec is None:
            debug["vfh_used"] = False
            return OutdoorControlCommand(linear, angular, reason, debug=debug)

        selected_heading, selected_clearance, forced_stop = self._select_vfh_heading(
            clearance=clearance_vec,
            bin_centers=bin_centers_vec,
            goal_error_rad=bearing_error,
        )

        center_idx = int(np.argmin(np.abs(bin_centers_vec)))
        forward_clearance = self._forward_clearance(clearance_vec, center_idx)
        obstacle_ahead = forward_clearance < self.config.vfh_engage_distance_m
        clear_ahead = forward_clearance > self.config.vfh_clear_distance_m

        if obstacle_ahead:
            self._avoidance_active = True
        elif self._avoidance_active and clear_ahead:
            self._avoidance_active = False
            self._avoidance_heading_rad = None
        elif not self._avoidance_active:
            self._avoidance_heading_rad = None

        debug.update(
            {
                "forward_clearance_m": forward_clearance,
                "selected_heading_rad": selected_heading,
                "selected_clearance_m": selected_clearance,
                "avoidance_active": self._avoidance_active,
                "clearance": clearance_vec.tolist(),
                "bin_centers_rad": bin_centers_vec.tolist(),
            }
        )

        if forced_stop:
            self._avoidance_active = True
            self._avoidance_heading_rad = None
            debug["vfh_used"] = True
            debug["avoidance_active"] = True
            return OutdoorControlCommand(0.0, 0.0, "depth_stop", debug=debug)

        if not self._avoidance_active:
            debug["vfh_used"] = False
            return OutdoorControlCommand(linear, angular, reason, debug=debug)

        selected_heading = self._smooth_avoidance_heading(selected_heading)
        angular = clamp(
            self.config.angular_gain * selected_heading,
            -self.config.max_angular,
            self.config.max_angular,
        )
        linear = self.config.nominal_linear * self.config.avoidance_linear_scale
        linear *= max(0.0, math.cos(abs(selected_heading) / 2.0))

        if self._turn_in_place or abs(math.degrees(selected_heading)) >= self.config.in_place_turn_threshold_deg:
            linear = 0.0

        if forward_clearance < self.config.depth_stop_distance_m:
            linear = 0.0
            reason = "gps_vfh_turn_clear"
        elif forward_clearance < self.config.depth_slow_distance_m:
            slow_scale = clamp(
                (forward_clearance - self.config.depth_stop_distance_m)
                / max(1e-6, self.config.depth_slow_distance_m - self.config.depth_stop_distance_m),
                0.0,
                1.0,
            )
            linear *= max(0.25, slow_scale)
            reason = "gps_vfh_slow"
        else:
            reason = "gps_vfh_avoid"

        if linear > 0.0:
            linear = clamp(linear, self.config.min_linear, self.config.max_linear)

        debug.update(
            {
                "vfh_used": True,
                "selected_heading_rad": selected_heading,
                "avoidance_active": self._avoidance_active,
            }
        )
        return OutdoorControlCommand(linear, angular, reason, debug=debug)
