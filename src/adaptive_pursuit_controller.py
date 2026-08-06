"""Adaptive pursuit-style controller for no-GPS route repeat on rough terrain.

This controller is still pragmatic, but it moves the rover away from a brittle
align-then-drive policy toward continuous curvature tracking with explicit pivot
fallbacks and progress-aware speed scheduling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from local_controller import ControlCommand, wrap_angle_deg


@dataclass
class AdaptivePursuitControllerConfig:
    max_linear: float = 0.28
    min_linear: float = 0.10
    cautious_min_linear: float = 0.08
    max_angular: float = 0.34
    min_turn_angular: float = 0.08
    min_pivot_angular: float = 0.14
    pivot_heading_gain: float = 0.010
    pursuit_gain: float = 0.42
    pursuit_lookahead_scale: float = 2.4
    heading_filter_alpha: float = 0.40
    angular_rate_limit: float = 0.08
    turn_rate_damping_gain: float = 0.004
    gyro_drive_correction_deadband_dps: float = 3.0
    step_gain: float = 0.014
    lookahead_cap_steps: int = 12
    confidence_stop_threshold: float = 0.30
    confidence_cautious_threshold: float = 0.48
    low_confidence_linear_scale: float = 0.60
    pivot_enter_threshold_deg: float = 70.0
    pivot_exit_threshold_deg: float = 20.0
    max_pivot_ticks: int = 10
    slow_heading_threshold_deg: float = 18.0
    hard_heading_threshold_deg: float = 38.0
    very_hard_heading_threshold_deg: float = 60.0
    high_turn_rate_threshold_dps: float = 18.0
    high_turn_rate_linear_scale: float = 0.55
    held_previous_linear_scale: float = 0.60
    rpm_motion_threshold: float = 0.8
    rpm_low_linear_scale: float = 0.78
    no_progress_creep_ticks: int = 4
    no_progress_pivot_ticks: int = 9
    no_progress_linear_scale: float = 0.45
    no_progress_angular_scale: float = 1.25
    step_progress_epsilon: int = 1


class AdaptivePursuitController:
    """Curvature-style local controller for differential-drive route repeat."""

    def __init__(self, config: AdaptivePursuitControllerConfig):
        self.config = config
        self._pivot_mode = False
        self._pivot_ticks = 0
        self._filtered_heading_error = 0.0
        self._previous_angular = 0.0
        self._last_current_step: Optional[int] = None
        self._no_progress_ticks = 0

    def reset(self) -> None:
        self._pivot_mode = False
        self._pivot_ticks = 0
        self._filtered_heading_error = 0.0
        self._previous_angular = 0.0
        self._last_current_step = None
        self._no_progress_ticks = 0

    def _smooth_heading_error(self, heading_error: float) -> float:
        alpha = self.config.heading_filter_alpha
        self._filtered_heading_error = (
            alpha * heading_error + (1.0 - alpha) * self._filtered_heading_error
        )
        return self._filtered_heading_error

    def _rate_limit_angular(self, desired_angular: float) -> float:
        delta = desired_angular - self._previous_angular
        max_delta = self.config.angular_rate_limit
        if delta > max_delta:
            desired_angular = self._previous_angular + max_delta
        elif delta < -max_delta:
            desired_angular = self._previous_angular - max_delta
        self._previous_angular = desired_angular
        return desired_angular

    def _update_progress_state(self, current_step: int) -> None:
        if self._last_current_step is None:
            self._last_current_step = current_step
            self._no_progress_ticks = 0
            return

        if current_step >= self._last_current_step + self.config.step_progress_epsilon:
            self._no_progress_ticks = 0
        else:
            self._no_progress_ticks += 1
        self._last_current_step = current_step

    def _cautious_linear(self, step_gap: int, confidence: float, held_previous: bool) -> float:
        linear = min(
            self.config.max_linear * 0.55,
            self.config.min_linear + self.config.step_gain * max(1, step_gap),
        )
        if confidence < self.config.confidence_cautious_threshold:
            linear *= self.config.low_confidence_linear_scale
        if self._no_progress_ticks >= self.config.no_progress_creep_ticks:
            linear *= self.config.no_progress_linear_scale
        if held_previous:
            linear *= self.config.held_previous_linear_scale
        return max(self.config.cautious_min_linear, linear)

    def compute_command(
        self,
        controller_input: dict,
        observation_heading_deg: Optional[float] = None,
    ) -> ControlCommand:
        confidence = float(controller_input.get("confidence", 0.0))
        current_step = controller_input.get("current_step")
        subgoal_step = controller_input.get("subgoal_step")
        current_orientation = controller_input.get("current_orientation")
        subgoal_orientation = controller_input.get("subgoal_orientation")
        held_previous = bool(controller_input.get("held_previous", False))
        heading_rate_dps = float(controller_input.get("heading_rate_dps", 0.0) or 0.0)
        rpm_mean = float(controller_input.get("rpm_mean", 0.0) or 0.0)
        motion_state_stale = bool(controller_input.get("motion_state_stale", False))

        if motion_state_stale:
            self.reset()
            return ControlCommand(0.0, 0.0, "motion_state_stale_stop")

        if confidence < self.config.confidence_stop_threshold:
            self.reset()
            return ControlCommand(0.0, 0.0, "low_confidence_stop")

        if current_step is None or subgoal_step is None:
            self.reset()
            return ControlCommand(0.0, 0.0, "missing_step_info")

        step_gap = int(subgoal_step) - int(current_step)
        if step_gap <= 0:
            self._pivot_mode = False
            self._pivot_ticks = 0
            self._previous_angular = 0.0
            return ControlCommand(0.0, 0.0, "subgoal_reached_or_behind", debug={"step_gap": step_gap})

        self._update_progress_state(int(current_step))

        heading_reference = observation_heading_deg
        if heading_reference is None:
            heading_reference = current_orientation

        if heading_reference is None or subgoal_orientation is None:
            self._pivot_mode = False
            self._pivot_ticks = 0
            self._previous_angular = 0.0
            linear = self._cautious_linear(step_gap, confidence, held_previous)
            return ControlCommand(linear, 0.0, "no_heading_cautious_crawl")

        # Headings in this stack are compass-style: 0=N, 90=E, 180=S, 270=W.
        # A right turn increases compass heading, while a left turn decreases it.
        # Our angular convention is positive=left, negative=right, so the signed
        # control error must be current - target, not target - current.
        raw_heading_error = wrap_angle_deg(float(heading_reference) - float(subgoal_orientation))
        heading_error = self._smooth_heading_error(raw_heading_error)
        abs_raw_error = abs(raw_heading_error)
        abs_error = abs(heading_error)

        if self._pivot_mode:
            if abs_raw_error <= self.config.pivot_exit_threshold_deg:
                self._pivot_mode = False
                self._pivot_ticks = 0
        else:
            if (
                abs_raw_error >= self.config.pivot_enter_threshold_deg
                or (
                    self._no_progress_ticks >= self.config.no_progress_pivot_ticks
                    and abs_raw_error >= self.config.hard_heading_threshold_deg
                )
            ):
                self._pivot_mode = True
                self._pivot_ticks = 0

        if self._pivot_mode:
            self._pivot_ticks += 1
            if self._pivot_ticks > self.config.max_pivot_ticks:
                # Break out of endless spin loops with a small forward probe.
                self._pivot_mode = False
                self._pivot_ticks = 0
                linear = 0.06 if confidence >= self.config.confidence_stop_threshold else 0.0
                angular = self.config.min_turn_angular * (1.0 if heading_error > 0 else -1.0)
                angular = self._rate_limit_angular(angular)
                return ControlCommand(
                    linear=linear,
                    angular=angular,
                    reason="pivot_timeout_probe",
                    debug={
                        "step_gap": step_gap,
                        "heading_error_deg": heading_error,
                        "no_progress_ticks": self._no_progress_ticks,
                    },
                )

            angular = self.config.pivot_heading_gain * heading_error
            if abs_error > 1e-6 and abs(angular) < self.config.min_pivot_angular:
                angular = self.config.min_pivot_angular * (1.0 if heading_error > 0 else -1.0)
            angular -= self.config.turn_rate_damping_gain * heading_rate_dps
            angular = max(-self.config.max_angular, min(self.config.max_angular, angular))
            angular = self._rate_limit_angular(angular)
            return ControlCommand(
                0.0,
                angular,
                "pivot_to_subgoal",
                debug={
                    "step_gap": step_gap,
                    "heading_error_deg": heading_error,
                    "pivot_ticks": self._pivot_ticks,
                    "no_progress_ticks": self._no_progress_ticks,
                },
            )

        self._pivot_ticks = 0

        lookahead_steps = max(1, min(step_gap, self.config.lookahead_cap_steps))
        error_rad = math.radians(heading_error)
        lookahead_scale = 1.0 + (self.config.pursuit_lookahead_scale / float(lookahead_steps))
        angular = self.config.pursuit_gain * math.sin(error_rad) * lookahead_scale

        if abs(heading_rate_dps) > self.config.gyro_drive_correction_deadband_dps:
            angular -= self.config.turn_rate_damping_gain * heading_rate_dps

        if self._no_progress_ticks >= self.config.no_progress_creep_ticks:
            angular *= self.config.no_progress_angular_scale

        if abs_error > 1e-6 and abs(angular) < self.config.min_turn_angular and abs_error >= 8.0:
            angular = self.config.min_turn_angular * (1.0 if heading_error > 0 else -1.0)

        angular = max(-self.config.max_angular, min(self.config.max_angular, angular))
        angular = self._rate_limit_angular(angular)

        linear = min(
            self.config.max_linear,
            self.config.min_linear + self.config.step_gain * float(lookahead_steps),
        )

        reason = "pursuit_track"
        if confidence < self.config.confidence_cautious_threshold:
            linear *= self.config.low_confidence_linear_scale
            reason = "pursuit_track_low_conf"
        if abs_error > self.config.very_hard_heading_threshold_deg:
            linear *= 0.28
            reason = f"{reason}_very_hard_turn"
        elif abs_error > self.config.hard_heading_threshold_deg:
            linear *= 0.42
            reason = f"{reason}_hard_turn"
        elif abs_error > self.config.slow_heading_threshold_deg:
            linear *= 0.68
            reason = f"{reason}_slow_turn"
        if abs(heading_rate_dps) > self.config.high_turn_rate_threshold_dps:
            linear *= self.config.high_turn_rate_linear_scale
            reason = f"{reason}_high_turn_rate"
        if 0.0 < rpm_mean < self.config.rpm_motion_threshold:
            linear *= self.config.rpm_low_linear_scale
            reason = f"{reason}_low_rpm"
        if self._no_progress_ticks >= self.config.no_progress_creep_ticks:
            linear *= self.config.no_progress_linear_scale
            reason = f"{reason}_no_progress"
        if held_previous:
            linear *= self.config.held_previous_linear_scale
            reason = f"{reason}_held_previous"

        linear = max(max(0.06, self.config.cautious_min_linear * 0.9), min(self.config.max_linear, linear))
        return ControlCommand(
            linear=linear,
            angular=angular,
            reason=reason,
            debug={
                "step_gap": step_gap,
                "lookahead_steps": lookahead_steps,
                "heading_error_deg": heading_error,
                "heading_rate_dps": heading_rate_dps,
                "rpm_mean": rpm_mean,
                "no_progress_ticks": self._no_progress_ticks,
            },
        )
