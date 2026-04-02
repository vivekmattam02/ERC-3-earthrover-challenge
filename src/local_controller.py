"""Local controller baselines for ERC indoor corridor navigation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def wrap_angle_deg(delta: float) -> float:
    return ((delta + 180.0) % 360.0) - 180.0


@dataclass
class ControlCommand:
    linear: float
    angular: float
    reason: str
    debug: Optional[dict] = None


@dataclass
class SimpleLocalControllerConfig:
    max_linear: float = 0.40
    min_linear: float = 0.20
    max_angular: float = 0.34
    min_turn_angular: float = 0.12
    align_turn_angular: float = 0.22
    heading_gain: float = 0.010
    drive_heading_gain: float = 0.0
    gyro_drive_correction_gain: float = 0.008
    gyro_drive_correction_deadband_dps: float = 3.0
    step_gain: float = 0.02
    confidence_stop_threshold: float = 0.35
    confidence_slow_threshold: float = 0.6
    align_enter_threshold_deg: float = 45.0
    align_exit_threshold_deg: float = 20.0
    max_align_ticks: int = 8
    slow_heading_threshold_deg: float = 20.0
    hard_turn_threshold_deg: float = 55.0
    held_previous_linear_scale: float = 0.5
    heading_filter_alpha: float = 0.35
    angular_rate_limit: float = 0.10
    low_confidence_linear_scale: float = 0.7
    turn_rate_damping_gain: float = 0.004
    high_turn_rate_threshold_dps: float = 20.0
    high_turn_rate_linear_scale: float = 0.55
    rpm_motion_threshold: float = 2.0
    motion_stale_stop: bool = True
    no_progress_realign_ticks: int = 4
    no_progress_crawl_linear_scale: float = 0.5
    step_progress_epsilon: int = 1


class SimpleLocalController:
    """A simple heading-aware local controller for nearby graph subgoals.

    This is a pragmatic first controller, not the final learned policy.
    """

    def __init__(self, config: SimpleLocalControllerConfig):
        self.config = config
        self._align_mode = False
        self._align_ticks = 0
        self._filtered_heading_error = 0.0
        self._previous_angular = 0.0
        self._last_current_step: Optional[int] = None
        self._no_progress_ticks = 0
        self._turn_direction = 1.0
        self._align_start_heading_deg: Optional[float] = None
        self._align_target_delta_deg: Optional[float] = None

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
            self._align_mode = False
            self._align_ticks = 0
            self._previous_angular = 0.0
            self._align_start_heading_deg = None
            self._align_target_delta_deg = None
            return ControlCommand(0.0, 0.0, "motion_state_stale_stop")

        if confidence < self.config.confidence_stop_threshold:
            self._align_mode = False
            self._align_ticks = 0
            self._align_start_heading_deg = None
            self._align_target_delta_deg = None
            return ControlCommand(0.0, 0.0, "low_confidence_stop")

        if current_step is None or subgoal_step is None:
            self._align_mode = False
            self._align_start_heading_deg = None
            self._align_target_delta_deg = None
            return ControlCommand(0.0, 0.0, "missing_step_info", debug={"confidence": confidence})

        step_gap = int(subgoal_step) - int(current_step)
        if step_gap <= 0:
            self._align_mode = False
            self._align_start_heading_deg = None
            self._align_target_delta_deg = None
            return ControlCommand(0.0, 0.0, "subgoal_reached_or_behind", debug={"step_gap": step_gap})

        self._update_progress_state(int(current_step))

        heading_reference = observation_heading_deg
        if heading_reference is None:
            heading_reference = current_orientation

        if heading_reference is None or subgoal_orientation is None:
            self._align_mode = False
            linear = min(
                self.config.max_linear * 0.75,
                self.config.min_linear + self.config.step_gain * max(1, step_gap),
            )
            if confidence < self.config.confidence_slow_threshold:
                linear *= self.config.low_confidence_linear_scale
            if held_previous:
                linear *= self.config.held_previous_linear_scale
            self._previous_angular = 0.0
            self._align_start_heading_deg = None
            self._align_target_delta_deg = None
            linear = max(self.config.min_linear, linear)
            return ControlCommand(linear, 0.0, "no_heading_forward_crawl")

        desired_turn_delta = None
        if current_orientation is not None and subgoal_orientation is not None:
            desired_turn_delta = wrap_angle_deg(float(subgoal_orientation) - float(current_orientation))

        raw_heading_error = wrap_angle_deg(float(subgoal_orientation) - float(heading_reference))
        if self._align_mode and observation_heading_deg is not None and self._align_start_heading_deg is not None and self._align_target_delta_deg is not None:
            turned_so_far = wrap_angle_deg(float(observation_heading_deg) - float(self._align_start_heading_deg))
            raw_heading_error = wrap_angle_deg(float(self._align_target_delta_deg) - turned_so_far)
        elif desired_turn_delta is not None:
            raw_heading_error = desired_turn_delta

        heading_error = self._smooth_heading_error(raw_heading_error)

        if self._align_mode:
            self._align_mode = abs(heading_error) > self.config.align_exit_threshold_deg
        else:
            self._align_mode = abs(heading_error) > self.config.align_enter_threshold_deg
            if self._align_mode:
                self._align_start_heading_deg = observation_heading_deg
                self._align_target_delta_deg = desired_turn_delta

        if self._align_mode and self._align_ticks >= self.config.max_align_ticks:
            self._align_mode = False
            self._align_ticks = 0

        if self._align_mode:
            self._align_ticks += 1
            angular = self.config.heading_gain * heading_error
            if abs(heading_error) > 1e-6 and abs(angular) < self.config.min_turn_angular:
                angular = self.config.min_turn_angular * (1.0 if heading_error > 0 else -1.0)
            angular -= self.config.turn_rate_damping_gain * heading_rate_dps
            angular = max(-self.config.max_angular, min(self.config.max_angular, angular))
            angular = self._rate_limit_angular(angular)
            return ControlCommand(0.0, angular, "align_heading")

        self._align_ticks = 0
        self._previous_angular = 0.0
        self._align_start_heading_deg = None
        self._align_target_delta_deg = None

        desired_linear = min(
            self.config.max_linear,
            self.config.min_linear + self.config.step_gain * max(1, step_gap),
        )

        linear = desired_linear

        # Gyro-based heading correction: counteract unwanted rotation
        # during forward drive.  Compass is disabled indoors, so this is
        # the only course-correction signal.
        angular = 0.0
        if abs(heading_rate_dps) > self.config.gyro_drive_correction_deadband_dps:
            angular = -self.config.gyro_drive_correction_gain * heading_rate_dps
            angular = max(-self.config.max_angular * 0.5,
                          min(self.config.max_angular * 0.5, angular))

        if confidence < self.config.confidence_slow_threshold:
            linear *= self.config.low_confidence_linear_scale
        if abs(heading_rate_dps) > self.config.high_turn_rate_threshold_dps:
            linear *= self.config.high_turn_rate_linear_scale
        if 0.0 < rpm_mean < self.config.rpm_motion_threshold:
            linear *= 0.8
        if self._no_progress_ticks >= self.config.no_progress_realign_ticks:
            linear *= self.config.no_progress_crawl_linear_scale
        if held_previous:
            linear *= self.config.held_previous_linear_scale

        linear = max(self.config.min_linear, min(self.config.max_linear, linear))
        return ControlCommand(linear=linear, angular=angular, reason="drive_to_subgoal")
