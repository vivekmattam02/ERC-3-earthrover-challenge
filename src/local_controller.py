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


@dataclass
class SimpleLocalControllerConfig:
    max_linear: float = 0.35
    min_linear: float = 0.08
    max_angular: float = 0.6
    heading_gain: float = 0.015
    step_gain: float = 0.03
    confidence_stop_threshold: float = 0.35
    confidence_slow_threshold: float = 0.6
    turn_in_place_threshold_deg: float = 30.0
    held_previous_linear_scale: float = 0.5


class SimpleLocalController:
    """A simple heading-aware local controller for nearby graph subgoals.

    This is a pragmatic first controller, not the final learned policy.
    """

    def __init__(self, config: SimpleLocalControllerConfig):
        self.config = config

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

        if confidence < self.config.confidence_stop_threshold:
            return ControlCommand(0.0, 0.0, "low_confidence_stop")

        if current_step is None or subgoal_step is None:
            return ControlCommand(0.0, 0.0, "missing_step_info")

        step_gap = int(subgoal_step) - int(current_step)
        if step_gap <= 0:
            return ControlCommand(0.0, 0.0, "subgoal_reached_or_behind")

        heading_reference = observation_heading_deg
        if heading_reference is None:
            heading_reference = current_orientation

        heading_error = 0.0
        if heading_reference is not None and subgoal_orientation is not None:
            heading_error = wrap_angle_deg(float(subgoal_orientation) - float(heading_reference))

        angular = self.config.heading_gain * heading_error
        angular = max(-self.config.max_angular, min(self.config.max_angular, angular))

        linear = min(
            self.config.max_linear,
            self.config.min_linear + self.config.step_gain * max(1, step_gap),
        )

        if abs(heading_error) >= self.config.turn_in_place_threshold_deg:
            linear = 0.0
        elif confidence < self.config.confidence_slow_threshold:
            linear *= 0.6

        if held_previous:
            linear *= self.config.held_previous_linear_scale

        linear = max(0.0, min(self.config.max_linear, linear))
        return ControlCommand(linear=linear, angular=angular, reason="simple_heading_controller")
