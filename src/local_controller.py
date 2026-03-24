"""Local controller baselines for ERC indoor corridor navigation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pygame.math import clamp


def wrap_angle_deg(delta: float) -> float:
    """Wraps an angle difference in degrees to the range [-180, 180]."""
    return ((delta + 180.0) % 360.0) - 180.0


@dataclass
class LocalControllerInput:
    """Represents the high-level plan and localization information needed by the local controller.
    
    Attributes:
        current_node (int): The current graph node ID where the robot is localized.
        current_step (int): The current step index in the graph path.
        current_orientation (float): The robot's current orientation in degrees.
        target_node (int): The target graph node ID for the current navigation goal.
        target_step (int): The target step index in the graph path.
        subgoal_node (int): The intermediate subgoal graph node ID selected by the planner.
        subgoal_step (int): The step index of the subgoal in the graph path.
        subgoal_image_name (Optional[str]): The filename of the subgoal image.
        subgoal_image_path (Optional[str]): The path to the subgoal image.
        subgoal_image_rgb (Optional[bytes]): The subgoal image as bytes.
        subgoal_orientation (Optional[float]): The orientation of the subgoal in degrees.
        confidence (float): The localization confidence for the current node.
        held_previous (bool): Whether the temporal localizer is currently holding a previous estimate.
        stable_steps (int): The number of consecutive steps the robot has been localized to the same node.
        checkpoint_reached (bool): Whether the current node is a checkpoint that has been reached.
        next_active_checkpoint (Optional[int]): The ID of the next active checkpoint.
        path_found (bool): Whether a valid path to the target was found by the planner.
        path_error (Optional[str]): An error message if pathfinding failed, otherwise None.
        heading_rate_dps (float): The rate of change of the robot's heading in degrees per second.
        rpm_mean (float): The mean RPM of the robot's wheels.
        motion_state_stale (bool): Whether the motion state information is outdated.
        """

    observation_rgb: bytes
    current_node: int
    current_step: int
    current_orientation: float
    target_node: int
    target_step: int
    subgoal_node: int
    subgoal_step: int
    subgoal_image_name: Optional[str]
    subgoal_image_path: Optional[str]
    subgoal_image_rgb: Optional[bytes]
    subgoal_orientation: Optional[float]
    confidence: float
    held_previous: bool
    stable_steps: int
    checkpoint_reached: bool
    next_active_checkpoint: Optional[int]
    path_found: bool
    path_error: Optional[str]
    heading_rate_dps: float
    rpm_mean: float
    motion_state_stale: bool


@dataclass
class ControlCommand:
    """Represents a single low-level motor command for the robot.

    Attributes:
        linear (float): The target forward velocity (m/s).
        angular (float): The target rotational velocity (rad/s).
        reason (str): A string indicating the logic used to generate the command.
    """

    linear: float
    angular: float
    reason: str
    debug: Optional[dict] = None


@dataclass
class SimpleLocalControllerConfig:
    """Tuning parameters for the SimpleLocalController.

    Attributes:
        max_linear (float): Absolute maximum forward speed.
        min_linear (float): A base speed to ensure the robot always makes progress.
        max_angular (float): Absolute maximum rotational speed.
        heading_gain (float): Proportional gain for turning to face the subgoal (P-controller).
        step_gain (float): Proportional gain for increasing speed based on distance to subgoal.
        confidence_stop_threshold (float): If localization confidence is below this, stop.
        confidence_slow_threshold (float): If confidence is below this, reduce speed.
        turn_in_place_threshold_deg (float): If heading error exceeds this, stop moving forward
                                     and only rotate.
        held_previous_linear_scale (float): Speed multiplier when temporal localization is
                                    holding a previous estimate.
    """
    max_linear: float = 0.24
    min_linear: float = 0.06
    max_angular: float = 0.34
    min_turn_angular: float = 0.12
    align_turn_angular: float = 0.22
    heading_gain: float = 0.010
    drive_heading_gain: float = 0.007
    step_gain: float = 0.02
    confidence_stop_threshold: float = 0.35
    confidence_slow_threshold: float = 0.6
    align_enter_threshold_deg: float = 32.0
    align_exit_threshold_deg: float = 12.0
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

    This controller takes a high-level plan (containing the current location and a
    near-term subgoal) and computes the linear and angular velocities required
    to drive towards that subgoal.

    This is a pragmatic first controller, not the final learned policy.
    """

    def __init__(self, config: SimpleLocalControllerConfig):
        """Initializes the controller with its tuning parameters.

        Args:
            config (SimpleLocalControllerConfig): The configuration object for the controller.
        """
        self.config = config
        self._align_mode = False
        self._filtered_heading_error = 0.0
        self._previous_angular = 0.0
        self._last_current_step: Optional[int] = None
        self._no_progress_ticks = 0
        self._turn_direction = 1.0

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
        controller_input: LocalControllerInput,
        observation_heading_deg: Optional[float] = None,
        frame_rgb: Optional[object] = None,
    ) -> ControlCommand:
        """Computes a motor command from a high-level plan.

        Args:
            controller_input (LocalControllerInput): A plan object from `GraphPlanner.plan()`.
            observation_heading_deg (Optional[float]): The robot's current heading, which can
                                                       override the heading from the graph node.

        Returns:
            ControlCommand: A ControlCommand object with the calculated linear and angular velocities.
        """
        confidence = float(controller_input.confidence or 0.0)
        current_step = controller_input.current_step
        subgoal_step = controller_input.subgoal_step
        current_orientation = controller_input.current_orientation
        subgoal_orientation = controller_input.subgoal_orientation
        held_previous = bool(controller_input.held_previous)
        heading_rate_dps = float(controller_input.heading_rate_dps or 0.0)
        rpm_mean = float(controller_input.rpm_mean or 0.0)
        motion_state_stale = bool(controller_input.motion_state_stale)

        if confidence < self.config.confidence_stop_threshold:
            # If confidence is very low, stop completely.
            self._align_mode = False
            return ControlCommand(0.0, 0.0, "low_confidence_stop", debug={"confidence": confidence})

        if current_step is None or subgoal_step is None:
            self._align_mode = False
            return ControlCommand(0.0, 0.0, "missing_step_info", debug={"confidence": confidence})

        # --- 3. Calculate error terms ---
        step_gap = int(subgoal_step) - int(current_step)
        if step_gap <= 0:
            # If we are at or past the subgoal, stop.
            self._align_mode = False
            return ControlCommand(0.0, 0.0, "subgoal_reached_or_behind", debug={"step_gap": step_gap})

        self._update_progress_state(int(current_step))

        if motion_state_stale and self.config.motion_stale_stop:
            self._align_mode = False
            return ControlCommand(0.0, 0.0, "stale_motion_state_stop", debug={"step_gap": step_gap})

        # Use the most up-to-date heading available.
        heading_reference = observation_heading_deg
        if heading_reference is None:
            heading_reference = current_orientation

        # Calculate heading error: how much we need to turn to face the subgoal.
        if heading_reference is None or subgoal_orientation is None:
            self._align_mode = False
            linear = min(
                self.config.max_linear * 0.5,
                self.config.min_linear + self.config.step_gain * max(1, step_gap),
            )
            if confidence < self.config.confidence_slow_threshold:
                linear *= self.config.low_confidence_linear_scale
            if held_previous:
                linear *= self.config.held_previous_linear_scale
            self._previous_angular = 0.0
            return ControlCommand(
                max(0.0, linear),
                0.0,
                "no_heading_forward_crawl",
                debug={
                    "step_gap": step_gap,
                    "confidence": confidence,
                    "no_progress_ticks": self._no_progress_ticks,
                },
            )

        raw_heading_error = wrap_angle_deg(float(subgoal_orientation) - float(heading_reference))
        heading_error = self._smooth_heading_error(raw_heading_error)

        if self._align_mode:
            self._align_mode = abs(heading_error) > self.config.align_exit_threshold_deg
        else:
            self._align_mode = abs(heading_error) > self.config.align_enter_threshold_deg

        if self._no_progress_ticks >= self.config.no_progress_realign_ticks and abs(heading_error) > self.config.align_exit_threshold_deg:
            self._align_mode = True

        if self._align_mode:
            if abs(heading_error) > self.config.align_enter_threshold_deg:
                self._turn_direction = 1.0 if heading_error >= 0 else -1.0
            angular_mag = self.config.align_turn_angular
            if abs(heading_error) > self.config.hard_turn_threshold_deg:
                angular_mag = self.config.max_angular
            angular = angular_mag * self._turn_direction
            angular -= self.config.turn_rate_damping_gain * heading_rate_dps
            angular = clamp(angular, -self.config.max_angular, self.config.max_angular)
            angular = self._rate_limit_angular(angular)
            return ControlCommand(
                0.0,
                angular,
                "align_heading",
                debug={
                    "raw_heading_error_deg": raw_heading_error,
                    "filtered_heading_error_deg": heading_error,
                    "heading_rate_dps": heading_rate_dps,
                    "step_gap": step_gap,
                    "no_progress_ticks": self._no_progress_ticks,
                    "align_mode": self._align_mode,
                },
            )

        desired_linear = min(
            self.config.max_linear,
            self.config.min_linear + self.config.step_gain * step_gap,
        )
        heading_scale = 1.0
        if abs(heading_error) > self.config.slow_heading_threshold_deg:
            overflow = min(
                1.0,
                (abs(heading_error) - self.config.slow_heading_threshold_deg)
                / max(1.0, self.config.hard_turn_threshold_deg - self.config.slow_heading_threshold_deg),
            )
            heading_scale = max(0.25, 1.0 - 0.75 * overflow)

        linear = desired_linear * heading_scale
        angular = self.config.drive_heading_gain * heading_error
        angular -= self.config.turn_rate_damping_gain * heading_rate_dps
        angular = max(-self.config.max_angular * 0.7, min(self.config.max_angular * 0.7, angular))

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

        # --- 6. Finalize and return the command ---
        linear = clamp(linear, 0.0, self.config.max_linear)
        return ControlCommand(
            linear=linear,
            angular=angular,
            reason="drive_to_subgoal",
            debug={
                "raw_heading_error_deg": raw_heading_error,
                "filtered_heading_error_deg": heading_error,
                "heading_rate_dps": heading_rate_dps,
                "step_gap": step_gap,
                "no_progress_ticks": self._no_progress_ticks,
                "align_mode": self._align_mode,
                "rpm_mean": rpm_mean,
            },)

# class MBRALocalController(SimpleLocalController):
#     """A simple local controller that also considers the previous action."""

#     def compute_command(
#         self,
#         controller_input: dict,
#         observation_heading_deg: Optional[float] = None,
#     ) -> ControlCommand:
#         command = super().compute_command(controller_input, observation_heading_deg=observation_heading_deg)
#         # Here you could add additional logic to modify the command based on previous actions or other factors.
#         return command