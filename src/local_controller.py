"""Local controller baselines for ERC indoor corridor navigation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


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
        """

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

    def compute_command(self, controller_input: LocalControllerInput, observation_heading_deg: Optional[float] = None,) -> ControlCommand:
        """Computes a motor command from a high-level plan.

        Args:
            controller_input (dict): A plan dictionary from `GraphPlanner.plan()`.
                                     It must contain keys like `confidence`, `current_step`,
                                     `subgoal_step`, `current_orientation`, `subgoal_orientation`,
                                     and `held_previous`.
            observation_heading_deg (Optional[float]): The robot's current heading, which can
                                                       override the heading from the graph node.

        Returns:
            ControlCommand: A ControlCommand object with the calculated linear and angular velocities.
        """
        # --- 1. Extract inputs from the plan ---
        confidence: float = controller_input.confidence
        current_step: int = controller_input.current_step
        subgoal_step: int = controller_input.subgoal_step
        current_orientation: float = controller_input.current_orientation
        subgoal_orientation: Optional[float] = controller_input.subgoal_orientation
        held_previous: bool = controller_input.held_previous

        # --- 2. Safety checks based on localization confidence ---
        if confidence < self.config.confidence_stop_threshold:
            # If confidence is very low, stop completely.
            return ControlCommand(0.0, 0.0, "low_confidence_stop")

        if current_step is None or subgoal_step is None:
            return ControlCommand(0.0, 0.0, "missing_step_info")

        # --- 3. Calculate error terms ---
        step_gap = int(subgoal_step) - int(current_step)
        if step_gap <= 0:
            # If we are at or past the subgoal, stop.
            return ControlCommand(0.0, 0.0, "subgoal_reached_or_behind")

        # Use the most up-to-date heading available.
        heading_reference = observation_heading_deg
        if heading_reference is None:
            heading_reference = current_orientation

        # Calculate heading error: how much we need to turn to face the subgoal.
        heading_error = 0.0
        if heading_reference is not None and subgoal_orientation is not None:
            heading_error = wrap_angle_deg(float(subgoal_orientation) - float(heading_reference))

        # --- 4. Compute velocities using a simple proportional controller ---

        # Angular velocity is proportional to the heading error.
        angular = self.config.heading_gain * heading_error
        angular = max(-self.config.max_angular, min(self.config.max_angular, angular))

        # Linear velocity has a minimum base speed and increases with distance (step_gap).
        linear = min(
            self.config.max_linear,
            self.config.min_linear + self.config.step_gain * max(1, step_gap),
        )

        # --- 5. Apply special condition adjustments ---

        # If the heading error is large, prioritize turning by setting linear speed to zero.
        if abs(heading_error) >= self.config.turn_in_place_threshold_deg:
            linear = 0.0
        # If confidence is low (but not critically low), slow down.
        elif confidence < self.config.confidence_slow_threshold:
            linear *= 0.6

        # If the localizer is holding a past estimate, be more cautious.
        if held_previous:
            linear *= self.config.held_previous_linear_scale

        # --- 6. Finalize and return the command ---
        linear = max(0.0, min(self.config.max_linear, linear))
        return ControlCommand(linear=linear, angular=angular, reason="simple_heading_controller")
