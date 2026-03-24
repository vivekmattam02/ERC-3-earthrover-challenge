"""Controller-facing runtime coordinator for ERC indoor navigation.

This module combines:
- corridor localization
- graph planning
- checkpoint progression

It is the thin handoff layer between perception/planning and control, providing
a single entry point for a robot's main control loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from corridor_localizer import CorridorLocalizer, CorridorLocalizerConfig
from graph_planner import GraphPlan, GraphPlanner, GraphPlannerConfig
from local_controller import LocalControllerInput


@dataclass
class NavigationRuntimeConfig:
    """A unified configuration for the entire navigation stack.

    This dataclass collects all parameters needed to initialize the
    `CorridorLocalizer` and `GraphPlanner` modules.

    Attributes:
        database_npz (Path): Path to the `.npz` file containing the image embeddings for the environment.
        graph_json (Path): Path to the `.json` file representing the connectivity of the environment nodes.
        data_info_json (Optional[Path]): Optional path to a JSON file with additional data information.
        cosplace_repo (Optional[Path]): Optional path to the CosPlace repository, if needed.
        top_k (int): The number of top candidate images to consider during localization.
        max_step_jump (int): The maximum allowed jump in steps between consecutive localizations.
        jump_penalty (float): Penalty applied for jumps in estimated position.
        backward_penalty (float): Penalty for moving backward along the corridor.
        heading_penalty (float): Penalty for deviations from the expected heading.
        ambiguity_margin (float): The margin used to detect localization ambiguity.
        ambiguity_hold_min_jump (int): Minimum jump size to hold ambiguity state.
        max_subgoal_hops (int): The maximum number of hops to look ahead when selecting a subgoal.
        min_confidence_to_advance (float): The minimum confidence required to advance to the next checkpoint.
    """

    # --- Shared Paths ---
    database_npz: Path
    graph_json: Path
    data_info_json: Optional[Path] = None
    cosplace_repo: Optional[Path] = None

    # --- CorridorLocalizer Pass-through ---
    top_k: int = 5
    max_step_jump: int = 15
    jump_penalty: float = 0.05
    backward_penalty: float = 0.15
    heading_penalty: float = 0.002
    ambiguity_margin: float = 0.05
    ambiguity_hold_min_jump: int = 4

    # --- GraphPlanner Pass-through ---
    max_subgoal_search_hops: int = 3
    max_subgoal_cost_threshold: float = 100.
    min_confidence_to_advance: float = 0.55


class NavigationRuntime:
    """The main runtime API for the indoor navigation baseline.

    This class orchestrates the `localize -> plan` pipeline. It initializes all
    necessary components and provides a single `step` method that takes a camera
    frame and returns a dictionary containing the localization result, the
    high-level plan, and a tailored input dictionary for a local controller.
    """

    def __init__(self, config: NavigationRuntimeConfig):
        """Initializes the full navigation stack.

        Args:
            config (NavigationRuntimeConfig): The configuration object for the navigation stack.
        """
        self.config = config
        # Instantiate the CorridorLocalizer.
        self.localizer = CorridorLocalizer(
            CorridorLocalizerConfig(
                database_npz=config.database_npz,
                cosplace_repo=config.cosplace_repo,
                data_info_json=config.data_info_json,
                top_k=config.top_k,
                max_step_jump=config.max_step_jump,
                jump_penalty=config.jump_penalty,
                backward_penalty=config.backward_penalty,
                heading_penalty=config.heading_penalty,
                ambiguity_margin=config.ambiguity_margin,
                ambiguity_hold_min_jump=config.ambiguity_hold_min_jump,
            )
        )
        # Instantiate the GraphPlanner.
        self.planner = GraphPlanner(
            GraphPlannerConfig(
                graph_json=config.graph_json,
                data_info_json=config.data_info_json,
                max_subgoal_search_hops=config.max_subgoal_search_hops,
                max_subgoal_cost_threshold=config.max_subgoal_cost_threshold,
                min_confidence_to_advance=config.min_confidence_to_advance,
            )
        )

    def reset(self) -> None:
        """Resets the state of the underlying localizer."""
        self.localizer.reset()

    def set_checkpoints(self, checkpoint_steps: Optional[list[int]] = None, checkpoint_images: Optional[list[str]] = None,) -> None:
        """Sets a multi-stage route by defining a sequence of checkpoints.

        Args:
            checkpoint_steps (Optional[list[int]]): A list of step indices defining the route.
            checkpoint_images (Optional[list[str]]): A list of image filenames defining the route.
        """
        self.planner.set_checkpoints(checkpoint_steps=checkpoint_steps, checkpoint_images=checkpoint_images)

    def _load_subgoal_image(self, subgoal_image_path: Optional[str]) -> Optional[np.ndarray]:
        """Helper to load the subgoal image for visualization.

        Args:
            subgoal_image_path (Optional[str]): The path to the subgoal image.

        Returns:
            Optional[np.ndarray]: The loaded image as a NumPy array, or None if the path is invalid.
        """
        if not subgoal_image_path:
            return None
        path = Path(subgoal_image_path)
        if not path.is_file():
            return None
        image = Image.open(path).convert("RGB")
        return np.array(image, dtype=np.uint8)

    def step_to_target(self, frame_rgb: np.ndarray, target_node: Optional[int] = None, target_step: Optional[int] = None, 
                       target_image_name: Optional[str] = None, observation_heading_deg: Optional[float] = None, hops_ahead: Optional[int] = None,
                       load_subgoal_image: bool = True, ) -> dict:
        """Performs one full `localize -> plan` step toward a single, fixed target.

        Args:
            frame_rgb (np.ndarray): The current camera frame as a NumPy array.
            target_node (Optional[int]): Identifier for the final goal node.
            target_step (Optional[int]): The step index of the target.
            target_image_name (Optional[str]): The filename of the target image.
            observation_heading_deg (Optional[float]): The robot's current heading from a compass or IMU.
            hops_ahead (Optional[int]): Overrides default subgoal selection distance.
            load_subgoal_image (bool): If True, loads the subgoal's image for debugging.

        Returns:
            dict: A dictionary containing the `localization` result, the full `plan`,
                  and a smaller `controller_input` dictionary for the local controller.
        """
        # 1. Localize the robot using the current camera frame.
        localization = self.localizer.localize_frame(
            frame_rgb,
            observation_heading_deg=observation_heading_deg,
        )

        # 2. Generate a plan from the new location to the specified target.
        plan: GraphPlan = self.planner.plan(
            localization,
            target_node=target_node,
            target_step=target_step,
            target_image_name=target_image_name,
            hops_ahead=hops_ahead,
        )
        subgoal_image_rgb = self._load_subgoal_image(plan.subgoal_image_path) if load_subgoal_image else None

        # 3. Package the results for the end user and local controller.
        return {
            "localization": localization,
            "plan": plan,
            "controller_input": LocalControllerInput(
                current_node=plan.current_node,
                current_step=plan.current_step,
                current_orientation=localization.get("node_orientation"), # type: ignore
                target_node=plan.target_node,
                target_step=plan.target_step,
                subgoal_node=plan.subgoal_node,
                subgoal_step=plan.subgoal_step,
                subgoal_image_name=plan.subgoal_image_name,
                subgoal_image_path=plan.subgoal_image_path,
                subgoal_image_rgb=subgoal_image_rgb, # type: ignore
                subgoal_orientation=None if plan.subgoal_metadata is None else plan.subgoal_metadata.get("orientation"),
                confidence=localization["confidence"],
                held_previous=localization["held_previous"],
                stable_steps=localization["stable_steps"],
                checkpoint_reached=plan.checkpoint_reached,
                next_active_checkpoint=None,
            ),
        }

    def step_to_active_checkpoint(self, frame_rgb: np.ndarray, observation_heading_deg: Optional[float] = None, hops_ahead: Optional[int] = None,
                                  load_subgoal_image: bool = True, auto_advance_checkpoint: bool = False,) -> dict:
        """Performs one `localize -> plan` step toward the active checkpoint in a route.

        This is the primary method to use for executing multi-stage routes.

        Args:
            frame_rgb (np.ndarray): The current camera frame as a NumPy array.
            observation_heading_deg (Optional[float]): The robot's current heading from a compass or IMU.
            hops_ahead (Optional[int]): Overrides default subgoal selection distance.
            load_subgoal_image (bool): If True, loads the subgoal's image for debugging.
            auto_advance_checkpoint (bool): If True, automatically moves to the next
                                            checkpoint when the current one is reached.

        Returns:
            dict: A dictionary containing the `localization` result, the full `plan`,
                  and a smaller `controller_input` dictionary for the local controller,
                  including checkpoint status.
        """
        # 1. Localize the robot.
        localization = self.localizer.localize_frame(
            frame_rgb,
            observation_heading_deg=observation_heading_deg,
        )

        # 2. Plan a path to the currently active checkpoint.
        plan: GraphPlan = self.planner.plan_to_active_checkpoint(localization, hops_ahead=hops_ahead)

        # 3. If we've reached the checkpoint, automatically advance to the next one.
        if auto_advance_checkpoint and plan.checkpoint_reached:
            next_target = self.planner.advance_checkpoint()
        else:
            next_target = self.planner.get_active_checkpoint()
        subgoal_image_rgb = self._load_subgoal_image(plan.subgoal_image_path) if load_subgoal_image else None

        # 4. Package the results.
        return {
            "localization": localization,
            "plan": plan,
            "controller_input": LocalControllerInput(
                current_node=plan.current_node,
                current_step=plan.current_step,
                current_orientation=localization["node_orientation"],
                target_node=plan.target_node,
                target_step=plan.target_step,
                subgoal_node=plan.subgoal_node,
                subgoal_step=plan.subgoal_step,
                subgoal_image_name=plan.subgoal_image_name,
                subgoal_image_path=plan.subgoal_image_path,
                subgoal_image_rgb=subgoal_image_rgb, # type: ignore
                subgoal_orientation=None if plan.subgoal_metadata is None else plan.subgoal_metadata.get("orientation", 0.0),
                confidence=localization["confidence"],
                held_previous=localization["held_previous"],
                stable_steps=localization["stable_steps"],
                checkpoint_reached=plan.checkpoint_reached,
                next_active_checkpoint=next_target,
            ),
        }
