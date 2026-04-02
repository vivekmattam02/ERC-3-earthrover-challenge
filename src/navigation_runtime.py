"""Controller-facing runtime coordinator for ERC indoor navigation.

This module combines:
- corridor localization
- graph planning
- checkpoint progression

It is the thin handoff layer between perception/planning and control.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from corridor_localizer import CorridorLocalizer, CorridorLocalizerConfig
from graph_planner import GraphPlanner, GraphPlannerConfig


@dataclass
class NavigationRuntimeConfig:
    database_npz: Path
    graph_json: Path
    data_info_json: Optional[Path] = None
    cosplace_repo: Optional[Path] = None
    top_k: int = 5
    max_step_jump: int = 15
    jump_penalty: float = 0.05
    backward_penalty: float = 0.15
    heading_penalty: float = 0.002
    ambiguity_margin: float = 0.05
    ambiguity_hold_min_jump: int = 4
    max_subgoal_hops: int = 3
    min_confidence_to_advance: float = 0.55


class NavigationRuntime:
    """Controller-facing runtime API for the current indoor baseline."""

    def __init__(self, config: NavigationRuntimeConfig):
        self.config = config
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
        self.planner = GraphPlanner(
            GraphPlannerConfig(
                graph_json=config.graph_json,
                data_info_json=config.data_info_json,
                max_subgoal_hops=config.max_subgoal_hops,
                min_confidence_to_advance=config.min_confidence_to_advance,
            )
        )

    def reset(self) -> None:
        self.localizer.reset()

    def revert_localization(self) -> None:
        """Undo the last localization update (used after jump rejection)."""
        self.localizer.revert_last_update()

    def set_checkpoints(
        self,
        checkpoint_steps: Optional[list[int]] = None,
        checkpoint_images: Optional[list[str]] = None,
    ) -> None:
        self.planner.set_checkpoints(checkpoint_steps=checkpoint_steps, checkpoint_images=checkpoint_images)

    def _load_subgoal_image(self, subgoal_image_path: Optional[str]) -> Optional[np.ndarray]:
        if not subgoal_image_path:
            return None
        path = Path(subgoal_image_path)
        if not path.is_file():
            # Graph/database paths may be hardcoded to a different machine.
            # Try to resolve relative to this repo by finding "data/" in the path.
            path_str = str(path)
            marker = "/data/"
            idx = path_str.find(marker)
            if idx >= 0:
                repo_root = Path(__file__).resolve().parents[1]
                path = repo_root / path_str[idx + 1:]  # e.g. "data/corrider_extracted/..."
            if not path.is_file():
                return None
        image = Image.open(path).convert("RGB")
        return np.array(image, dtype=np.uint8)

    def step_to_target(
        self,
        frame_rgb: np.ndarray,
        target_node: Optional[int] = None,
        target_step: Optional[int] = None,
        target_image_name: Optional[str] = None,
        observation_heading_deg: Optional[float] = None,
        hops_ahead: Optional[int] = None,
        load_subgoal_image: bool = True,
        localization_step_min: Optional[int] = None,
        localization_step_max: Optional[int] = None,
    ) -> dict:
        localization = self.localizer.localize_frame(
            frame_rgb,
            observation_heading_deg=observation_heading_deg,
            step_min=localization_step_min,
            step_max=localization_step_max,
        )
        plan = self.planner.plan(
            localization,
            target_node=target_node,
            target_step=target_step,
            target_image_name=target_image_name,
            hops_ahead=hops_ahead,
        )
        subgoal_image_rgb = self._load_subgoal_image(plan["subgoal_image_path"]) if load_subgoal_image else None
        return {
            "localization": localization,
            "plan": plan,
            "controller_input": {
                "observation_rgb": frame_rgb,
                "current_node": plan["current_node"],
                "current_step": plan["current_step"],
                "current_orientation": localization.get("node_orientation"),
                "target_node": plan["target_node"],
                "target_step": plan["target_step"],
                "subgoal_node": plan["subgoal_node"],
                "subgoal_step": plan["subgoal_step"],
                "subgoal_image_name": plan["subgoal_image_name"],
                "subgoal_image_path": plan["subgoal_image_path"],
                "subgoal_image_rgb": subgoal_image_rgb,
                "subgoal_orientation": None if plan["subgoal_metadata"] is None else plan["subgoal_metadata"].get("orientation"),
                "confidence": localization["confidence"],
                "held_previous": localization["held_previous"],
                "stable_steps": localization["stable_steps"],
                "path_found": plan["path_found"],
                "path_error": plan["path_error"],
            },
        }

    def step_to_active_checkpoint(
        self,
        frame_rgb: np.ndarray,
        observation_heading_deg: Optional[float] = None,
        hops_ahead: Optional[int] = None,
        load_subgoal_image: bool = True,
        auto_advance_checkpoint: bool = False,
        localization_step_min: Optional[int] = None,
        localization_step_max: Optional[int] = None,
    ) -> dict:
        localization = self.localizer.localize_frame(
            frame_rgb,
            observation_heading_deg=observation_heading_deg,
            step_min=localization_step_min,
            step_max=localization_step_max,
        )
        plan = self.planner.plan_to_active_checkpoint(localization, hops_ahead=hops_ahead)
        if auto_advance_checkpoint and plan["checkpoint_reached"]:
            next_target = self.planner.advance_checkpoint()
        else:
            next_target = self.planner.get_active_checkpoint()
        subgoal_image_rgb = self._load_subgoal_image(plan["subgoal_image_path"]) if load_subgoal_image else None
        return {
            "localization": localization,
            "plan": plan,
            "controller_input": {
                "observation_rgb": frame_rgb,
                "current_node": plan["current_node"],
                "current_step": plan["current_step"],
                "current_orientation": localization.get("node_orientation"),
                "target_node": plan["target_node"],
                "target_step": plan["target_step"],
                "subgoal_node": plan["subgoal_node"],
                "subgoal_step": plan["subgoal_step"],
                "subgoal_image_name": plan["subgoal_image_name"],
                "subgoal_image_path": plan["subgoal_image_path"],
                "subgoal_image_rgb": subgoal_image_rgb,
                "subgoal_orientation": None if plan["subgoal_metadata"] is None else plan["subgoal_metadata"].get("orientation"),
                "confidence": localization["confidence"],
                "held_previous": localization["held_previous"],
                "stable_steps": localization["stable_steps"],
                "checkpoint_reached": plan["checkpoint_reached"],
                "next_active_checkpoint": next_target,
                "path_found": plan["path_found"],
                "path_error": plan["path_error"],
            },
        }
