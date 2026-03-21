"""MBRA/LogoNav-backed local controller adapter for the indoor runtime.

This module keeps the existing runtime contract:
`controller.compute_command(controller_input, observation_heading_deg, frame_rgb)`.

It adapts the goal-pose-conditioned deployment code from ``mbra_repo_1`` so it
can consume graph-planner subgoals instead of GPS waypoints.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Optional

import numpy as np
import torch
import yaml
from PIL import Image

from local_controller import ControlCommand, wrap_angle_deg


@dataclass
class MBRALocalControllerConfig:
    repo_root: Path
    model_config_path: Path
    checkpoint_path: Path
    device: Optional[str] = None
    confidence_stop_threshold: float = 0.35
    confidence_slow_threshold: float = 0.6
    held_previous_linear_scale: float = 0.5
    max_linear: float = 0.3
    max_angular: float = 0.3
    meters_per_step: float = 0.25
    metric_waypoint_spacing: float = 0.25
    goal_distance_clip_m: float = 30.0
    waypoint_index: int = 2
    min_turning_angular: float = 0.2
    min_turning_linear_threshold: float = 0.05
    min_turning_angular_threshold: float = 0.05
    inference_dt: float = 0.25


class MBRALocalController:
    """Controller adapter around the MBRA/LogoNav deployment model."""

    def __init__(self, config: MBRALocalControllerConfig):
        self.config = config
        self.device = self._resolve_device(config.device)
        self.model_params = self._load_yaml(config.model_config_path)
        self.image_size = tuple(self.model_params.get('image_size', [96, 96]))
        self.context_size = int(self.model_params.get('context_size', 0))
        self.normalize_actions = bool(self.model_params.get('normalize', False))
        self.context_queue: deque[Image.Image] = deque(maxlen=self.context_size + 1)

        utils_module = self._load_utils_module(config.repo_root)
        self._transform_images_mbra = utils_module.transform_images_mbra
        self._to_numpy = utils_module.to_numpy
        self._clip_angle = utils_module.clip_angle
        self.model = utils_module.load_model(
            str(config.checkpoint_path),
            self.model_params,
            self.device,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    def compute_command(
        self,
        controller_input: dict,
        observation_heading_deg: Optional[float] = None,
        frame_rgb: Optional[np.ndarray] = None,
    ) -> ControlCommand:
        confidence = float(controller_input.get('confidence', 0.0))
        if confidence < self.config.confidence_stop_threshold:
            return ControlCommand(0.0, 0.0, 'mbra_low_confidence_stop')

        if frame_rgb is None:
            return ControlCommand(0.0, 0.0, 'mbra_missing_frame')

        current_step = controller_input.get('current_step')
        subgoal_step = controller_input.get('subgoal_step')
        current_orientation = controller_input.get('current_orientation')
        subgoal_orientation = controller_input.get('subgoal_orientation')
        held_previous = bool(controller_input.get('held_previous', False))

        if current_step is None or subgoal_step is None:
            return ControlCommand(0.0, 0.0, 'mbra_missing_step_info')

        step_gap = int(subgoal_step) - int(current_step)
        if step_gap <= 0:
            return ControlCommand(0.0, 0.0, 'mbra_subgoal_reached_or_behind')

        heading_reference = observation_heading_deg
        if heading_reference is None:
            heading_reference = current_orientation
        if heading_reference is None or subgoal_orientation is None:
            return ControlCommand(0.0, 0.0, 'mbra_missing_heading_info')

        self.context_queue.append(self._prepare_frame(frame_rgb))
        if len(self.context_queue) <= self.context_size:
            return ControlCommand(0.0, 0.0, 'mbra_warming_up_context')

        try:
            obs_images = self._build_observation_tensor()
            goal_pose = self._build_goal_pose(
                step_gap=step_gap,
                heading_reference_deg=float(heading_reference),
                subgoal_orientation_deg=float(subgoal_orientation),
            )
            with torch.no_grad():
                waypoints = self.model(obs_images, goal_pose)
            waypoint_np = self._to_numpy(waypoints)
            linear, angular = self._waypoint_to_command(waypoint_np)
        except Exception as exc:
            return ControlCommand(0.0, 0.0, f'mbra_inference_error:{type(exc).__name__}')

        if confidence < self.config.confidence_slow_threshold:
            linear *= 0.6
        if held_previous:
            linear *= self.config.held_previous_linear_scale

        linear = float(np.clip(linear, 0.0, self.config.max_linear))
        angular = float(np.clip(angular, -self.config.max_angular, self.config.max_angular))
        return ControlCommand(linear=linear, angular=angular, reason='mbra_local_controller')

    @staticmethod
    def _resolve_device(explicit_device: Optional[str]) -> torch.device:
        if explicit_device:
            return torch.device(explicit_device)
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        with path.open('r', encoding='utf-8') as handle:
            return yaml.safe_load(handle)

    def _load_utils_module(self, repo_root: Path) -> ModuleType:
        deployment_dir = repo_root / 'mbra_repo_1' / 'deployment'
        train_dir = repo_root / 'mbra_repo_1' / 'train'
        for path in (deployment_dir, train_dir):
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)

        module_path = deployment_dir / 'utils_logonav.py'
        spec = importlib.util.spec_from_file_location('mbra_repo_1_utils_logonav', module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f'Unable to load MBRA utilities from {module_path}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _prepare_frame(self, frame_rgb: np.ndarray) -> Image.Image:
        image = Image.fromarray(frame_rgb.astype(np.uint8), mode='RGB')
        return image.resize(self.image_size).convert('RGB')

    def _build_observation_tensor(self) -> torch.Tensor:
        obs_images = self._transform_images_mbra(list(self.context_queue))
        obs_images = torch.split(obs_images, 3, dim=1)
        obs_images = torch.cat(obs_images, dim=1)
        return obs_images.to(self.device)

    def _build_goal_pose(
        self,
        step_gap: int,
        heading_reference_deg: float,
        subgoal_orientation_deg: float,
    ) -> torch.Tensor:
        goal_distance_m = min(self.config.goal_distance_clip_m, step_gap * self.config.meters_per_step)
        heading_error_deg = wrap_angle_deg(subgoal_orientation_deg - heading_reference_deg)
        heading_error_rad = math.radians(heading_error_deg)

        relative_x = goal_distance_m * math.cos(heading_error_rad)
        relative_y = goal_distance_m * math.sin(heading_error_rad)

        goal_pose_np = np.array(
            [
                relative_y / self.config.metric_waypoint_spacing,
                -relative_x / self.config.metric_waypoint_spacing,
                math.cos(heading_error_rad),
                math.sin(heading_error_rad),
            ],
            dtype=np.float32,
        )
        return torch.from_numpy(goal_pose_np).unsqueeze(0).float().to(self.device)

    def _waypoint_to_command(self, waypoints: np.ndarray) -> tuple[float, float]:
        chosen_waypoint = waypoints[0][self.config.waypoint_index].copy()
        if self.normalize_actions:
            chosen_waypoint[:2] *= (self.config.max_linear / 3.0)

        dx, dy, hx, hy = chosen_waypoint
        eps = 1e-8
        dt = self.config.inference_dt

        if abs(dx) < eps and abs(dy) < eps:
            linear = 0.0
            angular = self._clip_angle(np.arctan2(hy, hx)) / dt
        elif abs(dx) < eps:
            linear = 0.0
            angular = np.sign(dy) * np.pi / (2.0 * dt)
        else:
            linear = dx / dt
            angular = np.arctan(dy / dx) / dt

        linear = float(np.clip(linear, 0.0, 0.5))
        angular = float(np.clip(angular, -1.0, 1.0))
        linear, angular = self._apply_velocity_limits(linear, angular)
        return linear, angular

    def _apply_velocity_limits(self, linear: float, angular: float) -> tuple[float, float]:
        maxv = self.config.max_linear
        maxw = self.config.max_angular

        if abs(linear) <= maxv:
            if abs(angular) <= maxw:
                linear_limited = linear
                angular_limited = angular
            else:
                radius = linear / angular
                linear_limited = maxw * np.sign(linear) * abs(radius)
                angular_limited = maxw * np.sign(angular)
        else:
            if abs(angular) <= 1e-3:
                linear_limited = maxv * np.sign(linear)
                angular_limited = 0.0
            else:
                radius = linear / angular
                if abs(radius) >= maxv / maxw:
                    linear_limited = maxv * np.sign(linear)
                    angular_limited = maxv * np.sign(angular) / abs(radius)
                else:
                    linear_limited = maxw * np.sign(linear) * abs(radius)
                    angular_limited = maxw * np.sign(angular)

        if (
            linear_limited < self.config.min_turning_linear_threshold
            and self.config.min_turning_angular_threshold < abs(angular_limited) < self.config.min_turning_angular
        ):
            scale = self.config.min_turning_angular / abs(angular_limited)
            angular_limited = np.sign(angular_limited) * self.config.min_turning_angular
            linear_limited = linear_limited * scale

        return float(linear_limited), float(angular_limited)
