"""MBRA local controller wrapper for corridor subgoal following.

This module keeps MBRA in the only role that makes sense for this project:
short-horizon local control between nearby graph subgoals.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import torch
import yaml

from local_controller import ControlCommand

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
MBRA_DEPLOY_DIR = REPO_ROOT / "mbra_repo" / "deployment"
MBRA_TRAIN_DIR = REPO_ROOT / "mbra_repo" / "train"
for import_path in (MBRA_DEPLOY_DIR, MBRA_TRAIN_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from utils_logonav import load_model, to_numpy, transform_images, transform_images_mbra  # type: ignore


@dataclass
class MBRALocalControllerConfig:
    weights_path: Path = REPO_ROOT / "mbra_repo" / "deployment" / "model_weights" / "mbra.pth"
    model_config_path: Path = REPO_ROOT / "mbra_repo" / "train" / "config" / "MBRA.yaml"
    max_linear: float = 0.24
    max_angular: float = 0.34
    min_confidence: float = 0.45
    low_confidence_linear_scale: float = 0.7
    robot_size: float = 0.30
    delay_steps: float = 0.0
    linear_blend: float = 0.55
    angular_blend: float = 0.45


class MBRALocalController:
    """Thin MBRA inference wrapper matching the current controller interface."""

    def __init__(self, config: MBRALocalControllerConfig):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if not config.model_config_path.is_file():
            raise FileNotFoundError(f"MBRA config not found at {config.model_config_path}")
        if not config.weights_path.is_file():
            raise FileNotFoundError(
                f"MBRA weights not found at {config.weights_path}. Place mbra.pth there or pass --mbra-weights."
            )

        with config.model_config_path.open("r", encoding="utf-8") as handle:
            self.model_params = yaml.safe_load(handle)

        self.context_size = int(self.model_params["context_size"])
        self.image_size = list(self.model_params.get("image_size", [96, 96]))
        self.model = load_model(str(config.weights_path), self.model_params, self.device)
        self.model = self.model.to(self.device)
        self.model.eval()

        self._obs_history: deque[Image.Image] = deque(maxlen=self.context_size + 1)
        self._linear_history: deque[float] = deque(maxlen=6)
        self._angular_history: deque[float] = deque(maxlen=6)
        self._last_linear = 0.0
        self._last_angular = 0.0

    def reset(self) -> None:
        self._obs_history.clear()
        self._linear_history.clear()
        self._angular_history.clear()
        self._last_linear = 0.0
        self._last_angular = 0.0

    def _to_pil(self, image_rgb: np.ndarray) -> Image.Image:
        return Image.fromarray(np.asarray(image_rgb, dtype=np.uint8), mode="RGB").resize(self.image_size)

    def _append_command_history(self, linear: float, angular: float) -> None:
        self._linear_history.append(float(linear))
        self._angular_history.append(float(angular))

    def _vel_past_tensor(self) -> torch.Tensor:
        linear_hist = list(self._linear_history)
        angular_hist = list(self._angular_history)
        if len(linear_hist) < 6:
            linear_hist = [0.0] * (6 - len(linear_hist)) + linear_hist
        if len(angular_hist) < 6:
            angular_hist = [0.0] * (6 - len(angular_hist)) + angular_hist
        vel = np.array(linear_hist + angular_hist, dtype=np.float32).reshape(1, 12, 1)
        return torch.from_numpy(vel).to(self.device)

    def compute_command(
        self,
        controller_input: dict,
        observation_heading_deg: Optional[float] = None,
    ) -> ControlCommand:
        del observation_heading_deg  # MBRA uses image context + subgoal image here.

        confidence = float(controller_input.get("confidence", 0.0))
        if confidence < self.config.min_confidence:
            self._append_command_history(0.0, 0.0)
            return ControlCommand(0.0, 0.0, "mbra_low_confidence_stop")

        observation_rgb = controller_input.get("observation_rgb")
        subgoal_image_rgb = controller_input.get("subgoal_image_rgb")
        if observation_rgb is None or subgoal_image_rgb is None:
            self._append_command_history(0.0, 0.0)
            return ControlCommand(0.0, 0.0, "mbra_missing_images")

        self._obs_history.append(self._to_pil(observation_rgb))
        if len(self._obs_history) < self.context_size + 1:
            self._append_command_history(0.0, 0.0)
            return ControlCommand(0.0, 0.0, "mbra_warmup")

        obs_tensor = transform_images_mbra(list(self._obs_history)).to(self.device)
        obs_parts = torch.split(obs_tensor, 3, dim=1)
        obs_tensor = torch.cat(obs_parts, dim=1)

        goal_pil = self._to_pil(subgoal_image_rgb)
        goal_tensor = transform_images(goal_pil, self.image_size).to(self.device)

        robot_size = torch.full((1, 1, 1), self.config.robot_size, dtype=torch.float32, device=self.device)
        delay = torch.full((1, 1, 1), self.config.delay_steps, dtype=torch.float32, device=self.device)
        vel_past = self._vel_past_tensor()

        with torch.no_grad():
            linear_vel, angular_vel, _dist = self.model(obs_tensor, goal_tensor, robot_size, delay, vel_past)

        linear_cmd = float(to_numpy(linear_vel)[0, 0])
        angular_cmd = float(to_numpy(angular_vel)[0, 0])

        if confidence < 0.60:
            linear_cmd *= self.config.low_confidence_linear_scale

        linear_cmd = float(np.clip(linear_cmd, 0.0, self.config.max_linear))
        angular_cmd = float(np.clip(angular_cmd, -self.config.max_angular, self.config.max_angular))

        linear_cmd = self.config.linear_blend * linear_cmd + (1.0 - self.config.linear_blend) * self._last_linear
        angular_cmd = self.config.angular_blend * angular_cmd + (1.0 - self.config.angular_blend) * self._last_angular

        self._last_linear = linear_cmd
        self._last_angular = angular_cmd
        self._append_command_history(linear_cmd, angular_cmd)

        return ControlCommand(linear_cmd, angular_cmd, "mbra_controller")
