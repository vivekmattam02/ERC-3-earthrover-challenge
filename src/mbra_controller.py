"""MBRA local controller wrapper for corridor subgoal following.

This module keeps MBRA in the only role that makes sense for this project:
short-horizon local control between nearby graph subgoals.

Key deployment details from the original MBRA training code:
- vel_past should use FIXED values (linear=0.5, angular=0.0), NOT feedback
  from the model's own predictions. The model handles temporal context
  through its 6-frame observation window.
- robot_size=0.3, delay=0.0 are the reference deployment values.
- Output index [0] of the 8-step trajectory is the immediate next command.
- Linear output range: [0, 0.5] m/s. Angular: [-1.0, 1.0] rad/s.
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
    max_linear: float = 0.40
    min_linear: float = 0.18
    max_angular: float = 0.34
    min_confidence: float = 0.45
    low_confidence_linear_scale: float = 0.7
    robot_size: float = 0.30
    delay_steps: float = 0.0
    # Fixed vel_past values matching the MBRA authors' deployment reference.
    # These tell the model "the robot is going straight at moderate speed."
    vel_past_linear: float = 0.5
    vel_past_angular: float = 0.0


class MBRALocalController:
    """Thin MBRA inference wrapper matching the current controller interface."""

    def __init__(self, config: MBRALocalControllerConfig):
        self.config = config
        # Check CUDA device capability — fall back to CPU if incompatible
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            try:
                cap = torch.cuda.get_device_capability(0)
                arch_list = torch.cuda.get_arch_list()
                gpu_major = cap[0]
                supported = any(
                    int(a.split("_")[1][0]) <= gpu_major for a in arch_list if a.startswith("sm_")
                )
                if supported:
                    t = torch.zeros(1, device="cuda:0")
                    del t
                    self.device = torch.device("cuda:0")
                else:
                    print(f"CUDA sm_{cap[0]}{cap[1]} not in {arch_list}. MBRA falling back to CPU.")
            except Exception:
                print("CUDA test failed. MBRA falling back to CPU.")

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

        # Pre-build the fixed vel_past tensor (matches MBRA authors' deployment reference).
        linear_past = [config.vel_past_linear] * 6
        angular_past = [config.vel_past_angular] * 6
        vel_np = np.array(linear_past + angular_past, dtype=np.float32).reshape(1, 12, 1)
        self._vel_past_fixed = torch.from_numpy(vel_np).to(self.device)

    def reset(self) -> None:
        self._obs_history.clear()

    def _to_pil(self, image_rgb: np.ndarray) -> Image.Image:
        return Image.fromarray(np.asarray(image_rgb, dtype=np.uint8), mode="RGB").resize(self.image_size)

    def compute_command(
        self,
        controller_input: dict,
        observation_heading_deg: Optional[float] = None,
    ) -> ControlCommand:
        del observation_heading_deg  # MBRA uses image context + subgoal image.

        confidence = float(controller_input.get("confidence", 0.0))
        if confidence < self.config.min_confidence:
            return ControlCommand(0.0, 0.0, "mbra_low_confidence_stop")

        observation_rgb = controller_input.get("observation_rgb")
        subgoal_image_rgb = controller_input.get("subgoal_image_rgb")
        if observation_rgb is None or subgoal_image_rgb is None:
            return ControlCommand(0.0, 0.0, "mbra_missing_images")

        obs_pil = self._to_pil(observation_rgb)
        self._obs_history.append(obs_pil)
        if len(self._obs_history) < self.context_size + 1:
            # Pre-fill context with copies of current frame so MBRA starts immediately
            while len(self._obs_history) < self.context_size + 1:
                self._obs_history.appendleft(obs_pil)

        # Observation: context_size+1 images → [1, 3*(context_size+1), 96, 96]
        obs_tensor = transform_images_mbra(list(self._obs_history)).to(self.device)

        # Goal: single subgoal image → [1, 3, 96, 96]
        goal_pil = self._to_pil(subgoal_image_rgb)
        goal_tensor = transform_images(goal_pil, self.image_size).to(self.device)

        robot_size = torch.full((1, 1, 1), self.config.robot_size, dtype=torch.float32, device=self.device)
        delay = torch.full((1, 1, 1), self.config.delay_steps, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            linear_vel, angular_vel, _dist = self.model(
                obs_tensor, goal_tensor, robot_size, delay, self._vel_past_fixed
            )

        # Index [0, 0] = first batch, first timestep = immediate next command
        linear_cmd = float(to_numpy(linear_vel)[0, 0])
        angular_cmd = float(to_numpy(angular_vel)[0, 0])

        if confidence < 0.60:
            linear_cmd *= self.config.low_confidence_linear_scale

        linear_cmd = float(np.clip(linear_cmd, 0.0, self.config.max_linear))
        angular_cmd = float(np.clip(angular_cmd, -self.config.max_angular, self.config.max_angular))

        # Allow MBRA to output low/zero linear — it may be stopping intentionally
        # for obstacles or confusion. Forcing forward causes crashes.
        if linear_cmd < 0.02:
            linear_cmd = 0.0

        return ControlCommand(linear_cmd, angular_cmd, "mbra_controller")
