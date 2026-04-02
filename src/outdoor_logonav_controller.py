"""Outdoor LogoNav controller wrapper.

This module adapts the teammate's GPS-conditioned learned outdoor controller
into the shared outdoor runtime interface.

Important convention:
- the runtime may use a math-angle heading convention for classical control
- this controller intentionally expects raw SDK ``orientation_deg`` and converts
  it internally using the original LogoNav script's convention
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml
from PIL import Image as PILImage

from outdoor_gps_controller import OutdoorControlCommand


REPO_ROOT = Path(__file__).resolve().parents[1]
MBRA_DEPLOY_DIR = REPO_ROOT / "mbra_repo" / "deployment"
MBRA_TRAIN_DIR = REPO_ROOT / "mbra_repo" / "train"
MBRA_CONFIG_DIR = MBRA_TRAIN_DIR / "config"
MBRA_WEIGHTS_DIR = MBRA_DEPLOY_DIR / "model_weights"

for path in (MBRA_DEPLOY_DIR, MBRA_TRAIN_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils_logonav import load_model, to_numpy, transform_images_mbra  # type: ignore


def clip_angle(angle_rad: float) -> float:
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi


def calculate_relative_position(x_a: float, y_a: float, x_b: float, y_b: float) -> tuple[float, float]:
    return x_b - x_a, y_b - y_a


def calculate_distance(x_a: float, y_a: float, x_b: float, y_b: float) -> float:
    return math.hypot(x_b - x_a, y_b - y_a)


def rotate_to_local_frame(delta_x: float, delta_y: float, heading_a_rad: float) -> tuple[float, float]:
    relative_x = delta_x * math.cos(heading_a_rad) + delta_y * math.sin(heading_a_rad)
    relative_y = -delta_x * math.sin(heading_a_rad) + delta_y * math.cos(heading_a_rad)
    return relative_x, relative_y


def lerp(start: float, end: float, alpha: float) -> float:
    return start + alpha * (end - start)


def clamp_delta(previous: float, target: float, max_step: float) -> float:
    return previous + np.clip(target - previous, -max_step, max_step)


@dataclass
class OutdoorLogoNavControllerConfig:
    weights_path: Path = MBRA_WEIGHTS_DIR / "logonav.pth"
    config_path: Path = MBRA_CONFIG_DIR / "LogoNav.yaml"
    device: str = "cpu"
    image_size: tuple[int, int] = (96, 96)
    metric_waypoint_spacing: float = 0.25
    goal_distance_cap_m: float = 30.0
    goal_update_distance_m: float = 5.0
    final_goal_stop_distance_m: float = 1.0
    max_linear: float = 0.3
    max_angular: float = 0.3
    command_smoothing: float = 0.35
    max_linear_step: float = 0.05
    max_angular_step: float = 0.10
    angular_deadband: float = 0.03
    linear_deadband: float = 0.02
    turn_slowdown_gain: float = 0.65
    min_turn_scale: float = 0.25
    goal_heading_gain: float = 0.90
    goal_heading_blend: float = 0.65
    max_goal_heading_bias: float = 0.30
    min_goal_turn_scale: float = 0.35


class OutdoorLogoNavController:
    """Shared-runtime wrapper around the teammate's GPS-conditioned LogoNav policy."""

    def __init__(self, config: OutdoorLogoNavControllerConfig):
        self.config = config
        self.device = self._resolve_device(config.device)
        self.model_params = yaml.safe_load(config.config_path.read_text())
        self.context_size = int(self.model_params["context_size"])
        self.model = load_model(str(config.weights_path), self.model_params, self.device)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.context_queue: list[PILImage.Image] = []
        self.goal_utm: Optional[tuple[float, float]] = None
        self.goal_compass_rad: float = 0.0
        self.prev_linear_cmd = 0.0
        self.prev_angular_cmd = 0.0

    def _resolve_device(self, requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested)

    def reset(self) -> None:
        self.context_queue = []
        self.prev_linear_cmd = 0.0
        self.prev_angular_cmd = 0.0

    def update_goal(self, goal_utm: tuple[float, float], goal_compass_rad: float = 0.0) -> None:
        self.goal_utm = (float(goal_utm[0]), float(goal_utm[1]))
        self.goal_compass_rad = float(goal_compass_rad)

    def _prepare_image(self, frame_rgb: np.ndarray) -> PILImage.Image:
        image = PILImage.fromarray(frame_rgb.astype(np.uint8), mode="RGB")
        return image.resize(self.config.image_size).convert("RGB")

    def _apply_smoothing(self, linear_cmd: float, angular_cmd: float) -> tuple[float, float]:
        linear_smooth = lerp(self.prev_linear_cmd, linear_cmd, self.config.command_smoothing)
        angular_smooth = lerp(self.prev_angular_cmd, angular_cmd, self.config.command_smoothing)

        linear_smooth = clamp_delta(self.prev_linear_cmd, linear_smooth, self.config.max_linear_step)
        angular_smooth = clamp_delta(self.prev_angular_cmd, angular_smooth, self.config.max_angular_step)

        if abs(linear_smooth) < self.config.linear_deadband:
            linear_smooth = 0.0
        if abs(angular_smooth) < self.config.angular_deadband:
            angular_smooth = 0.0

        self.prev_linear_cmd = float(linear_smooth)
        self.prev_angular_cmd = float(angular_smooth)
        return self.prev_linear_cmd, self.prev_angular_cmd

    def compute_command(
        self,
        *,
        frame_rgb: np.ndarray,
        current_utm: tuple[float, float],
        orientation_deg: float,
    ) -> OutdoorControlCommand:
        if self.goal_utm is None:
            return OutdoorControlCommand(0.0, 0.0, "logonav_missing_goal")

        image = self._prepare_image(frame_rgb)
        if len(self.context_queue) < self.context_size + 1:
            self.context_queue.append(image)
        else:
            self.context_queue.pop(0)
            self.context_queue.append(image)

        if len(self.context_queue) <= self.context_size:
            self.prev_linear_cmd = 0.0
            self.prev_angular_cmd = 0.0
            return OutdoorControlCommand(
                0.0,
                0.0,
                "logonav_warmup_context",
                debug={"context_len": len(self.context_queue), "context_size": self.context_size},
            )

        cur_x, cur_y = float(current_utm[0]), float(current_utm[1])
        goal_x, goal_y = float(self.goal_utm[0]), float(self.goal_utm[1])
        distance_to_target = calculate_distance(cur_x, cur_y, goal_x, goal_y)
        current_heading_math_rad = clip_angle(math.radians(90.0 - float(orientation_deg)))
        goal_bearing_math_rad = math.atan2(goal_y - cur_y, goal_x - cur_x)
        bearing_error_rad = clip_angle(goal_bearing_math_rad - current_heading_math_rad)

        if distance_to_target < self.config.final_goal_stop_distance_m:
            self.prev_linear_cmd = 0.0
            self.prev_angular_cmd = 0.0
            return OutdoorControlCommand(
                0.0,
                0.0,
                "logonav_goal_reached",
                debug={"distance_to_goal_m": distance_to_target},
            )

        cur_compass = -float(orientation_deg) / 180.0 * math.pi
        delta_x, delta_y = calculate_relative_position(cur_x, cur_y, goal_x, goal_y)
        relative_x, relative_y = rotate_to_local_frame(delta_x, delta_y, cur_compass)

        distance_goal = math.hypot(relative_x, relative_y)
        if distance_goal > self.config.goal_distance_cap_m:
            relative_x = relative_x / distance_goal * self.config.goal_distance_cap_m
            relative_y = relative_y / distance_goal * self.config.goal_distance_cap_m

        goal_pose = np.array(
            [
                relative_y / self.config.metric_waypoint_spacing,
                -relative_x / self.config.metric_waypoint_spacing,
                np.cos(self.goal_compass_rad - cur_compass),
                np.sin(self.goal_compass_rad - cur_compass),
            ]
        )

        obs_images = transform_images_mbra(self.context_queue)
        obs_images = torch.split(obs_images, 3, dim=1)
        obs_images = torch.cat(obs_images, dim=1).to(self.device)
        goal_pose_torch = torch.from_numpy(goal_pose).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            waypoints = self.model(obs_images, goal_pose_torch)
        waypoints = to_numpy(waypoints)
        chosen_waypoint = waypoints[0][2].copy()
        chosen_waypoint[:2] *= (self.config.max_linear / 3.0)

        dx, dy, hx, hy = chosen_waypoint
        eps = 1e-8
        dt = 1 / 4

        if abs(dx) < eps and abs(dy) < eps:
            linear_vel = 0.0
            angular_vel = clip_angle(np.arctan2(hy, hx)) / dt
        elif abs(dx) < eps:
            linear_vel = 0.0
            angular_vel = np.sign(dy) * np.pi / (2 * dt)
        else:
            linear_vel = dx / dt
            angular_vel = np.arctan2(dy, dx) / dt

        linear_vel = float(np.clip(linear_vel, 0.0, 0.5))
        angular_vel = float(np.clip(angular_vel, -1.0, 1.0))

        turn_scale = max(
            self.config.min_turn_scale,
            1.0 - self.config.turn_slowdown_gain * min(1.0, abs(angular_vel)),
        )
        linear_vel *= turn_scale

        # Keep the learned policy obstacle-aware, but add a bounded GPS heading bias
        # so it still makes monotonic progress toward the active waypoint.
        goal_angular_bias = float(np.clip(
            self.config.goal_heading_gain * bearing_error_rad,
            -self.config.max_goal_heading_bias,
            self.config.max_goal_heading_bias,
        ))
        goal_turn_scale = max(
            self.config.min_goal_turn_scale,
            math.cos(min(abs(bearing_error_rad), math.pi / 2.0) / 2.0),
        )
        linear_vel *= goal_turn_scale
        angular_vel = (1.0 - self.config.goal_heading_blend) * angular_vel + self.config.goal_heading_blend * goal_angular_bias

        maxv = self.config.max_linear
        maxw = self.config.max_angular
        if abs(linear_vel) <= maxv:
            if abs(angular_vel) <= maxw:
                linear_limited = linear_vel
                angular_limited = angular_vel
            else:
                rd = linear_vel / angular_vel
                linear_limited = maxw * np.sign(linear_vel) * abs(rd)
                angular_limited = maxw * np.sign(angular_vel)
        else:
            if abs(angular_vel) <= 0.001:
                linear_limited = maxv * np.sign(linear_vel)
                angular_limited = 0.0
            else:
                rd = linear_vel / angular_vel
                if abs(rd) >= maxv / maxw:
                    linear_limited = maxv * np.sign(linear_vel)
                    angular_limited = maxv * np.sign(angular_vel) / abs(rd)
                else:
                    linear_limited = maxw * np.sign(linear_vel) * abs(rd)
                    angular_limited = maxw * np.sign(angular_vel)

        if linear_limited < 0.05 and 0.05 < abs(angular_limited) < 0.2:
            angular_limited = np.sign(angular_limited) * 0.2
            linear_limited = linear_limited * 0.2 / abs(angular_limited)

        linear_cmd, angular_cmd = self._apply_smoothing(float(linear_limited), float(angular_limited))
        return OutdoorControlCommand(
            linear=float(linear_cmd),
            angular=float(angular_cmd),
            reason="logonav_follow",
            debug={
                "distance_to_goal_m": distance_to_target,
                "context_len": len(self.context_queue),
                "goal_pose": goal_pose.tolist(),
                "goal_bearing_rad": float(goal_bearing_math_rad),
                "bearing_error_rad": float(bearing_error_rad),
                "raw_waypoint": chosen_waypoint.tolist(),
                "raw_linear": float(linear_limited),
                "raw_angular": float(angular_limited),
                "goal_angular_bias": float(goal_angular_bias),
                "orientation_deg": float(orientation_deg),
                "cur_compass_rad": float(cur_compass),
                "current_heading_math_rad": float(current_heading_math_rad),
            },
        )
