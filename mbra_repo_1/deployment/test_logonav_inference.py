import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_DIR = REPO_ROOT / "deployment"
TRAIN_DIR = REPO_ROOT / "train"
if str(DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_DIR))
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from utils_logonav import load_model, transform_images_mbra


def make_synthetic_context(num_frames: int, image_size=(96, 96)):
    width, height = image_size
    frames = []
    x = np.arange(width, dtype=np.uint8)[None, :]
    y = np.arange(height, dtype=np.uint8)[:, None]
    for idx in range(num_frames):
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        arr[..., 0] = (x + idx * 7) % 256
        arr[..., 1] = ((y * 2) + idx * 11) % 256
        arr[..., 2] = (
            (arr[..., 0].astype(np.uint16) + arr[..., 1].astype(np.uint16)) // 2
        ).astype(np.uint8)
        frames.append(Image.fromarray(arr, "RGB"))
    return frames


def waypoint_to_cmd(chosen_waypoint: np.ndarray):
    waypoint = chosen_waypoint.copy()
    waypoint[:2] *= 0.3 / 3.0
    dx, dy, hx, hy = waypoint

    eps = 1e-8
    dt = 1 / 4
    if abs(dx) < eps and abs(dy) < eps:
        linear_vel = 0.0
        angular_vel = np.arctan2(hy, hx) / dt
    elif abs(dx) < eps:
        linear_vel = 0.0
        angular_vel = np.sign(dy) * np.pi / (2 * dt)
    else:
        linear_vel = dx / dt
        angular_vel = np.arctan(dy / dx) / dt

    linear_vel = float(np.clip(linear_vel, 0.0, 0.5))
    angular_vel = float(np.clip(angular_vel, -1.0, 1.0))
    return linear_vel, angular_vel


class TestLogoNavInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cpu")
        with open(REPO_ROOT / "train" / "config" / "LogoNav.yaml", "r", encoding="utf-8") as f:
            cls.config = yaml.safe_load(f)

        cls.model = load_model(
            str(REPO_ROOT / "deployment" / "model_weights" / "logonav.pth"),
            cls.config,
            cls.device,
        )
        cls.model.eval()

    def test_goal_pose_produces_waypoint_and_action(self):
        context = make_synthetic_context(
            self.config["context_size"] + 1,
            tuple(self.config["image_size"]),
        )
        obs_images = transform_images_mbra(context).to(self.device)
        goal_pose = torch.tensor([[4.0, -2.0, 1.0, 0.0]], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            waypoints = self.model(obs_images, goal_pose)
        self.assertEqual(tuple(waypoints.shape), (1, self.config["len_traj_pred"], 4))
        self.assertTrue(torch.isfinite(waypoints).all().item())

        heading_norms = torch.linalg.norm(waypoints[0, :, 2:], dim=-1)
        self.assertTrue(torch.allclose(heading_norms, torch.ones_like(heading_norms), atol=1e-5))

        chosen_waypoint = waypoints[0, 2].cpu().numpy()
        linear_vel, angular_vel = waypoint_to_cmd(chosen_waypoint)
        print("The chosen waypoint is:", chosen_waypoint)
        print("The linear velocity is:", linear_vel)
        print("The angular velocity is:", angular_vel)

        self.assertTrue(np.isfinite(chosen_waypoint).all())
        self.assertTrue(np.isfinite([linear_vel, angular_vel]).all())
        self.assertGreaterEqual(linear_vel, 0.0)
        self.assertLessEqual(linear_vel, 0.5)
        self.assertGreaterEqual(angular_vel, -1.0)
        self.assertLessEqual(angular_vel, 1.0)

        np.testing.assert_allclose(
            chosen_waypoint,
            np.array([2.4785933, -0.77315855, 0.63244015, -0.77460927], dtype=np.float32),
            rtol=1e-4,
            atol=1e-4,
        )
        self.assertAlmostEqual(linear_vel, 0.5, places=6)
        self.assertAlmostEqual(angular_vel, -1.0, places=6)


if __name__ == "__main__":
    unittest.main()
