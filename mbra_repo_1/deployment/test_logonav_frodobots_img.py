import base64
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_DIR = REPO_ROOT / "deployment"
TRAIN_DIR = REPO_ROOT / "train"
if str(DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_DIR))
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

IMPORT_ERROR = None
try:
    import numpy as np
    from PIL import Image
    import torch
    import yaml

    from LogoNav_frodobots_img import MBRAImageRunner, load_memory_nodes_from_h5
    from utils_logonav import load_model, transform_images_mbra
except (ImportError, ModuleNotFoundError) as exc:
    IMPORT_ERROR = exc

H5_LOG_PATH = Path("/home/zihan/ERC-3-earthrover-challenge/nyu-earthrover/logs/corrider.h5")


def make_synthetic_image(shift: int, image_size=(96, 96)):
    width, height = image_size
    x = np.arange(width, dtype=np.uint8)[None, :]
    y = np.arange(height, dtype=np.uint8)[:, None]
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[..., 0] = (x + shift * 9) % 256
    arr[..., 1] = ((y * 3) + shift * 13) % 256
    arr[..., 2] = (
        (arr[..., 0].astype(np.uint16) + 2 * arr[..., 1].astype(np.uint16)) // 3
    ).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def encode_image_to_base64(pil_img):
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class TestLogoNavFrodobotsImageInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if IMPORT_ERROR is not None:
            raise unittest.SkipTest(f"Test dependencies unavailable: {IMPORT_ERROR}")

        cls.device = torch.device("cpu")
        with open(REPO_ROOT / "train" / "config" / "MBRA.yaml", "r", encoding="utf-8") as f:
            cls.config = yaml.safe_load(f)

        cls.model = load_model(
            str(REPO_ROOT / "deployment" / "model_weights" / "mbra.pth"),
            cls.config,
            cls.device,
        )
        cls.model.eval()

    def test_load_memory_nodes_from_h5_builds_topological_graph(self):
        if not H5_LOG_PATH.exists():
            raise unittest.SkipTest(f"H5 log not found: {H5_LOG_PATH}")

        nodes = load_memory_nodes_from_h5(
            H5_LOG_PATH,
            group_name="front_frames",
            stride=20,
            max_nodes=6,
        )

        self.assertEqual(len(nodes), 6)
        self.assertEqual([node["idx"] for node in nodes], list(range(6)))
        self.assertTrue(all(node["img"].size == tuple(self.config["image_size"]) for node in nodes))
        self.assertTrue(all(np.isfinite(node["embed"]).all() for node in nodes))
        self.assertTrue(all("front_frames" in node["path"] for node in nodes))
        self.assertTrue(all(nodes[i]["source_idx"] < nodes[i + 1]["source_idx"] for i in range(len(nodes) - 1)))

    def test_similarity_selects_matching_h5_node(self):
        if not H5_LOG_PATH.exists():
            raise unittest.SkipTest(f"H5 log not found: {H5_LOG_PATH}")

        nodes = load_memory_nodes_from_h5(
            H5_LOG_PATH,
            group_name="front_frames",
            stride=15,
            max_nodes=8,
        )
        probe_idx = 3
        runner = MBRAImageRunner(
            model=self.model,
            model_type_norm=str(self.config["model_type"]).lower(),
            context_size=int(self.config["context_size"]),
            goal_img_tensor=transform_images_mbra([nodes[0]["img"]]).to(self.device),
            device=self.device,
            base_url="http://unit-test.local",
            tick_rate=3.0,
            memory_nodes=nodes,
            lookahead=1,
            relocalize_window=len(nodes),
            arrive_thresh=1.1,
            lost_thresh=0.0,
        )

        goal_tensor, best_idx, best_score, target_idx = runner._select_goal_from_memory(nodes[probe_idx]["img"])

        self.assertIsNotNone(goal_tensor)
        self.assertEqual(best_idx, probe_idx)
        self.assertGreater(best_score, 0.999)
        self.assertEqual(target_idx, min(probe_idx + 1, len(nodes) - 1))
        self.assertEqual(runner.closest_idx, probe_idx)

    def test_policy_calc_accepts_current_and_target_images(self):
        goal_img = make_synthetic_image(shift=99, image_size=tuple(self.config["image_size"]))
        goal_img_tensor = transform_images_mbra([goal_img]).to(self.device)
        runner = MBRAImageRunner(
            model=self.model,
            model_type_norm=str(self.config["model_type"]).lower(),
            context_size=int(self.config["context_size"]),
            goal_img_tensor=goal_img_tensor,
            device=self.device,
            base_url="http://unit-test.local",
            tick_rate=3.0,
        )

        frames = [
            {"front_frame": encode_image_to_base64(make_synthetic_image(shift=i, image_size=tuple(self.config["image_size"])))}
            for i in range(self.config["context_size"] + 1)
        ]

        with patch("LogoNav_frodobots_img.request_json", side_effect=frames):
            for _ in range(self.config["context_size"]):
                linear_vel, angular_vel = runner.policy_calc()
                self.assertEqual(linear_vel, 0.0)
                self.assertEqual(angular_vel, 0.0)

            linear_vel, angular_vel = runner.policy_calc()
            print(f"Linear Velocity: {linear_vel}, Angular Velocity: {angular_vel}")

        self.assertTrue(np.isfinite([linear_vel, angular_vel]).all())
        self.assertGreaterEqual(linear_vel, 0.0)
        self.assertLessEqual(linear_vel, 0.5)
        self.assertGreaterEqual(angular_vel, -1.0)
        self.assertLessEqual(angular_vel, 1.0)
        self.assertEqual(len(runner.context_queue), self.config["context_size"] + 1)


if __name__ == "__main__":
    unittest.main()
