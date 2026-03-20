'''
How to run:
python /home/zihan/ERC-3-earthrover-challenge/mbra_repo_1/deployment/LogoNav_frodobots_img.py \
  --memory-h5 /path/to/rover_log.h5 \
  --memory-group front_frames \
  --memory-stride 5 \
  --lookahead 1 \
  --base-url http://127.0.0.1:8000 \
  --cpu
'''
import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import h5py
import numpy as np
import requests
import torch
import yaml
from PIL import Image as PILImage

from utils_logonav import load_model, to_numpy, transform_images_mbra

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO_ROOT / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))


def decode_from_base64(encoded_image):
    image_bytes = base64.b64decode(encoded_image)
    return PILImage.open(io.BytesIO(image_bytes))


def request_json(url, method="get", data=None, timeout=10):
    try:
        if method == "get":
            resp = requests.get(url, timeout=timeout)
        else:
            resp = requests.post(url, data=data, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"HTTP request failed: {method.upper()} {url}: {exc}") from exc

    body = (resp.text or "").strip().replace("\n", " ")
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code} from {method.upper()} {url}: {body[:200]}")

    try:
        return resp.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Non-JSON response from {method.upper()} {url} (status={resp.status_code}): {body[:200]}"
        ) from exc


def waypoint_to_cmd(chosen_waypoint):
    chosen_waypoint = chosen_waypoint.copy()
    chosen_waypoint[:2] *= (0.3 / 3.0)
    dx, dy, hx, hy = chosen_waypoint

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

    linear_vel = float(np.clip(linear_vel, 0, 0.5))
    angular_vel = float(np.clip(angular_vel, -1.0, 1.0))
    return linear_vel, angular_vel


def make_embed(pil_img):
    arr = np.asarray(pil_img.resize((96, 96)).convert("RGB"), dtype=np.float32) / 255.0
    emb = arr.reshape(-1)
    norm = np.linalg.norm(emb) + 1e-8
    return emb / norm


def decode_h5_frame(grp, idx):
    if "data" not in grp or grp["data"].shape[0] == 0:
        return None

    frame_bytes = bytes(grp["data"][idx])
    encoded = np.frombuffer(frame_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return PILImage.fromarray(img_rgb)


def load_memory_nodes(memory_dir):
    memory_dir = Path(memory_dir)
    if not memory_dir.exists():
        raise FileNotFoundError(f"memory_dir not found: {memory_dir}")

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    img_paths = [p for p in sorted(memory_dir.iterdir()) if p.suffix.lower() in exts]
    if not img_paths:
        raise RuntimeError(f"No memory images found in {memory_dir}")

    nodes = []
    for idx, path in enumerate(img_paths):
        img = PILImage.open(path).convert("RGB").resize((96, 96))
        nodes.append({"idx": idx, "path": str(path), "img": img, "embed": make_embed(img)})
    return nodes


def load_memory_nodes_from_h5(
    h5_path,
    group_name="front_frames",
    stride=1,
    max_nodes: Optional[int] = None,
):
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"h5 file not found: {h5_path}")
    if stride <= 0:
        raise ValueError("stride must be >= 1")

    nodes = []
    with h5py.File(h5_path, "r") as h5:
        if group_name not in h5:
            raise KeyError(f"group '{group_name}' not found in {h5_path}")
        grp = h5[group_name]
        if "data" not in grp or grp["data"].shape[0] == 0:
            raise RuntimeError(f"No frames found in {h5_path}:{group_name}")

        timestamps = grp["timestamps"][:] if "timestamps" in grp else None
        total_frames = grp["data"].shape[0]
        for src_idx in range(0, total_frames, stride):
            img = decode_h5_frame(grp, src_idx)
            if img is None:
                continue

            img = img.convert("RGB").resize((96, 96))
            timestamp = float(timestamps[src_idx]) if timestamps is not None else float(src_idx)
            nodes.append(
                {
                    "idx": len(nodes),
                    "source_idx": src_idx,
                    "timestamp": timestamp,
                    "path": f"{h5_path}:{group_name}:{src_idx}",
                    "img": img,
                    "embed": make_embed(img),
                }
            )
            if max_nodes is not None and len(nodes) >= max_nodes:
                break

    if not nodes:
        raise RuntimeError(f"No decodable frames found in {h5_path}:{group_name}")
    return nodes


class MBRAImageRunner:
    def __init__(
        self,
        model,
        model_type_norm,
        context_size,
        goal_img_tensor,
        device,
        base_url,
        tick_rate,
        memory_nodes=None,
        lookahead=1,
        relocalize_window=40,
        arrive_thresh=0.92,
        lost_thresh=0.55,
    ):
        self.model = model
        self.model_type_norm = model_type_norm
        self.context_size = context_size
        self.goal_img_tensor = goal_img_tensor
        self.device = device
        self.base_url = base_url.rstrip("/")
        self.tick_rate = tick_rate
        self.context_queue = []
        self.linear = 0.0
        self.angular = 0.0
        self.memory_nodes = memory_nodes
        self.lookahead = lookahead
        self.relocalize_window = relocalize_window
        self.arrive_thresh = arrive_thresh
        self.lost_thresh = lost_thresh
        self.closest_idx = 0
        self.state = "SINGLE_GOAL"
        self.last_log_time = 0.0

        if self.memory_nodes is not None:
            self.memory_embeds = np.stack([n["embed"] for n in self.memory_nodes], axis=0)
            self.memory_goal_tensors = [transform_images_mbra([n["img"]]) for n in self.memory_nodes]
            self.state = "TRACKING"

    def _policy_from_mbra(self, obs_images, goal_img_tensor):
        b = obs_images.shape[0]
        rsize = 0.3 * torch.ones(b, 1, 1, device=self.device)
        delay = torch.zeros(b, 1, 1, device=self.device)
        linear_vel_old = 0.5 * torch.ones(b, 6, device=self.device)
        angular_vel_old = torch.zeros(b, 6, device=self.device)
        vel_past = torch.cat((linear_vel_old, angular_vel_old), dim=1).unsqueeze(2)
        with torch.no_grad():
            linear_seq, angular_seq, _ = self.model(obs_images, goal_img_tensor, rsize, delay, vel_past)
        idx = min(2, linear_seq.shape[1] - 1)
        linear = float(torch.clamp(linear_seq[0, idx], 0.0, 0.5).item())
        angular = float(torch.clamp(angular_seq[0, idx], -1.0, 1.0).item())
        return linear, angular

    def _policy_from_waypoints(self, obs_images, goal_img_tensor):
        with torch.no_grad():
            out = self.model(obs_images, goal_img_tensor)
        waypoints = out[-1] if isinstance(out, tuple) else out
        waypoints = to_numpy(waypoints)
        idx = min(2, waypoints.shape[1] - 1)
        return waypoint_to_cmd(waypoints[0][idx])

    def _select_goal_from_memory(self, curr_img):
        curr_embed = make_embed(curr_img)
        n = len(self.memory_nodes)
        if n == 0:
            raise RuntimeError("memory_nodes is empty")

        lo = max(0, self.closest_idx - self.relocalize_window)
        hi = min(n, self.closest_idx + self.relocalize_window + 1)
        local_embeds = self.memory_embeds[lo:hi]
        sims = local_embeds @ curr_embed
        best_local = int(np.argmax(sims))
        best_score = float(sims[best_local])
        best_idx = lo + best_local

        # Fallback to global search if local window is not confident.
        if best_score < self.lost_thresh:
            sims_all = self.memory_embeds @ curr_embed
            best_idx = int(np.argmax(sims_all))
            best_score = float(sims_all[best_idx])

        if best_score < self.lost_thresh:
            self.state = "LOST"
            return None, best_idx, best_score, best_idx

        self.closest_idx = best_idx
        target_idx = min(self.closest_idx + self.lookahead, n - 1)
        target_score = float(curr_embed @ self.memory_embeds[target_idx])
        if target_score >= self.arrive_thresh and target_idx < n - 1:
            target_idx += 1

        self.state = "TRACKING"
        goal_tensor = self.memory_goal_tensors[target_idx].to(self.device)
        return goal_tensor, best_idx, best_score, target_idx

    def policy_calc(self):
        try:
            response = request_json(f"{self.base_url}/v2/front", method="get")
            if "front_frame" not in response:
                raise RuntimeError(f"/v2/front missing 'front_frame': {response}")
            img = decode_from_base64(response["front_frame"]).resize((96, 96)).convert("RGB")
        except Exception as exc:
            print(f"[WARN] Front frame unavailable ({exc}); zero command this cycle.")
            return 0.0, 0.0

        if len(self.context_queue) < self.context_size + 1:
            self.context_queue.append(img)
        else:
            self.context_queue.pop(0)
            self.context_queue.append(img)

        if len(self.context_queue) <= self.context_size:
            return 0.0, 0.0

        obs_images = transform_images_mbra(self.context_queue).to(self.device)
        goal_img_tensor = self.goal_img_tensor

        if self.memory_nodes is not None:
            selected = self._select_goal_from_memory(img)
            if selected[0] is None:
                now = time.time()
                if now - self.last_log_time > 1.0:
                    print(f"[INFO] state=LOST closest_idx={selected[1]} score={selected[2]:.3f}")
                    self.last_log_time = now
                return 0.0, 0.0
            goal_img_tensor, best_idx, best_score, target_idx = selected
            now = time.time()
            if now - self.last_log_time > 1.0:
                print(
                    f"[INFO] state={self.state} closest_idx={best_idx} "
                    f"target_idx={target_idx} score={best_score:.3f}"
                )
                self.last_log_time = now

        if self.model_type_norm == "mbra":
            return self._policy_from_mbra(obs_images, goal_img_tensor)
        return self._policy_from_waypoints(obs_images, goal_img_tensor)

    def control_send(self, linear, angular):
        data = json.dumps({"command": {"linear": linear, "angular": angular}})
        try:
            request_json(f"{self.base_url}/control", method="post", data=data)
        except Exception as exc:
            print(f"[WARN] /control unavailable ({exc}); command skipped.")

    def run(self):
        loop_time = 1.0 / self.tick_rate
        start_time = time.time()
        while True:
            now = time.time()
            if now - start_time > loop_time:
                self.linear, self.angular = self.policy_calc()
                start_time = time.time()
            self.control_send(self.linear, self.angular)


def main():
    parser = argparse.ArgumentParser(description="Image-goal conditioned runner for MBRA/LogoNav.")
    parser.add_argument("--goal-image", type=str, default=None, help="Path to a single goal image.")
    parser.add_argument("--memory-dir", type=str, default=None, help="Directory of topological memory images.")
    parser.add_argument("--memory-h5", type=str, default=None, help="HDF5 log file used to build topological memory nodes.")
    parser.add_argument(
        "--memory-group",
        type=str,
        default="front_frames",
        help="Frame group inside the HDF5 log used for topological memory.",
    )
    parser.add_argument(
        "--memory-stride",
        type=int,
        default=1,
        help="Sample every Nth HDF5 frame when building the topological graph.",
    )
    parser.add_argument(
        "--memory-max-nodes",
        type=int,
        default=None,
        help="Optional cap on the number of HDF5 frames used as topological nodes.",
    )
    parser.add_argument("--lookahead", type=int, default=1, help="Use node (closest + lookahead) as goal.")
    parser.add_argument("--relocalize-window", type=int, default=40, help="Local search window around current index.")
    parser.add_argument("--arrive-thresh", type=float, default=0.92, help="Similarity threshold to advance target.")
    parser.add_argument("--lost-thresh", type=float, default=0.55, help="Similarity threshold below which robot stops.")
    parser.add_argument(
        "--model-config",
        type=str,
        default=str(REPO_ROOT / "train" / "config" / "MBRA.yaml"),
        help="Config path (MBRA.yaml for mbra.pth).",
    )
    parser.add_argument(
        "--model-weights",
        type=str,
        default=str(REPO_ROOT / "deployment" / "model_weights" / "mbra.pth"),
        help="Checkpoint path.",
    )
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--tick-rate", type=float, default=3.0)
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")

    with open(args.model_config, "r", encoding="utf-8") as f:
        model_params = yaml.safe_load(f)
    model_type_norm = str(model_params["model_type"]).lower()
    context_size = int(model_params["context_size"])

    model = load_model(args.model_weights, model_params, device).to(device)
    model.eval()
    print(f"[INFO] Loaded model_type={model_params['model_type']} from {args.model_weights}")

    memory_sources = [args.memory_dir is not None, args.memory_h5 is not None]
    if sum(memory_sources) > 1:
        raise ValueError("Provide only one of --memory-dir or --memory-h5.")
    if args.memory_dir is None and args.memory_h5 is None and args.goal_image is None:
        raise ValueError("Provide one of --goal-image, --memory-dir, or --memory-h5.")

    memory_nodes = None
    if args.memory_dir is not None:
        memory_nodes = load_memory_nodes(args.memory_dir)
        goal_img_tensor = transform_images_mbra([memory_nodes[0]["img"]]).to(device)
        print(f"[INFO] Loaded {len(memory_nodes)} memory nodes from {args.memory_dir}")
    elif args.memory_h5 is not None:
        memory_nodes = load_memory_nodes_from_h5(
            args.memory_h5,
            group_name=args.memory_group,
            stride=args.memory_stride,
            max_nodes=args.memory_max_nodes,
        )
        goal_img_tensor = transform_images_mbra([memory_nodes[0]["img"]]).to(device)
        print(
            f"[INFO] Loaded {len(memory_nodes)} memory nodes from "
            f"{args.memory_h5}:{args.memory_group} with stride={args.memory_stride}"
        )
    else:
        goal_img = PILImage.open(args.goal_image).convert("RGB").resize((96, 96))
        goal_img_tensor = transform_images_mbra([goal_img]).to(device)

    runner = MBRAImageRunner(
        model=model,
        model_type_norm=model_type_norm,
        context_size=context_size,
        goal_img_tensor=goal_img_tensor,
        device=device,
        base_url=args.base_url,
        tick_rate=args.tick_rate,
        memory_nodes=memory_nodes,
        lookahead=args.lookahead,
        relocalize_window=args.relocalize_window,
        arrive_thresh=args.arrive_thresh,
        lost_thresh=args.lost_thresh,
    )
    runner.run()


if __name__ == "__main__":
    main()
