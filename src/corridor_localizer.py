"""Runtime-facing corridor localizer for ERC indoor navigation.

This module turns the baseline database and temporal localizer into a single
API that planning and integration code can call directly.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline import (  # type: ignore
    DescriptorConfig,
    crop_image,
    descriptor_distance_search,
    get_device,
    load_cosplace_model,
    load_descriptor_archive,
    load_descriptor_config,
    make_cosplace_transform,
)
from temporal_localization import TemporalLocalizer, TemporalLocalizerConfig


@dataclass
class CorridorLocalizerConfig:
    database_npz: Path
    cosplace_repo: Optional[Path] = None
    data_info_json: Optional[Path] = None
    top_k: int = 10
    max_step_jump: int = 20
    jump_penalty: float = 0.05
    backward_penalty: float = 0.15
    heading_penalty: float = 0.002
    ambiguity_margin: float = 0.05
    ambiguity_hold_min_jump: int = 4


class CorridorLocalizer:
    """Planning-facing localizer for known-corridor runtime use."""

    def __init__(self, config: CorridorLocalizerConfig):
        self.config = config
        self.device = self._select_device()
        self.descriptor_config: DescriptorConfig = load_descriptor_config(config.database_npz)
        self.model = load_cosplace_model(config.cosplace_repo, self.descriptor_config, self.device)
        self.transform = make_cosplace_transform(self.descriptor_config)
        self.descriptors, self.image_names, self.image_paths = load_descriptor_archive(config.database_npz)

        self.step_by_index: dict[int, int] = {}
        for idx, image_name in enumerate(self.image_names):
            try:
                self.step_by_index[idx] = int(Path(image_name).stem)
            except ValueError:
                continue

        self.image_meta_by_name: dict[str, dict] = {}
        self.heading_by_index: dict[int, float] = {}
        if config.data_info_json is not None and config.data_info_json.is_file():
            with config.data_info_json.open("r", encoding="utf-8") as handle:
                data_info = json.load(handle)
            for entry in data_info:
                image_name = entry["image"]
                self.image_meta_by_name[image_name] = entry
            for idx, image_name in enumerate(self.image_names):
                entry = self.image_meta_by_name.get(image_name)
                orientation = None if entry is None else entry.get("orientation")
                if orientation is not None:
                    self.heading_by_index[idx] = float(orientation)

        self.temporal_localizer = TemporalLocalizer(
            TemporalLocalizerConfig(
                top_k=config.top_k,
                max_step_jump=config.max_step_jump,
                jump_penalty=config.jump_penalty,
                backward_penalty=config.backward_penalty,
                heading_penalty=config.heading_penalty,
                ambiguity_margin=config.ambiguity_margin,
                ambiguity_hold_min_jump=config.ambiguity_hold_min_jump,
            )
        )

    def _select_device(self) -> torch.device:
        requested_device = os.getenv("ERC_LOCALIZER_DEVICE")
        if requested_device:
            requested_device = requested_device.strip().lower()
            if requested_device == "cuda" and not torch.cuda.is_available():
                print("ERC_LOCALIZER_DEVICE=cuda requested, but CUDA is unavailable. Falling back to CPU.")
                return torch.device("cpu")
            return torch.device(requested_device)

        device = get_device()
        if device.type != "cuda":
            return device

        try:
            major, minor = torch.cuda.get_device_capability(0)
            arch_list = torch.cuda.get_arch_list()
            supported = any(
                int(a.split("_")[1][0]) <= major for a in arch_list if a.startswith("sm_")
            )
            if not supported:
                print(
                    "CUDA device capability "
                    f"sm_{major}{minor} is not covered by arch list {arch_list}. "
                    "Falling back to CPU for localization."
                )
                return torch.device("cpu")
            # Smoke-test a small tensor op
            t = torch.zeros(1, device="cuda:0")
            del t
        except Exception as exc:
            print(f"Unable to validate CUDA device compatibility ({exc}). Falling back to CPU.")
            return torch.device("cpu")

        return device

    def _move_model_to_device(self, device: torch.device) -> None:
        if self.device.type == device.type:
            return
        self.device = device
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"Corridor localizer switched to {self.device.type.upper()} inference.")

    def reset(self) -> None:
        self.temporal_localizer = TemporalLocalizer(self.temporal_localizer.config)

    def revert_last_update(self) -> None:
        """Undo the last localize call's effect on temporal state."""
        self.temporal_localizer.revert_state()

    def preprocess_pil(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        image = crop_image(
            image,
            self.descriptor_config.crop_top_ratio,
            self.descriptor_config.crop_bottom_ratio,
        )
        tensor = self.transform(image)
        return tensor.unsqueeze(0)

    def encode_pil(self, image: Image.Image) -> np.ndarray:
        tensor = self.preprocess_pil(image)
        try:
            tensor = tensor.to(self.device)
            with torch.no_grad():
                descriptor = self.model(tensor)
                descriptor = torch.nn.functional.normalize(descriptor, p=2, dim=1)
        except Exception as exc:
            if self.device.type != "cuda":
                raise
            print(f"CUDA localization inference failed ({exc}). Retrying on CPU.")
            self._move_model_to_device(torch.device("cpu"))
            tensor = tensor.to(self.device)
            with torch.no_grad():
                descriptor = self.model(tensor)
                descriptor = torch.nn.functional.normalize(descriptor, p=2, dim=1)
        return descriptor.cpu().numpy()[0].astype(np.float32)

    def localize_frame(
        self,
        frame_rgb: np.ndarray,
        observation_heading_deg: Optional[float] = None,
        step_min: Optional[int] = None,
        step_max: Optional[int] = None,
    ) -> dict:
        image = Image.fromarray(frame_rgb.astype(np.uint8), mode="RGB")
        return self.localize_pil(
            image,
            observation_heading_deg=observation_heading_deg,
            step_min=step_min,
            step_max=step_max,
        )

    def localize_image_path(
        self,
        image_path: Path,
        observation_heading_deg: Optional[float] = None,
        step_min: Optional[int] = None,
        step_max: Optional[int] = None,
    ) -> dict:
        image = Image.open(image_path).convert("RGB")
        return self.localize_pil(
            image,
            observation_heading_deg=observation_heading_deg,
            step_min=step_min,
            step_max=step_max,
        )

    def localize_pil(
        self,
        image: Image.Image,
        observation_heading_deg: Optional[float] = None,
        step_min: Optional[int] = None,
        step_max: Optional[int] = None,
    ) -> dict:
        query_desc = self.encode_pil(image)
        search_top_k = self.config.top_k
        if step_min is not None or step_max is not None:
            search_top_k = min(len(self.descriptors), max(self.config.top_k * 40, 200))
        candidates = descriptor_distance_search(self.descriptors, query_desc, top_k=search_top_k)

        candidate_rows = []
        for index, distance in candidates:
            image_name = self.image_names[index]
            step = self.step_by_index.get(int(index))
            if step is not None:
                if step_min is not None and step < step_min:
                    continue
                if step_max is not None and step > step_max:
                    continue
            candidate_rows.append(
                {
                    "index": int(index),
                    "distance": float(distance),
                    "image_name": image_name,
                    "image_path": self.image_paths[index],
                    "step": step,
                    "orientation": self.heading_by_index.get(int(index)),
                }
            )

        if not candidate_rows and (step_min is not None or step_max is not None):
            for index, distance in candidates[: self.config.top_k]:
                image_name = self.image_names[index]
                candidate_rows.append(
                    {
                        "index": int(index),
                        "distance": float(distance),
                        "image_name": image_name,
                        "image_path": self.image_paths[index],
                        "step": self.step_by_index.get(int(index)),
                        "orientation": self.heading_by_index.get(int(index)),
                    }
                )

        self.temporal_localizer.save_state()
        temporal_state = self.temporal_localizer.update(
            candidate_rows,
            observation_heading=observation_heading_deg,
            node_heading_lookup=self.heading_by_index,
        )

        node_index = temporal_state["node_index"]
        node_step = self.step_by_index.get(int(node_index)) if node_index is not None else None
        node_name = self.image_names[int(node_index)] if node_index is not None else None
        node_path = self.image_paths[int(node_index)] if node_index is not None else None
        node_orientation = self.heading_by_index.get(int(node_index)) if node_index is not None else None

        return {
            "node_index": node_index,
            "node_step": node_step,
            "node_image_name": node_name,
            "node_image_path": node_path,
            "node_orientation": node_orientation,
            "confidence": temporal_state["confidence"],
            "stable_steps": temporal_state.get("stable_steps"),
            "held_previous": temporal_state.get("held_previous"),
            "reason": temporal_state.get("reason"),
            "best_candidate": temporal_state.get("best_candidate"),
            "second_candidate": temporal_state.get("second_candidate"),
            "candidates": temporal_state.get("scored_candidates", candidate_rows),
        }
