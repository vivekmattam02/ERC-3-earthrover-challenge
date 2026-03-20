"""Runtime-facing corridor localizer for ERC indoor navigation.

This module turns the baseline database and temporal localizer into a single
API that planning and integration code can call directly.
"""

from __future__ import annotations

import json
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
    """Configuration for the main CorridorLocalizer.

    Attributes:
        database_npz (Path): Path to the `descriptors.npz` file created by the baseline.
        cosplace_repo (Optional[Path]): Optional local path to the CosPlace repository.
        data_info_json (Optional[Path]): Optional path to `data_info.json` for heading/orientation data.
        top_k (int): Number of initial candidates to retrieve from the database.
        max_step_jump (int): Max allowed jump in step numbers between consecutive frames.
        jump_penalty (float): Penalty applied to candidates based on their step distance.
        backward_penalty (float): Additional penalty for moving backward in step numbers.
        heading_penalty (float): Penalty for mismatch between observation and candidate heading.
        ambiguity_margin (float): Confidence margin to detect ambiguous localization.
        ambiguity_hold_min_jump (int): Minimum step jump to trigger an ambiguity hold.
    """

    database_npz: Path
    cosplace_repo: Optional[Path] = None
    data_info_json: Optional[Path] = None
    top_k: int = 5
    max_step_jump: int = 15
    jump_penalty: float = 0.05
    backward_penalty: float = 0.15
    heading_penalty: float = 0.002
    ambiguity_margin: float = 0.05
    ambiguity_hold_min_jump: int = 4


class CorridorLocalizer:
    """A high-level, runtime-ready interface for corridor localization.

    This class wraps the entire localization pipeline:
    1.  Loads a pre-built descriptor database and CosPlace model.
    2.  Accepts single frames (as images or arrays).
    3.  Computes a global descriptor for the frame.
    4.  Retrieves the top-K nearest neighbors from the database.
    5.  Uses a `TemporalLocalizer` to filter these candidates and produce a
        stable, temporally consistent location estimate (graph node).
    """

    def __init__(self, config: CorridorLocalizerConfig):
        """Initializes the localizer and loads all necessary artifacts.

        Args:
            config (CorridorLocalizerConfig): The configuration object for the localizer.
        """
        self.config = config
        self.device = get_device()

        # Load the core components from the pre-built baseline database.
        self.descriptor_config: DescriptorConfig = load_descriptor_config(config.database_npz)
        self.model = load_cosplace_model(config.cosplace_repo, self.descriptor_config, self.device)
        self.transform = make_cosplace_transform(self.descriptor_config)
        self.descriptors, self.image_names, self.image_paths = load_descriptor_archive(config.database_npz)

        # Create a mapping from graph node index to the original trajectory step number.
        self.step_by_index: dict[int, int] = {}
        for idx, image_name in enumerate(self.image_names):
            try:
                # Assumes filenames are numeric (e.g., "123.jpg").
                self.step_by_index[idx] = int(Path(image_name).stem)
            except ValueError:
                continue

        # Load optional metadata, primarily for heading information.
        self.image_meta_by_name: dict[str, dict] = {}
        self.heading_by_index: dict[int, float] = {}
        if config.data_info_json is not None and config.data_info_json.is_file():
            with config.data_info_json.open("r", encoding="utf-8") as handle:
                data_info = json.load(handle)
            for entry in data_info:
                image_name = entry["image"]
                self.image_meta_by_name[image_name] = entry
            # Create a mapping from node index to heading/orientation.
            for idx, image_name in enumerate(self.image_names):
                entry = self.image_meta_by_name.get(image_name)
                orientation = None if entry is None else entry.get("orientation")
                if orientation is not None:
                    self.heading_by_index[idx] = float(orientation)

        # Instantiate the temporal filter with parameters from the main config.
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

    def reset(self) -> None:
        """Resets the temporal localizer's state.

        Call this if the robot's location is teleported or becomes unknown,
        to clear any history and start localization from scratch.
        """
        self.temporal_localizer = TemporalLocalizer(self.temporal_localizer.config)

    def preprocess_pil(self, image: Image.Image) -> torch.Tensor:
        """Applies the standard CosPlace preprocessing to a PIL image.

        Args:
            image (Image.Image): The input PIL image.

        Returns:
            torch.Tensor: The preprocessed image as a tensor, ready for the model.
        """
        image = image.convert("RGB")
        image = crop_image(
            image,
            self.descriptor_config.crop_top_ratio,
            self.descriptor_config.crop_bottom_ratio,
        )
        tensor = self.transform(image)
        return tensor.unsqueeze(0)  # Add batch dimension

    def encode_pil(self, image: Image.Image) -> np.ndarray:
        """Computes a global descriptor for a single PIL image.

        Args:
            image (Image.Image): The input PIL image.

        Returns:
            np.ndarray: The computed global descriptor as a NumPy array.
        """
        tensor = self.preprocess_pil(image).to(self.device)
        with torch.no_grad():
            descriptor = self.model(tensor)
            # Normalize to a unit vector for distance comparison.
            descriptor = torch.nn.functional.normalize(descriptor, p=2, dim=1)
        return descriptor.cpu().numpy()[0].astype(np.float32)

    def localize_frame(
        self,
        frame_rgb: np.ndarray,
        observation_heading_deg: Optional[float] = None,
    ) -> dict:
        """Localizes a single frame provided as a NumPy RGB array.

        Args:
            frame_rgb (np.ndarray): The input frame as a NumPy RGB array.
            observation_heading_deg (Optional[float]): The robot's current heading in degrees.

        Returns:
            dict: The localization result dictionary.
        """
        image = Image.fromarray(frame_rgb.astype(np.uint8), mode="RGB")
        return self.localize_pil(image, observation_heading_deg=observation_heading_deg)

    def localize_image_path(
        self,
        image_path: Path,
        observation_heading_deg: Optional[float] = None,
    ) -> dict:
        """Localizes a single frame loaded from a file path.

        Args:
            image_path (Path): The path to the input image file.
            observation_heading_deg (Optional[float]): The robot's current heading in degrees.

        Returns:
            dict: The localization result dictionary.
        """
        image = Image.open(image_path).convert("RGB")
        return self.localize_pil(image, observation_heading_deg=observation_heading_deg)

    def localize_pil(
        self,
        image: Image.Image,
        observation_heading_deg: Optional[float] = None,
    ) -> dict:
        """The core localization logic for a single PIL image.

        Args:
            image (Image.Image): The input PIL image for the current camera view.
            observation_heading_deg (Optional[float]): The robot's current heading in degrees, if available.

        Returns:
            dict: A dictionary containing the final localization result, including the
                  estimated node index, confidence, and diagnostic information from
                  the temporal localizer.
        """
        # 1. Compute the descriptor for the current query image.
        query_desc = self.encode_pil(image)

        # 2. Find the top-K nearest neighbors in the descriptor database.
        # This is the "raw" place recognition result for the current frame.
        candidates = descriptor_distance_search(self.descriptors, query_desc, top_k=self.config.top_k)

        # 3. Format candidates and add metadata for the temporal filter.
        candidate_rows = []
        for index, distance in candidates:
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

        # 4. Update the temporal localizer with the new candidates.
        # This applies penalties and smoothing to select the most likely candidate
        # based on history, preventing flickering and spurious jumps.
        temporal_state = self.temporal_localizer.update(
            candidate_rows,
            observation_heading=observation_heading_deg,
            node_heading_lookup=self.heading_by_index,
        )

        # 5. Package the final, filtered result into a comprehensive dictionary.
        node_index = temporal_state["node_index"]
        node_step = self.step_by_index.get(int(node_index)) if node_index is not None else None
        node_name = self.image_names[int(node_index)] if node_index is not None else None
        node_path = self.image_paths[int(node_index)] if node_index is not None else None
        node_orientation = self.heading_by_index.get(int(node_index)) if node_index is not None else None

        return {
            # The final, stable estimate of the robot's location.
            "node_index": node_index,
            # Associated metadata for the estimated node.
            "node_step": node_step,
            "node_image_name": node_name,
            "node_image_path": node_path,
            "node_orientation": node_orientation,
            # Diagnostics from the temporal filter.
            "confidence": temporal_state["confidence"],
            "stable_steps": temporal_state.get("stable_steps"),
            "held_previous": temporal_state.get("held_previous"),
            "reason": temporal_state.get("reason"),
            "best_candidate": temporal_state.get("best_candidate"),
            "second_candidate": temporal_state.get("second_candidate"),
            "candidates": temporal_state.get("scored_candidates", candidate_rows),
        }
