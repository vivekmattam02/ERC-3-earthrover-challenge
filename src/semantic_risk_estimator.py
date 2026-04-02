"""Runtime semantic risk estimator for ERC-3 outdoor navigation.

This module wraps the offline-validated SegFormer scoring logic into a small
runtime component. It is intentionally weak: it is designed to provide a soft
angular bias only, not a stop or speed override.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

try:
    from transformers import AutoModelForUniversalSegmentation
except Exception:
    AutoModelForUniversalSegmentation = None


DRIVABLE_LABELS = {"road", "earth", "path", "sidewalk", "dirt_track"}
NEUTRAL_LABELS = {"grass", "field"}
HAZARD_LABELS = {"person", "animal", "pole", "wall", "fence"}
CAUTION_LABELS = {"tree", "plant"}
IGNORE_LABELS = {"sky"}


def normalize_label(label: str) -> str:
    raw = str(label).strip().lower().replace('-', ' ').replace('_', ' ').replace('/', ' ')
    compact = ' '.join(raw.split())
    if not compact:
        return compact
    if any(tok in compact for tok in ("person", "pedestrian", "walker")):
        return "person"
    if any(tok in compact for tok in ("animal", "dog", "cat", "horse", "cow", "sheep")):
        return "animal"
    if "sidewalk" in compact or "footway" in compact or "walkway" in compact or "pedestrian area" in compact:
        return "sidewalk"
    if "crosswalk" in compact:
        return "path"
    if compact == "path" or "bike path" in compact or "cycleway" in compact or "trail" in compact:
        return "path"
    if any(tok in compact for tok in ("road", "street", "lane", "driveable", "drivable", "parking")):
        return "road"
    if "dirt" in compact and "track" in compact:
        return "dirt_track"
    if compact in {"earth", "soil", "sand", "gravel"}:
        return "earth"
    if any(tok in compact for tok in ("pole", "signpost", "utility pole", "traffic sign pole", "streetlight pole")):
        return "pole"
    if "fence" in compact or "guard rail" in compact or "barrier" in compact:
        return "fence"
    if any(tok in compact for tok in ("wall", "building facade", "building wall")):
        return "wall"
    if any(tok in compact for tok in ("tree", "trunk")):
        return "tree"
    if any(tok in compact for tok in ("plant", "bush", "shrub", "vegetation")):
        return "plant"
    if compact in {"grass", "field", "sky"}:
        return compact
    return compact.replace(' ', '_')


@dataclass
class SemanticRiskResult:
    hard_alerts: list[str]
    vegetation_blocked: bool
    risk_score: float
    bias_angular: float
    drivable_center: float
    caution_center: float
    person_center: float
    animal_center: float
    road_center: float
    sidewalk_center: float
    path_center: float
    debug: dict[str, Any] = field(default_factory=dict)


class SemanticRiskEstimator:
    """Estimate semantic risk from a forward RGB frame."""

    def __init__(
        self,
        model_id: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        device: str = "cpu",
    ) -> None:
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_id = model_id
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.backend = "semantic"
        try:
            self.model = AutoModelForSemanticSegmentation.from_pretrained(model_id).to(self.device)
        except Exception:
            if AutoModelForUniversalSegmentation is None:
                raise
            self.model = AutoModelForUniversalSegmentation.from_pretrained(model_id).to(self.device)
            self.backend = "universal"
        self.model.eval()
        self.id2label = {int(k): normalize_label(v) for k, v in self.model.config.id2label.items()}

        # Offline-validated raised corridor.
        self.roi_top_frac = 0.40
        self.roi_bottom_frac = 0.80
        self.roi_left_frac = 0.30
        self.roi_right_frac = 0.70

        self.person_thresh = 0.002
        self.animal_thresh = 0.002
        self.pole_thresh = 0.002
        self.wall_thresh = 0.010
        self.drive_thresh = 0.10
        self.caution_thresh = 0.60
        self.max_bias = 0.50

    def _build_masks(self, h: int, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        top = int(h * self.roi_top_frac)
        bottom = int(h * self.roi_bottom_frac)
        left = int(w * self.roi_left_frac)
        right = int(w * self.roi_right_frac)

        roi = np.zeros((h, w), dtype=bool)
        roi[top:bottom, left:right] = True

        center_left = left + int(0.25 * (right - left))
        center_right = right - int(0.25 * (right - left))
        center = np.zeros((h, w), dtype=bool)
        center[top:bottom, center_left:center_right] = True

        mid = left + (right - left) // 2
        left_half = np.zeros((h, w), dtype=bool)
        right_half = np.zeros((h, w), dtype=bool)
        left_half[top:bottom, left:mid] = True
        right_half[top:bottom, mid:right] = True
        return roi, center, left_half, right_half

    def _region_fracs(self, seg: np.ndarray, mask: np.ndarray) -> dict[str, float]:
        vals = seg[mask]
        uniq, counts = np.unique(vals, return_counts=True)
        total = float(counts.sum()) if len(counts) else 1.0
        out: dict[str, float] = {}
        for idx, count in zip(uniq, counts):
            name = self.id2label.get(int(idx), normalize_label(str(int(idx))))
            out[name] = out.get(name, 0.0) + float(count) / total
        return out

    def _sum_labels(self, fracs: dict[str, float], labels: set[str]) -> float:
        return sum(fracs.get(label, 0.0) for label in labels)

    def _side_free_score(self, side_fracs: dict[str, float], *, hard_mode: bool) -> float:
        drivable = self._sum_labels(side_fracs, DRIVABLE_LABELS)
        neutral = self._sum_labels(side_fracs, NEUTRAL_LABELS)
        caution = self._sum_labels(side_fracs, CAUTION_LABELS)
        person = side_fracs.get("person", 0.0)
        animal = side_fracs.get("animal", 0.0)
        pole = side_fracs.get("pole", 0.0)
        wall = side_fracs.get("wall", 0.0) + side_fracs.get("fence", 0.0)

        free = drivable + 0.30 * neutral
        free -= 0.60 * caution
        if hard_mode:
            free -= 4.0 * (person + animal)
            free -= 3.0 * (pole + wall)
        return free

    def _risk_and_alerts(self, center_fracs: dict[str, float]) -> tuple[float, list[str], bool, float, float, float, float, float, float, float]:
        person = center_fracs.get("person", 0.0)
        animal = center_fracs.get("animal", 0.0)
        pole = center_fracs.get("pole", 0.0)
        wall = center_fracs.get("wall", 0.0) + center_fracs.get("fence", 0.0)
        road_center = center_fracs.get("road", 0.0)
        sidewalk_center = center_fracs.get("sidewalk", 0.0)
        path_center = center_fracs.get("path", 0.0)
        drivable_center = self._sum_labels(center_fracs, DRIVABLE_LABELS)
        caution_center = self._sum_labels(center_fracs, CAUTION_LABELS)

        score = 0.0
        alerts: list[str] = []

        if person > self.person_thresh:
            score += 0.55 + 18.0 * person
            alerts.append("person")
        if animal > self.animal_thresh:
            score += 0.55 + 18.0 * animal
            alerts.append("animal")
        if pole > self.pole_thresh:
            score += 0.35 + 10.0 * pole
            alerts.append("pole")
        if wall > self.wall_thresh:
            score += 0.30 + 8.0 * wall
            alerts.append("wall")

        vegetation_blocked = drivable_center < self.drive_thresh and caution_center > self.caution_thresh
        if vegetation_blocked:
            score += 0.45 + 0.50 * max(0.0, caution_center - self.caution_thresh)

        return (
            max(0.0, score),
            alerts,
            vegetation_blocked,
            drivable_center,
            caution_center,
            person,
            animal,
            road_center,
            sidewalk_center,
            path_center,
        )

    def _bias(self, left_fracs: dict[str, float], right_fracs: dict[str, float], *, active: bool, hard_mode: bool) -> float:
        if not active:
            return 0.0
        left_free = self._side_free_score(left_fracs, hard_mode=hard_mode)
        right_free = self._side_free_score(right_fracs, hard_mode=hard_mode)
        diff = left_free - right_free
        scale = max(0.25, abs(left_free) + abs(right_free))
        bias = diff / scale
        return float(max(-self.max_bias, min(self.max_bias, bias)))

    def estimate(self, rgb: np.ndarray) -> SemanticRiskResult:
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape (H, W, 3), got {rgb.shape}")

        pil = Image.fromarray(rgb)
        inputs = self.processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        if hasattr(self.processor, "post_process_semantic_segmentation"):
            seg = self.processor.post_process_semantic_segmentation(outputs, target_sizes=[rgb.shape[:2]])[0].cpu().numpy().astype(np.int32)
        else:
            logits = outputs.logits
            up = torch.nn.functional.interpolate(
                logits,
                size=rgb.shape[:2],
                mode="bilinear",
                align_corners=False,
            )
            seg = up.argmax(dim=1)[0].cpu().numpy().astype(np.int32)

        roi, center, left_half, right_half = self._build_masks(seg.shape[0], seg.shape[1])
        center_fracs = self._region_fracs(seg, center)
        left_fracs = self._region_fracs(seg, left_half)
        right_fracs = self._region_fracs(seg, right_half)
        roi_fracs = self._region_fracs(seg, roi)

        (
            risk_score,
            hard_alerts,
            vegetation_blocked,
            drivable_center,
            caution_center,
            person_center,
            animal_center,
            road_center,
            sidewalk_center,
            path_center,
        ) = self._risk_and_alerts(center_fracs)
        hard_mode = bool(hard_alerts)
        bias = self._bias(left_fracs, right_fracs, active=hard_mode or vegetation_blocked, hard_mode=hard_mode)

        left_hazard = self._sum_labels(left_fracs, HAZARD_LABELS)
        right_hazard = self._sum_labels(right_fracs, HAZARD_LABELS)
        left_caution = self._sum_labels(left_fracs, CAUTION_LABELS)
        right_caution = self._sum_labels(right_fracs, CAUTION_LABELS)

        debug = {
            "sem_risk": round(risk_score, 2),
            "sem_bias_deg": round(bias * 57.2958, 1),
            "sem_bias_active": bool((hard_alerts or vegetation_blocked) and abs(bias) > 1e-3),
            "sem_event": hard_alerts[0] if hard_alerts else ("veg" if vegetation_blocked else ""),
            "sem_drivable_ctr": round(drivable_center, 2),
            "sem_caution_ctr": round(caution_center, 2),
            "sem_person_ctr": round(person_center, 3),
            "sem_animal_ctr": round(animal_center, 3),
            "sem_road_ctr": round(road_center, 2),
            "sem_sidewalk_ctr": round(sidewalk_center, 2),
            "sem_path_ctr": round(path_center, 2),
            "sem_left_hazard": round(left_hazard, 2),
            "sem_right_hazard": round(right_hazard, 2),
            "sem_left_caution": round(left_caution, 2),
            "sem_right_caution": round(right_caution, 2),
            "sem_model": self.model_id,
            "sem_backend": self.backend,
            "sem_top_labels": [
                f"{name}:{frac:.1%}"
                for name, frac in sorted(roi_fracs.items(), key=lambda kv: kv[1], reverse=True)[:5]
            ],
        }

        return SemanticRiskResult(
            hard_alerts=hard_alerts,
            vegetation_blocked=vegetation_blocked,
            risk_score=risk_score,
            bias_angular=bias,
            drivable_center=drivable_center,
            caution_center=caution_center,
            person_center=person_center,
            animal_center=animal_center,
            road_center=road_center,
            sidewalk_center=sidewalk_center,
            path_center=path_center,
            debug=debug,
        )
