from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VisionSafetyConfig:
    min_brightness: float = 42.0
    max_dark_fraction: float = 0.65
    max_glare_fraction: float = 0.12
    min_texture_score: float = 8.0
    consecutive_bad_ticks_to_stop: int = 3
    consecutive_clear_ticks_to_reset: int = 1


@dataclass
class VisionSafetyResult:
    mean_brightness: float
    dark_fraction: float
    glare_fraction: float
    texture_score: float
    emergency_stop: bool
    reason: str


class VisionSafetyMonitor:
    """Simple night-time image-quality safety gate.

    This module is deliberately conservative. It does not try to improve the
    image; it decides whether the current frame is too dark / washed out /
    detail-poor to trust visual navigation safely.
    """

    def __init__(self, config: VisionSafetyConfig | None = None) -> None:
        self.config = config or VisionSafetyConfig()
        self._bad_ticks = 0
        self._clear_ticks = 0

    @staticmethod
    def _grayscale(rgb: np.ndarray) -> np.ndarray:
        rgb_f = rgb.astype(np.float32)
        return 0.299 * rgb_f[..., 0] + 0.587 * rgb_f[..., 1] + 0.114 * rgb_f[..., 2]

    @staticmethod
    def _texture_score(gray: np.ndarray) -> float:
        dx = np.abs(np.diff(gray, axis=1))
        dy = np.abs(np.diff(gray, axis=0))
        score = 0.0
        if dx.size:
            score += float(dx.mean())
        if dy.size:
            score += float(dy.mean())
        return score

    def update(self, rgb: np.ndarray) -> VisionSafetyResult:
        gray = self._grayscale(rgb)
        mean_brightness = float(gray.mean())
        dark_fraction = float((gray < 35.0).mean())
        glare_fraction = float((gray > 245.0).mean())
        texture_score = self._texture_score(gray)

        reasons: list[str] = []
        if mean_brightness < self.config.min_brightness and dark_fraction > self.config.max_dark_fraction:
            reasons.append('too_dark')
        if glare_fraction > self.config.max_glare_fraction and texture_score < self.config.min_texture_score:
            reasons.append('glare')
        if texture_score < self.config.min_texture_score and dark_fraction > 0.45:
            reasons.append('low_detail')

        if reasons:
            self._bad_ticks += 1
            self._clear_ticks = 0
        else:
            self._clear_ticks += 1
            if self._clear_ticks >= self.config.consecutive_clear_ticks_to_reset:
                self._bad_ticks = 0

        emergency_stop = self._bad_ticks >= self.config.consecutive_bad_ticks_to_stop
        reason = '|'.join(reasons) if emergency_stop else ''
        return VisionSafetyResult(
            mean_brightness=mean_brightness,
            dark_fraction=dark_fraction,
            glare_fraction=glare_fraction,
            texture_score=texture_score,
            emergency_stop=emergency_stop,
            reason=reason,
        )

    def reset(self) -> None:
        self._bad_ticks = 0
        self._clear_ticks = 0
