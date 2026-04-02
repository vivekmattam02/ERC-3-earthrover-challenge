"""Outdoor traversability module for ERC outdoor navigation.

Converts a Depth Anything V2 depth map into a local obstacle mask and a
recommended safe heading. This is the "local safety layer" that sits between
the depth estimator and the GPS / LogoNav controller.

Key difference from the old --depth-safety VFH:
  - The old VFH used ``depth_crop_bottom=0.60``, which reads the *bottom* 60 %
    of the image.  On this platform that is the ground directly underfoot, so
    it always reads ~0.32 m and triggers DEPSTOP on open terrain.
  - This module reads a *middle band* of the image (rows crop_top_frac to
    crop_bot_frac, defaulting to 15 %–60 % from the top).  That band captures
    tree trunks, walls, barriers, and other obstacles at rover-eye level while
    ignoring sky and the ground patch immediately in front.

Typical usage (called from live_outdoor_runtime.py):
    traversability = OutdoorTraversability(OutdoorTraversabilityConfig())
    result = traversability.compute(depth_map, goal_bearing_error_rad)
    # result.linear_scale  — multiply controller linear speed by this
    # result.angular_override — if not None, steer toward this angle (rad)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def _wrap_angle_rad(delta: float) -> float:
    return (delta + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class OutdoorTraversabilityConfig:
    num_bins: int = 16
    fov_horizontal_deg: float = 90.0

    # Vertical band for obstacle detection (fractions of image height from top).
    # Rows from crop_top_frac to crop_bot_frac are analysed.
    # - Sky and far horizon: top ~15 %  → excluded (above crop_top_frac)
    # - Tree trunks / walls at 1–5 m: rows ~15–60 %  → included
    # - Ground directly underfoot:  rows ~60–100 %  → excluded (below crop_bot_frac)
    crop_top_frac: float = 0.15
    crop_bot_frac: float = 0.60

    # A bin is "blocked" if its minimum depth is below this distance.
    obstacle_distance_m: float = 1.5

    # Forward clearance thresholds (applied to the ±forward_bin_half_window bins).
    stop_distance_m: float = 0.60    # zero linear when forward clearance < this
    slow_distance_m: float = 1.20    # scale linear down when forward clearance < this
    slow_linear_min: float = 0.35    # minimum linear scale while slowing (never below this)

    # Forward bin window: how many bins left and right of centre count as "forward".
    forward_bin_half_window: int = 1

    # Obstacle memory: use the per-bin minimum over the last N depth frames.
    # This makes obstacles "stick" for a few seconds so the robot does not drive
    # back into something it just turned away from.
    memory_frames: int = 4


@dataclass
class TraversabilityResult:
    clearance: np.ndarray            # (num_bins,) clearance per angular bin [m]
    bin_centers: np.ndarray          # (num_bins,) bin centre angles [rad]
    forward_clearance: float         # min clearance in forward-facing bins [m]
    safe_heading_rad: float          # recommended heading relative to robot [rad]
    forward_blocked: bool            # True → forward direction is obstructed
    all_blocked: bool                # True → every bin is obstructed
    linear_scale: float              # multiply controller linear speed by this
    angular_override: Optional[float]  # if set, steer toward this angle [rad]
    debug: dict = field(default_factory=dict)


class OutdoorTraversability:
    """Local traversability layer for unknown outdoor terrain."""

    def __init__(self, config: OutdoorTraversabilityConfig) -> None:
        self.config = config
        self._history: deque[np.ndarray] = deque(maxlen=config.memory_frames)

    def _clearance_from_depth(
        self, depth_map: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-bin clearance from the middle band of a depth map."""
        h, w = depth_map.shape

        row_start = int(h * self.config.crop_top_frac)
        row_end   = int(h * self.config.crop_bot_frac)
        row_end   = max(row_end, row_start + 1)
        band = depth_map[row_start:row_end, :]

        fov_rad = math.radians(self.config.fov_horizontal_deg)
        fx = w / (2.0 * math.tan(fov_rad / 2.0))
        cx = w / 2.0

        u = np.arange(w, dtype=np.float32)
        yaw_per_col = np.arctan((u - cx) / fx)

        bin_edges   = np.linspace(-fov_rad / 2.0, fov_rad / 2.0, self.config.num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        clearance = np.full(self.config.num_bins, 20.0, dtype=np.float32)
        for b in range(self.config.num_bins):
            mask = (yaw_per_col >= bin_edges[b]) & (yaw_per_col < bin_edges[b + 1])
            cols = np.where(mask)[0]
            if len(cols) > 0:
                vals  = band[:, cols]
                valid = vals[vals > 0.0]
                if len(valid) > 0:
                    clearance[b] = float(np.percentile(valid, 10))

        return clearance, bin_centers

    def _forward_clearance(self, clearance: np.ndarray) -> float:
        center = len(clearance) // 2
        half   = self.config.forward_bin_half_window
        lo     = max(0, center - half)
        hi     = min(len(clearance), center + half + 1)
        return float(np.min(clearance[lo:hi]))

    def compute(
        self,
        depth_map: np.ndarray,
        goal_bearing_error_rad: float,
    ) -> TraversabilityResult:
        """
        Compute local traversability from a depth map.

        Args:
            depth_map: (H, W) metric depth [m] from Depth Anything V2.
            goal_bearing_error_rad: direction to GPS goal relative to current
                heading (positive = turn left / counter-clockwise).

        Returns:
            TraversabilityResult with clearance, safe heading, and command
            modifiers (linear_scale, angular_override).
        """
        current_clearance, bin_centers = self._clearance_from_depth(depth_map)

        # Keep a rolling min-pool so obstacles are "sticky" for memory_frames ticks.
        self._history.append(current_clearance.copy())
        if len(self._history) > 1:
            persistent = np.min(np.stack(list(self._history)), axis=0)
        else:
            persistent = current_clearance

        fwd_clearance  = self._forward_clearance(persistent)
        fwd_blocked    = fwd_clearance < self.config.obstacle_distance_m
        open_mask      = persistent >= self.config.obstacle_distance_m
        all_blocked    = not bool(np.any(open_mask))

        # Pick the open bin closest to the GPS goal direction.
        if all_blocked:
            best_idx     = int(np.argmax(persistent))
            safe_heading = float(bin_centers[best_idx])
        else:
            open_angles  = bin_centers[open_mask]
            costs        = np.abs([_wrap_angle_rad(float(a) - goal_bearing_error_rad)
                                   for a in open_angles])
            safe_heading = float(open_angles[int(np.argmin(costs))])

        # Compute speed scale and angular override.
        linear_scale     = 1.0
        angular_override: Optional[float] = None

        if fwd_clearance < self.config.stop_distance_m or all_blocked:
            linear_scale     = 0.0
            angular_override = safe_heading
        elif fwd_clearance < self.config.slow_distance_m:
            t            = (fwd_clearance - self.config.stop_distance_m) / max(
                1e-6, self.config.slow_distance_m - self.config.stop_distance_m
            )
            linear_scale = self.config.slow_linear_min + (1.0 - self.config.slow_linear_min) * t
            if fwd_blocked:
                angular_override = safe_heading
        elif fwd_blocked:
            linear_scale     = max(self.config.slow_linear_min, 0.70)
            angular_override = safe_heading

        debug = {
            "trav_fwd_m":        round(fwd_clearance, 2),
            "trav_fwd_blocked":  fwd_blocked,
            "trav_all_blocked":  all_blocked,
            "trav_safe_hdg":     round(math.degrees(safe_heading), 1),
            "trav_lin_scale":    round(linear_scale, 2),
        }

        return TraversabilityResult(
            clearance        = persistent,
            bin_centers      = bin_centers,
            forward_clearance= fwd_clearance,
            safe_heading_rad = safe_heading,
            forward_blocked  = fwd_blocked,
            all_blocked      = all_blocked,
            linear_scale     = linear_scale,
            angular_override = angular_override,
            debug            = debug,
        )
