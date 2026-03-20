"""Temporal localization utilities for ERC indoor corridor navigation.

This module stabilizes frame-wise retrieval by combining:
- descriptor distance
- continuity in node index / graph progression
- optional heading consistency
- ambiguity handling that avoids jumping on weak evidence
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class TemporalLocalizerConfig:
    top_k: int = 5
    max_step_jump: int = 15
    distance_weight: float = 1.0
    jump_penalty: float = 0.05
    backward_penalty: float = 0.15
    heading_penalty: float = 0.002
    ambiguity_margin: float = 0.05
    hold_on_ambiguity: bool = True
    ambiguity_hold_min_jump: int = 4


@dataclass
class TemporalState:
    node_index: Optional[int] = None
    confidence: float = 0.0
    stable_steps: int = 0


def wrap_angle_deg(delta: float) -> float:
    return ((delta + 180.0) % 360.0) - 180.0


class TemporalLocalizer:
    def __init__(self, config: TemporalLocalizerConfig):
        self.config = config
        self.state = TemporalState()

    def _heading_cost(self, candidate_heading: Optional[float], observation_heading: Optional[float]) -> float:
        if candidate_heading is None or observation_heading is None:
            return 0.0
        delta = abs(wrap_angle_deg(candidate_heading - observation_heading))
        return self.config.heading_penalty * delta

    def _continuity_cost(self, candidate_index: int) -> float:
        previous_index = self.state.node_index
        if previous_index is None:
            return 0.0

        delta = candidate_index - previous_index
        jump_mag = abs(delta)
        penalty = 0.0

        if jump_mag > self.config.max_step_jump:
            penalty += self.config.jump_penalty * (jump_mag - self.config.max_step_jump)
        if delta < 0:
            penalty += self.config.backward_penalty * abs(delta)

        return penalty

    def update(
        self,
        candidates: list[dict],
        observation_heading: Optional[float] = None,
        node_heading_lookup: Optional[dict[int, float]] = None,
    ) -> dict:
        if not candidates:
            return {
                "node_index": self.state.node_index,
                "confidence": 0.0,
                "held_previous": True,
                "reason": "no_candidates",
            }

        scored: list[dict] = []
        for candidate in candidates[: self.config.top_k]:
            candidate_index = int(candidate["index"])
            candidate_heading = None
            if node_heading_lookup is not None:
                candidate_heading = node_heading_lookup.get(candidate_index)

            score = self.config.distance_weight * float(candidate["distance"])
            score += self._continuity_cost(candidate_index)
            score += self._heading_cost(candidate_heading, observation_heading)

            scored.append(
                {
                    **candidate,
                    "score": score,
                    "candidate_heading": candidate_heading,
                }
            )

        scored.sort(key=lambda item: item["score"])
        best = scored[0]
        second = scored[1] if len(scored) > 1 else None

        held_previous = False
        reason = "best_candidate"

        if (
            self.config.hold_on_ambiguity
            and self.state.node_index is not None
            and second is not None
            and abs(float(second["score"]) - float(best["score"])) < self.config.ambiguity_margin
            and int(best["index"]) != self.state.node_index
            and abs(int(best["index"]) - int(self.state.node_index)) >= self.config.ambiguity_hold_min_jump
        ):
            held_previous = True
            chosen_index = self.state.node_index
            reason = "held_on_ambiguity"
        else:
            chosen_index = int(best["index"])

        if self.state.node_index == chosen_index:
            stable_steps = self.state.stable_steps + 1
        else:
            stable_steps = 1

        confidence = 1.0 / (1.0 + float(best["score"]))
        self.state = TemporalState(
            node_index=chosen_index,
            confidence=confidence,
            stable_steps=stable_steps,
        )

        return {
            "node_index": chosen_index,
            "confidence": confidence,
            "stable_steps": stable_steps,
            "held_previous": held_previous,
            "reason": reason,
            "best_candidate": best,
            "second_candidate": second,
            "scored_candidates": scored,
        }
