"""Temporal localization utilities for ERC indoor corridor navigation.

This module stabilizes frame-wise retrieval by combining:
- descriptor distance (the raw image similarity)
- continuity cost (penalizing large or backward jumps in the graph)
- optional heading consistency cost (penalizing heading mismatches)
- ambiguity handling (holding the previous estimate on weak evidence)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class TemporalLocalizerConfig:
    """Tuning parameters for the temporal localization filter.

    Attributes:
        top_k (int): Number of raw candidates to consider from the input list.
        max_step_jump (int): A jump in node index greater than this starts incurring a penalty.
        distance_weight (float): Multiplier for the raw descriptor distance in the score.
        jump_penalty (float): Penalty multiplier for jumps exceeding `max_step_jump`.
        backward_penalty (float): Additional penalty for moving to a smaller node index.
        heading_penalty (float): Penalty multiplier for the difference in heading (in degrees).
        ambiguity_margin (float): If the scores of the top two candidates are within this
                                  margin, the situation is considered ambiguous.
        hold_on_ambiguity (bool): If True, the localizer will stick with its previous
                                  estimate during ambiguous situations.
        ambiguity_hold_min_jump (int): A hold is only triggered if the ambiguous new
                                       candidate is at least this many nodes away.
    """

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
    """Tracks the localizer's state between updates.

    Attributes:
        node_index (Optional[int]): The best estimate of the robot's location from the last step.
        confidence (float): The confidence score of the last estimate.
        stable_steps (int): How many consecutive frames the estimate has been stable.
    """

    node_index: Optional[int] = None
    confidence: float = 0.0
    stable_steps: int = 0


def wrap_angle_deg(delta: float) -> float:
    """Wraps an angle difference in degrees to the range [-180, 180].

    Args:
        delta (float): The angle difference in degrees.

    Returns:
        float: The wrapped angle.
    """
    return ((delta + 180.0) % 360.0) - 180.0


class TemporalLocalizer:
    """A stateful filter to stabilize localization estimates over time.

    This class takes a list of raw, per-frame localization candidates and
    produces a single, more stable estimate by penalizing candidates that are
    inconsistent with the previous state (e.g., too far away, backward, or
    at a different heading).
    """

    def __init__(self, config: TemporalLocalizerConfig):
        """Initializes the localizer with its configuration and a fresh state.

        Args:
            config (TemporalLocalizerConfig): The configuration object for the temporal localizer.
        """
        self.config = config
        self.state = TemporalState()

    def _heading_cost(self, candidate_heading: Optional[float], observation_heading: Optional[float]) -> float:
        """Calculates a penalty based on heading difference.

        Args:
            candidate_heading (Optional[float]): The heading of the candidate node in degrees.
            observation_heading (Optional[float]): The observed heading of the robot in degrees.

        Returns:
            float: The calculated heading cost.
        """
        if candidate_heading is None or observation_heading is None:
            return 0.0
        delta = abs(wrap_angle_deg(candidate_heading - observation_heading))
        return self.config.heading_penalty * delta

    def _continuity_cost(self, candidate_index: int) -> float:
        """Calculates a penalty for jumping too far or moving backward.

        Args:
            candidate_index (int): The index of the candidate node.

        Returns:
            float: The calculated continuity cost.
        """
        previous_index = self.state.node_index
        if previous_index is None:
            # No penalty on the first frame.
            return 0.0

        delta = candidate_index - previous_index
        jump_mag = abs(delta)
        penalty = 0.0

        # Penalize jumps larger than the allowed maximum.
        if jump_mag > self.config.max_step_jump:
            penalty += self.config.jump_penalty * (jump_mag - self.config.max_step_jump)
        # Add an additional, harsher penalty for any backward movement.
        if delta < 0:
            penalty += self.config.backward_penalty * abs(delta)

        return penalty

    def update(
        self,
        candidates: list[dict],
        observation_heading: Optional[float] = None,
        node_heading_lookup: Optional[dict[int, float]] = None,
    ) -> dict:
        """Processes a new set of candidates and updates the localization state.

        Args:
            candidates (list[dict]): A list of raw candidate dictionaries from a descriptor search.
                                     Each dictionary should contain at least 'index' and 'distance'.
            observation_heading (Optional[float]): The robot's current heading from a compass or IMU.
            node_heading_lookup (Optional[dict[int, float]]): A dictionary mapping node indices to their headings.

        Returns:
            dict: A dictionary containing the filtered localization result and
                  diagnostic information about the decision process. Keys include:
                  'node_index', 'confidence', 'stable_steps', 'held_previous', 'reason',
                  'best_candidate', 'second_candidate', 'scored_candidates'.
        """
        if not candidates:
            # If no candidates are provided, hold the previous state with zero confidence.
            return {
                "node_index": self.state.node_index,
                "confidence": 0.0,
                "held_previous": True,
                "reason": "no_candidates",
            }

        # --- 1. Score all candidates ---
        # The score is a combination of the raw descriptor distance and penalties.
        # A lower score is better.
        scored: list[dict] = []
        for candidate in candidates[: self.config.top_k]:
            candidate_index = int(candidate["index"])
            candidate_heading = None
            if node_heading_lookup is not None:
                candidate_heading = node_heading_lookup.get(candidate_index)

            # Start with the descriptor distance.
            score = self.config.distance_weight * float(candidate["distance"])
            # Add penalty for being discontinuous with the last frame.
            score += self._continuity_cost(candidate_index)
            # Add penalty for heading mismatch.
            score += self._heading_cost(candidate_heading, observation_heading)

            scored.append(
                {
                    **candidate,
                    "score": score,
                    "candidate_heading": candidate_heading,
                }
            )

        # --- 2. Find the best and second-best candidates ---
        scored.sort(key=lambda item: item["score"])
        best = scored[0]
        second = scored[1] if len(scored) > 1 else None

        held_previous = False
        reason = "best_candidate"  # Assume we'll take the best candidate.

        # --- 3. Check for ambiguity and decide whether to hold the previous estimate ---
        # The situation is ambiguous if:
        #   a) We have a valid previous state to hold on to.
        #   b) The second-best candidate is nearly as good as the best one.
        #   c) The new best candidate is different from our previous state.
        #   d) The new best candidate represents a significant jump, making it risky to switch.
        if (
            self.config.hold_on_ambiguity
            and self.state.node_index is not None
            and second is not None
            and abs(float(second["score"]) - float(best["score"])) < self.config.ambiguity_margin
            and int(best["index"]) != self.state.node_index
            and abs(int(best["index"]) - int(self.state.node_index)) >= self.config.ambiguity_hold_min_jump
        ):
            # If all conditions are met, ignore the new candidates and hold the previous state.
            held_previous = True
            chosen_index = self.state.node_index
            reason = "held_on_ambiguity"
        else:
            # Otherwise, accept the new best candidate.
            chosen_index = int(best["index"])

        # --- 4. Update the internal state ---
        if self.state.node_index == chosen_index:
            # If the estimate is unchanged, increment the stability counter.
            stable_steps = self.state.stable_steps + 1
        else:
            # If the estimate has changed, reset the counter.
            stable_steps = 1

        # Confidence is inversely related to the best score.
        confidence = 1.0 / (1.0 + float(best["score"]))
        self.state = TemporalState(
            node_index=chosen_index,
            confidence=confidence,
            stable_steps=stable_steps,
        )

        # --- 5. Return the comprehensive result ---
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
