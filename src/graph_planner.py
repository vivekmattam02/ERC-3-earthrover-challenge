"""Runtime graph planner for ERC indoor corridor navigation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import networkx as nx
from networkx.readwrite import json_graph


@dataclass
class GraphPlannerConfig:
    """Configuration for the GraphPlanner.

    Attributes:
        graph_json (Path): Path to the `place_graph.json` file created by the baseline.
        data_info_json (Optional[Path]): Optional path to `data_info.json` for metadata lookup.
        max_subgoal_hops (int): Default number of nodes to look ahead on the path
                                to select a short-term subgoal.
        min_confidence_to_advance (float): Minimum localization confidence required
                                           to mark a checkpoint as reached.
    """

    graph_json: Path
    data_info_json: Optional[Path] = None
    max_subgoal_hops: int = 3
    min_confidence_to_advance: float = 0.55


class GraphPlanner:
    """A high-level path planner operating on a pre-built navigation graph.

    This class is responsible for:
    1.  Loading the graph and associated metadata.
    2.  Managing a sequence of goal "checkpoints" for long-horizon navigation.
    3.  Computing the shortest path between the current location and a target.
    4.  Selecting a short-term "subgoal" a few steps along the path for the
        local controller to execute.
    """

    def __init__(self, config: GraphPlannerConfig):
        """Initializes the planner and loads graph artifacts.

        Args:
            config (GraphPlannerConfig): The configuration object for the planner.
        """
        self.config = config
        self.graph = self._load_graph(config.graph_json)

        # Create lookup dictionaries for efficient mapping between different
        # identifiers (node index, image name, step number).
        self.node_to_name: dict[int, str] = {}
        self.node_to_path: dict[int, str] = {}
        self.node_to_step: dict[int, int] = {}
        self.step_to_node: dict[int, int] = {}
        self.image_to_node: dict[str, int] = {}

        for node, attrs in self.graph.nodes(data=True):
            node_id = int(node)
            name = attrs.get("name")
            path = attrs.get("path")
            if name is not None:
                self.node_to_name[node_id] = str(name)
                self.image_to_node[str(name)] = node_id
                try:
                    step = int(Path(str(name)).stem)
                    self.node_to_step[node_id] = step
                    self.step_to_node[step] = node_id
                except ValueError:
                    # Non-numeric filenames are ignored for step mapping.
                    pass
            if path is not None:
                self.node_to_path[node_id] = str(path)

        # Load optional metadata (e.g., from data_info.json) if provided.
        self.image_meta_by_name: dict[str, dict] = {}
        if config.data_info_json is not None and config.data_info_json.is_file():
            with config.data_info_json.open("r", encoding="utf-8") as handle:
                data_info = json.load(handle)
            for entry in data_info:
                self.image_meta_by_name[entry["image"]] = entry

        # State for multi-stage navigation.
        self.checkpoints: list[int] = []
        self.active_checkpoint_index: int = 0

    def _load_graph(self, path: Path) -> nx.Graph:
        """Loads a networkx graph from a JSON file.

        Args:
            path (Path): The path to the graph JSON file.

        Returns:
            nx.Graph: The loaded graph.
        """
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        graph = json_graph.node_link_graph(data)
        return graph

    def resolve_target_node(
        self,
        target_node: Optional[int] = None,
        target_step: Optional[int] = None,
        target_image_name: Optional[str] = None,
    ) -> int:
        """Finds a graph node index from various possible identifiers.

        Args:
            target_node (Optional[int]): A direct node index.
            target_step (Optional[int]): A step number to look up.
            target_image_name (Optional[str]): An image filename to look up.

        Returns:
            int: The resolved graph node index.

        Raises:
            KeyError: If the specified identifier doesn't map to a known node.
            ValueError: If no target identifier is provided.
        """
        if target_node is not None:
            return int(target_node)
        if target_step is not None:
            node = self.step_to_node.get(int(target_step))
            if node is None:
                raise KeyError(f"No node found for target step {target_step}")
            return node
        if target_image_name is not None:
            node = self.image_to_node.get(target_image_name)
            if node is None:
                raise KeyError(f"No node found for target image {target_image_name}")
            return node
        raise ValueError("One of target_node, target_step, or target_image_name is required")

    def set_checkpoints(
        self,
        checkpoint_steps: Optional[list[int]] = None,
        checkpoint_images: Optional[list[str]] = None,
    ) -> None:
        """Defines a multi-stage route using a sequence of checkpoints.

        Args:
            checkpoint_steps (Optional[list[int]]): A list of step numbers defining the route.
            checkpoint_images (Optional[list[str]]): A list of image filenames defining the route.
        """
        if checkpoint_steps is None and checkpoint_images is None:
            raise ValueError("Provide checkpoint_steps or checkpoint_images")

        nodes: list[int] = []
        if checkpoint_steps is not None:
            for step in checkpoint_steps:
                nodes.append(self.resolve_target_node(target_step=step))
        if checkpoint_images is not None:
            for image_name in checkpoint_images:
                nodes.append(self.resolve_target_node(target_image_name=image_name))

        self.checkpoints = nodes
        self.active_checkpoint_index = 0

    def get_active_checkpoint(self) -> Optional[int]:
        """Returns the node index of the current checkpoint, or None if finished.

        Returns:
            Optional[int]: The active checkpoint node index, or None if the route is complete.
        """
        if 0 <= self.active_checkpoint_index < len(self.checkpoints):
            return self.checkpoints[self.active_checkpoint_index]
        return None

    def advance_checkpoint(self) -> Optional[int]:
        """Moves to the next checkpoint in the sequence.

        Returns:
            Optional[int]: The new active checkpoint node index, or None if the route is complete.
        """
        if self.active_checkpoint_index < len(self.checkpoints):
            self.active_checkpoint_index += 1
        return self.get_active_checkpoint()

    def shortest_path(self, current_node: int, target_node: int) -> list[int]:
        """Calculates the shortest path between two nodes in the graph.

        Args:
            current_node (int): The starting node index.
            target_node (int): The target node index.

        Returns:
            list[int]: A list of node indices representing the shortest path.
        """
        if current_node not in self.graph:
            raise KeyError(f"Current node {current_node} not in graph")
        if target_node not in self.graph:
            raise KeyError(f"Target node {target_node} not in graph")
        # Uses networkx's implementation of Dijkstra's algorithm.
        path = nx.shortest_path(self.graph, source=current_node, target=target_node)
        return [int(node) for node in path]

    def choose_subgoal_node(self, path_nodes: list[int], hops_ahead: Optional[int] = None) -> int:
        """Selects a short-term subgoal from a path.

        The subgoal is a node a few steps ahead on the path. This provides a
        stable, near-term target for the local controller, rather than having
        it aim for the final, distant goal.

        Args:
            path_nodes (list[int]): The sequence of nodes in the full path.
            hops_ahead (Optional[int]): The number of hops to look ahead for the subgoal.
                                       If None, uses the default from the config.

        Returns:
            int: The selected subgoal node index.
        """
        if not path_nodes:
            raise ValueError("path_nodes cannot be empty")
        hops = self.config.max_subgoal_hops if hops_ahead is None else max(0, int(hops_ahead))
        # Clamp index to be within the path's bounds.
        idx = min(hops, len(path_nodes) - 1)
        return int(path_nodes[idx])

    def checkpoint_reached(self, localization_result: dict, target_node: int) -> bool:
        """Checks if the robot has confidently reached the target node.

        Args:
            localization_result (dict): The output from the CorridorLocalizer.
            target_node (int): The target node to check against.

        Returns:
            bool: True if the robot is at the target node with sufficient confidence.
        """
        current_node = localization_result.get("node_index")
        if current_node is None:
            return False
        confidence = float(localization_result.get("confidence", 0.0))
        # The robot must be at the target node with sufficient confidence.
        return int(current_node) == int(target_node) and confidence >= self.config.min_confidence_to_advance

    def plan(
        self,
        localization_result: dict,
        target_node: Optional[int] = None,
        target_step: Optional[int] = None,
        target_image_name: Optional[str] = None,
        hops_ahead: Optional[int] = None,
    ) -> dict:
        """Generates a plan from the current location to a specified target.

        Args:
            localization_result (dict): The output from `CorridorLocalizer.localize_*`.
            target_node (Optional[int]): The final goal node index.
            target_step (Optional[int]): The final goal step number.
            target_image_name (Optional[str]): The final goal image name.
            hops_ahead (Optional[int]): Overrides the default number of hops for subgoal selection.

        Returns:
            A dictionary containing the full plan, including the path and the
            selected subgoal with its associated metadata.
        """
        current_node = localization_result.get("node_index")
        if current_node is None:
            raise ValueError("Localization result does not contain node_index")

        # 1. Determine the final target node from the provided identifiers.
        resolved_target = self.resolve_target_node(
            target_node=target_node,
            target_step=target_step,
            target_image_name=target_image_name,
        )

        # 2. Find the shortest sequence of nodes to the target.
        path_nodes = self.shortest_path(int(current_node), resolved_target)

        # 3. Choose a short-term subgoal from the path.
        subgoal_node = self.choose_subgoal_node(path_nodes, hops_ahead=hops_ahead)

        # 4. Package all information into a plan dictionary for the controller.
        return {
            "current_node": int(current_node),
            "current_step": self.node_to_step.get(int(current_node)),
            "target_node": resolved_target,
            "target_step": self.node_to_step.get(resolved_target),
            "path_nodes": path_nodes,
            "path_steps": [self.node_to_step.get(node) for node in path_nodes],
            "subgoal_node": subgoal_node,
            "subgoal_step": self.node_to_step.get(subgoal_node),
            "subgoal_image_name": self.node_to_name.get(subgoal_node),
            "subgoal_image_path": self.node_to_path.get(subgoal_node),
            "subgoal_metadata": self.image_meta_by_name.get(self.node_to_name.get(subgoal_node, "")),
            "checkpoint_reached": self.checkpoint_reached(localization_result, resolved_target),
        }

    def plan_to_active_checkpoint(self, localization_result: dict, hops_ahead: Optional[int] = None) -> dict:
        """Convenience method to plan to the currently active checkpoint."""
        target = self.get_active_checkpoint()
        if target is None:
            raise ValueError("No active checkpoint configured")
        return self.plan(localization_result, target_node=target, hops_ahead=hops_ahead)
