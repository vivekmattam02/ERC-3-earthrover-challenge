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
    graph_json: Path
    data_info_json: Optional[Path] = None
    max_subgoal_hops: int = 3
    min_confidence_to_advance: float = 0.55
    checkpoint_reach_tolerance: int = 3


class GraphPlanner:
    """Planner-facing wrapper around the corridor graph artifacts."""

    def __init__(self, config: GraphPlannerConfig):
        self.config = config
        self.graph = self._load_graph(config.graph_json)
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
                    pass
            if path is not None:
                self.node_to_path[node_id] = str(path)

        self.image_meta_by_name: dict[str, dict] = {}
        if config.data_info_json is not None and config.data_info_json.is_file():
            with config.data_info_json.open("r", encoding="utf-8") as handle:
                data_info = json.load(handle)
            for entry in data_info:
                self.image_meta_by_name[entry["image"]] = entry

        self.checkpoints: list[int] = []
        self.active_checkpoint_index: int = 0

    def _load_graph(self, path: Path) -> nx.Graph:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        graph = json_graph.node_link_graph(data, edges="links")
        return graph

    def resolve_target_node(
        self,
        target_node: Optional[int] = None,
        target_step: Optional[int] = None,
        target_image_name: Optional[str] = None,
    ) -> int:
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
        if 0 <= self.active_checkpoint_index < len(self.checkpoints):
            return self.checkpoints[self.active_checkpoint_index]
        return None

    def advance_checkpoint(self) -> Optional[int]:
        if self.active_checkpoint_index < len(self.checkpoints):
            self.active_checkpoint_index += 1
        return self.get_active_checkpoint()

    def shortest_path(self, current_node: int, target_node: int) -> Optional[list[int]]:
        if current_node not in self.graph:
            raise KeyError(f"Current node {current_node} not in graph")
        if target_node not in self.graph:
            raise KeyError(f"Target node {target_node} not in graph")
        try:
            path = nx.shortest_path(self.graph, source=current_node, target=target_node)
        except nx.NetworkXNoPath:
            return None
        return [int(node) for node in path]

    def choose_subgoal_node(self, path_nodes: list[int], hops_ahead: Optional[int] = None) -> int:
        if not path_nodes:
            raise ValueError("path_nodes cannot be empty")
        hops = self.config.max_subgoal_hops if hops_ahead is None else max(0, int(hops_ahead))
        idx = min(hops, len(path_nodes) - 1)
        return int(path_nodes[idx])

    def checkpoint_reached(self, localization_result: dict, target_node: int) -> bool:
        current_node = localization_result.get("node_index")
        if current_node is None:
            return False
        confidence = float(localization_result.get("confidence", 0.0))
        if confidence < self.config.min_confidence_to_advance:
            return False
        cur_step = self.node_to_step.get(int(current_node))
        tgt_step = self.node_to_step.get(int(target_node))
        if cur_step is not None and tgt_step is not None:
            return abs(cur_step - tgt_step) <= self.config.checkpoint_reach_tolerance
        return int(current_node) == int(target_node)

    def plan(
        self,
        localization_result: dict,
        target_node: Optional[int] = None,
        target_step: Optional[int] = None,
        target_image_name: Optional[str] = None,
        hops_ahead: Optional[int] = None,
    ) -> dict:
        current_node = localization_result.get("node_index")
        if current_node is None:
            raise ValueError("Localization result does not contain node_index")

        resolved_target = self.resolve_target_node(
            target_node=target_node,
            target_step=target_step,
            target_image_name=target_image_name,
        )
        path_nodes = self.shortest_path(int(current_node), resolved_target)
        if path_nodes is None:
            # Even with no path (e.g. already past the target in a directed graph),
            # check if the checkpoint should be considered reached.
            reached = self.checkpoint_reached(localization_result, resolved_target)
            return {
                "current_node": int(current_node),
                "current_step": self.node_to_step.get(int(current_node)),
                "target_node": resolved_target,
                "target_step": self.node_to_step.get(resolved_target),
                "path_nodes": [],
                "path_steps": [],
                "subgoal_node": None,
                "subgoal_step": None,
                "subgoal_image_name": None,
                "subgoal_image_path": None,
                "subgoal_metadata": None,
                "checkpoint_reached": reached,
                "path_found": False,
                "path_error": f"no_path:{int(current_node)}->{resolved_target}",
            }

        subgoal_node = self.choose_subgoal_node(path_nodes, hops_ahead=hops_ahead)

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
            "path_found": True,
            "path_error": None,
        }

    def plan_to_active_checkpoint(self, localization_result: dict, hops_ahead: Optional[int] = None) -> dict:
        target = self.get_active_checkpoint()
        if target is None:
            raise ValueError("No active checkpoint configured")
        return self.plan(localization_result, target_node=target, hops_ahead=hops_ahead)
