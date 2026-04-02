"""OpenStreetMap pedestrian router for ERC outdoor missions.

This module queries Overpass, builds a lightweight pedestrian graph,
and returns intermediate GPS waypoints suitable for local outdoor controllers.

Design goals:
- keep the API independent of any specific controller
- prefer pedestrian-friendly paths when possible
- still allow road-like fallbacks with higher traversal cost
- return waypoint lists that can be fed directly into the outdoor runtime
- never make OSM availability a single point of mission failure
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import requests


LatLon = tuple[float, float]
BBox = tuple[float, float, float, float]


def haversine_m(a: LatLon, b: LatLon) -> float:
    """Great-circle distance in meters."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    return 2.0 * 6371000.0 * math.asin(math.sqrt(h))


def compute_bbox(start: LatLon, goal: LatLon, buffer_m: float = 300.0) -> BBox:
    """Compute a simple lat/lon bounding box with meter padding."""
    min_lat = min(start[0], goal[0])
    max_lat = max(start[0], goal[0])
    min_lon = min(start[1], goal[1])
    max_lon = max(start[1], goal[1])

    center_lat = 0.5 * (min_lat + max_lat)
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = max(1.0, 111320.0 * math.cos(math.radians(center_lat)))

    lat_pad = buffer_m / meters_per_deg_lat
    lon_pad = buffer_m / meters_per_deg_lon
    return (
        min_lat - lat_pad,
        min_lon - lon_pad,
        max_lat + lat_pad,
        max_lon + lon_pad,
    )


@dataclass
class OSMRoutingConfig:
    overpass_url: str = "https://overpass-api.de/api/interpreter"
    timeout_s: float = 20.0
    query_timeout_s: int = 25
    request_retries: int = 2
    retry_backoff_s: float = 1.0
    buffer_m: float = 300.0
    max_segment_m: float = 20.0
    min_waypoint_spacing_m: float = 8.0
    max_snap_distance_m: float = 60.0
    allowed_highways: tuple[str, ...] | None = None
    highway_cost_multipliers: dict[str, float] = field(
        default_factory=lambda: {
            "footway": 1.0,
            "pedestrian": 1.0,
            "path": 1.05,
            "living_street": 1.20,
            "cycleway": 1.60,
            "service": 1.75,
            "track": 1.90,
            "residential": 2.20,
            "unclassified": 2.40,
            "tertiary": 2.80,
            "secondary": 4.00,
            "primary": 6.00,
            "road": 3.00,
        }
    )


@dataclass
class OSMRouteResult:
    waypoints: list[LatLon]
    node_path: list[int]
    total_distance_m: float
    debug: dict[str, Any]


def build_overpass_query(bbox: BBox, query_timeout_s: int, allowed_highways: tuple[str, ...] | None = None) -> str:
    south, west, north, east = bbox
    if allowed_highways:
        highway_regex = "|".join(allowed_highways)
    else:
        highway_regex = (
            "footway|path|pedestrian|living_street|cycleway|service|track|"
            "residential|unclassified|tertiary|secondary|primary|road"
        )
    return f"""
[out:json][timeout:{int(query_timeout_s)}];
(
  way["highway"~"{highway_regex}"]({south},{west},{north},{east});
);
(._;>;);
out body;
""".strip()


def _straight_line_result(start: LatLon, goal: LatLon, *, error: Optional[str] = None) -> OSMRouteResult:
    debug = {
        "routing": "fallback_straight_line",
        "route_distance_m": haversine_m(start, goal),
        "waypoints_count": 2,
    }
    if error is not None:
        debug["fallback_error"] = error
    return OSMRouteResult(
        waypoints=[start, goal],
        node_path=[],
        total_distance_m=haversine_m(start, goal),
        debug=debug,
    )


def fetch_overpass_json(
    start: LatLon,
    goal: LatLon,
    config: OSMRoutingConfig,
    *,
    session: Optional[requests.Session] = None,
) -> dict[str, Any]:
    bbox = compute_bbox(start, goal, buffer_m=config.buffer_m)
    query = build_overpass_query(bbox, query_timeout_s=config.query_timeout_s, allowed_highways=config.allowed_highways)
    client = session or requests.Session()

    last_error: Optional[Exception] = None
    attempts = max(1, int(config.request_retries))
    for attempt in range(attempts):
        try:
            response = client.post(
                config.overpass_url,
                data=query.encode("utf-8"),
                timeout=config.timeout_s,
                headers={"Content-Type": "text/plain"},
            )
            response.raise_for_status()
            payload = response.json()
            if "elements" not in payload:
                raise RuntimeError("Overpass response missing 'elements'")
            return payload
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(config.retry_backoff_s * (attempt + 1))

    if last_error is None:
        raise RuntimeError("Unknown Overpass fetch failure")
    raise last_error


def _is_oneway(tags: dict[str, Any]) -> bool:
    value = str(tags.get("oneway", "")).lower()
    return value in {"yes", "true", "1"}


def _node_latlon(node: dict[str, Any]) -> LatLon:
    return float(node["lat"]), float(node["lon"])


def build_pedestrian_graph(
    payload: dict[str, Any],
    config: OSMRoutingConfig,
) -> tuple[dict[int, LatLon], dict[int, list[tuple[int, float]]], dict[str, Any]]:
    elements = payload.get("elements", [])
    nodes: dict[int, LatLon] = {}
    adjacency: dict[int, list[tuple[int, float]]] = {}
    ways_used = 0

    for element in elements:
        if element.get("type") == "node":
            nodes[int(element["id"])] = _node_latlon(element)

    for element in elements:
        if element.get("type") != "way":
            continue
        tags = element.get("tags", {}) or {}
        highway = str(tags.get("highway", "")).lower()
        if config.allowed_highways is not None and highway not in config.allowed_highways:
            continue
        if highway not in config.highway_cost_multipliers:
            continue

        node_ids = [int(nid) for nid in element.get("nodes", []) if int(nid) in nodes]
        if len(node_ids) < 2:
            continue

        multiplier = float(config.highway_cost_multipliers[highway])
        oneway = _is_oneway(tags)

        for a, b in zip(node_ids[:-1], node_ids[1:]):
            pa = nodes[a]
            pb = nodes[b]
            distance = haversine_m(pa, pb)
            cost = distance * multiplier
            adjacency.setdefault(a, []).append((b, cost))
            adjacency.setdefault(b, [])
            if not oneway:
                adjacency[b].append((a, cost))
                adjacency.setdefault(a, adjacency.get(a, []))
        ways_used += 1

    debug = {
        "elements_total": len(elements),
        "nodes_total": len(nodes),
        "ways_used": ways_used,
        "graph_nodes": len(adjacency),
    }
    return nodes, adjacency, debug


def nearest_graph_node(latlon: LatLon, nodes: dict[int, LatLon]) -> tuple[int, float]:
    if not nodes:
        raise RuntimeError("OSM graph is empty")
    best_id = -1
    best_distance = float("inf")
    for node_id, node_latlon in nodes.items():
        d = haversine_m(latlon, node_latlon)
        if d < best_distance:
            best_distance = d
            best_id = node_id
    return best_id, best_distance


def astar_path(
    start_id: int,
    goal_id: int,
    nodes: dict[int, LatLon],
    adjacency: dict[int, list[tuple[int, float]]],
) -> tuple[list[int], float]:
    open_heap: list[tuple[float, int]] = []
    heapq.heappush(open_heap, (0.0, start_id))

    came_from: dict[int, int] = {}
    g_score: dict[int, float] = {start_id: 0.0}
    closed: set[int] = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal_id:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, g_score[goal_id]

        closed.add(current)
        for neighbor, edge_cost in adjacency.get(current, []):
            tentative = g_score[current] + edge_cost
            if tentative < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                heuristic = haversine_m(nodes[neighbor], nodes[goal_id])
                heapq.heappush(open_heap, (tentative + heuristic, neighbor))

    raise RuntimeError(f"No pedestrian route found between graph nodes {start_id} and {goal_id}")


def _interpolate_segment(a: LatLon, b: LatLon, max_segment_m: float) -> list[LatLon]:
    distance = haversine_m(a, b)
    if distance <= max_segment_m:
        return [b]
    pieces = max(1, int(math.ceil(distance / max_segment_m)))
    out: list[LatLon] = []
    for idx in range(1, pieces + 1):
        t = idx / pieces
        lat = a[0] + t * (b[0] - a[0])
        lon = a[1] + t * (b[1] - a[1])
        out.append((lat, lon))
    return out


def _densify_path(points: list[LatLon], max_segment_m: float) -> list[LatLon]:
    if len(points) <= 1:
        return points[:]
    out = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        out.extend(_interpolate_segment(a, b, max_segment_m=max_segment_m))
    return out


def _thin_waypoints(points: list[LatLon], min_spacing_m: float) -> list[LatLon]:
    if len(points) <= 2:
        return points[:]
    out = [points[0]]
    for point in points[1:-1]:
        if haversine_m(out[-1], point) >= min_spacing_m:
            out.append(point)
    out.append(points[-1])
    return out


def route_from_overpass_payload(
    start: LatLon,
    goal: LatLon,
    payload: dict[str, Any],
    config: Optional[OSMRoutingConfig] = None,
) -> OSMRouteResult:
    config = config or OSMRoutingConfig()
    nodes, adjacency, debug = build_pedestrian_graph(payload, config)
    start_id, start_snap_m = nearest_graph_node(start, nodes)
    goal_id, goal_snap_m = nearest_graph_node(goal, nodes)

    if start_snap_m > config.max_snap_distance_m:
        raise RuntimeError(f"Start is too far from OSM pedestrian graph ({start_snap_m:.1f} m)")
    if goal_snap_m > config.max_snap_distance_m:
        raise RuntimeError(f"Goal is too far from OSM pedestrian graph ({goal_snap_m:.1f} m)")

    node_path, weighted_cost = astar_path(start_id, goal_id, nodes, adjacency)
    raw_path = [start] + [nodes[node_id] for node_id in node_path] + [goal]
    densified = _densify_path(raw_path, max_segment_m=config.max_segment_m)
    thinned = _thin_waypoints(densified, min_spacing_m=config.min_waypoint_spacing_m)

    route_distance = 0.0
    for a, b in zip(thinned[:-1], thinned[1:]):
        route_distance += haversine_m(a, b)

    route_debug = {
        **debug,
        "routing": "osm_astar",
        "start_snap_m": start_snap_m,
        "goal_snap_m": goal_snap_m,
        "graph_start_id": start_id,
        "graph_goal_id": goal_id,
        "node_path_len": len(node_path),
        "weighted_cost": weighted_cost,
        "route_distance_m": route_distance,
        "waypoints_count": len(thinned),
    }

    return OSMRouteResult(
        waypoints=thinned,
        node_path=node_path,
        total_distance_m=route_distance,
        debug=route_debug,
    )


def get_pedestrian_route(
    start: LatLon,
    goal: LatLon,
    config: Optional[OSMRoutingConfig] = None,
    *,
    session: Optional[requests.Session] = None,
    allow_fallback: bool = True,
) -> OSMRouteResult:
    config = config or OSMRoutingConfig()
    try:
        payload = fetch_overpass_json(start, goal, config, session=session)
        return route_from_overpass_payload(start, goal, payload, config=config)
    except Exception as exc:
        if not allow_fallback:
            raise
        return _straight_line_result(start, goal, error=str(exc))
