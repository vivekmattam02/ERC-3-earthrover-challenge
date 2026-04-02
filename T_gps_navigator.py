"""GPS waypoint navigation utilities for ERC outdoor missions.

Pure-Python GPS math (no utm dependency). Provides:
- haversine distance and bearing
- heading error computation
- waypoint manager for sequential checkpoint navigation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# GPS coordinate math
# ---------------------------------------------------------------------------

EARTH_RADIUS_M = 6_371_000.0


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters between two lat/lon points."""
    rlat1, rlon1 = math.radians(lat1), math.radians(lon1)
    rlat2, rlon2 = math.radians(lat2), math.radians(lon2)
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return EARTH_RADIUS_M * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def haversine_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial compass bearing (degrees, 0=N 90=E) from point 1 to point 2."""
    rlat1, rlon1 = math.radians(lat1), math.radians(lon1)
    rlat2, rlon2 = math.radians(lat2), math.radians(lon2)
    dlon = rlon2 - rlon1
    x = math.sin(dlon) * math.cos(rlat2)
    y = math.cos(rlat1) * math.sin(rlat2) - math.sin(rlat1) * math.cos(rlat2) * math.cos(dlon)
    bearing_rad = math.atan2(x, y)
    return bearing_rad * 180.0 / math.pi % 360.0


def wrap_angle_deg(delta: float) -> float:
    """Wrap angle to [-180, 180] degrees."""
    return ((delta + 180.0) % 360.0) - 180.0


def heading_error_deg(target_bearing: float, current_heading: float) -> float:
    """Signed heading error in degrees.

    Positive → target is clockwise (right) of current heading.
    Negative → target is counter-clockwise (left) of current heading.
    """
    return wrap_angle_deg(target_bearing - current_heading)


def gps_valid(lat: Optional[float], lon: Optional[float]) -> bool:
    """Return True if lat/lon look like a real outdoor GPS fix."""
    if lat is None or lon is None:
        return False
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return False
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return False
    if lat == 0.0 and lon == 0.0:
        return False
    if lat < -80.0 or lat > 84.0:
        return False
    if lon < -180.0 or lon > 180.0:
        return False
    return True


# ---------------------------------------------------------------------------
# Waypoint / Checkpoint manager
# ---------------------------------------------------------------------------

@dataclass
class Checkpoint:
    """A single mission checkpoint."""
    sequence: int
    latitude: float
    longitude: float
    checkpoint_id: Optional[int] = None


@dataclass
class GPSWaypointManager:
    """Manages sequential checkpoint navigation for an outdoor mission."""

    checkpoints: list[Checkpoint] = field(default_factory=list)
    current_index: int = 0
    # Distance thresholds
    arrival_radius_m: float = 8.0       # close enough to attempt /checkpoint-reached
    slowdown_radius_m: float = 20.0     # start slowing down

    def load_from_sdk(self, checkpoints_list: list[dict]) -> None:
        """Load checkpoints from SDK /checkpoints-list response."""
        self.checkpoints = []
        for cp in sorted(checkpoints_list, key=lambda c: c.get("sequence", 0)):
            self.checkpoints.append(Checkpoint(
                sequence=int(cp["sequence"]),
                latitude=float(cp["latitude"]),
                longitude=float(cp["longitude"]),
                checkpoint_id=cp.get("id"),
            ))
        self.current_index = 0

    @property
    def active_checkpoint(self) -> Optional[Checkpoint]:
        if 0 <= self.current_index < len(self.checkpoints):
            return self.checkpoints[self.current_index]
        return None

    @property
    def is_mission_complete(self) -> bool:
        return self.current_index >= len(self.checkpoints)

    @property
    def total_checkpoints(self) -> int:
        return len(self.checkpoints)

    @property
    def checkpoints_reached(self) -> int:
        return min(self.current_index, len(self.checkpoints))

    def advance(self) -> Optional[Checkpoint]:
        """Move to the next checkpoint. Returns the new active checkpoint or None."""
        self.current_index += 1
        return self.active_checkpoint

    def compute_nav_state(
        self,
        current_lat: float,
        current_lon: float,
        current_heading_deg: float,
    ) -> dict:
        """Compute navigation state relative to the active checkpoint.

        Returns dict with:
            distance_m, bearing_deg, heading_error_deg,
            within_arrival, within_slowdown,
            checkpoint_sequence, checkpoint_lat, checkpoint_lon
        """
        cp = self.active_checkpoint
        if cp is None:
            return {"mission_complete": True}

        dist = haversine_distance(current_lat, current_lon, cp.latitude, cp.longitude)
        bearing = haversine_bearing(current_lat, current_lon, cp.latitude, cp.longitude)
        h_error = heading_error_deg(bearing, current_heading_deg)

        return {
            "mission_complete": False,
            "distance_m": dist,
            "bearing_deg": bearing,
            "heading_error_deg": h_error,
            "within_arrival": dist <= self.arrival_radius_m,
            "within_slowdown": dist <= self.slowdown_radius_m,
            "checkpoint_sequence": cp.sequence,
            "checkpoint_lat": cp.latitude,
            "checkpoint_lon": cp.longitude,
            "checkpoint_index": self.current_index,
            "total_checkpoints": len(self.checkpoints),
        }
