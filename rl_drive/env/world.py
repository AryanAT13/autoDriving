"""Mutable simulation state: the ego vehicle, surrounding traffic and signal phases."""

from __future__ import annotations

from dataclasses import dataclass

GREEN, YELLOW, RED = 0, 1, 2
PHASE_NAMES = ("green", "yellow", "red")


@dataclass
class Vehicle:
    x: float
    lane: int
    speed: float


class World:
    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng
        self.steps = 0
        self.ego = Vehicle(0.0, cfg.lanes // 2, 2 * cfg.speed_unit)
        self.ego_speed_level = 2
        cycle = cfg.light_green + cfg.light_yellow + cfg.light_red
        self.light_offsets = [int(rng.integers(cycle)) for _ in cfg.light_positions]
        self.traffic: list[Vehicle] = []
        for _ in range(cfg.n_traffic):
            self._spawn(initial=True)

    def _spawn(self, initial: bool = False) -> None:
        cfg = self.cfg
        low = self.ego.x - cfg.spawn_behind if initial else self.ego.x + cfg.spawn_min
        high = self.ego.x + cfg.spawn_max
        for _ in range(30):
            lane = int(self.rng.integers(cfg.lanes))
            x = float(self.rng.uniform(low, high))
            if self._is_free(x, lane):
                level = int(self.rng.integers(cfg.traffic_speed_min, cfg.traffic_speed_max + 1))
                self.traffic.append(Vehicle(x, lane, level * cfg.speed_unit))
                return

    def _is_free(self, x: float, lane: int) -> bool:
        clearance = self.cfg.car_length + self.cfg.min_spawn_gap
        if lane == self.ego.lane and abs(x - self.ego.x) < clearance:
            return False
        return all(not (v.lane == lane and abs(v.x - x) < clearance) for v in self.traffic)

    def lead(self, x: float, lane: int, exclude: Vehicle | None = None,
             include_ego: bool = False) -> Vehicle | None:
        """Nearest vehicle in front in the given lane, or None if the lane is clear."""
        best, best_gap = None, float("inf")
        candidates = list(self.traffic)
        if include_ego:
            candidates.append(self.ego)
        for v in candidates:
            if v is exclude or v.lane != lane:
                continue
            delta = v.x - x
            if 0.0 < delta < best_gap:
                best, best_gap = v, delta
        return best

    def gap_ahead(self, x: float, lane: int, exclude: Vehicle | None = None,
                  include_ego: bool = False) -> float:
        vehicle = self.lead(x, lane, exclude, include_ego)
        return float("inf") if vehicle is None else vehicle.x - x

    def traffic_moves(self) -> list[float]:
        """Per-vehicle displacement for this step; traffic brakes to keep a safe gap."""
        cfg = self.cfg
        moves = []
        for v in self.traffic:
            gap = self.gap_ahead(v.x, v.lane, exclude=v, include_ego=True)
            if gap > cfg.traffic_safe_gap:
                moves.append(v.speed)
            else:
                moves.append(min(v.speed, max(0.0, gap - cfg.car_length)))
        return moves

    def recycle(self) -> None:
        """Respawn vehicles that have dropped behind, keeping traffic density constant."""
        for v in list(self.traffic):
            if v.x < self.ego.x - self.cfg.despawn_behind:
                self.traffic.remove(v)
                self._spawn()

    def light_phase(self, index: int) -> int:
        cfg = self.cfg
        cycle = cfg.light_green + cfg.light_yellow + cfg.light_red
        t = (self.steps + self.light_offsets[index]) % cycle
        if t < cfg.light_green:
            return GREEN
        if t < cfg.light_green + cfg.light_yellow:
            return YELLOW
        return RED

    def next_light(self) -> tuple[float, int | None]:
        """Distance and phase of the nearest signal ahead of the ego."""
        best_distance, best_phase = float("inf"), None
        for i, position in enumerate(self.cfg.light_positions):
            distance = position - self.ego.x
            if 0.0 <= distance < best_distance:
                best_distance, best_phase = distance, self.light_phase(i)
        return best_distance, best_phase
