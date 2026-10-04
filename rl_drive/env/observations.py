"""The two observation encodings. Both read the same world, so agents stay comparable."""

from __future__ import annotations

import numpy as np

from .world import GREEN, RED, YELLOW

NEAR, MED, FAR, CLEAR = 0, 1, 2, 3
STEADY, CLOSING, OPENING = 0, 1, 2  # index 0 is the neutral value the ablation falls back to
BLOCKED, FREE, NO_LANE = 0, 1, 2
L_NONE, L_GREEN, L_YELLOW, L_RED = 0, 1, 2, 3

GAP_NAMES = ("near", "med", "far", "clear")
CLOSING_NAMES = ("steady", "closing", "opening")
SIDE_NAMES = ("blocked", "free", "no_lane")
LIGHT_NAMES = ("none", "green", "yellow", "red")
FEATURE_NAMES = ("lane", "speed", "gap_ahead", "closing", "left", "right", "light")

CONTINUOUS_DIM = 15
_PHASE_TO_FEATURE = {GREEN: L_GREEN, YELLOW: L_YELLOW, RED: L_RED}


def _gap_bucket(cfg, gap: float) -> int:
    if gap < cfg.gap_near:
        return NEAR
    if gap < cfg.gap_med:
        return MED
    if gap < cfg.gap_far:
        return FAR
    return CLEAR


def _closing(world, lead) -> int:
    """Whether the gap to the lead vehicle is shrinking. Distance alone is not enough."""
    if world.cfg.closing_buckets < 2:
        return STEADY
    if lead is None:
        return OPENING
    delta = lead.speed - world.ego.speed
    threshold = world.cfg.speed_unit / 2.0
    if delta < -threshold:
        return CLOSING
    if delta > threshold:
        return OPENING
    return STEADY


def _side(world, lane: int) -> int:
    """Whether an adjacent lane is safe to merge into; the window grows with ego speed."""
    cfg = world.cfg
    if not 0 <= lane < cfg.lanes:
        return NO_LANE
    ahead = cfg.side_ahead + world.ego.speed
    for v in world.traffic:
        if v.lane == lane and -cfg.side_behind <= v.x - world.ego.x <= ahead:
            return BLOCKED
    return FREE


def _light_feature(cfg, distance: float, phase: int | None) -> int:
    if phase is None or distance > cfg.light_lookahead:
        return L_NONE
    return _PHASE_TO_FEATURE[phase]


class DiscreteEncoder:
    """Buckets the world into a small tuple so tabular methods stay tractable."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.dims = (cfg.lanes, cfg.max_speed_level + 1, 4, cfg.closing_buckets, 3, 3, 4)
        self.n = int(np.prod(self.dims))

    @property
    def reachable(self) -> int:
        """Side occupancy is constrained by the lane index, so the raw table size overstates |S|."""
        _, speeds, gaps, closings, _, _, lights = self.dims
        sides = sum((1 if lane == 0 else 2) * (1 if lane == self.cfg.lanes - 1 else 2)
                    for lane in range(self.cfg.lanes))
        return sides * speeds * gaps * closings * lights

    def features(self, world) -> tuple[int, ...]:
        lead = world.lead(world.ego.x, world.ego.lane)
        gap = float("inf") if lead is None else lead.x - world.ego.x
        distance, phase = world.next_light()
        return (
            world.ego.lane,
            world.ego_speed_level,
            _gap_bucket(self.cfg, gap),
            _closing(world, lead),
            _side(world, world.ego.lane - 1),
            _side(world, world.ego.lane + 1),
            _light_feature(self.cfg, distance, phase),
        )

    def encode(self, world) -> int:
        return int(np.ravel_multi_index(self.features(world), self.dims))

    def decode(self, state: int) -> tuple[int, ...]:
        return tuple(int(i) for i in np.unravel_index(state, self.dims))

    def describe(self, state: int) -> str:
        lane, speed, gap, closing, left, right, light = self.decode(state)
        return (f"lane={lane} speed={speed} gap={GAP_NAMES[gap]} {CLOSING_NAMES[closing]} "
                f"left={SIDE_NAMES[left]} right={SIDE_NAMES[right]} light={LIGHT_NAMES[light]}")


class ContinuousEncoder:
    """Normalised feature vector for function-approximation agents."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.n = CONTINUOUS_DIM

    def encode(self, world) -> np.ndarray:
        cfg = self.cfg
        scale = cfg.gap_far
        out = np.zeros(CONTINUOUS_DIM, dtype=np.float32)
        out[0] = world.ego_speed_level / cfg.max_speed_level
        out[1 + world.ego.lane] = 1.0
        for i, lane in enumerate((world.ego.lane, world.ego.lane - 1, world.ego.lane + 1)):
            out[4 + i] = _relative_gap(world, lane, ahead=True, scale=scale)
        for i, lane in enumerate((world.ego.lane - 1, world.ego.lane + 1)):
            out[7 + i] = _relative_gap(world, lane, ahead=False, scale=scale)
        lead = world.lead(world.ego.x, world.ego.lane)
        relative = 0.0 if lead is None else (lead.speed - world.ego.speed)
        span = cfg.max_speed_level * cfg.speed_unit
        out[9] = float(np.clip(0.5 + relative / (2.0 * span), 0.0, 1.0))
        distance, phase = world.next_light()
        out[10] = min(distance, cfg.light_lookahead) / cfg.light_lookahead
        out[11 + _light_feature(cfg, distance, phase)] = 1.0
        return out


def _relative_gap(world, lane: int, ahead: bool, scale: float) -> float:
    if not 0 <= lane < world.cfg.lanes:
        return 0.0
    best = float("inf")
    for v in world.traffic:
        if v.lane != lane:
            continue
        delta = (v.x - world.ego.x) if ahead else (world.ego.x - v.x)
        if delta >= 0.0:
            best = min(best, delta)
    return float(min(best, scale) / scale)
