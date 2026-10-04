"""Configuration for the simulation, the reward function and the learning agents."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class EnvConfig:
    """Road layout, vehicle kinematics, traffic behaviour and observation bucketing."""

    lanes: int = 3
    road_length: float = 1200.0
    max_steps: int = 300
    car_length: float = 10.0
    speed_unit: float = 4.0
    max_speed_level: int = 4

    n_traffic: int = 12
    traffic_speed_min: int = 1
    traffic_speed_max: int = 3
    spawn_min: float = 80.0
    spawn_max: float = 440.0
    spawn_behind: float = 120.0
    despawn_behind: float = 140.0
    min_spawn_gap: float = 30.0
    traffic_safe_gap: float = 35.0

    closing_buckets: int = 1  # 3 adds relative lead speed; measured to cost more than it gains
    gap_near: float = 20.0
    gap_med: float = 45.0
    gap_far: float = 80.0
    side_behind: float = 15.0
    side_ahead: float = 25.0

    light_positions: tuple[float, ...] = (300.0, 600.0, 900.0)
    light_green: int = 18
    light_yellow: int = 4
    light_red: int = 12
    light_lookahead: float = 60.0


@dataclass
class RewardConfig:
    """Per-step reward terms. Progress and the step penalty together rule out idling."""

    progress: float = 0.1
    step_penalty: float = -0.25
    lane_change: float = -0.2
    off_road: float = -5.0
    red_light: float = -20.0
    collision: float = -75.0
    goal: float = 50.0


@dataclass
class AgentConfig:
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 3000
    optimistic_init: float = 0.0
    planning_steps: int = 10


def apply_overrides(cfg, overrides: dict):
    """Set dataclass fields from a dict, rejecting unknown keys."""
    known = {f.name for f in fields(cfg)}
    for key, value in overrides.items():
        if key not in known:
            raise KeyError(f"unknown config field {key!r} for {type(cfg).__name__}")
        setattr(cfg, key, value)
    return cfg


ALGO_DEFAULTS = Path(__file__).resolve().parent.parent / "configs" / "algos.json"


def algo_defaults(algo: str) -> dict:
    """Per-algorithm hyperparameters. SARSA in particular needs a lower exploration rate."""
    if not ALGO_DEFAULTS.exists():
        return {}
    return json.loads(ALGO_DEFAULTS.read_text()).get(algo, {})


def load_configs(path: str | Path | None, algo: str | None = None):
    """Build configs from defaults, then per-algorithm tuning, then a JSON preset."""
    env, reward, agent = EnvConfig(), RewardConfig(), AgentConfig()
    if algo:
        apply_overrides(agent, algo_defaults(algo))
    if path:
        data = json.loads(Path(path).read_text())
        apply_overrides(env, data.get("env", {}))
        apply_overrides(reward, data.get("reward", {}))
        apply_overrides(agent, data.get("agent", {}))
    return env, reward, agent


def dump_configs(env: EnvConfig, reward: RewardConfig, agent: AgentConfig) -> dict:
    return {"env": asdict(env), "reward": asdict(reward), "agent": asdict(agent)}
