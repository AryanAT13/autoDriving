"""Gymnasium environment: a three-lane road with traffic and signals."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..config import EnvConfig, RewardConfig
from .observations import CONTINUOUS_DIM, ContinuousEncoder, DiscreteEncoder
from .rewards import StepEvents, step_reward
from .world import PHASE_NAMES, RED, World

MAINTAIN, ACCELERATE, BRAKE, LANE_LEFT, LANE_RIGHT = range(5)
ACTION_NAMES = ("MAINTAIN", "ACCELERATE", "BRAKE", "LANE_LEFT", "LANE_RIGHT")
N_ACTIONS = len(ACTION_NAMES)


class DrivingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, env_cfg=None, reward_cfg=None, obs_mode: str = "discrete"):
        if obs_mode not in ("discrete", "continuous"):
            raise ValueError(f"obs_mode must be 'discrete' or 'continuous', got {obs_mode!r}")
        self.cfg = env_cfg or EnvConfig()
        self.reward_cfg = reward_cfg or RewardConfig()
        self.obs_mode = obs_mode
        self.encoder = (DiscreteEncoder(self.cfg) if obs_mode == "discrete"
                        else ContinuousEncoder(self.cfg))
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = (
            spaces.Discrete(self.encoder.n) if obs_mode == "discrete"
            else spaces.Box(0.0, 1.0, (CONTINUOUS_DIM,), dtype=np.float32)
        )
        self.world: World | None = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.world = World(self.cfg, self.np_random)
        return self.encoder.encode(self.world), {}

    def step(self, action: int):
        cfg, world = self.cfg, self.world
        events = StepEvents()

        if action == ACCELERATE:
            world.ego_speed_level = min(cfg.max_speed_level, world.ego_speed_level + 1)
        elif action == BRAKE:
            world.ego_speed_level = max(0, world.ego_speed_level - 1)
        elif action in (LANE_LEFT, LANE_RIGHT):
            target = world.ego.lane + (-1 if action == LANE_LEFT else 1)
            if 0 <= target < cfg.lanes:
                world.ego.lane = target
                events.lane_changed = True
            else:
                events.off_road = True

        world.ego.speed = world.ego_speed_level * cfg.speed_unit
        events.speed_level = world.ego_speed_level

        moves = world.traffic_moves()
        events.collision = self._collides(moves)

        start_x = world.ego.x
        world.ego.x += world.ego.speed
        for vehicle, delta in zip(world.traffic, moves):
            vehicle.x += delta
        events.ran_red = self._ran_red(start_x, world.ego.x)
        world.steps += 1

        events.goal = bool(world.ego.x >= cfg.road_length and not events.collision)
        reward = step_reward(self.reward_cfg, events)
        terminated = bool(events.collision or events.goal)
        truncated = bool(not terminated and world.steps >= cfg.max_steps)
        if not terminated:
            world.recycle()
        return self.encoder.encode(world), reward, terminated, truncated, self._info(events)

    def _collides(self, moves: list[float]) -> bool:
        """Swept check: at these speeds a vehicle can pass clean through another in one step."""
        world, length = self.world, self.cfg.car_length
        for vehicle, delta in zip(world.traffic, moves):
            if vehicle.lane != world.ego.lane:
                continue
            before = vehicle.x - world.ego.x
            after = (vehicle.x + delta) - (world.ego.x + world.ego.speed)
            if abs(before) < length or abs(after) < length or (before > 0.0) != (after > 0.0):
                return True
        return False

    def _ran_red(self, start_x: float, end_x: float) -> bool:
        for i, position in enumerate(self.cfg.light_positions):
            if start_x < position <= end_x and self.world.light_phase(i) == RED:
                return True
        return False

    def _info(self, events: StepEvents) -> dict:
        return {
            "collision": events.collision,
            "goal": events.goal,
            "ran_red": events.ran_red,
            "off_road": events.off_road,
            "lane_changed": events.lane_changed,
            "speed_level": events.speed_level,
            "x": self.world.ego.x,
            "lane": self.world.ego.lane,
            "steps": self.world.steps,
        }

    def frame(self) -> dict:
        """Serialisable snapshot for the browser renderer."""
        world = self.world
        return {
            "step": world.steps,
            "lanes": self.cfg.lanes,
            "road_length": self.cfg.road_length,
            "ego": {"x": world.ego.x, "lane": world.ego.lane, "speed": world.ego_speed_level},
            "traffic": [{"x": v.x, "lane": v.lane} for v in world.traffic],
            "lights": [{"x": p, "phase": PHASE_NAMES[world.light_phase(i)]}
                       for i, p in enumerate(self.cfg.light_positions)],
        }
