"""Non-learning reference policies: the performance floor the learners must beat."""

from __future__ import annotations

import numpy as np

from ..env.driving_env import ACCELERATE, BRAKE, LANE_LEFT, LANE_RIGHT, MAINTAIN
from ..env.observations import CLEAR, CLOSING, FAR, FREE, L_RED, L_YELLOW, MED, NEAR


class _Fixed:
    obs_mode = "discrete"

    def observe(self, *args) -> None:
        pass

    def end_episode(self) -> None:
        pass

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, data: dict) -> None:
        pass


class RandomAgent(_Fixed):
    def __init__(self, env, cfg=None, rng=None):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.n_actions = env.action_space.n

    def act(self, obs, greedy: bool = False) -> int:
        return int(self.rng.integers(self.n_actions))


class AlwaysAccelerate(_Fixed):
    def __init__(self, env, cfg=None, rng=None):
        pass

    def act(self, obs, greedy: bool = False) -> int:
        return ACCELERATE


class ScriptedSafeDriver(_Fixed):
    """Hand-written rules over the discrete state.

    Its job is to prove the observation is sufficient to drive safely, so a failure to
    learn is an agent problem and not an unsolvable environment.
    """

    def __init__(self, env, cfg=None, rng=None):
        self.encoder = env.encoder
        self.max_speed = env.cfg.max_speed_level

    def act(self, obs, greedy: bool = False) -> int:
        _, speed, gap, closing, left, right, light = self.encoder.decode(obs)
        if light in (L_YELLOW, L_RED):
            return BRAKE if speed > 0 else MAINTAIN
        if gap == NEAR or (gap == MED and closing == CLOSING):
            if left == FREE:
                return LANE_LEFT
            if right == FREE:
                return LANE_RIGHT
            return BRAKE
        if gap == MED and speed >= 3:
            return BRAKE
        if gap == CLEAR and speed < self.max_speed:
            return ACCELERATE
        if gap == FAR and speed < 2:
            return ACCELERATE
        return MAINTAIN
