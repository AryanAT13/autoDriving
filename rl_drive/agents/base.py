"""Agent interface, exploration schedule and the shared epsilon-greedy table."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Agent(Protocol):
    """Three methods cover every algorithm: TD updates in observe, MC in end_episode."""

    obs_mode: str

    def act(self, obs, greedy: bool = False) -> int: ...

    def observe(self, obs, action, reward, next_obs, next_action,
                terminated: bool, truncated: bool) -> None: ...

    def end_episode(self) -> None: ...


class EpsilonSchedule:
    """Exponential decay reaching `end` after `episodes` episodes."""

    def __init__(self, start: float, end: float, episodes: int):
        self.start, self.end = start, end
        self.rate = (end / start) ** (1.0 / max(1, episodes)) if start > 0.0 else 0.0
        self.value = start

    def update(self, episode: int) -> float:
        self.value = max(self.end, self.start * self.rate**episode)
        return self.value


class TabularPolicy:
    """Epsilon-greedy action selection over a Q-table, shared by the tabular learners."""

    obs_mode = "discrete"

    def __init__(self, env, cfg, rng=None):
        self.cfg = cfg
        self.rng = rng if rng is not None else np.random.default_rng()
        self.n_actions = env.action_space.n
        self.q = np.zeros((env.encoder.n, self.n_actions), dtype=np.float64)
        self.epsilon = EpsilonSchedule(cfg.epsilon_start, cfg.epsilon_end,
                                       cfg.epsilon_decay_episodes)
        self.last_td = 0.0

    def act(self, obs, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon.value:
            return int(self.rng.integers(self.n_actions))
        values = self.q[obs]
        if greedy:
            # Deterministic tie-break so evaluation on fixed seeds is reproducible.
            return int(np.argmax(values))
        best = np.flatnonzero(values == values.max())
        return int(self.rng.choice(best))

    def end_episode(self) -> None:
        pass

    def state_dict(self) -> dict:
        return {"q": self.q}

    def load_state_dict(self, data: dict) -> None:
        self.q = np.asarray(data["q"])
