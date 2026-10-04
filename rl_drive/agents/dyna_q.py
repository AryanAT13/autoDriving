"""Model-based planning: learn a transition model and replay it between real steps."""

from __future__ import annotations

from collections import deque

import numpy as np

from .tabular import QLearning


class DynaQ(QLearning):
    """Each real step is followed by `planning_steps` updates drawn from a learned model.

    The book's model is deterministic, keeping only the last transition per state-action
    pair. That assumption fails here: the state buckets alias many traffic configurations,
    so replaying one sample misrepresents the real distribution. Keeping up to
    `model_capacity` transitions per pair and sampling among them restores the benefit.
    Set `model_capacity = 1` for the textbook behaviour.
    """

    def __init__(self, env, cfg, rng=None):
        super().__init__(env, cfg, rng)
        self.model: dict[tuple[int, int], deque] = {}
        self.keys: list[tuple[int, int]] = []

    def observe(self, obs, action, reward, next_obs, next_action, terminated, truncated):
        super().observe(obs, action, reward, next_obs, next_action, terminated, truncated)
        self._remember(int(obs), int(action), float(reward), int(next_obs), bool(terminated))
        self._plan()

    def _remember(self, state, action, reward, next_state, terminated) -> None:
        key = (state, action)
        bucket = self.model.get(key)
        if bucket is None:
            bucket = self.model[key] = deque(maxlen=max(1, self.cfg.model_capacity))
            self.keys.append(key)
        bucket.append((reward, next_state, terminated))

    def _plan(self) -> None:
        if not self.keys or self.cfg.planning_steps <= 0:
            return
        for index in self.rng.integers(len(self.keys), size=self.cfg.planning_steps):
            key = self.keys[index]
            bucket = self.model[key]
            reward, next_state, terminated = bucket[int(self.rng.integers(len(bucket)))]
            self._planning_update(key[0], key[1], reward, next_state, terminated)

    def _planning_update(self, state, action, reward, next_state, terminated) -> None:
        target = reward if terminated else reward + self.cfg.gamma * self.q[next_state].max()
        self.q[state, action] += self.cfg.alpha * (target - self.q[state, action])


class DynaQPlus(DynaQ):
    """Dyna-Q with an exploration bonus for pairs not tried recently.

    The bonus applies during planning only. Untried actions are not seeded into the model,
    so this is the simpler of the two variants in the book.
    """

    def __init__(self, env, cfg, rng=None):
        super().__init__(env, cfg, rng)
        self.last_tried = np.zeros_like(self.q)
        self.step_count = 0

    def observe(self, obs, action, reward, next_obs, next_action, terminated, truncated):
        self.step_count += 1
        self.last_tried[int(obs), int(action)] = self.step_count
        super().observe(obs, action, reward, next_obs, next_action, terminated, truncated)

    def _planning_update(self, state, action, reward, next_state, terminated) -> None:
        elapsed = self.step_count - self.last_tried[state, action]
        bonus = self.cfg.dyna_kappa * np.sqrt(max(0.0, elapsed))
        super()._planning_update(state, action, reward + bonus, next_state, terminated)
