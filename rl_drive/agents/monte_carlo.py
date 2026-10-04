"""First-visit Monte Carlo control: no bootstrapping, updates only once an episode ends.

A truncated episode is treated as complete, which biases its return slightly low.
"""

from __future__ import annotations

import numpy as np

from .base import TabularPolicy


class MonteCarloControl(TabularPolicy):
    def __init__(self, env, cfg, rng=None):
        super().__init__(env, cfg, rng)
        self.counts = np.zeros_like(self.q)
        self.trajectory: list[tuple[int, int, float]] = []

    def observe(self, obs, action, reward, next_obs, next_action, terminated, truncated):
        self.trajectory.append((int(obs), int(action), float(reward)))

    def end_episode(self) -> None:
        if not self.trajectory:
            return
        returns = np.empty(len(self.trajectory))
        total = 0.0
        for t in range(len(self.trajectory) - 1, -1, -1):
            total = self.trajectory[t][2] + self.cfg.gamma * total
            returns[t] = total

        first_visit: dict[tuple[int, int], int] = {}
        for t, (state, action, _) in enumerate(self.trajectory):
            first_visit.setdefault((state, action), t)

        errors = []
        for (state, action), t in first_visit.items():
            self.counts[state, action] += 1
            error = returns[t] - self.q[state, action]
            self.q[state, action] += error / self.counts[state, action]
            errors.append(abs(error))
        self.last_td = float(np.mean(errors))
        self.trajectory.clear()

    def state_dict(self) -> dict:
        return {"q": self.q, "counts": self.counts}

    def load_state_dict(self, data: dict) -> None:
        self.q = np.asarray(data["q"])
        self.counts = np.asarray(data["counts"])
