"""Temporal-difference control. The three variants differ only in the bootstrap target."""

from __future__ import annotations

import numpy as np

from .base import TabularPolicy


class TabularTD(TabularPolicy):
    def __init__(self, env, cfg, rng=None):
        super().__init__(env, cfg, rng)
        if cfg.optimistic_init:
            self.q.fill(cfg.optimistic_init)

    def observe(self, obs, action, reward, next_obs, next_action, terminated, truncated):
        # Bootstrap on truncation but not on termination: a time limit is not an absorbing state.
        target = reward
        if not terminated:
            target += self.cfg.gamma * self._bootstrap(next_obs, next_action)
        error = target - self.q[obs, action]
        self.q[obs, action] += self.cfg.alpha * error
        self.last_td = abs(error)

    def _bootstrap(self, next_obs, next_action) -> float:
        raise NotImplementedError


class QLearning(TabularTD):
    """Off-policy: bootstraps from the best next action regardless of what it will do."""

    def _bootstrap(self, next_obs, next_action) -> float:
        return float(self.q[next_obs].max())


class Sarsa(TabularTD):
    """On-policy: bootstraps from the action actually taken, so exploration risk is priced in."""

    def _bootstrap(self, next_obs, next_action) -> float:
        return float(self.q[next_obs, next_action])


class ExpectedSarsa(TabularTD):
    """On-policy, but averages over the policy instead of sampling it, cutting variance."""

    def _bootstrap(self, next_obs, next_action) -> float:
        values = self.q[next_obs]
        epsilon = self.epsilon.value
        probs = np.full(self.n_actions, epsilon / self.n_actions)
        best = np.flatnonzero(values == values.max())
        probs[best] += (1.0 - epsilon) / len(best)
        return float(probs @ values)
