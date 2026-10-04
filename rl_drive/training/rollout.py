"""A single episode. The one place the agent interface meets the environment."""

from __future__ import annotations

import numpy as np

from .metrics import EpisodeTally


def run_episode(env, agent, seed=None, greedy: bool = False, learn: bool = True,
                frames: list | None = None):
    obs, _ = env.reset(seed=seed)
    # Choosing the next action before updating is what lets SARSA share this loop.
    action = agent.act(obs, greedy=greedy)
    tally = EpisodeTally()
    errors = []
    if frames is not None:
        frames.append(env.frame())

    while True:
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_action = agent.act(next_obs, greedy=greedy)
        if learn:
            agent.observe(obs, action, reward, next_obs, next_action, terminated, truncated)
            errors.append(getattr(agent, "last_td", 0.0))
        if frames is not None:
            frames.append(env.frame())

        stats = tally.add(reward, info)

        obs, action = next_obs, next_action
        if terminated or truncated:
            break

    if learn:
        agent.end_episode()
    stats.mean_td = float(np.mean(errors)) if errors else 0.0
    return stats
