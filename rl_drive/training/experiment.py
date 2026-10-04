"""Multi-seed runs and the aggregation used for reporting.

A single-seed RL curve is not evidence, so every reported number here is a mean over
seeds with a confidence interval.
"""

from __future__ import annotations

import numpy as np

from ..agents import REGISTRY, make_agent
from ..env import DrivingEnv
from .evaluator import evaluate
from .trainer import train

_Z = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}


def confidence_interval(values, confidence: float = 0.95) -> float:
    """Half-width of the interval about the mean, using the normal approximation."""
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return 0.0
    return float(_Z[confidence] * values.std(ddof=1) / np.sqrt(values.size))


def obs_mode_for(algo: str) -> str:
    return getattr(REGISTRY[algo], "obs_mode", "discrete")


def run_seeds(algo, episodes, seeds, env_cfg=None, reward_cfg=None, agent_cfg=None,
              eval_every=0, eval_episodes=30, final_episodes=200, on_seed=None) -> dict:
    """Train `algo` once per seed, returning each run's eval curve and final metrics."""
    mode = obs_mode_for(algo)
    curves, finals, training = [], [], []
    for seed in seeds:
        env = DrivingEnv(env_cfg, reward_cfg, mode)
        env.reset(seed=seed)
        eval_env = DrivingEnv(env_cfg, reward_cfg, mode)
        agent = make_agent(algo, env, agent_cfg, np.random.default_rng(seed))
        history, evals = train(env, agent, episodes, eval_env=eval_env,
                               eval_every=eval_every, eval_episodes=eval_episodes)
        curves.append(evals)
        finals.append(evaluate(eval_env, agent, episodes=final_episodes))
        training.append(float(np.mean([h.collision for h in history])))
        if on_seed:
            on_seed(algo, seed, finals[-1])
    return {"algo": algo, "seeds": list(seeds), "curves": curves,
            "finals": finals, "training_collisions": training}


def evaluate_seeds(algo, seeds, env_cfg=None, reward_cfg=None, episodes=200) -> dict:
    """Score a non-learning baseline across seeds, matching `run_seeds` output."""
    env = DrivingEnv(env_cfg, reward_cfg, obs_mode_for(algo))
    finals = [evaluate(env, make_agent(algo, env, None, np.random.default_rng(seed)),
                       episodes=episodes) for seed in seeds]
    return {"algo": algo, "seeds": list(seeds), "curves": [],
            "finals": finals, "training_collisions": []}


def summarise_seeds(result: dict, metric: str = "reward") -> dict:
    values = [final[metric] for final in result["finals"]]
    return {"mean": float(np.mean(values)), "ci": confidence_interval(values),
            "values": values}


def curve_band(result: dict, metric: str = "reward"):
    """Episode axis, per-seed mean and CI half-width at each evaluation point."""
    if not result["curves"]:
        return np.array([]), np.array([]), np.array([])
    episodes = np.array([point["episode"] for point in result["curves"][0]])
    matrix = np.array([[point[metric] for point in curve] for curve in result["curves"]])
    band = np.array([confidence_interval(column) for column in matrix.T])
    return episodes, matrix.mean(axis=0), band
