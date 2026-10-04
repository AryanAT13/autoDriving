"""Exploration-free evaluation on a fixed held-out seed set."""

from __future__ import annotations

from .metrics import summarise
from .rollout import run_episode

EVAL_SEED_BASE = 10_000


def eval_seeds(n: int, base: int = EVAL_SEED_BASE) -> list[int]:
    return [base + i for i in range(n)]


def evaluate(env, agent, episodes: int = 20, seeds=None) -> dict:
    seeds = seeds if seeds is not None else eval_seeds(episodes)
    stats = [run_episode(env, agent, seed=s, greedy=True, learn=False) for s in seeds]
    return summarise(stats)
