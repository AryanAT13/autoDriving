"""The single training loop shared by every agent."""

from __future__ import annotations

from .evaluator import evaluate
from .rollout import run_episode


def train(env, agent, episodes: int, eval_env=None, eval_every: int = 0,
          eval_episodes: int = 20, logger=None, on_episode=None):
    """Run `episodes` of training, periodically scoring the greedy policy.

    `on_episode(stats, evaluation)` is the hook the web server streams from.
    """
    history, evaluations = [], []
    schedule = getattr(agent, "epsilon", None)

    for episode in range(episodes):
        if schedule is not None:
            schedule.update(episode)
        stats = run_episode(env, agent)
        stats.episode = episode
        stats.epsilon = schedule.value if schedule is not None else 0.0
        history.append(stats)
        if logger:
            logger.log_episode(stats)

        evaluation = None
        if eval_every and (episode + 1) % eval_every == 0:
            evaluation = evaluate(eval_env or env, agent, eval_episodes)
            evaluation["episode"] = episode
            evaluations.append(evaluation)
            if logger:
                logger.log_eval(episode, evaluation)
        if on_episode:
            on_episode(stats, evaluation)

    return history, evaluations
