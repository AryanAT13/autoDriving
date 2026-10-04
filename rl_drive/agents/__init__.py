from .actor_critic import ActorCritic
from .base import Agent, EpsilonSchedule, TabularPolicy
from .baselines import AlwaysAccelerate, RandomAgent, ScriptedSafeDriver
from .dyna_q import DynaQ, DynaQPlus
from .monte_carlo import MonteCarloControl
from .tabular import ExpectedSarsa, QLearning, Sarsa

REGISTRY = {
    "random": RandomAgent,
    "always_accelerate": AlwaysAccelerate,
    "scripted": ScriptedSafeDriver,
    "q_learning": QLearning,
    "sarsa": Sarsa,
    "expected_sarsa": ExpectedSarsa,
    "monte_carlo": MonteCarloControl,
    "dyna_q": DynaQ,
    "dyna_q_plus": DynaQPlus,
    "actor_critic": ActorCritic,
}

BASELINES = ("random", "always_accelerate", "scripted")
LEARNERS = ("q_learning", "sarsa", "expected_sarsa", "monte_carlo",
            "dyna_q", "dyna_q_plus", "actor_critic")


def make_agent(name: str, env, cfg=None, rng=None):
    if name not in REGISTRY:
        raise KeyError(f"unknown agent {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name](env, cfg, rng)


__all__ = ["Agent", "EpsilonSchedule", "TabularPolicy", "REGISTRY", "BASELINES",
           "LEARNERS", "make_agent"]
