import numpy as np
import pytest

from rl_drive.agents import make_agent
from rl_drive.config import AgentConfig
from rl_drive.env import DrivingEnv
from rl_drive.training import evaluate, train


@pytest.fixture
def env():
    e = DrivingEnv()
    e.reset(seed=0)
    return e


def test_zero_planning_is_exactly_q_learning():
    """The sample-efficiency comparison is only controlled if this holds."""
    tables = []
    for algo in ("q_learning", "dyna_q"):
        env = DrivingEnv()
        env.reset(seed=0)
        agent = make_agent(algo, env, AgentConfig(planning_steps=0), np.random.default_rng(0))
        train(env, agent, episodes=200)
        tables.append(agent.q)
    assert np.array_equal(tables[0], tables[1])


def test_model_records_transitions(env):
    agent = make_agent("dyna_q", env, AgentConfig(planning_steps=0), np.random.default_rng(0))
    agent.observe(3, 1, 2.0, 7, 0, False, False)
    assert agent.keys == [(3, 1)]
    assert list(agent.model[(3, 1)]) == [(2.0, 7, False)]


def test_sampled_model_keeps_the_most_recent_transitions(env):
    agent = make_agent("dyna_q", env, AgentConfig(planning_steps=0, model_capacity=3),
                       np.random.default_rng(0))
    for next_state in range(10):
        agent.observe(3, 1, 1.0, next_state, 0, False, False)
    bucket = agent.model[(3, 1)]
    assert len(bucket) == 3
    assert [t[1] for t in bucket] == [7, 8, 9]


def test_deterministic_model_keeps_only_the_last_transition(env):
    """model_capacity = 1 is the textbook Dyna-Q model."""
    agent = make_agent("dyna_q", env, AgentConfig(planning_steps=0, model_capacity=1),
                       np.random.default_rng(0))
    agent.observe(3, 1, 1.0, 5, 0, False, False)
    agent.observe(3, 1, 9.0, 6, 0, False, False)
    assert list(agent.model[(3, 1)]) == [(9.0, 6, False)]


def test_planning_updates_q_without_new_experience(env):
    cfg = AgentConfig(alpha=0.5, gamma=0.9, planning_steps=0)
    agent = make_agent("dyna_q", env, cfg, np.random.default_rng(0))
    agent.observe(1, 0, 10.0, 2, 0, False, False)
    before = agent.q[1, 0]
    agent.cfg.planning_steps = 50
    agent._plan()
    assert agent.q[1, 0] > before


def test_planning_is_a_no_op_before_any_experience(env):
    agent = make_agent("dyna_q", env, AgentConfig(planning_steps=20), np.random.default_rng(0))
    agent._plan()
    assert np.all(agent.q == 0.0)


def test_dyna_q_plus_bonus_grows_with_staleness(env):
    cfg = AgentConfig(alpha=1.0, gamma=0.0, planning_steps=0, dyna_kappa=1.0)
    agent = make_agent("dyna_q_plus", env, cfg, np.random.default_rng(0))

    agent.step_count, agent.last_tried[1, 0] = 0, 0
    agent._planning_update(1, 0, 0.0, 2, True)
    assert agent.q[1, 0] == pytest.approx(0.0)

    agent.step_count = 100
    agent._planning_update(1, 0, 0.0, 2, True)
    assert agent.q[1, 0] == pytest.approx(10.0)


@pytest.mark.parametrize("algo", ["dyna_q", "dyna_q_plus"])
def test_planning_agents_learn(algo, env):
    eval_env = DrivingEnv()
    cfg = AgentConfig(epsilon_decay_episodes=1500, planning_steps=5)
    agent = make_agent(algo, env, cfg, np.random.default_rng(0))
    train(env, agent, episodes=2500)
    metrics = evaluate(eval_env, agent, episodes=40)
    assert metrics["reward"] > -40.0
    assert metrics["collision_rate"] < 0.6
