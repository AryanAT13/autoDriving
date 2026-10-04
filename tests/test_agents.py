import numpy as np
import pytest

from rl_drive.agents import LEARNERS, REGISTRY, make_agent
from rl_drive.agents.base import EpsilonSchedule
from rl_drive.agents.tabular import ExpectedSarsa, QLearning, Sarsa
from rl_drive.config import AgentConfig
from rl_drive.env import DrivingEnv
from rl_drive.training import (RunLogger, evaluate, load_checkpoint, obs_mode_for,
                               run_episode, train)

GREEDY = AgentConfig(alpha=1.0, gamma=0.9, epsilon_start=0.0, epsilon_end=0.0)


@pytest.fixture
def env():
    e = DrivingEnv()
    e.reset(seed=0)
    return e


def test_epsilon_schedule_decays_to_end():
    schedule = EpsilonSchedule(1.0, 0.05, 1000)
    assert schedule.update(0) == pytest.approx(1.0)
    assert schedule.update(1000) == pytest.approx(0.05)
    assert schedule.update(5000) == pytest.approx(0.05)
    assert schedule.update(500) > 0.05


def test_greedy_action_is_argmax(env):
    agent = QLearning(env, GREEDY, np.random.default_rng(0))
    agent.q[7] = [0.0, 0.0, 9.0, 0.0, 0.0]
    assert agent.act(7, greedy=True) == 2


def test_terminated_does_not_bootstrap_but_truncated_does(env):
    """A time limit is not an absorbing state; bootstrapping must survive truncation."""
    terminal = QLearning(env, GREEDY, np.random.default_rng(0))
    terminal.q[5] = 10.0
    terminal.observe(0, 0, 1.0, 5, 0, terminated=True, truncated=False)
    assert terminal.q[0, 0] == pytest.approx(1.0)

    cut_off = QLearning(env, GREEDY, np.random.default_rng(0))
    cut_off.q[5] = 10.0
    cut_off.observe(0, 0, 1.0, 5, 0, terminated=False, truncated=True)
    assert cut_off.q[0, 0] == pytest.approx(1.0 + 0.9 * 10.0)


def test_td_variants_use_different_targets(env):
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    q = QLearning(env, GREEDY, np.random.default_rng(0))
    q.q[5] = values
    q.observe(0, 0, 0.0, 5, 1, False, False)
    assert q.q[0, 0] == pytest.approx(0.9 * values.max())

    sarsa = Sarsa(env, GREEDY, np.random.default_rng(0))
    sarsa.q[5] = values
    sarsa.observe(0, 0, 0.0, 5, 1, False, False)
    assert sarsa.q[0, 0] == pytest.approx(0.9 * values[1])

    cfg = AgentConfig(alpha=1.0, gamma=0.9, epsilon_start=0.5, epsilon_end=0.5)
    expected = ExpectedSarsa(env, cfg, np.random.default_rng(0))
    expected.q[5] = values
    expected.observe(0, 0, 0.0, 5, 1, False, False)
    target = 0.5 / 5 * values.sum() + 0.5 * values.max()
    assert expected.q[0, 0] == pytest.approx(0.9 * target)


def test_monte_carlo_updates_only_at_episode_end(env):
    agent = make_agent("monte_carlo", env, AgentConfig(gamma=1.0), np.random.default_rng(0))
    agent.observe(3, 1, 2.0, 4, 0, False, False)
    agent.observe(4, 0, 5.0, 5, 0, True, False)
    assert np.all(agent.q == 0.0)

    agent.end_episode()
    assert agent.q[3, 1] == pytest.approx(7.0)
    assert agent.q[4, 0] == pytest.approx(5.0)
    assert agent.trajectory == []


def test_monte_carlo_first_visit_uses_first_occurrence(env):
    agent = make_agent("monte_carlo", env, AgentConfig(gamma=1.0), np.random.default_rng(0))
    for reward in (1.0, 1.0, 1.0):
        agent.observe(3, 1, reward, 3, 1, False, False)
    agent.end_episode()
    assert agent.q[3, 1] == pytest.approx(3.0)
    assert agent.counts[3, 1] == 1


def _agent_env(name):
    """Each agent must be given the encoder its obs_mode declares."""
    env = DrivingEnv(obs_mode=obs_mode_for(name))
    env.reset(seed=0)
    return env


@pytest.mark.parametrize("name", LEARNERS)
def test_learner_runs_an_episode_and_updates(name):
    env = _agent_env(name)
    agent = make_agent(name, env, AgentConfig(), np.random.default_rng(0))
    before = {key: value.copy() for key, value in agent.state_dict().items()}
    stats = run_episode(env, agent)
    assert stats.steps > 0
    assert any(not np.array_equal(before[key], value)
               for key, value in agent.state_dict().items())


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_agent_exposes_the_interface(name):
    env = _agent_env(name)
    agent = make_agent(name, env, AgentConfig(), np.random.default_rng(0))
    assert agent.obs_mode in ("discrete", "continuous")
    for method in ("act", "observe", "end_episode", "state_dict", "load_state_dict"):
        assert callable(getattr(agent, method))


def test_scripted_driver_solves_the_environment(env):
    """Phase 1 gate: if the hand-written policy cannot drive safely, nothing can learn."""
    agent = make_agent("scripted", env, None, np.random.default_rng(0))
    metrics = evaluate(env, agent, episodes=40)
    assert metrics["collision_rate"] == 0.0
    assert metrics["goal_rate"] == 1.0
    assert metrics["reward"] > 30.0


def test_random_baseline_is_poor(env):
    agent = make_agent("random", env, None, np.random.default_rng(0))
    metrics = evaluate(env, agent, episodes=40)
    assert metrics["collision_rate"] > 0.8


def test_q_learning_improves_over_random(env):
    eval_env = DrivingEnv()
    cfg = AgentConfig(epsilon_decay_episodes=2000)
    agent = make_agent("q_learning", env, cfg, np.random.default_rng(0))
    train(env, agent, episodes=3000)

    trained = evaluate(eval_env, agent, episodes=40)
    random_agent = make_agent("random", eval_env, None, np.random.default_rng(0))
    baseline = evaluate(eval_env, random_agent, episodes=40)

    assert trained["reward"] > baseline["reward"] + 50.0
    assert trained["collision_rate"] < 0.4
    assert trained["goal_rate"] > 0.5


def test_checkpoint_roundtrip(env, tmp_path):
    agent = make_agent("q_learning", env, AgentConfig(), np.random.default_rng(0))
    train(env, agent, episodes=50)
    logger = RunLogger(tmp_path, "test")
    logger.save_checkpoint(agent, {"algo": "q_learning"})
    logger.close()

    restored = make_agent("q_learning", env, AgentConfig(), np.random.default_rng(0))
    restored.load_state_dict(load_checkpoint(logger.path))
    assert np.array_equal(agent.q, restored.q)
