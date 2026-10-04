import csv
import json

import numpy as np
import pytest

from rl_drive.agents import make_agent
from rl_drive.cli import main
from rl_drive.config import AgentConfig, dump_configs, load_configs
from rl_drive.env import DrivingEnv
from rl_drive.training import (RunLogger, eval_seeds, evaluate, moving_average,
                               run_episode, summarise, train)


@pytest.fixture
def env():
    e = DrivingEnv()
    e.reset(seed=0)
    return e


def test_run_episode_reports_consistent_stats(env):
    agent = make_agent("scripted", env, None, np.random.default_rng(0))
    stats = run_episode(env, agent, seed=11, greedy=True, learn=False)
    assert stats.steps > 0
    assert 0.0 <= stats.mean_speed <= env.cfg.max_speed_level
    assert not (stats.collision and stats.goal)


def test_rollout_records_frames(env):
    agent = make_agent("scripted", env, None, np.random.default_rng(0))
    frames = []
    stats = run_episode(env, agent, seed=11, greedy=True, learn=False, frames=frames)
    assert len(frames) == stats.steps + 1
    json.dumps(frames[0])


def test_evaluation_is_reproducible(env):
    agent = make_agent("scripted", env, None, np.random.default_rng(0))
    assert evaluate(env, agent, episodes=10) == evaluate(env, agent, episodes=10)


def test_evaluation_does_not_train(env):
    agent = make_agent("q_learning", env, AgentConfig(), np.random.default_rng(0))
    train(env, agent, episodes=20)
    before = agent.q.copy()
    evaluate(env, agent, episodes=10)
    assert np.array_equal(before, agent.q)


def test_eval_seeds_are_held_out():
    """Evaluation seeds must not overlap the seeds training happens to visit."""
    assert eval_seeds(5) == [10_000, 10_001, 10_002, 10_003, 10_004]


def test_train_decays_epsilon_and_calls_back(env):
    agent = make_agent("q_learning", env, AgentConfig(epsilon_decay_episodes=50),
                       np.random.default_rng(0))
    seen = []
    history, evaluations = train(env, agent, episodes=20, eval_every=10, eval_episodes=3,
                                 on_episode=lambda s, e: seen.append((s.episode, e)))
    assert len(history) == 20 and len(seen) == 20
    assert len(evaluations) == 2
    assert [e["episode"] for e in evaluations] == [9, 19]
    assert history[0].epsilon > history[-1].epsilon


def test_train_without_eval_produces_no_evaluations(env):
    agent = make_agent("sarsa", env, AgentConfig(), np.random.default_rng(0))
    history, evaluations = train(env, agent, episodes=5)
    assert len(history) == 5 and evaluations == []


def test_summarise_and_moving_average():
    assert summarise([]) == {}
    stats = [run_episode(DrivingEnv(), make_agent("scripted", DrivingEnv(), None,
                                                  np.random.default_rng(0)),
                         seed=s, greedy=True, learn=False) for s in (1, 2, 3)]
    metrics = summarise(stats)
    assert metrics["episodes"] == 3
    assert 0.0 <= metrics["collision_rate"] <= 1.0
    assert np.allclose(moving_average([1, 2, 3, 4], window=2), [1.5, 2.5, 3.5])


def test_run_logger_writes_expected_files(env, tmp_path):
    agent = make_agent("q_learning", env, AgentConfig(), np.random.default_rng(0))
    logger = RunLogger(tmp_path, "unit")
    logger.save_config(dump_configs(*load_configs(None)))
    train(env, agent, episodes=6, eval_every=3, eval_episodes=2, logger=logger)
    logger.save_checkpoint(agent, {"algo": "q_learning"})
    logger.close()

    assert (logger.path / "config.json").exists()
    assert (logger.path / "policy.npz").exists()
    assert json.loads((logger.path / "meta.json").read_text())["algo"] == "q_learning"
    with open(logger.path / "train.csv") as handle:
        assert len(list(csv.DictReader(handle))) == 6
    with open(logger.path / "eval.csv") as handle:
        assert len(list(csv.DictReader(handle))) == 2


def test_cli_train_eval_and_policy(tmp_path, capsys):
    main(["train", "--algo", "q_learning", "--episodes", "40", "--eval-every", "20",
          "--eval-episodes", "2", "--final-episodes", "5", "--out", str(tmp_path),
          "--tag", "clitest"])
    run_dir = next(tmp_path.glob("clitest_*"))

    main(["eval", "--run", str(run_dir), "--episodes", "5"])
    main(["policy", "--run", str(run_dir)])
    output = capsys.readouterr().out
    assert "collision rate" in output and "greedy policy" in output


def test_cli_baseline(capsys):
    main(["baseline", "--algo", "scripted", "--episodes", "5"])
    assert "goal rate" in capsys.readouterr().out
