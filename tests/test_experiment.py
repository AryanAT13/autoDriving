import numpy as np
import pytest

from rl_drive.config import AgentConfig, EnvConfig
from rl_drive.training import (confidence_interval, curve_band, evaluate_seeds,
                               obs_mode_for, run_seeds, summarise_seeds)


def test_confidence_interval_matches_the_normal_approximation():
    values = [10.0, 12.0, 14.0, 11.0, 13.0]
    expected = 1.96 * np.std(values, ddof=1) / np.sqrt(len(values))
    assert confidence_interval(values) == pytest.approx(expected)


def test_confidence_interval_degenerate_cases():
    assert confidence_interval([3.0]) == 0.0
    assert confidence_interval([]) == 0.0
    assert confidence_interval([5.0, 5.0, 5.0]) == pytest.approx(0.0)


def test_obs_mode_lookup():
    assert obs_mode_for("q_learning") == "discrete"
    assert obs_mode_for("actor_critic") == "continuous"


def test_run_seeds_shapes_and_reproducibility():
    kwargs = dict(episodes=60, seeds=(0, 1), env_cfg=EnvConfig(),
                  agent_cfg=AgentConfig(epsilon_decay_episodes=40),
                  eval_every=30, eval_episodes=3, final_episodes=5)
    result = run_seeds("q_learning", **kwargs)
    assert len(result["curves"]) == 2
    assert len(result["finals"]) == 2
    assert len(result["training_collisions"]) == 2
    assert [point["episode"] for point in result["curves"][0]] == [29, 59]

    repeat = run_seeds("q_learning", **kwargs)
    assert [f["reward"] for f in repeat["finals"]] == [f["reward"] for f in result["finals"]]


def test_run_seeds_reports_progress():
    seen = []
    run_seeds("q_learning", episodes=20, seeds=(0,), agent_cfg=AgentConfig(),
              final_episodes=3, on_seed=lambda a, s, m: seen.append((a, s)))
    assert seen == [("q_learning", 0)]


def test_curve_band_dimensions():
    result = run_seeds("q_learning", episodes=60, seeds=(0, 1, 2),
                       agent_cfg=AgentConfig(), eval_every=20, eval_episodes=3,
                       final_episodes=3)
    episodes, mean, band = curve_band(result, "reward")
    assert episodes.shape == mean.shape == band.shape == (3,)
    assert np.all(band >= 0.0)


def test_curve_band_is_empty_without_evaluations():
    result = evaluate_seeds("random", seeds=(0, 1), episodes=3)
    episodes, mean, band = curve_band(result, "reward")
    assert episodes.size == 0 and mean.size == 0 and band.size == 0


def test_evaluate_seeds_matches_run_seeds_shape():
    result = evaluate_seeds("scripted", seeds=(0, 1, 2), episodes=5)
    assert len(result["finals"]) == 3
    assert result["curves"] == []
    summary = summarise_seeds(result)
    assert summary["mean"] > 0.0
    assert len(summary["values"]) == 3


def test_summarise_seeds_picks_the_requested_metric():
    result = evaluate_seeds("scripted", seeds=(0, 1), episodes=5)
    assert summarise_seeds(result, "collision_rate")["mean"] == pytest.approx(0.0)


def test_curve_persistence_round_trip(tmp_path):
    """The replot path must rebuild exactly what was trained, without retraining."""
    from experiments.compare_algorithms import load_curves, save_curves

    original = {"q_learning": run_seeds("q_learning", episodes=40, seeds=(0, 1),
                                        agent_cfg=AgentConfig(), eval_every=20,
                                        eval_episodes=3, final_episodes=3)}
    path = tmp_path / "curves.csv"
    save_curves(original, path)
    restored = load_curves(path)

    assert set(restored) == {"q_learning"}
    assert restored["q_learning"]["seeds"] == [0, 1]
    before = curve_band(original["q_learning"], "reward")
    after = curve_band(restored["q_learning"], "reward")
    for lhs, rhs in zip(before, after):
        assert np.allclose(lhs, rhs)


def test_save_curves_skips_agents_without_curves(tmp_path):
    from experiments.compare_algorithms import save_curves

    path = tmp_path / "curves.csv"
    save_curves({"random": evaluate_seeds("random", seeds=(0,), episodes=2)}, path)
    assert not path.exists()
