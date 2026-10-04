import numpy as np
import pytest

from rl_drive.agents import make_agent
from rl_drive.agents.actor_critic import ActorCritic
from rl_drive.config import AgentConfig
from rl_drive.env import DrivingEnv
from rl_drive.training import run_episode

EPS = 1e-6


def build(hidden=32, **kwargs):
    env = DrivingEnv(obs_mode="continuous")
    env.reset(seed=0)
    cfg = AgentConfig(hidden_units=hidden, **kwargs)
    return env, make_agent("actor_critic", env, cfg, np.random.default_rng(0))


def randomise(agent, seed=2):
    rng = np.random.default_rng(seed)
    for net in (agent.actor, agent.critic):
        net.w2 = rng.normal(size=net.w2.shape) * 0.5
        if net.hidden:
            net.w1 = rng.normal(size=net.w1.shape) * 0.5
    return np.append(rng.random(15), 1.0)


def finite_difference(agent, net, objective):
    """Numerical gradient of `objective` with respect to every weight in `net`."""
    grads = {}
    for name in ("w1", "w2"):
        weights = getattr(net, name)
        if weights is None:
            continue
        numeric = np.zeros_like(weights)
        for index in np.ndindex(weights.shape):
            weights[index] += EPS
            up = objective()
            weights[index] -= 2 * EPS
            down = objective()
            weights[index] += EPS
            numeric[index] = (up - down) / (2 * EPS)
        grads[name] = numeric
    return grads


def analytic(net, x, grad_out):
    before = {"w1": None if net.w1 is None else net.w1.copy(), "w2": net.w2.copy()}
    _, h = net.forward(x)
    net.ascend(x, h, grad_out)
    out = {"w2": net.w2 - before["w2"]}
    if net.w1 is not None:
        out["w1"] = net.w1 - before["w1"]
        net.w1 = before["w1"]
    net.w2 = before["w2"]
    return out


@pytest.mark.parametrize("hidden", [0, 8])
def test_score_function_gradient_matches_finite_difference(hidden):
    """d/dtheta log pi(a|s), the core of the policy gradient."""
    _, agent = build(hidden=hidden)
    x, action = randomise(agent), 2

    def objective():
        logits, _ = agent.actor.forward(x)
        return float(np.log(agent._softmax(logits)[action]))

    logits, _ = agent.actor.forward(x)
    probs = agent._softmax(logits)
    score = -probs.copy()
    score[action] += 1.0

    numeric = finite_difference(agent, agent.actor, objective)
    exact = analytic(agent.actor, x, score)
    for name in numeric:
        assert np.allclose(exact[name], numeric[name], atol=1e-7), name


@pytest.mark.parametrize("hidden", [0, 8])
def test_entropy_gradient_matches_finite_difference(hidden):
    """Without this term the policy goes deterministic and stops exploring."""
    _, agent = build(hidden=hidden)
    x = randomise(agent)

    def objective():
        logits, _ = agent.actor.forward(x)
        probs = agent._softmax(logits)
        return -float(probs @ np.log(probs + 1e-12))

    logits, _ = agent.actor.forward(x)
    probs = agent._softmax(logits)
    log_probs = np.log(probs + 1e-12)
    entropy = -float(probs @ log_probs)
    grad = -probs * (log_probs + entropy)

    numeric = finite_difference(agent, agent.actor, objective)
    exact = analytic(agent.actor, x, grad)
    for name in numeric:
        assert np.allclose(exact[name], numeric[name], atol=1e-7), name


@pytest.mark.parametrize("hidden", [0, 32])
def test_policy_is_a_probability_distribution(hidden):
    env, agent = build(hidden=hidden)
    randomise(agent)
    obs, _ = env.reset(seed=1)
    logits, _ = agent.actor.forward(agent._features(obs))
    probs = agent._softmax(logits)
    assert probs.shape == (env.action_space.n,)
    assert np.all(probs > 0.0)
    assert probs.sum() == pytest.approx(1.0)


def test_greedy_action_is_deterministic_argmax():
    env, agent = build()
    randomise(agent)
    obs, _ = env.reset(seed=1)
    logits, _ = agent.actor.forward(agent._features(obs))
    expected = int(np.argmax(logits))
    assert {agent.act(obs, greedy=True) for _ in range(20)} == {expected}


def test_terminated_does_not_bootstrap_but_truncated_does():
    _, terminal = build(hidden=0, gamma=0.9, alpha_value=1.0, alpha_policy=0.0,
                        entropy_beta=0.0, reward_scale=1.0)
    _, cut_off = build(hidden=0, gamma=0.9, alpha_value=1.0, alpha_policy=0.0,
                       entropy_beta=0.0, reward_scale=1.0)
    for agent in (terminal, cut_off):
        agent.critic.w2 = np.full_like(agent.critic.w2, 0.5)

    obs = np.zeros(15, dtype=np.float32)
    next_obs = np.ones(15, dtype=np.float32)
    terminal.observe(obs, 0, 1.0, next_obs, 0, True, False)
    cut_off.observe(obs, 0, 1.0, next_obs, 0, False, True)
    assert not np.allclose(terminal.critic.w2, cut_off.critic.w2)


def test_observe_updates_both_networks():
    env, agent = build()
    before = (agent.actor.w1.copy(), agent.critic.w1.copy(), agent.critic.w2.copy())
    run_episode(env, agent)
    assert not np.array_equal(agent.actor.w1, before[0])
    assert not np.array_equal(agent.critic.w1, before[1])
    assert not np.array_equal(agent.critic.w2, before[2])


@pytest.mark.parametrize("hidden", [0, 32])
def test_checkpoint_roundtrip(hidden):
    env, agent = build(hidden=hidden)
    run_episode(env, agent)
    _, restored = build(hidden=hidden)
    restored.load_state_dict(agent.state_dict())
    assert np.array_equal(agent.actor.w2, restored.actor.w2)
    assert np.array_equal(agent.critic.w2, restored.critic.w2)
    if hidden:
        assert np.array_equal(agent.actor.w1, restored.actor.w1)


def test_uses_the_continuous_encoder():
    assert ActorCritic.obs_mode == "continuous"
