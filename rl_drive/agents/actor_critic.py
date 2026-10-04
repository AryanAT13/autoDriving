"""One-step actor-critic over the continuous encoder, with explicit gradients.

The policy and the value function are separate networks, each either linear or carrying
one tanh hidden layer (`hidden_units`). The gradients are written out rather than
autodiffed: they are short, and they are the part of the method worth being able to defend.

Two details decide whether this learns at all. Rewards are scaled before use, because at a
raw collision penalty of -75 the softmax saturates within a few hundred steps at any
learning rate. And the policy's entropy is regularised, without which it goes deterministic
early and can never discover the goal.
"""

from __future__ import annotations

import numpy as np


class _Network:
    """Linear map, or one tanh hidden layer when `hidden` > 0. Ascends its own gradient."""

    def __init__(self, n_in: int, hidden: int, n_out: int, rng):
        self.hidden = hidden
        if hidden:
            self.w1 = rng.normal(0.0, 1.0 / np.sqrt(n_in), (n_in, hidden))
            self.w2 = np.zeros((hidden, n_out))
        else:
            self.w1 = None
            self.w2 = np.zeros((n_in, n_out))

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = np.tanh(x @ self.w1) if self.hidden else x
        return h @ self.w2, h

    def ascend(self, x: np.ndarray, h: np.ndarray, grad_out: np.ndarray) -> None:
        """`grad_out` is d(objective)/d(output), already scaled by its learning rate."""
        if self.hidden:
            dh = self.w2 @ grad_out
            self.w2 += np.outer(h, grad_out)
            self.w1 += np.outer(x, dh * (1.0 - h * h))
        else:
            self.w2 += np.outer(h, grad_out)


class ActorCritic:
    obs_mode = "continuous"

    def __init__(self, env, cfg, rng=None):
        self.cfg = cfg
        self.rng = rng if rng is not None else np.random.default_rng()
        self.n_actions = env.action_space.n
        n_features = env.encoder.n + 1  # trailing bias term
        self.actor = _Network(n_features, cfg.hidden_units, self.n_actions, self.rng)
        self.critic = _Network(n_features, cfg.hidden_units, 1, self.rng)
        self.last_td = 0.0

    def _features(self, obs) -> np.ndarray:
        return np.append(np.asarray(obs, dtype=np.float64), 1.0)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        exp = np.exp(logits - logits.max())
        return exp / exp.sum()

    def act(self, obs, greedy: bool = False) -> int:
        logits, _ = self.actor.forward(self._features(obs))
        if greedy:
            return int(np.argmax(logits))
        return int(self.rng.choice(self.n_actions, p=self._softmax(logits)))

    def observe(self, obs, action, reward, next_obs, next_action, terminated, truncated):
        features = self._features(obs)
        value, h_critic = self.critic.forward(features)
        target = reward * self.cfg.reward_scale
        if not terminated:
            target += self.cfg.gamma * float(self.critic.forward(self._features(next_obs))[0][0])
        error = target - float(value[0])
        self.critic.ascend(features, h_critic, np.array([self.cfg.alpha_value * error]))

        logits, h_actor = self.actor.forward(features)
        probs = self._softmax(logits)
        score = -probs.copy()
        score[action] += 1.0  # d/dz log pi(a|s)
        log_probs = np.log(probs + 1e-12)
        entropy = -float(probs @ log_probs)
        entropy_grad = -probs * (log_probs + entropy)  # d/dz H(pi)
        self.actor.ascend(features, h_actor,
                          self.cfg.alpha_policy * error * score
                          + self.cfg.entropy_beta * entropy_grad)
        self.last_td = abs(error)

    def end_episode(self) -> None:
        pass

    def state_dict(self) -> dict:
        state = {"actor_w2": self.actor.w2, "critic_w2": self.critic.w2}
        if self.actor.hidden:
            state["actor_w1"] = self.actor.w1
            state["critic_w1"] = self.critic.w1
        return state

    def load_state_dict(self, data: dict) -> None:
        self.actor.w2 = np.asarray(data["actor_w2"])
        self.critic.w2 = np.asarray(data["critic_w2"])
        if "actor_w1" in data:
            self.actor.w1 = np.asarray(data["actor_w1"])
            self.critic.w1 = np.asarray(data["critic_w1"])
