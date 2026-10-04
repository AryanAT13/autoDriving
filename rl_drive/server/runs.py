"""Background training runs: start them, watch their metrics, stop them, replay them."""

from __future__ import annotations

import threading
import uuid
from dataclasses import asdict

import numpy as np

from ..agents import make_agent
from ..config import apply_overrides, load_configs
from ..env import DrivingEnv
from ..training import obs_mode_for, run_episode, train

MAX_RUNS = 20
BLOCK = 25  # episodes aggregated into one streamed point


class TrainingRun:
    """One training job on its own thread.

    Training is CPU-bound, so it cannot share the event loop. The thread owns the agent;
    readers take a snapshot of its weights rather than touching it directly.

    Per-episode results are aggregated into blocks of `BLOCK` before anyone sees them. A
    long run produces episodes far faster than a browser can plot them, and the chart only
    needs the trend. `blocks` and `evals` are append-only and read by slicing, both atomic
    under the GIL, so no lock is needed and the event loop never blocks on the trainer.
    """

    def __init__(self, algo: str, episodes: int, seed: int = 0, eval_every: int = 250,
                 eval_episodes: int = 20, overrides: dict | None = None):
        self.id = uuid.uuid4().hex[:8]
        self.algo = algo
        self.episodes = episodes
        self.seed = seed
        self.eval_every = eval_every
        self.eval_episodes = eval_episodes
        self.status = "pending"
        self.error: str | None = None
        self.completed = 0
        self.blocks: list[dict] = []
        self.evals: list[dict] = []

        self._window: list = []
        self._stop = threading.Event()

        self.env_cfg, self.reward_cfg, self.agent_cfg = load_configs(None, algo=algo)
        apply_overrides(self.agent_cfg, overrides or {})
        mode = obs_mode_for(algo)
        self._env = DrivingEnv(self.env_cfg, self.reward_cfg, mode)
        self._env.reset(seed=seed)
        self._eval_env = DrivingEnv(self.env_cfg, self.reward_cfg, mode)
        self._agent = make_agent(algo, self._env, self.agent_cfg, np.random.default_rng(seed))

    def start(self) -> "TrainingRun":
        self.status = "running"
        threading.Thread(target=self._train, daemon=True).start()
        return self

    def _train(self) -> None:
        try:
            train(self._env, self._agent, self.episodes, eval_env=self._eval_env,
                  eval_every=self.eval_every, eval_episodes=self.eval_episodes,
                  on_episode=self._record, should_stop=self._stop.is_set)
            self._flush()
            self.status = "stopped" if self._stop.is_set() else "done"
        except Exception as exc:  # surfaced to the client rather than lost in a thread
            self._flush()
            self.status, self.error = "error", f"{type(exc).__name__}: {exc}"

    def _flush(self) -> None:
        """Emit the trailing partial block so the last episodes are not dropped."""
        if self._window:
            self.blocks.append(_aggregate(self._window))
            self._window = []

    def _record(self, stats, evaluation) -> None:
        self._window.append(stats)
        self.completed += 1
        if len(self._window) >= BLOCK:
            self.blocks.append(_aggregate(self._window))
            self._window = []
        if evaluation:
            self.evals.append(evaluation)

    def stop(self) -> None:
        self._stop.set()

    def since(self, block_cursor: int, eval_cursor: int) -> tuple[list, list]:
        return self.blocks[block_cursor:], self.evals[eval_cursor:]

    def summary(self) -> dict:
        latest = self.evals[-1] if self.evals else None
        return {"id": self.id, "algo": self.algo, "status": self.status,
                "episode": self.completed, "episodes": self.episodes, "seed": self.seed,
                "latest_eval": latest, "error": self.error,
                "config": asdict(self.agent_cfg)}

    def rollout(self, seed: int | None = None) -> dict:
        """One greedy episode with the policy as it stands.

        The weights are copied first; while training continues that copy is a live preview
        and may straddle an update, which is fine for watching and never used for metrics.
        """
        weights = {key: np.array(value, copy=True)
                   for key, value in self._agent.state_dict().items()}
        env = DrivingEnv(self.env_cfg, self.reward_cfg, obs_mode_for(self.algo))
        agent = make_agent(self.algo, env, self.agent_cfg, np.random.default_rng(0))
        if weights:
            agent.load_state_dict(weights)
        frames: list[dict] = []
        stats = run_episode(env, agent, seed=seed, greedy=True, learn=False, frames=frames)
        return {"frames": frames, "stats": asdict(stats)}


def _aggregate(window: list) -> dict:
    """One streamed point: the block's mean, labelled with its last episode."""
    size = len(window)
    return {
        "episode": window[-1].episode,
        "reward": sum(s.reward for s in window) / size,
        "collision_rate": sum(s.collision for s in window) / size,
        "mean_speed": sum(s.mean_speed for s in window) / size,
        "epsilon": window[-1].epsilon,
    }


class RunManager:
    """Keeps the most recent runs addressable; older finished ones are discarded."""

    def __init__(self, limit: int = MAX_RUNS):
        self.limit = limit
        self._runs: dict[str, TrainingRun] = {}
        self._lock = threading.Lock()

    def start(self, **kwargs) -> TrainingRun:
        run = TrainingRun(**kwargs)
        with self._lock:
            self._runs[run.id] = run
            self._prune()
        return run.start()

    def _prune(self) -> None:
        finished = [r for r in self._runs.values() if r.status not in ("running", "pending")]
        while len(self._runs) > self.limit and finished:
            self._runs.pop(finished.pop(0).id, None)

    def get(self, run_id: str) -> TrainingRun | None:
        return self._runs.get(run_id)

    def list(self) -> list[dict]:
        return [run.summary() for run in reversed(list(self._runs.values()))]

    def stop_all(self) -> None:
        for run in self._runs.values():
            run.stop()
