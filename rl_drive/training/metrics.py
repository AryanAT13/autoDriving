"""Per-episode statistics, aggregation and run persistence."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

import numpy as np


@dataclass
class EpisodeStats:
    episode: int = 0
    steps: int = 0
    reward: float = 0.0
    collision: bool = False
    goal: bool = False
    red_lights: int = 0
    off_road: int = 0
    lane_changes: int = 0
    mean_speed: float = 0.0
    epsilon: float = 0.0
    mean_td: float = 0.0


def summarise(stats: list[EpisodeStats]) -> dict:
    """Collapse a batch of episodes into the scalar metrics we report."""
    if not stats:
        return {}
    rewards = np.array([s.reward for s in stats])
    return {
        "episodes": len(stats),
        "reward": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "collision_rate": float(np.mean([s.collision for s in stats])),
        "goal_rate": float(np.mean([s.goal for s in stats])),
        "red_lights": float(np.mean([s.red_lights for s in stats])),
        "lane_changes": float(np.mean([s.lane_changes for s in stats])),
        "mean_speed": float(np.mean([s.mean_speed for s in stats])),
        "steps": float(np.mean([s.steps for s in stats])),
    }


def moving_average(values, window: int = 100) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        window = max(1, len(values))
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


class RunLogger:
    """Writes one directory per run: config, per-episode CSV, eval CSV, checkpoint."""

    def __init__(self, root: str | Path, name: str):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = Path(root) / f"{name}_{stamp}"
        self.path.mkdir(parents=True, exist_ok=True)
        self._train_file = None
        self._train_writer = None
        self._eval_file = None
        self._eval_writer = None

    def save_config(self, data: dict) -> None:
        (self.path / "config.json").write_text(json.dumps(data, indent=2, default=str))

    def log_episode(self, stats: EpisodeStats) -> None:
        if self._train_writer is None:
            self._train_file = open(self.path / "train.csv", "w", newline="")
            self._train_writer = csv.DictWriter(
                self._train_file, fieldnames=[f.name for f in fields(EpisodeStats)])
            self._train_writer.writeheader()
        self._train_writer.writerow(asdict(stats))

    def log_eval(self, episode: int, metrics: dict) -> None:
        row = {"episode": episode, **metrics}
        if self._eval_writer is None:
            self._eval_file = open(self.path / "eval.csv", "w", newline="")
            self._eval_writer = csv.DictWriter(self._eval_file, fieldnames=list(row))
            self._eval_writer.writeheader()
        self._eval_writer.writerow(row)
        self._eval_file.flush()

    def save_checkpoint(self, agent, meta: dict | None = None) -> None:
        state = agent.state_dict()
        if state:
            np.savez_compressed(self.path / "policy.npz", **state)
        if meta:
            (self.path / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    def close(self) -> None:
        for handle in (self._train_file, self._eval_file):
            if handle:
                handle.close()


def load_checkpoint(path: str | Path) -> dict:
    with np.load(Path(path) / "policy.npz") as data:
        return {key: data[key] for key in data.files}
