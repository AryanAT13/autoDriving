"""FastAPI backend: start training runs, stream their metrics, replay their policies."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..agents import BASELINES, LEARNERS, REGISTRY, make_agent
from ..config import apply_overrides, load_configs
from ..env import ACTION_NAMES, DrivingEnv
from ..policy_view import policy_table
from ..training import obs_mode_for, run_episode
from .runs import RunManager, SessionStore

WEB_DIR = Path(__file__).resolve().parents[2] / "web"
TUNABLE = ("alpha", "gamma", "epsilon_start", "epsilon_end", "epsilon_decay_episodes",
           "planning_steps", "model_capacity", "alpha_policy", "alpha_value",
           "entropy_beta", "hidden_units")
POLL_SECONDS = 0.25


class DriveRequest(BaseModel):
    action: int = Field(..., ge=0, le=4)


class TrainRequest(BaseModel):
    algo: str
    episodes: int = Field(5000, ge=1, le=200_000)
    seed: int = Field(0, ge=0)
    eval_every: int = Field(250, ge=0, le=10_000)
    eval_episodes: int = Field(20, ge=1, le=200)
    overrides: dict[str, float] = Field(default_factory=dict)


manager = RunManager()
sessions = SessionStore()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    manager.stop_all()  # do not leave training threads running past shutdown


app = FastAPI(title="RL Driving Simulation", lifespan=lifespan)


@app.get("/api/algorithms")
def algorithms() -> dict:
    """Learners with their per-algorithm defaults, plus the non-learning baselines."""
    entries = []
    for algo in LEARNERS:
        _, _, agent_cfg = load_configs(None, algo=algo)
        defaults = {key: getattr(agent_cfg, key) for key in TUNABLE}
        entries.append({"name": algo, "obs_mode": obs_mode_for(algo), "defaults": defaults})
    return {"learners": entries, "baselines": list(BASELINES), "actions": list(ACTION_NAMES)}


@app.get("/api/environment")
def environment() -> dict:
    env_cfg, reward_cfg, _ = load_configs(None)
    return {"env": asdict(env_cfg), "reward": asdict(reward_cfg)}


@app.post("/api/runs")
def start_run(request: TrainRequest) -> dict:
    if request.algo not in LEARNERS:
        raise HTTPException(400, f"unknown algorithm {request.algo!r}")
    try:
        run = manager.start(algo=request.algo, episodes=request.episodes, seed=request.seed,
                            eval_every=request.eval_every, eval_episodes=request.eval_episodes,
                            overrides=request.overrides)
    except KeyError as exc:
        raise HTTPException(400, f"unknown setting {exc}") from exc
    except (TypeError, ValueError) as exc:  # a setting the agent cannot be built with
        raise HTTPException(400, f"invalid settings: {exc}") from exc
    return run.summary()


@app.get("/api/runs")
def list_runs() -> dict:
    return {"runs": manager.list()}


def _require(run_id: str):
    run = manager.get(run_id)
    if run is None:
        raise HTTPException(404, f"no run {run_id!r}")
    return run


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    return _require(run_id).summary()


@app.post("/api/runs/{run_id}/stop")
def stop_run(run_id: str) -> dict:
    run = _require(run_id)
    run.stop()
    return run.summary()


@app.get("/api/runs/{run_id}/episode")
def run_episode_for(run_id: str, seed: int | None = None) -> dict:
    return _require(run_id).rollout(seed=seed)


@app.get("/api/episode")
def baseline_episode(algo: str = "scripted", seed: int | None = None) -> dict:
    """One greedy episode from a non-learning policy, for comparison against a run."""
    if algo not in BASELINES:
        raise HTTPException(400, f"{algo!r} is not a baseline; use /api/runs/<id>/episode")
    env_cfg, reward_cfg, _ = load_configs(None)
    env = DrivingEnv(env_cfg, reward_cfg, obs_mode_for(algo))
    agent = make_agent(algo, env, None, np.random.default_rng(seed or 0))
    frames: list[dict] = []
    stats = run_episode(env, agent, seed=seed, greedy=True, learn=False, frames=frames)
    return {"frames": frames, "stats": asdict(stats)}


@app.get("/api/runs/{run_id}/policy")
def run_policy(run_id: str, light: str = "none", closing: str = "steady") -> dict:
    """The learned policy as a table a person can read and sanity-check."""
    run = _require(run_id)
    agent = run.agent_for_inspection()
    if not hasattr(agent, "q"):
        raise HTTPException(400, f"{run.algo} uses function approximation, not a Q-table")
    try:
        return policy_table(run.encoder, agent.q, run.env_cfg.max_speed_level,
                            light=light, closing=closing)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/drive")
def start_drive(seed: int | None = None) -> dict:
    """Begin a human-driven episode on the same road the agents train on."""
    env_cfg, reward_cfg, _ = load_configs(None)
    return sessions.start(env_cfg, reward_cfg, seed).view()


@app.post("/api/drive/{session_id}")
def drive_step(session_id: str, request: DriveRequest) -> dict:
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(404, f"no drive session {session_id!r}")
    return session.step(request.action)


@app.websocket("/ws/runs/{run_id}")
async def stream_run(websocket: WebSocket, run_id: str) -> None:
    run = manager.get(run_id)
    if run is None:
        await websocket.close(code=4004)
        return
    await websocket.accept()
    blocks_sent = evals_sent = 0
    try:
        while True:
            blocks, evals = run.since(blocks_sent, evals_sent)
            blocks_sent += len(blocks)
            evals_sent += len(evals)
            # Read the summary once: deciding on a fresh run.status here would let a run
            # that finishes mid-iteration break the loop after a payload saying "running",
            # leaving the client stuck on a status that never arrives.
            summary = run.summary()
            await websocket.send_json({"blocks": blocks, "evals": evals, "summary": summary})
            if summary["status"] not in ("running", "pending"):
                break
            await asyncio.sleep(POLL_SECONDS)
    except WebSocketDisconnect:
        pass


if WEB_DIR.is_dir():  # mounted last so it does not shadow the API routes
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
