import time

import pytest
from fastapi.testclient import TestClient

from rl_drive.server import app
from rl_drive.server.runs import BLOCK, TrainingRun

TERMINAL = ("done", "stopped", "error")


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def wait_for_terminal(run, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if run.status in TERMINAL:
            return run.status
        time.sleep(0.02)
    raise AssertionError(f"run still {run.status} after {timeout}s")


def test_algorithms_lists_learners_with_defaults(client):
    data = client.get("/api/algorithms").json()
    names = [entry["name"] for entry in data["learners"]]
    assert "q_learning" in names and "actor_critic" in names
    assert "scripted" in data["baselines"]
    actor = next(e for e in data["learners"] if e["name"] == "actor_critic")
    assert actor["obs_mode"] == "continuous"
    assert "alpha_policy" in actor["defaults"]


def test_environment_describes_the_road(client):
    env = client.get("/api/environment").json()["env"]
    assert env["lanes"] == 3 and env["road_length"] > 0


def test_index_and_assets_are_served(client):
    assert "RL Driving Simulation" in client.get("/").text
    for asset in ("app.js", "renderer.js", "styles.css"):
        assert client.get(f"/{asset}").status_code == 200


def test_baseline_episode_returns_renderable_frames(client):
    payload = client.get("/api/episode", params={"algo": "scripted", "seed": 5}).json()
    assert payload["stats"]["steps"] == len(payload["frames"]) - 1
    frame = payload["frames"][0]
    assert {"step", "lanes", "road_length", "ego", "traffic", "lights"} <= set(frame)
    assert frame["ego"]["lane"] in range(frame["lanes"])


def test_baseline_endpoint_rejects_learners(client):
    assert client.get("/api/episode", params={"algo": "q_learning"}).status_code == 400


def test_unknown_algorithm_is_rejected(client):
    assert client.post("/api/runs", json={"algo": "nope", "episodes": 10}).status_code == 400


def test_unknown_override_is_rejected(client):
    response = client.post("/api/runs", json={"algo": "q_learning", "episodes": 10,
                                              "overrides": {"not_a_setting": 1}})
    assert response.status_code == 400


def test_episode_bounds_are_validated(client):
    assert client.post("/api/runs", json={"algo": "q_learning", "episodes": 0}).status_code == 422


def test_unknown_run_is_404(client):
    assert client.get("/api/runs/deadbeef").status_code == 404
    assert client.post("/api/runs/deadbeef/stop").status_code == 404


def test_websocket_rejects_unknown_run(client):
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/runs/deadbeef") as ws:
            ws.receive_json()


def test_run_lifecycle_and_overrides(client):
    run = client.post("/api/runs", json={"algo": "q_learning", "episodes": 60,
                                         "eval_every": 30, "eval_episodes": 3,
                                         "overrides": {"alpha": 0.42}}).json()
    assert run["config"]["alpha"] == 0.42
    from rl_drive.server.app import manager
    wait_for_terminal(manager.get(run["id"]))

    final = client.get(f"/api/runs/{run['id']}").json()
    assert final["status"] == "done" and final["episode"] == 60
    assert final["latest_eval"]["episodes"] == 3
    assert run["id"] in [r["id"] for r in client.get("/api/runs").json()["runs"]]


def test_trained_rollout_is_reproducible(client):
    run = client.post("/api/runs", json={"algo": "q_learning", "episodes": 40,
                                         "eval_every": 0}).json()
    from rl_drive.server.app import manager
    wait_for_terminal(manager.get(run["id"]))
    first = client.get(f"/api/runs/{run['id']}/episode", params={"seed": 3}).json()
    second = client.get(f"/api/runs/{run['id']}/episode", params={"seed": 3}).json()
    assert first["stats"] == second["stats"]
    assert len(first["frames"]) == first["stats"]["steps"] + 1


def test_stop_halts_training_early(client):
    run = client.post("/api/runs", json={"algo": "q_learning", "episodes": 500_00,
                                         "eval_every": 0}).json()
    from rl_drive.server.app import manager
    handle = manager.get(run["id"])
    time.sleep(0.3)
    stopped = client.post(f"/api/runs/{run['id']}/stop").json()
    assert stopped["status"] in ("running", "stopped")
    assert wait_for_terminal(handle) == "stopped"
    assert handle.completed < 50_000


@pytest.mark.parametrize("episodes", [40, 120, 260])
def test_last_websocket_message_is_always_terminal(client, episodes):
    """The client must never be left on a non-terminal status after a run ends."""
    run = client.post("/api/runs", json={"algo": "q_learning", "episodes": episodes,
                                         "eval_every": 0}).json()
    with client.websocket_connect(f"/ws/runs/{run['id']}") as ws:
        last = None
        deadline = time.time() + 30
        while time.time() < deadline:
            last = ws.receive_json()
            if last["summary"]["status"] in TERMINAL:
                break
        assert last is not None
        assert last["summary"]["status"] in TERMINAL
        assert last["summary"]["episode"] == episodes


def test_blocks_cover_every_episode_including_a_partial_tail():
    """A run whose length is not a multiple of BLOCK must still report its last episodes."""
    episodes = BLOCK * 2 + 7
    run = TrainingRun(algo="q_learning", episodes=episodes, eval_every=0).start()
    wait_for_terminal(run)
    assert len(run.blocks) == 3
    assert run.blocks[-1]["episode"] == episodes - 1
    assert run.completed == episodes


def test_streamed_blocks_are_far_fewer_than_episodes(client):
    """Browsers cannot plot every episode of a long run; the server aggregates first."""
    run = client.post("/api/runs", json={"algo": "q_learning", "episodes": 400,
                                         "eval_every": 0}).json()
    with client.websocket_connect(f"/ws/runs/{run['id']}") as ws:
        blocks = 0
        while True:
            message = ws.receive_json()
            blocks += len(message["blocks"])
            if message["summary"]["status"] in TERMINAL:
                break
    assert blocks == 400 // BLOCK
    assert blocks < 400


@pytest.mark.parametrize("algo,override", [
    ("actor_critic", {"hidden_units": 32}),
    ("dyna_q", {"planning_steps": 5, "model_capacity": 10}),
    ("q_learning", {"epsilon_decay_episodes": 100}),
])
def test_integer_overrides_survive_the_json_round_trip(client, algo, override):
    """These arrive as floats over JSON and must not reach numpy that way."""
    response = client.post("/api/runs", json={"algo": algo, "episodes": 30,
                                              "eval_every": 0, "overrides": override})
    assert response.status_code == 200, response.text
    from rl_drive.server.app import manager
    run = manager.get(response.json()["id"])
    assert wait_for_terminal(run) == "done"
    assert run.error is None


def test_impossible_setting_is_a_client_error_not_a_crash(client):
    response = client.post("/api/runs", json={"algo": "actor_critic", "episodes": 10,
                                              "overrides": {"hidden_units": -5}})
    assert response.status_code == 400
    assert "invalid settings" in response.json()["detail"]
