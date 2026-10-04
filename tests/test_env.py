import numpy as np
import pytest

from rl_drive.config import EnvConfig
from rl_drive.env import DrivingEnv
from rl_drive.env.driving_env import ACCELERATE, BRAKE, LANE_LEFT, LANE_RIGHT, MAINTAIN
from rl_drive.env.world import RED, Vehicle


@pytest.fixture
def env():
    e = DrivingEnv()
    e.reset(seed=0)
    return e


def test_state_space_size(env):
    assert env.encoder.n == 2160
    assert env.observation_space.n == 2160
    assert env.action_space.n == 5


def test_reset_is_reproducible():
    def rollout(seed):
        e = DrivingEnv()
        obs, _ = e.reset(seed=seed)
        rng = np.random.default_rng(1)
        trace = [obs]
        for _ in range(50):
            obs, reward, term, trunc, _ = e.step(int(rng.integers(5)))
            trace.append((obs, round(reward, 6)))
            if term or trunc:
                break
        return trace

    assert rollout(3) == rollout(3)
    assert rollout(3) != rollout(4)


def test_accelerate_and_brake_change_speed(env):
    env.world.traffic.clear()
    start = env.world.ego_speed_level
    env.step(ACCELERATE)
    assert env.world.ego_speed_level == start + 1
    env.step(BRAKE)
    assert env.world.ego_speed_level == start


def test_speed_is_clamped(env):
    env.world.traffic.clear()
    for _ in range(10):
        env.step(ACCELERATE)
    assert env.world.ego_speed_level == env.cfg.max_speed_level
    for _ in range(10):
        env.step(BRAKE)
    assert env.world.ego_speed_level == 0


def test_lane_change_and_off_road(env):
    env.world.traffic.clear()
    env.world.ego.lane = 0
    _, _, _, _, info = env.step(LANE_LEFT)
    assert info["off_road"] and env.world.ego.lane == 0

    _, _, _, _, info = env.step(LANE_RIGHT)
    assert info["lane_changed"] and env.world.ego.lane == 1


def test_collision_terminates_with_penalty(env):
    env.world.traffic = [Vehicle(env.world.ego.x + 5.0, env.world.ego.lane, 0.0)]
    _, reward, terminated, truncated, info = env.step(MAINTAIN)
    assert info["collision"] and terminated and not truncated
    assert reward < env.reward_cfg.collision + 10.0


def test_swept_collision_catches_pass_through(env):
    """A fast ego must not tunnel past a slow car between two discrete positions."""
    env.world.ego.x, env.world.ego.lane, env.world.ego.speed = 0.0, 1, 30.0
    env.world.traffic = [Vehicle(11.0, 1, 0.0)]
    assert env._collides([0.0])


def test_no_collision_in_other_lane(env):
    env.world.ego.lane = 1
    env.world.traffic = [Vehicle(env.world.ego.x + 2.0, 2, 0.0)]
    _, _, _, _, info = env.step(MAINTAIN)
    assert not info["collision"]


def test_goal_terminates(env):
    env.world.traffic.clear()
    env.world.ego.x = env.cfg.road_length - 5.0
    _, reward, terminated, truncated, info = env.step(MAINTAIN)
    assert info["goal"] and terminated and not truncated
    assert reward > env.reward_cfg.goal - 10.0


def test_truncation_is_not_termination():
    cfg = EnvConfig(max_steps=5, n_traffic=0, road_length=1e9)
    env = DrivingEnv(cfg)
    env.reset(seed=0)
    for _ in range(4):
        _, _, terminated, truncated, _ = env.step(MAINTAIN)
        assert not terminated and not truncated
    _, _, terminated, truncated, _ = env.step(MAINTAIN)
    assert truncated and not terminated


def test_red_light_violation(env):
    env.world.traffic.clear()
    env.world.steps = 0
    env.world.light_offsets = [25] * len(env.cfg.light_positions)
    assert env.world.light_phase(0) == RED
    env.world.ego.x = env.cfg.light_positions[0] - 5.0
    _, _, _, _, info = env.step(MAINTAIN)
    assert info["ran_red"]


def test_green_light_is_not_a_violation(env):
    env.world.traffic.clear()
    env.world.steps = 0
    env.world.light_offsets = [0] * len(env.cfg.light_positions)
    env.world.ego.x = env.cfg.light_positions[0] - 5.0
    _, _, _, _, info = env.step(MAINTAIN)
    assert not info["ran_red"]


def test_traffic_density_is_maintained(env):
    for _ in range(100):
        _, _, terminated, truncated, _ = env.step(ACCELERATE)
        if terminated or truncated:
            env.reset(seed=1)
        assert len(env.world.traffic) == env.cfg.n_traffic


def test_frame_is_serialisable(env):
    import json
    frame = env.frame()
    json.dumps(frame)
    assert frame["lanes"] == env.cfg.lanes
    assert len(frame["traffic"]) == env.cfg.n_traffic
    assert len(frame["lights"]) == len(env.cfg.light_positions)


def test_invalid_obs_mode_rejected():
    with pytest.raises(ValueError):
        DrivingEnv(obs_mode="pixels")
