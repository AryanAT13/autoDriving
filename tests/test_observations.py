import numpy as np

from rl_drive.config import EnvConfig
from rl_drive.env import DrivingEnv
from rl_drive.env.observations import (BLOCKED, CLEAR, CLOSING, CONTINUOUS_DIM, FREE,
                                       L_NONE, L_RED, NEAR, NO_LANE, OPENING,
                                       ContinuousEncoder)
from rl_drive.env.world import Vehicle


def test_encode_decode_roundtrip():
    env = DrivingEnv()
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(200):
        state = env.encoder.encode(env.world)
        assert env.encoder.decode(state) == env.encoder.features(env.world)
        _, _, term, trunc, _ = env.step(int(rng.integers(5)))
        if term or trunc:
            env.reset()


def test_features_stay_within_dims():
    env = DrivingEnv()
    env.reset(seed=2)
    rng = np.random.default_rng(2)
    for _ in range(500):
        for value, dim in zip(env.encoder.features(env.world), env.encoder.dims):
            assert 0 <= value < dim
        _, _, term, trunc, _ = env.step(int(rng.integers(5)))
        if term or trunc:
            env.reset()


def test_edge_lanes_report_no_lane():
    env = DrivingEnv()
    env.reset(seed=0)
    env.world.ego.lane = 0
    assert env.encoder.features(env.world)[4] == NO_LANE
    env.world.ego.lane = env.cfg.lanes - 1
    assert env.encoder.features(env.world)[5] == NO_LANE


def test_gap_bucket_and_side_occupancy():
    env = DrivingEnv()
    env.reset(seed=0)
    env.world.ego.lane, env.world.ego.x = 1, 0.0
    env.world.traffic = [Vehicle(10.0, 1, 0.0)]
    assert env.encoder.features(env.world)[2] == NEAR

    env.world.traffic = []
    assert env.encoder.features(env.world)[2] == CLEAR

    env.world.traffic = [Vehicle(5.0, 0, 0.0)]
    assert env.encoder.features(env.world)[4] == BLOCKED
    assert env.encoder.features(env.world)[5] == FREE


def test_light_feature_only_within_lookahead():
    env = DrivingEnv()
    env.reset(seed=0)
    env.world.steps = 0
    env.world.light_offsets = [25] * len(env.cfg.light_positions)
    env.world.ego.x = env.cfg.light_positions[0] - env.cfg.light_lookahead - 1.0
    assert env.encoder.features(env.world)[6] == L_NONE
    env.world.ego.x = env.cfg.light_positions[0] - 5.0
    assert env.encoder.features(env.world)[6] == L_RED


def test_continuous_encoding_is_bounded():
    env = DrivingEnv(obs_mode="continuous")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (CONTINUOUS_DIM,) and obs.dtype == np.float32
    rng = np.random.default_rng(0)
    for _ in range(300):
        obs, _, term, trunc, _ = env.step(int(rng.integers(5)))
        assert obs.shape == (CONTINUOUS_DIM,)
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
        assert env.observation_space.contains(obs)
        if term or trunc:
            env.reset()


def test_both_encoders_read_the_same_world():
    """Comparability between tabular and function-approximation agents depends on this."""
    cfg = EnvConfig()
    env = DrivingEnv(cfg)
    env.reset(seed=5)
    continuous = ContinuousEncoder(cfg).encode(env.world)
    lane, speed, *_ = env.encoder.features(env.world)
    assert continuous[1 + lane] == 1.0
    assert np.isclose(continuous[0], speed / cfg.max_speed_level)


def test_closing_feature_tracks_relative_speed():
    """Distance alone cannot say whether a gap is shrinking; this feature can."""
    env = DrivingEnv(EnvConfig(closing_buckets=3))
    env.reset(seed=0)
    env.world.ego.lane, env.world.ego.x = 1, 0.0
    env.world.ego_speed_level, env.world.ego.speed = 3, 3 * env.cfg.speed_unit

    env.world.traffic = [Vehicle(60.0, 1, 1 * env.cfg.speed_unit)]
    assert env.encoder.features(env.world)[3] == CLOSING

    env.world.traffic = [Vehicle(60.0, 1, 4 * env.cfg.speed_unit)]
    assert env.encoder.features(env.world)[3] == OPENING

    env.world.traffic = []
    assert env.encoder.features(env.world)[3] == OPENING


def test_reachable_state_count_is_an_upper_bound():
    env = DrivingEnv()
    env.reset(seed=0)
    assert env.encoder.reachable == 640
    assert env.encoder.reachable < env.encoder.n

    seen = set()
    rng = np.random.default_rng(0)
    for episode in range(60):
        env.reset(seed=episode)
        while True:
            seen.add(env.encoder.encode(env.world))
            _, _, term, trunc, _ = env.step(int(rng.integers(5)))
            if term or trunc:
                break
    assert len(seen) <= env.encoder.reachable


def test_ablated_closing_feature_is_neutral():
    """With the feature off the index must be STEADY, not a value rules would act on."""
    from rl_drive.env.observations import STEADY

    env = DrivingEnv(EnvConfig(closing_buckets=1))
    env.reset(seed=0)
    assert env.encoder.n == 2160
    env.world.ego.lane, env.world.ego.x = 1, 0.0
    env.world.traffic = [Vehicle(60.0, 1, 0.0)]
    assert env.encoder.features(env.world)[3] == STEADY
