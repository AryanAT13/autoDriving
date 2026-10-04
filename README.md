# RL Driving Simulation

Reinforcement learning agents that learn to drive a three-lane road with traffic and
signals. The simulation exists to serve the RL, not the other way round: it is fast,
fully reproducible, and exposes the same world through two observation encodings so
tabular and function-approximation agents stay directly comparable.

**Status: Phases 1 and 2 complete** — environment, baselines, four learning algorithms,
training harness, CLI and test suite.

## Setup

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
```

All commands below assume `./.venv/bin/python`.

## Quickstart

```bash
./.venv/bin/python -m rl_drive.cli baseline
```

```bash
./.venv/bin/python -m rl_drive.cli train --algo q_learning --episodes 40000 --alpha 0.2 --eps-decay 25000
```

```bash
./.venv/bin/python -m rl_drive.cli policy --run runs/q_learning_s0_<timestamp>
```

```bash
./.venv/bin/python -m pytest -q
```

## The decision problem

A three-lane road 1200 units long, 12 traffic vehicles that spawn ahead and recycle
behind, and three traffic lights on a green/yellow/red cycle. The episode ends on a
collision, on reaching the goal, or after 300 steps.

**Actions (5):** `MAINTAIN`, `ACCELERATE`, `BRAKE`, `LANE_LEFT`, `LANE_RIGHT`.

**Discrete state (2160 entries, 640 reachable):**

| Feature | Values |
| --- | --- |
| lane | left / middle / right |
| speed level | 0-4 |
| gap ahead | near / med / far / clear |
| left lane | blocked / free / no lane |
| right lane | blocked / free / no lane |
| light ahead | none / green / yellow / red |

Side occupancy is constrained by the lane index, so only 640 of the 2160 table entries
are physically reachable.

A sixth feature carrying the lead vehicle's relative speed (`closing / steady / opening`)
is implemented and switchable via `closing_buckets`, but is **off by default**: measured
over matched runs it never paid for its 3x larger state space. See Findings below.

**Continuous state (15 floats)** for function approximation: normalised speed, lane
one-hot, gaps ahead and behind in the adjacent lanes, relative lead speed, distance to
the next light and its phase one-hot.

**Rewards:**

| Event | Value |
| --- | --- |
| progress | `+0.1 x speed` |
| time step | `-0.25` |
| lane change | `-0.2` |
| off-road attempt | `-5` |
| red-light violation | `-20` |
| collision | `-75`, terminates |
| goal reached | `+50`, terminates |

Progress is proportional to speed, but total distance is fixed, so the progress term is
roughly constant across policies. The step penalty is what actually makes speed pay. The
pair rules out the degenerate "stop forever and never crash" policy, which scores -74.9.

A time limit is not an absorbing state, so updates bootstrap on truncation but not on
termination.

## Results

Three seeds, 20000 training episodes, scored greedily on 200 held-out episodes.
Return is mean +/- standard deviation across seeds.

| Agent | Return | Collisions | Goal | Speed | Training collisions |
| --- | --- | --- | --- | --- | --- |
| random | -95.6 +/- 2.2 | 98.2% | 1.7% | 2.07 | - |
| always_accelerate | -75.2 +/- 0.0 | 100.0% | 0.0% | 3.91 | - |
| crawler (never moves) | -74.9 | 0.0% | 0.0% | 0.00 | - |
| `monte_carlo` | -6.8 +/- 6.5 | 16.5% | 81.8% | 1.83 | 62.0% |
| `sarsa` | 34.0 +/- 3.5 | 0.2% | 99.2% | 2.06 | **32.0%** |
| `expected_sarsa` | 38.6 +/- 0.8 | 0.0% | 99.8% | 2.26 | 39.1% |
| `q_learning` | **42.2 +/- 0.8** | 0.0% | 100.0% | 2.59 | 58.2% |
| scripted (hand-written) | 43.5 +/- 0.0 | 0.0% | 100.0% | 2.24 | - |

Calibration: on an empty road with no signals the scripted driver scores 61.3, matching
the hand-computed ceiling of 61.2. Traffic costs roughly 14 points and the signals 4, so
43.5 is close to what this traffic density allows.

Q-Learning reaches parity with a hand-tuned rule-based controller from zero prior
knowledge, having started 138 points below it.

## Findings

**On-policy methods trade return for safety during training.** Q-Learning ends higher
(42.2 vs 34.0) but collides in 58.2% of training episodes against SARSA's 32.0%. Because
SARSA bootstraps from the action it will actually take, it prices in its own exploration
and learns a slower, more cautious policy (speed 2.06 vs 2.59). Expected SARSA lands
between the two on both axes, as theory predicts. This is the cliff-walking result in a
driving setting.

**SARSA's caution scales with the exploration rate.** At `epsilon_end = 0.05` it degrades
into a crawling policy that never arrives (return -44, speed 0.69): assuming it will keep
taking random actions forever, crawling really is its best option. At 0.01 it recovers to
34.0. Per-algorithm exploration rates live in `configs/algos.json`.

**Monte Carlo is far weaker than TD here.** Without bootstrapping it needs complete
episodes and inherits their full variance, finishing at -6.8 with a 16.5% collision rate.

**Adding relative speed to the tabular state did not pay for itself.** It tripled the
state space and never won on matched runs (35.1 vs 40.1 at 20k episodes; 35.1 vs 35.9 at
50k). It is retained behind `closing_buckets` as an ablation knob, and the continuous
encoder keeps the feature for function approximation in Phase 3.

**Reward shaping needed rebalancing.** Because total distance is fixed, the
speed-proportional progress term is near-constant across policies; the step penalty is
what actually makes speed pay. At the original `-0.05` the agent had almost no reason to
hurry. Raising it to `-0.25` and the collision cost to `-75` puts the degenerate
"never move" policy (-74.9) level with crashing, while discounting keeps crashing
strictly worse than crawling, so there is no incentive to end an episode early.

## Algorithms

| Agent | Family |
| --- | --- |
| `random`, `always_accelerate`, `scripted` | non-learning baselines |
| `q_learning` | off-policy TD control |
| `sarsa` | on-policy TD control |
| `expected_sarsa` | on-policy TD, expectation instead of a sample |
| `monte_carlo` | first-visit MC control |

The three TD variants differ only in their bootstrap target and share one file.
`scripted` is a hand-written rule-based driver reading the same discrete state as the
learners; its job is to prove the observation is sufficient to drive safely, so a failure
to learn is an agent problem rather than an unsolvable environment.

Per-algorithm hyperparameters are in `configs/algos.json` and are applied automatically;
explicit CLI flags still override them.

## Layout

```
rl_drive/
  config.py           dataclass defaults, JSON overrides
  env/
    world.py          road, traffic, signals
    observations.py   discrete + continuous encoders
    rewards.py        every reward term
    driving_env.py    Gymnasium Env
  agents/
    base.py           Agent protocol, epsilon schedule, Q-table policy
    baselines.py      random / always-accelerate / scripted
    tabular.py        Q-Learning, SARSA, Expected SARSA
    monte_carlo.py
  training/
    rollout.py        one episode
    evaluator.py      greedy evaluation on held-out seeds
    trainer.py        the training loop
    metrics.py        stats, CSV logging, checkpoints
  cli.py
configs/              JSON presets
tests/
runs/                 outputs (gitignored)
```

## Reproducibility

`reset(seed=k)` is bit-for-bit reproducible. Evaluation always runs greedily on a fixed
held-out seed set starting at 10000, with deterministic tie-breaking, so repeated
evaluations of the same policy return identical numbers. Each run writes `config.json`,
`train.csv`, `eval.csv`, `meta.json` and `policy.npz` to its own directory under `runs/`.

## Roadmap

- **Phase 3** — Dyna-Q / Dyna-Q+, Actor-Critic on the continuous encoder, multi-seed
  experiment suite and report figures.
- **Phase 4** — FastAPI backend and browser dashboard: live canvas, live charts, controls.
- **Phase 5** — manual-drive human baseline, hyperparameter sweeps, learning-proof artifacts.
