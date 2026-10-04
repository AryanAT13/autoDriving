# RL Driving Simulation

Reinforcement learning agents that learn to drive a three-lane road with traffic and
signals. The simulation exists to serve the RL, not the other way round: it is fast,
fully reproducible, and exposes the same world through two observation encodings so
tabular and function-approximation agents stay directly comparable.

**Status: Phases 1-4 complete** — environment, baselines, seven learning algorithms
spanning TD control, Monte Carlo, model-based planning and policy gradient, plus the
training harness, CLI, multi-seed experiment suite, test suite, and a browser dashboard
that trains an agent live and animates it driving.

## Setup

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
```

All commands below assume `./.venv/bin/python`.

## Quickstart

Run the dashboard and open http://127.0.0.1:8000:

```bash
./.venv/bin/python -m rl_drive.cli serve
```

Everything below also works headless from the command line:

```bash
./.venv/bin/python -m rl_drive.cli baseline
```

```bash
./.venv/bin/python -m rl_drive.cli train --algo q_learning --episodes 20000
```

Hyperparameters come from `configs/algos.json` per algorithm; CLI flags override them.
Inspect what a finished run learned:

```bash
./.venv/bin/python -m rl_drive.cli policy --run runs/q_learning_s0_<timestamp>
```

```bash
./.venv/bin/python -m pytest -q
```

Compare every algorithm across seeds and write the table and figures:

```bash
./.venv/bin/python experiments/compare_algorithms.py --episodes 20000 --seeds 5
```

That writes `comparison.csv`, `curves.csv` and three figures. Redraw the figures from the
saved curves without retraining:

```bash
./.venv/bin/python experiments/compare_algorithms.py --replot figures
```

Sweep a single hyperparameter:

```bash
./.venv/bin/python experiments/sweep.py --algo dyna_q --param planning_steps --values 0 5 20 50
```

## Dashboard

Pick an algorithm, adjust the settings that matter for it, press Train, and watch the
greedy-evaluation return climb and the collision rate fall while it learns. Press Watch
agent at any point to replay a greedy episode with the policy as it currently stands, or
Watch scripted to see the hand-written driver on the same road for comparison. Stop ends a
run early and keeps what it learned.

Training is CPU-bound, so each run owns a thread; the API only ever reads from it.
Episodes are aggregated server-side into blocks of 25 before being streamed, because a
long run produces episodes far faster than a browser can plot them.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/algorithms` | learners, their obs mode, and per-algorithm defaults |
| `GET /api/environment` | road and reward settings, for the renderer |
| `POST /api/runs` | start a run; body sets algorithm, episodes, seed and overrides |
| `GET /api/runs`, `GET /api/runs/{id}` | run list and status |
| `POST /api/runs/{id}/stop` | end a run early |
| `GET /api/runs/{id}/episode` | one greedy episode with the current policy, as frames |
| `GET /api/episode?algo=scripted` | the same for a non-learning baseline |
| `WS /ws/runs/{id}` | live stream of aggregated blocks, evaluations and status |

Interactive API docs are at `/docs`. The frontend is plain HTML, CSS and JavaScript with
Chart.js from a CDN, so there is no build step and nothing to install for it.

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

Five seeds, 20000 training episodes, scored greedily on 200 held-out episodes.
Return is mean +/- 95% confidence interval across seeds. Reproduce with
`experiments/compare_algorithms.py`; figures land in `figures/`.

| Agent | Return | Collisions | Goal | Speed | Collisions while training |
| --- | --- | --- | --- | --- | --- |
| random | -95.6 +/- 1.8 | 98.4% | 1.4% | 2.09 | - |
| always_accelerate | -75.2 +/- 0.0 | 100.0% | 0.0% | 3.91 | - |
| crawler (never moves) | -74.9 | 0.0% | 0.0% | 0.00 | - |
| `monte_carlo` | -2.3 +/- 8.1 | 19.4% | 79.3% | 1.95 | 59.1% |
| `actor_critic` | 18.5 +/- 25.8 | 0.0% | 86.5% | 2.06 | **10.8%** |
| `sarsa` | 33.0 +/- 3.1 | 0.6% | 99.1% | 1.95 | **32.0%** |
| `dyna_q` | 34.5 +/- 7.0 | 1.6% | 98.4% | 2.51 | 61.7% |
| `dyna_q_plus` | 37.6 +/- 2.1 | 0.9% | 98.4% | 2.58 | 61.6% |
| `expected_sarsa` | 40.5 +/- 0.9 | 0.0% | 100.0% | 2.45 | 39.9% |
| `q_learning` | **40.5 +/- 1.5** | 0.1% | 99.9% | 2.49 | 57.0% |
| scripted (hand-written) | 43.5 +/- 0.0 | 0.0% | 100.0% | 2.24 | - |

Calibration: on an empty road with no signals the scripted driver scores 61.3, matching
the hand-computed ceiling of 61.2. Traffic costs roughly 14 points and the signals 4, so
43.5 is close to what this traffic density allows.

Q-Learning comes within 3 points of a hand-tuned rule-based controller from zero prior
knowledge, having started 136 points below it. Every learner except Monte Carlo ends with
a collision rate under 2%.

## Findings

**On-policy methods trade return for safety during training.** Q-Learning ends higher
(40.5 vs 33.0) but collides in 57.0% of training episodes against SARSA's 32.0%. Because
SARSA bootstraps from the action it will actually take, it prices in its own exploration
and learns a slower, more cautious policy (speed 1.95 vs 2.49). Expected SARSA sits between
the two on training collisions at 39.9% while matching Q-Learning's final return, which is
the variance reduction it is meant to buy. This is the cliff-walking result in a driving
setting.

**SARSA's caution scales with the exploration rate.** At `epsilon_end = 0.05` it degrades
into a crawling policy that never arrives (return -44, speed 0.69): assuming it will keep
taking random actions forever, crawling really is its best option. At 0.01 it recovers to
33.0. Per-algorithm exploration rates live in `configs/algos.json`.

**Monte Carlo is far weaker than TD here.** Without bootstrapping it needs complete
episodes and inherits their full variance, finishing at -2.3 with a 19.4% collision rate,
the only learner that still crashes often.

**Adding relative speed to the tabular state did not pay for itself.** It tripled the
state space and never won on matched runs (35.1 vs 40.1 at 20k episodes; 35.1 vs 35.9 at
50k). It is retained behind `closing_buckets` as an ablation knob, and the continuous
encoder keeps the feature, where the actor-critic generalises across states instead of
visiting each one.

**Reward shaping needed rebalancing.** Because total distance is fixed, the
speed-proportional progress term is near-constant across policies; the step penalty is
what actually makes speed pay. At the original `-0.05` the agent had almost no reason to
hurry. Raising it to `-0.25` and the collision cost to `-75` puts the degenerate
"never move" policy (-74.9) level with crashing, while discounting keeps crashing
strictly worse than crawling, so there is no incentive to end an episode early.

**Dyna-Q's textbook model is destructive here, and a sampled model fixes it.** Sutton and
Barto's model is deterministic: one stored transition per state-action pair. The discrete
state buckets alias many different traffic configurations, so replaying that single sample
misrepresents the real distribution, and each planning step injects the error again. With
`model_capacity = 1` and 20 planning steps the agent ends at **-12.8**, far worse than no
planning at all. Keeping the 20 most recent transitions per pair and sampling among them
recovers it to **38.3**.

**Planning only pays when experience is the bottleneck.** At 20000 episodes with epsilon
decaying over 12000, planning changes almost nothing: performance is gated by the
exploration schedule, not by how fast Q converges. Re-asked in a sample-limited regime
(1200 episodes, epsilon decayed over 300, five seeds), planning is worth a great deal, and
the curve is an inverted U:

| planning steps | 0 | 5 | 20 | 50 |
| --- | --- | --- | --- | --- |
| final return | 9.8 +/- 9.6 | **24.4 +/- 5.2** | 11.3 +/- 7.4 | 9.5 +/- 9.3 |

Five planning steps give two and a half times the return of none, with the tightest
interval of the four. Beyond that it decays back: with an approximate model, over-planning
amplifies model error rather than extracting more signal. See
`figures/sweep_dyna_q_planning_steps.png`.

**The actor-critic needed reward scaling, entropy regularisation, and a learning rate
scaled to match.** At raw rewards the softmax saturates within a few hundred steps at any
learning rate, because a collision contributes -75 to the gradient. Scaling rewards by
0.05 fixes that but shrinks the TD error twentyfold, and at the original `alpha_policy` the
policy stopped moving entirely: entropy sat at exactly ln(5) = 1.609, the uniform maximum,
for 5000 episodes while the critic learned correctly. Raising `alpha_policy` to 0.05
restored learning. Entropy regularisation is what stops the policy going deterministic
early, and its weight is a balance, not a floor: at `entropy_beta = 0.02` the policy is
pinned near uniform, and at 0.2 for `alpha_policy` it collapses to a single action.

**Dyna-Q+ is both better and far steadier than Dyna-Q.** 37.6 +/- 2.1 against
34.5 +/- 7.0, on an environment that is stationary and therefore not what the exploration
bonus was designed for. The bonus still helps because the *model* goes stale: as the policy
shifts, pairs it stopped visiting keep old transitions, and the staleness term pulls the
agent back to refresh them. Dyna-Q's wide interval comes from one seed finishing at 20.8
while the rest land near 38.

**A single seed is not evidence, and policy gradient is where that bites.** On seed 0 the
linear actor-critic scored 29.0, matching the hidden-layer version. Across five seeds it
averages **-24.6 +/- 41.9** with two outright collapses, against **18.5 +/- 25.8** for one
32-unit tanh hidden layer. `hidden_units` is therefore 32 by default, with 0 kept as the
ablation.

Even settled, the actor-critic is bimodal: per-seed returns are
`[44.7, 36.7, 38.3, -16.9, -10.1]`. Three seeds beat Q-Learning's average and two fail to
arrive, which averages to a mediocre number that describes none of the five runs. Its
interval is an order of magnitude wider than Q-Learning's, the expected contrast between
policy gradient and tabular TD. Note also that it has the *lowest* training collision rate
of any learner at 10.8%: a stochastic entropy-regularised policy never commits hard enough
to crash often, and its two bad seeds fail by crawling rather than by crashing.

**Streaming every episode to the browser does not work, and the fix is server-side.**
Q-Learning produces well over a thousand episodes a second. Pushing each one over the
websocket meant roughly 300 KB of JSON four times a second, which the browser's main
thread could not parse and plot fast enough to keep up. Aggregating into blocks of 25
before sending cuts the traffic and the memory about twenty-five fold and loses nothing:
the chart was only plotting a running mean anyway.

**One websocket bug was a read-twice race.** The stream handler sent `run.summary()` and
then separately re-read `run.status` to decide whether to stop looping. A run that
finished between those two reads produced a payload saying "running" followed by the
server closing the loop, so the browser sat on a status that would never arrive. Reading
the summary once and deciding from that value fixes it; `test_last_websocket_message_is_always_terminal`
guards it.

## Algorithms

| Agent | Family |
| --- | --- |
| `random`, `always_accelerate`, `scripted` | non-learning baselines |
| `q_learning` | off-policy TD control |
| `sarsa` | on-policy TD control |
| `expected_sarsa` | on-policy TD, expectation instead of a sample |
| `monte_carlo` | first-visit MC control |
| `dyna_q` | model-based planning over a learned transition model |
| `dyna_q_plus` | Dyna-Q with an exploration bonus for stale state-action pairs |
| `actor_critic` | one-step policy gradient with a learned value baseline |

The three TD variants differ only in their bootstrap target and share one file.
`scripted` is a hand-written rule-based driver reading the same discrete state as the
learners; its job is to prove the observation is sufficient to drive safely, so a failure
to learn is an agent problem rather than an unsolvable environment.

`actor_critic` is the only agent that reads the continuous encoder, and the only one
whose gradients are written out by hand rather than inherited from a Q-table update.

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
    dyna_q.py         Dyna-Q and Dyna-Q+
    actor_critic.py   policy gradient, explicit gradients
  training/
    rollout.py        one episode
    evaluator.py      greedy evaluation on held-out seeds
    trainer.py        the training loop
    experiment.py     multi-seed runs and aggregation
    metrics.py        stats, CSV logging, checkpoints
  server/
    app.py            FastAPI routes, websocket stream, static mount
    runs.py           background training runs and their aggregation
  cli.py
web/                  dashboard: index.html, app.js, renderer.js, styles.css
configs/              JSON presets, including per-algorithm hyperparameters
experiments/          comparison and sweep scripts, figure styling
tests/
runs/, figures/       outputs (gitignored)
```

## Reproducibility

`reset(seed=k)` is bit-for-bit reproducible. Evaluation always runs greedily on a fixed
held-out seed set starting at 10000, with deterministic tie-breaking, so repeated
evaluations of the same policy return identical numbers. Each run writes `config.json`,
`train.csv`, `eval.csv`, `meta.json` and `policy.npz` to its own directory under `runs/`.

## Roadmap

- **Phase 5** — manual-drive human baseline, hyperparameter sweeps, learning-proof artifacts.
