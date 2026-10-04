"""Command-line entry point: train, evaluate and inspect driving agents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .agents import BASELINES, LEARNERS, REGISTRY, make_agent
from .config import (AgentConfig, EnvConfig, RewardConfig, apply_overrides,
                     dump_configs, load_configs)
from .env import ACTION_NAMES, DrivingEnv
from .env.observations import CLOSING_NAMES, FREE, GAP_NAMES, LIGHT_NAMES
from .training import RunLogger, evaluate, load_checkpoint, train

METRIC_ORDER = ("reward", "reward_std", "collision_rate", "goal_rate",
                "mean_speed", "steps", "red_lights", "lane_changes")


def _print_metrics(title: str, metrics: dict) -> None:
    print(f"\n{title}")
    print(f"  return         {metrics['reward']:8.2f} +/- {metrics['reward_std']:.2f}")
    print(f"  collision rate {metrics['collision_rate']:8.1%}")
    print(f"  goal rate      {metrics['goal_rate']:8.1%}")
    print(f"  mean speed     {metrics['mean_speed']:8.2f}")
    print(f"  steps          {metrics['steps']:8.1f}")
    print(f"  red lights/ep  {metrics['red_lights']:8.2f}")


def _agent_config(args) -> tuple:
    env_cfg, reward_cfg, agent_cfg = load_configs(getattr(args, "config", None),
                                                  algo=getattr(args, "algo", None))
    overrides = {
        "alpha": getattr(args, "alpha", None),
        "gamma": getattr(args, "gamma", None),
        "epsilon_start": getattr(args, "eps_start", None),
        "epsilon_end": getattr(args, "eps_end", None),
        "epsilon_decay_episodes": getattr(args, "eps_decay", None),
    }
    apply_overrides(agent_cfg, {k: v for k, v in overrides.items() if v is not None})
    return env_cfg, reward_cfg, agent_cfg


def load_run(path: str | Path):
    """Rebuild the environment and agent stored in a run directory."""
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    data = json.loads((path / "config.json").read_text())
    env_cfg = apply_overrides(EnvConfig(), data["env"])
    env_cfg.light_positions = tuple(env_cfg.light_positions)
    reward_cfg = apply_overrides(RewardConfig(), data["reward"])
    agent_cfg = apply_overrides(AgentConfig(), data["agent"])
    obs_mode = getattr(REGISTRY[meta["algo"]], "obs_mode", "discrete")
    env = DrivingEnv(env_cfg, reward_cfg, obs_mode)
    agent = make_agent(meta["algo"], env, agent_cfg, np.random.default_rng(0))
    agent.load_state_dict(load_checkpoint(path))
    return env, agent, meta


def cmd_train(args) -> None:
    env_cfg, reward_cfg, agent_cfg = _agent_config(args)
    obs_mode = getattr(REGISTRY[args.algo], "obs_mode", "discrete")
    env = DrivingEnv(env_cfg, reward_cfg, obs_mode)
    eval_env = DrivingEnv(env_cfg, reward_cfg, obs_mode)
    agent = make_agent(args.algo, env, agent_cfg, np.random.default_rng(args.seed))

    logger = RunLogger(args.out, args.tag or f"{args.algo}_s{args.seed}")
    logger.save_config(dump_configs(env_cfg, reward_cfg, agent_cfg))
    print(f"training {args.algo} for {args.episodes} episodes -> {logger.path}")

    def on_episode(stats, evaluation):
        if evaluation:
            print(f"  ep {stats.episode + 1:6d}  eps={stats.epsilon:.3f}  "
                  f"return={evaluation['reward']:7.2f}  "
                  f"collision={evaluation['collision_rate']:5.1%}  "
                  f"goal={evaluation['goal_rate']:5.1%}  "
                  f"speed={evaluation['mean_speed']:4.2f}", flush=True)

    train(env, agent, args.episodes, eval_env=eval_env, eval_every=args.eval_every,
          eval_episodes=args.eval_episodes, logger=logger, on_episode=on_episode)

    final = evaluate(eval_env, agent, episodes=args.final_episodes)
    logger.save_checkpoint(agent, {"algo": args.algo, "seed": args.seed,
                                   "episodes": args.episodes, "final": final})
    logger.close()
    _print_metrics(f"{args.algo} final (greedy, {args.final_episodes} held-out episodes)", final)


def cmd_baseline(args) -> None:
    env_cfg, reward_cfg, _ = load_configs(args.config)
    env = DrivingEnv(env_cfg, reward_cfg)
    names = (args.algo,) if args.algo else BASELINES
    for name in names:
        agent = make_agent(name, env, None, np.random.default_rng(args.seed))
        _print_metrics(name, evaluate(env, agent, episodes=args.episodes))


def cmd_eval(args) -> None:
    env, agent, meta = load_run(args.run)
    _print_metrics(f"{meta['algo']} @ {args.run}", evaluate(env, agent, episodes=args.episodes))


def cmd_policy(args) -> None:
    """Print the greedy action for an interpretable slice of the state space."""
    env, agent, meta = load_run(args.run)
    light = LIGHT_NAMES.index(args.light)
    closing = CLOSING_NAMES.index(args.closing)
    if closing >= env.encoder.dims[3]:
        closing = 0  # the relative-speed feature is ablated in this run
    print(f"\n{meta['algo']}: greedy policy | lane=middle, both sides free, "
          f"lead={args.closing}, light={args.light}")
    print("  speed |" + "".join(f"{name:>14}" for name in GAP_NAMES))
    print("  " + "-" * (6 + 14 * len(GAP_NAMES)))
    for speed in range(env.cfg.max_speed_level + 1):
        actions = []
        for gap in range(len(GAP_NAMES)):
            state = int(np.ravel_multi_index(
                (1, speed, gap, closing, FREE, FREE, light), env.encoder.dims))
            actions.append(ACTION_NAMES[int(np.argmax(agent.q[state]))])
        print(f"  {speed:5d} |" + "".join(f"{a:>14}" for a in actions))
    visited = int((agent.q != 0).any(axis=1).sum())
    print(f"\n  states with learned values: {visited} / {env.encoder.reachable} reachable "
          f"({agent.q.shape[0]} in the table)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rl_drive", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="train a learning agent")
    train_p.add_argument("--algo", choices=LEARNERS, required=True)
    train_p.add_argument("--episodes", type=int, default=20000)
    train_p.add_argument("--seed", type=int, default=0)
    train_p.add_argument("--alpha", type=float)
    train_p.add_argument("--gamma", type=float)
    train_p.add_argument("--eps-start", dest="eps_start", type=float)
    train_p.add_argument("--eps-end", dest="eps_end", type=float)
    train_p.add_argument("--eps-decay", dest="eps_decay", type=int)
    train_p.add_argument("--eval-every", type=int, default=1000)
    train_p.add_argument("--eval-episodes", type=int, default=30)
    train_p.add_argument("--final-episodes", type=int, default=200)
    train_p.add_argument("--config")
    train_p.add_argument("--out", default="runs")
    train_p.add_argument("--tag")
    train_p.set_defaults(func=cmd_train)

    base_p = sub.add_parser("baseline", help="score the non-learning reference policies")
    base_p.add_argument("--algo", choices=BASELINES)
    base_p.add_argument("--episodes", type=int, default=200)
    base_p.add_argument("--seed", type=int, default=0)
    base_p.add_argument("--config")
    base_p.set_defaults(func=cmd_baseline)

    eval_p = sub.add_parser("eval", help="score a saved run on held-out seeds")
    eval_p.add_argument("--run", required=True)
    eval_p.add_argument("--episodes", type=int, default=200)
    eval_p.set_defaults(func=cmd_eval)

    pol_p = sub.add_parser("policy", help="print a readable slice of the learned policy")
    pol_p.add_argument("--run", required=True)
    pol_p.add_argument("--light", choices=LIGHT_NAMES, default="none")
    pol_p.add_argument("--closing", choices=CLOSING_NAMES, default="steady")
    pol_p.set_defaults(func=cmd_policy)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
