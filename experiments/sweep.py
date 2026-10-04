"""Sweep one agent hyperparameter across seeds and plot its effect.

    python experiments/sweep.py --algo dyna_q --param planning_steps --values 0 5 20 50
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from experiments.plotting import MUTED, colour, finish, new_axes
from rl_drive.config import apply_overrides, load_configs
from rl_drive.training import run_seeds, summarise_seeds


def parse(text: str):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", required=True)
    parser.add_argument("--param", required=True)
    parser.add_argument("--values", nargs="+", required=True)
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE",
                        help="further agent config fields, held fixed across the sweep")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--final-episodes", type=int, default=200)
    parser.add_argument("--config")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = tuple(range(args.seeds))
    values = [parse(v) for v in args.values]
    fixed = {key: parse(val) for key, val in
             (pair.split("=", 1) for pair in args.overrides)}
    rows = []

    for value in values:
        env_cfg, reward_cfg, agent_cfg = load_configs(args.config, algo=args.algo)
        apply_overrides(agent_cfg, {**fixed, args.param: value})
        result = run_seeds(args.algo, args.episodes, seeds, env_cfg, reward_cfg, agent_cfg,
                           final_episodes=args.final_episodes)
        summary = summarise_seeds(result)
        finals = result["finals"]
        rows.append({
            args.param: value,
            "return": round(summary["mean"], 2),
            "return_ci": round(summary["ci"], 2),
            "collision_rate": round(float(np.mean([f["collision_rate"] for f in finals])), 4),
            "goal_rate": round(float(np.mean([f["goal_rate"] for f in finals])), 4),
        })
        print(f"  {args.param}={value}  return={summary['mean']:7.2f} +/-{summary['ci']:5.2f}",
              flush=True)

    stem = f"sweep_{args.algo}_{args.param}"
    with open(out / f"{stem}.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    positions = np.arange(len(values))
    fig, ax = new_axes(width=7.5, height=4.5)
    ax.errorbar(positions, [r["return"] for r in rows], yerr=[r["return_ci"] for r in rows],
                color=colour(args.algo), linewidth=2.0, marker="o", markersize=8,
                capsize=4, ecolor="#9a9a95", elinewidth=1.2, zorder=3)
    for x, row in zip(positions, rows):
        # offset sideways so the label never lands on a vertical error bar
        ax.annotate(f"{row['return']:.1f}", (x, row["return"]), textcoords="offset points",
                    xytext=(17, -3), ha="left", fontsize=9, color=MUTED)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(v) for v in values])
    ax.margins(x=0.12, y=0.2)
    finish(fig, ax, f"{args.algo}: effect of {args.param}", args.param,
           f"final return ({args.seeds} seeds, 95% CI)", out / f"{stem}.png")
    print(f"\nwrote {out}/{stem}.csv and {out}/{stem}.png")


if __name__ == "__main__":
    main()
