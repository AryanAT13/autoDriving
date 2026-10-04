"""Train every algorithm across seeds, then write the comparison table and figures.

    python experiments/compare_algorithms.py --episodes 20000 --seeds 5
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from experiments.plotting import BASELINE, MUTED, colour, finish, new_axes
from rl_drive.agents import BASELINES, LEARNERS
from rl_drive.config import load_configs
from rl_drive.training import curve_band, evaluate_seeds, run_seeds, summarise_seeds

COLUMNS = ("algorithm", "return", "return_ci", "collision_rate", "goal_rate",
           "mean_speed", "steps", "training_collisions")


def collect(args) -> dict:
    seeds = tuple(range(args.seeds))
    results = {}
    for algo in args.baselines:
        env_cfg, reward_cfg, _ = load_configs(args.config)
        results[algo] = evaluate_seeds(algo, seeds, env_cfg, reward_cfg, args.final_episodes)
        print(f"  {algo:16s} baseline  return={summarise_seeds(results[algo])['mean']:7.2f}",
              flush=True)
    for algo in args.algos:
        env_cfg, reward_cfg, agent_cfg = load_configs(args.config, algo=algo)
        results[algo] = run_seeds(
            algo, args.episodes, seeds, env_cfg, reward_cfg, agent_cfg,
            eval_every=args.eval_every, eval_episodes=args.eval_episodes,
            final_episodes=args.final_episodes,
            on_seed=lambda a, s, m: print(f"  {a:16s} seed {s}  return={m['reward']:7.2f}  "
                                          f"collision={m['collision_rate']:5.1%}", flush=True))
    return results


def write_table(results: dict, path: Path) -> list[dict]:
    rows = []
    for algo, result in results.items():
        summary = summarise_seeds(result)
        finals = result["finals"]
        rows.append({
            "algorithm": algo,
            "return": round(summary["mean"], 2),
            "return_ci": round(summary["ci"], 2),
            "collision_rate": round(float(np.mean([f["collision_rate"] for f in finals])), 4),
            "goal_rate": round(float(np.mean([f["goal_rate"] for f in finals])), 4),
            "mean_speed": round(float(np.mean([f["mean_speed"] for f in finals])), 3),
            "steps": round(float(np.mean([f["steps"] for f in finals])), 1),
            "training_collisions": (round(float(np.mean(result["training_collisions"])), 4)
                                    if result["training_collisions"] else ""),
        })
    rows.sort(key=lambda row: row["return"])
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_curve(results: dict, metric: str, title: str, ylabel: str, path: Path) -> None:
    fig, ax = new_axes()
    is_rate = metric.endswith("_rate")
    for algo, result in results.items():
        episodes, mean, band = curve_band(result, metric)
        if episodes.size == 0:
            continue
        shade = colour(algo)
        lower = mean - band
        if is_rate:  # a proportion has no negative values for the band to imply
            lower = np.maximum(lower, 0.0)
        ax.fill_between(episodes, lower, mean + band, color=shade, alpha=0.15, linewidth=0)
        ax.plot(episodes, mean, color=shade, linewidth=2.0, label=algo, zorder=3)
    legend = ax.legend(frameon=False, fontsize=9, loc="best")
    for text in legend.get_texts():
        text.set_color(MUTED)
    finish(fig, ax, title, "training episode", ylabel, path)


def save_curves(results: dict, path: Path) -> None:
    """Persist per-seed curves so figures can be redrawn without retraining."""
    rows = [{"algorithm": algo, "seed": seed, **point}
            for algo, result in results.items()
            for seed, curve in zip(result["seeds"], result["curves"])
            for point in curve]
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_curves(path: Path) -> dict:
    by_algo: dict[str, dict[int, list]] = {}
    with open(path) as handle:
        for row in csv.DictReader(handle):
            algo, seed = row.pop("algorithm"), int(row.pop("seed"))
            point = {key: float(value) for key, value in row.items()}
            point["episode"] = int(point["episode"])
            by_algo.setdefault(algo, {}).setdefault(seed, []).append(point)
    return {algo: {"algo": algo, "seeds": sorted(seeds),
                   "curves": [seeds[s] for s in sorted(seeds)],
                   "finals": [], "training_collisions": []}
            for algo, seeds in by_algo.items()}


def plot_summary(rows: list[dict], path: Path) -> None:
    fig, ax = new_axes(height=0.5 * len(rows) + 1.8)
    names = [row["algorithm"] for row in rows]
    values = [row["return"] for row in rows]
    errors = [row["return_ci"] for row in rows]
    colours = [colour(name) if name in LEARNERS else BASELINE for name in names]
    ax.barh(names, values, xerr=errors, color=colours, height=0.6,
            error_kw={"ecolor": "#9a9a95", "elinewidth": 1.2, "capsize": 3}, zorder=3)
    ax.axvline(0.0, color="#d5d4d0", linewidth=1.0)
    span = max(abs(min(values)), abs(max(values))) or 1.0
    for name, value, error in zip(names, values, errors):
        offset = 0.03 * span + error  # clear the whisker, not just the bar end
        ax.text(value + (offset if value >= 0 else -offset), name, f"{value:.1f}",
                va="center", ha="left" if value >= 0 else "right", fontsize=9, color=MUTED)
    ax.margins(x=0.22)
    finish(fig, ax, "Final greedy return by agent", "mean return (95% CI)", "", path)


def draw(results: dict, rows: list[dict], out: Path) -> None:
    plot_curve(results, "reward", "Greedy evaluation return during training",
               "return on held-out seeds", out / "learning_curves.png")
    plot_curve(results, "collision_rate", "Collision rate during training",
               "collisions per episode", out / "collision_curves.png")
    plot_summary(rows, out / "final_comparison.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replot", metavar="DIR",
                        help="redraw figures from a previous run's CSVs, without training")
    parser.add_argument("--algos", nargs="+", default=list(LEARNERS))
    parser.add_argument("--baselines", nargs="+", default=list(BASELINES))
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--final-episodes", type=int, default=200)
    parser.add_argument("--config")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()

    if args.replot:
        out = Path(args.replot)
        with open(out / "comparison.csv") as handle:
            rows = [{key: (value if key == "algorithm" else float(value or 0.0))
                     for key, value in row.items()} for row in csv.DictReader(handle)]
        draw(load_curves(out / "curves.csv"), rows, out)
        print(f"redrew figures in {out}")
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(args.algos)} algorithms x {args.seeds} seeds x {args.episodes} episodes",
          flush=True)

    results = collect(args)
    rows = write_table(results, out / "comparison.csv")
    save_curves(results, out / "curves.csv")
    draw(results, rows, out)

    width = max(len(row["algorithm"]) for row in rows)
    print(f"\n{'agent':>{width}}  {'return':>16}  {'collision':>9}  {'goal':>7}  {'speed':>6}")
    for row in reversed(rows):
        print(f"{row['algorithm']:>{width}}  {row['return']:9.2f} +/-{row['return_ci']:5.2f}  "
              f"{row['collision_rate']:8.1%}  {row['goal_rate']:6.1%}  {row['mean_speed']:6.2f}")
    print(f"\nwrote {out}/comparison.csv, {out}/curves.csv and 3 figures")


if __name__ == "__main__":
    main()
