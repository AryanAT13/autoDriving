"""Shared figure styling. Colour follows the algorithm, never its rank in a chart."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e5e4e0"
BASELINE = "#8a8a85"

# Fixed categorical slots, validated for colour-vision deficiency on this surface.
PALETTE = {
    "q_learning": "#2a78d6",
    "sarsa": "#eb6834",
    "expected_sarsa": "#1baf7a",
    "monte_carlo": "#eda100",
    "dyna_q": "#e87ba4",
    "dyna_q_plus": "#008300",
    "actor_critic": "#4a3aa7",
}


def colour(algo: str) -> str:
    return PALETTE.get(algo, BASELINE)


def new_axes(width=9.0, height=5.0):
    fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d5d4d0")
    ax.tick_params(colors=MUTED, labelsize=9)
    return fig, ax


def finish(fig, ax, title: str, xlabel: str, ylabel: str, path) -> None:
    ax.set_title(title, color=INK, fontsize=13, pad=12, loc="left")
    ax.set_xlabel(xlabel, color=MUTED, fontsize=10)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
