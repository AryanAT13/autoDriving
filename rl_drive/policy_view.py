"""A readable slice of a learned Q-table, shared by the CLI and the dashboard."""

from __future__ import annotations

import numpy as np

from .env.driving_env import ACTION_NAMES
from .env.observations import CLOSING_NAMES, FREE, GAP_NAMES, LIGHT_NAMES


def policy_table(encoder, q: np.ndarray, max_speed: int,
                 light: str = "none", closing: str = "steady") -> dict:
    """Greedy action per (speed, gap ahead), mid-lane with both neighbours free.

    This is the slice a person can check against their own driving intuition, which is
    what makes a learned policy inspectable rather than just a number.
    """
    if light not in LIGHT_NAMES:
        raise ValueError(f"light must be one of {LIGHT_NAMES}")
    if closing not in CLOSING_NAMES:
        raise ValueError(f"closing must be one of {CLOSING_NAMES}")

    light_index = LIGHT_NAMES.index(light)
    # the relative-speed feature is off by default, leaving a single valid index
    closing_index = min(CLOSING_NAMES.index(closing), encoder.dims[3] - 1)

    rows = []
    for speed in range(max_speed + 1):
        actions = []
        for gap in range(len(GAP_NAMES)):
            state = int(np.ravel_multi_index(
                (1, speed, gap, closing_index, FREE, FREE, light_index), encoder.dims))
            actions.append(ACTION_NAMES[int(np.argmax(q[state]))])
        rows.append({"speed": speed, "actions": actions})

    return {
        "gaps": list(GAP_NAMES),
        "rows": rows,
        "light": light,
        "closing": CLOSING_NAMES[closing_index],
        "states_learned": int((q != 0).any(axis=1).sum()),
        "states_reachable": encoder.reachable,
        "states_total": int(q.shape[0]),
    }
