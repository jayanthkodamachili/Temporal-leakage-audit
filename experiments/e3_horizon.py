"""E3: how much the leak is worth at each forecast horizon.

This was written to test a guess that turned out to be wrong, and the real answer is more
useful than the guess. The expectation was that a longer horizon makes the honest problem
harder and so makes a leaking feature worth more. What happens instead is that the value
of the leak collapses after horizon one.

The reason is specific and worth stating plainly. The worst feature in the example
pipeline reads the target over the window [-1, +1]. At horizon one the label sits at
offset +1, so the window lands exactly on it. At horizon two the label has moved to +2 and
the window no longer reaches it. The definitions are byte for byte identical at every
horizon, and what they are worth differs by a factor of sixty.

The audit flags the feature at every horizon, correctly, because reading the target past
the prediction moment is never legitimate. But the audit cannot predict what it is worth,
and this experiment is why the README says so.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import table, write  # noqa: E402

from leakage_audit.catalog import example_pipeline  # noqa: E402
from leakage_audit.dataset import generate_panel  # noqa: E402
from leakage_audit.evaluate import WALK_FORWARD, score_one  # noqa: E402

SEEDS = (20260219, 11, 404, 2027)
HORIZONS = (1, 2, 3, 5, 7)
ENTITIES, PERIODS, NOISE, FOLDS = 80, 200, 1.2, 5


def run() -> dict:
    cells = []
    for horizon in HORIZONS:
        pipeline = example_pipeline(horizon=horizon)
        declared, honest = [], []
        for seed in SEEDS:
            panel = generate_panel(ENTITIES, PERIODS, NOISE, seed)
            declared.append(score_one(pipeline, panel, WALK_FORWARD, as_of=False,
                                      folds=FOLDS, seed=seed).r2)
            honest.append(score_one(pipeline, panel, WALK_FORWARD, as_of=True,
                                    folds=FOLDS, seed=seed).r2)
        declared_mean, honest_mean = float(np.mean(declared)), float(np.mean(honest))
        # Does the worst feature's window actually reach the label at this horizon?
        target_feature = pipeline.get("target_rolling_3d")
        cells.append({
            "horizon": horizon,
            "window_reaches_label": target_feature.window_end >= horizon,
            "declared": declared_mean,
            "honest": honest_mean,
            "inflation": declared_mean - honest_mean,
            "inflation_multiple": (
                declared_mean / honest_mean if honest_mean > 0 else float("inf")
            ),
        })

    first, last = cells[0], cells[-1]
    return {
        "experiment": "e3_horizon",
        "question": "Does a longer forecast horizon make the leak more valuable?",
        "seeds": list(SEEDS),
        "cells": cells,
        "shortest_horizon": first["horizon"],
        "shortest_inflation": first["inflation"],
        "shortest_multiple": first["inflation_multiple"],
        "longest_horizon": last["horizon"],
        "longest_inflation": last["inflation"],
        "longest_multiple": last["inflation_multiple"],
        "honest_decay": first["honest"] - last["honest"],
        "horizons_reaching_label": sum(1 for c in cells if c["window_reaches_label"]),
        "horizon_count": len(cells),
        "worth_ratio_first_to_rest": (
            first["inflation"] / max(c["inflation"] for c in cells[1:])
        ),
    }


def main() -> None:
    payload = run()
    rows = [
        [
            f"{c['horizon']}",
            "yes" if c["window_reaches_label"] else "no",
            f"{c['declared']:.4f}",
            f"{c['honest']:.4f}",
            f"{c['inflation']:+.4f}",
        ]
        for c in payload["cells"]
    ]
    print(f"E3 what the leak is worth at each horizon, {len(SEEDS)} seeds, walk-forward\n")
    print(table(["horizon", "window hits label", "as written", "as-of", "inflation"],
                rows, [9, 19, 12, 10, 12]))
    print()
    print(
        f"the target window reaches the label at "
        f"{payload['horizons_reaching_label']} of {payload['horizon_count']} horizons. "
        f"Where it does, the leak is worth {payload['shortest_inflation']:+.4f}; "
        f"everywhere else at most "
        f"{max(c['inflation'] for c in payload['cells'][1:]):+.4f}, a ratio of "
        f"{payload['worth_ratio_first_to_rest']:.0f} from identical definitions"
    )
    path = write("e3_horizon", payload)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
