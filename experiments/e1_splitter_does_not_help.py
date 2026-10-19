"""E1: the headline. Does changing the validation splitter recover a leak?

Four cells, one panel, one model. Two splitters crossed with two materialisations of the
same feature specification. The comparison teams reach for is the left column, top against
bottom: shuffled k-fold against walk-forward, both on the features as written. The
comparison that matters is left against right.

Averaged over seeds, because a single panel can flatter either answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import table, write  # noqa: E402

from leakage_audit.catalog import example_pipeline, honest_pipeline  # noqa: E402
from leakage_audit.dataset import generate_panel  # noqa: E402
from leakage_audit.evaluate import SHUFFLED, WALK_FORWARD, compare  # noqa: E402

SEEDS = (20260219, 11, 404, 2027, 90210, 13, 777, 31337)
ENTITIES, PERIODS, NOISE, FOLDS = 80, 180, 1.2, 5


def sweep(pipeline, label: str) -> dict:
    cells = {(s, a): [] for s in (SHUFFLED, WALK_FORWARD) for a in (False, True)}
    inflations, recoveries = [], []
    for seed in SEEDS:
        panel = generate_panel(ENTITIES, PERIODS, NOISE, seed)
        comparison = compare(pipeline, panel, folds=FOLDS, seed=seed)
        for key in cells:
            cells[key].append(comparison.get(*key).r2)
        inflations.append(comparison.inflation)
        recoveries.append(comparison.splitter_recovery)

    mean_inflation = float(np.mean(inflations))
    mean_recovery = float(np.mean(recoveries))
    return {
        "label": label,
        "means": {f"{s}|{a}": float(np.mean(v)) for (s, a), v in cells.items()},
        "inflation_mean": mean_inflation,
        "inflation_spread": float(max(inflations) - min(inflations)),
        "recovery_mean": mean_recovery,
        "recovery_share": mean_recovery / mean_inflation if mean_inflation else 0.0,
    }


def run() -> dict:
    leaky = sweep(example_pipeline(), "as written")
    honest = sweep(honest_pipeline(), "repaired")
    return {
        "experiment": "e1_splitter_does_not_help",
        "question": "Does a time-ordered splitter recover a leaking feature definition?",
        "seeds": list(SEEDS),
        "folds": FOLDS,
        "leaky": leaky,
        "honest": honest,
        "declared_walk_forward": leaky["means"][f"{WALK_FORWARD}|False"],
        "asof_walk_forward": leaky["means"][f"{WALK_FORWARD}|True"],
        "declared_shuffled": leaky["means"][f"{SHUFFLED}|False"],
        "inflation": leaky["inflation_mean"],
        "recovery": leaky["recovery_mean"],
        "recovery_share": leaky["recovery_share"],
        "honest_inflation": honest["inflation_mean"],
        "inflation_spread": leaky["inflation_spread"],
    }


def main() -> None:
    payload = run()
    rows = []
    for splitter, name in ((SHUFFLED, "shuffled k-fold"), (WALK_FORWARD, "walk-forward")):
        rows.append([
            name,
            f"{payload['leaky']['means'][f'{splitter}|False']:.4f}",
            f"{payload['leaky']['means'][f'{splitter}|True']:.4f}",
            f"{payload['honest']['means'][f'{splitter}|False']:.4f}",
        ])
    print(f"E1 four cells, {len(SEEDS)} seeds, mean r2\n")
    print(table(
        ["splitter", "leaky pipeline", "same, as-of", "repaired pipeline"],
        rows, [18, 16, 14, 20],
    ))
    print()
    print(
        f"the leak is worth {payload['inflation']:+.4f} r2; changing the splitter "
        f"recovers {payload['recovery']:+.4f}, which is "
        f"{payload['recovery_share']:.1%} of it"
    )
    path = write("e1_splitter_does_not_help", payload)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
