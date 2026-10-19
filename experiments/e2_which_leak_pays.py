"""E2: which of the four problems is actually worth anything.

An audit that reports four violations invites the question of where to start. Dropping one
offending feature at a time and rescoring answers it with a number, and the answer is not
the ordering the severity ranking would suggest, which is worth knowing before telling
anyone the ranking is a priority list.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import table, write  # noqa: E402

from leakage_audit.audit import audit  # noqa: E402
from leakage_audit.catalog import example_pipeline  # noqa: E402
from leakage_audit.dataset import generate_panel  # noqa: E402
from leakage_audit.evaluate import WALK_FORWARD, score_one  # noqa: E402
from leakage_audit.features import Pipeline  # noqa: E402

SEEDS = (20260219, 11, 404, 2027, 90210)
ENTITIES, PERIODS, NOISE, FOLDS = 80, 180, 1.2, 5


def score_without(dropped: tuple[str, ...]) -> float:
    kept = tuple(f for f in example_pipeline().features if f.name not in dropped)
    pipeline = Pipeline(features=kept, horizon=1)
    values = []
    for seed in SEEDS:
        panel = generate_panel(ENTITIES, PERIODS, NOISE, seed)
        values.append(score_one(pipeline, panel, WALK_FORWARD, as_of=False,
                                folds=FOLDS, seed=seed).r2)
    return float(np.mean(values))


def run() -> dict:
    report = audit(example_pipeline())
    full = score_without(())
    floor = score_without(report.offending_features)

    cells = []
    for violation in report.violations:
        without = score_without((violation.feature,))
        cells.append({
            "feature": violation.feature,
            "kind": violation.kind,
            "severity_rank": violation.severity_rank,
            "score_without": without,
            "worth": full - without,
        })

    by_worth = sorted(cells, key=lambda c: c["worth"], reverse=True)
    by_severity = sorted(cells, key=lambda c: c["severity_rank"])
    return {
        "experiment": "e2_which_leak_pays",
        "question": "Which violation is actually inflating the score?",
        "seeds": list(SEEDS),
        "full_score": full,
        "all_leaks_removed": floor,
        "cells": cells,
        "most_valuable_leak": by_worth[0]["feature"],
        "most_valuable_worth": by_worth[0]["worth"],
        "least_valuable_leak": by_worth[-1]["feature"],
        "least_valuable_worth": by_worth[-1]["worth"],
        "severity_matches_worth": [c["feature"] for c in by_severity]
        == [c["feature"] for c in by_worth],
        "worst_by_severity": by_severity[0]["feature"],
    }


def main() -> None:
    payload = run()
    rows = [
        [c["feature"], c["kind"], f"{c['score_without']:.4f}", f"{c['worth']:+.4f}"]
        for c in sorted(payload["cells"], key=lambda c: c["worth"], reverse=True)
    ]
    print(f"E2 what each violation is worth, {len(SEEDS)} seeds, walk-forward\n")
    print(table(["feature dropped", "kind", "score without", "worth"],
                rows, [22, 18, 15, 10]))
    print()
    print(
        f"full pipeline {payload['full_score']:.4f}, every leak removed "
        f"{payload['all_leaks_removed']:.4f}; severity ordering matches value ordering: "
        f"{payload['severity_matches_worth']}"
    )
    path = write("e2_which_leak_pays", payload)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
