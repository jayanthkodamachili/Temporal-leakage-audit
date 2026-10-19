"""E4: does the instrument actually work, on pipelines nobody designed to be caught?

The example pipeline was written by the same person who wrote the rules, which proves
nothing. This generates random pipelines with randomly injected problems, audits each one,
and compares what the audit reported against what was injected. Recall and precision are
the result, and both are counted per feature rather than per pipeline so one sprawling
pipeline cannot carry the score.

A perfect score here would be suspicious, and it is worth saying in advance exactly what
it would and would not mean. The generator injects leaks by violating the same conditions
the audit checks, so a clean sweep demonstrates that the implementation has no bugs across
thousands of randomly shaped pipelines, including inheritance through dependency edges. It
demonstrates nothing at all about whether the rules are complete. A leak of a kind neither
the generator nor the audit models is invisible to both, and this experiment would score it
as a success.

That limit is not a caveat bolted on afterwards; it is the reason recall is reported here
rather than in the README's headline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import table, write  # noqa: E402

from leakage_audit.audit import audit  # noqa: E402
from leakage_audit.features import EVENTS, TARGET, FeatureSpec, Pipeline  # noqa: E402

TRIALS = 400
SEED = 460_000
SIZES = (3, 6, 12, 24)


def random_feature(rng: np.random.Generator, index: int, leak: bool) -> tuple[FeatureSpec, bool]:
    """Build one feature, leaking or not, and say which it is."""
    source = TARGET if rng.random() < 0.3 else EVENTS
    if not leak:
        end = int(rng.integers(-5, 0))                 # strictly before the prediction
        lag = int(rng.integers(0, max(1, -end)))       # lag that still lands in time
        start = end - int(rng.integers(0, 6))
        return FeatureSpec(
            name=f"f{index}", source=source, window_start=start,
            window_end=end, ingestion_lag=lag,
        ), False

    style = rng.integers(0, 3)
    if style == 0:                                      # window reaching past the moment
        end = int(rng.integers(1, 5))
        start, lag = end - int(rng.integers(1, 7)), 0
    elif style == 1:                                    # lag outruns the window
        end = int(rng.integers(-3, 0))
        lag = -end + int(rng.integers(1, 4))
        start = end - int(rng.integers(0, 5))
    else:                                               # the target, at or after the label
        source = TARGET
        end = int(rng.integers(0, 3))
        start, lag = end - int(rng.integers(0, 4)), 0
    return FeatureSpec(
        name=f"f{index}", source=source, window_start=start,
        window_end=end, ingestion_lag=lag,
    ), True


def one_trial(rng: np.random.Generator, size: int, leak_rate: float) -> tuple[int, int, int]:
    """Return (true positives, false negatives, false positives) for one pipeline."""
    specs, truth = [], {}
    for index in range(size):
        should_leak = rng.random() < leak_rate
        feature, leaks = random_feature(rng, index, should_leak)
        specs.append(feature)
        truth[feature.name] = leaks

    # A dependency edge onto an earlier feature, so inheritance is exercised too. The
    # truth for a dependent is the truth of what it is built from.
    specs_with_edges = []
    for index, feature in enumerate(specs):
        if index > 0 and rng.random() < 0.3:
            parent = specs[int(rng.integers(0, index))].name
            feature = FeatureSpec(
                name=feature.name, source=feature.source,
                window_start=feature.window_start, window_end=feature.window_end,
                ingestion_lag=feature.ingestion_lag, depends_on=(parent,),
            )
            truth[feature.name] = truth[feature.name] or truth[parent]
        specs_with_edges.append(feature)

    report = audit(Pipeline(features=tuple(specs_with_edges)))
    flagged = set(report.offending_features)

    true_positive = sum(1 for n, leaks in truth.items() if leaks and n in flagged)
    false_negative = sum(1 for n, leaks in truth.items() if leaks and n not in flagged)
    false_positive = sum(1 for n, leaks in truth.items() if not leaks and n in flagged)
    return true_positive, false_negative, false_positive


def run() -> dict:
    rng = np.random.default_rng(SEED)

    cells = []
    for size in SIZES:
        tp = fn = fp = 0
        for _ in range(TRIALS):
            a, b, c = one_trial(rng, size, leak_rate=0.35)
            tp, fn, fp = tp + a, fn + b, fp + c
        cells.append({
            "pipeline_size": size,
            "true_positives": tp,
            "false_negatives": fn,
            "false_positives": fp,
            "recall": tp / (tp + fn) if tp + fn else 1.0,
            "precision": tp / (tp + fp) if tp + fp else 1.0,
        })

    clean_flagged = 0
    for _ in range(TRIALS):
        _, _, fp = one_trial(rng, 8, leak_rate=0.0)
        clean_flagged += 1 if fp else 0

    return {
        "experiment": "e4_audit_recall",
        "question": "Does the audit find injected leaks it was not designed around?",
        "trials_per_size": TRIALS,
        "cells": cells,
        "overall_recall": sum(c["true_positives"] for c in cells)
        / max(1, sum(c["true_positives"] + c["false_negatives"] for c in cells)),
        "overall_precision": sum(c["true_positives"] for c in cells)
        / max(1, sum(c["true_positives"] + c["false_positives"] for c in cells)),
        "clean_pipelines_tested": TRIALS,
        "clean_pipelines_flagged": clean_flagged,
        "clean_false_alarm_rate": clean_flagged / TRIALS,
        "injected_leaks_total": sum(
            c["true_positives"] + c["false_negatives"] for c in cells
        ),
        "measures": "implementation correctness against the rules, not rule completeness",
    }


def main() -> None:
    payload = run()
    rows = [
        [
            f"{c['pipeline_size']}",
            f"{c['true_positives']}",
            f"{c['false_negatives']}",
            f"{c['false_positives']}",
            f"{c['recall']:.4f}",
            f"{c['precision']:.4f}",
        ]
        for c in payload["cells"]
    ]
    print(f"E4 audit recall on random pipelines, {TRIALS} trials per size\n")
    print(table(["features", "found", "missed", "false", "recall", "precision"],
                rows, [10, 8, 9, 8, 10, 11]))
    print()
    print(
        f"overall recall {payload['overall_recall']:.4f}, precision "
        f"{payload['overall_precision']:.4f}; of "
        f"{payload['clean_pipelines_tested']} pipelines with nothing wrong, "
        f"{payload['clean_pipelines_flagged']} were flagged"
    )
    print(
        f"this covers {payload['injected_leaks_total']} injected leaks and measures "
        f"{payload['measures']}"
    )
    path = write("e4_audit_recall", payload)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
