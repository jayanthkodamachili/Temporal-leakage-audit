"""Command line entry point.

The default run is the audit alone, with no data and no model, because that is the claim:
this finds leakage by reading, and reading is cheap enough to put in a pre-commit hook.
The scoring comparison is behind a flag, and is there to demonstrate what the audit saves
you from rather than to do the finding.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from leakage_audit.audit import AuditReport, audit
from leakage_audit.catalog import example_pipeline, honest_pipeline
from leakage_audit.dataset import generate_panel
from leakage_audit.evaluate import SHUFFLED, WALK_FORWARD, Comparison, compare
from leakage_audit.features import SpecError

EXIT_CLEAN = 0
EXIT_LEAK_FOUND = 1
EXIT_BAD_PIPELINE = 2
EXIT_INTERNAL_ERROR = 3

PIPELINES = {"example": example_pipeline, "honest": honest_pipeline}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="leak-audit",
        description="Find feature definitions that cannot be computed at prediction time.",
    )
    parser.add_argument("--pipeline", choices=sorted(PIPELINES), default="example")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument(
        "--score",
        action="store_true",
        help="also train models to show what the leak is worth; needs data and is slow",
    )
    parser.add_argument("--entities", type=int, default=80)
    parser.add_argument("--periods", type=int, default=180)
    parser.add_argument("--noise", type=float, default=1.2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260219)
    parser.add_argument("--json", action="store_true", help="emit machine readable output")
    return parser


def render_audit(report: AuditReport) -> str:
    pipeline = report.pipeline
    lines = [
        f"pipeline of {len(pipeline)} features, horizon {pipeline.horizon}",
        "",
    ]
    if report.clean:
        lines.append("no violations: every feature is computable when it claims to be")
        return "\n".join(lines)

    lines.append(f"{'feature':<22}{'kind':<20}detail")
    lines.append("-" * 96)
    for violation in report.violations:
        lines.append(f"{violation.feature:<22}{violation.kind:<20}{violation.detail}")
    lines.append("-" * 96)
    lines.append(
        f"{len(report.offending_features)} of {len(pipeline)} features cannot be "
        f"computed at prediction time"
    )
    lines.append("safe: " + (", ".join(report.safe_features) or "none"))
    return "\n".join(lines)


def render_scores(comparison: Comparison) -> str:
    lines = [
        "",
        f"{'':<18}{'as declared':>14}{'as-of correct':>16}",
        "-" * 48,
    ]
    for splitter, label in ((SHUFFLED, "shuffled k-fold"), (WALK_FORWARD, "walk-forward")):
        declared = comparison.get(splitter, False).r2
        honest = comparison.get(splitter, True).r2
        lines.append(f"{label:<18}{declared:>14.4f}{honest:>16.4f}")
    lines.append("-" * 48)
    lines.append(
        f"the leak is worth {comparison.inflation:+.4f} r2 under walk-forward validation"
    )
    lines.append(
        f"switching splitter recovers {comparison.splitter_recovery:+.4f} of that, "
        f"which is {comparison.recovery_share:.1%} of the inflation"
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        pipeline = PIPELINES[args.pipeline](horizon=args.horizon)
    except SpecError as exc:
        print(f"invalid pipeline: {exc}", file=sys.stderr)
        return EXIT_BAD_PIPELINE

    try:
        report = audit(pipeline)
        comparison = None
        if args.score:
            panel = generate_panel(args.entities, args.periods, args.noise, args.seed)
            comparison = compare(pipeline, panel, folds=args.folds, seed=args.seed)
    except Exception as exc:  # noqa: BLE001
        print(f"audit failed to complete: {exc}", file=sys.stderr)
        return EXIT_INTERNAL_ERROR

    if args.json:
        payload = report.as_dict()
        if comparison is not None:
            payload["comparison"] = comparison.as_dict()
        print(json.dumps(payload, indent=2))
    else:
        print(render_audit(report))
        if comparison is not None:
            print(render_scores(comparison))

    return EXIT_CLEAN if report.clean else EXIT_LEAK_FOUND


if __name__ == "__main__":
    raise SystemExit(main())
