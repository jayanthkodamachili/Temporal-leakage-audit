"""The static audit: the only instrument here that touches no data.

This is the argument of the repository in one module. Every other way of finding temporal
leakage works by noticing that a score is too good, which requires knowing what score would
have been reasonable, which is the thing nobody knows in advance. The audit instead reads
the feature definitions and asks a question that has an answer: at the moment a prediction
is made, is this value knowable.

Because it reads definitions, it runs before a model is trained, on a pipeline with no data
behind it yet, and it returns the same verdict whether the leak helped the score or not.
"""

from __future__ import annotations

from dataclasses import dataclass

from leakage_audit.features import TARGET, FeatureSpec, Pipeline

# Ordered worst first. A feature reading the target's future is a different category of
# problem from one that is merely two days early, and a report that mixes them wastes the
# reader's attention on the cheap finding.
SEVERITY_ORDER = (
    "target_future",
    "future_window",
    "ingestion_lag",
    "inherited",
    "cycle",
    "missing_dependency",
)


@dataclass(frozen=True)
class Violation:
    """One reason a feature cannot be computed when it claims to be."""

    feature: str
    kind: str
    detail: str
    offending_offset: int | None = None

    @property
    def severity_rank(self) -> int:
        return SEVERITY_ORDER.index(self.kind)

    def as_dict(self) -> dict:
        return {
            "feature": self.feature,
            "kind": self.kind,
            "detail": self.detail,
            "offending_offset": self.offending_offset,
        }


@dataclass(frozen=True)
class AuditReport:
    """What the audit found, and what it could not look at."""

    pipeline: Pipeline
    violations: tuple[Violation, ...]

    @property
    def clean(self) -> bool:
        return not self.violations

    @property
    def offending_features(self) -> tuple[str, ...]:
        return tuple(sorted({v.feature for v in self.violations}))

    @property
    def safe_features(self) -> tuple[str, ...]:
        offending = set(self.offending_features)
        return tuple(n for n in self.pipeline.names if n not in offending)

    def by_kind(self, kind: str) -> tuple[Violation, ...]:
        return tuple(v for v in self.violations if v.kind == kind)

    @property
    def worst_kind(self) -> str | None:
        return min(self.violations, key=lambda v: v.severity_rank).kind if self.violations else None

    def as_dict(self) -> dict:
        return {
            "pipeline": self.pipeline.as_dict(),
            "violations": [v.as_dict() for v in self.violations],
            "clean": self.clean,
            "offending_features": list(self.offending_features),
            "safe_features": list(self.safe_features),
            "worst_kind": self.worst_kind,
            "counts_by_kind": {
                kind: len(self.by_kind(kind))
                for kind in SEVERITY_ORDER
                if self.by_kind(kind)
            },
        }


def _own_window_violations(feature: FeatureSpec) -> list[Violation]:
    """Checks that need only the feature's own definition."""
    found: list[Violation] = []

    if feature.source == TARGET and feature.window_end >= 0:
        found.append(Violation(
            feature=feature.name,
            kind="target_future",
            detail=(
                f"reads the target through offset {feature.window_end}, which is the "
                "label itself or later"
            ),
            offending_offset=feature.window_end,
        ))
    elif feature.window_end > 0:
        found.append(Violation(
            feature=feature.name,
            kind="future_window",
            detail=(
                f"window ends at offset {feature.window_end}, after the prediction is made"
            ),
            offending_offset=feature.window_end,
        ))
    elif feature.available_from > 0:
        found.append(Violation(
            feature=feature.name,
            kind="ingestion_lag",
            detail=(
                f"window ends at offset {feature.window_end} but the source lags "
                f"{feature.ingestion_lag}, so the value lands at offset "
                f"{feature.available_from}"
            ),
            offending_offset=feature.available_from,
        ))

    return found


def audit(pipeline: Pipeline) -> AuditReport:
    """Read the pipeline and report every feature that cannot be computed in time.

    Structural problems are reported first and stop the availability walk, because
    availability is inherited through the graph and inheritance is meaningless when the
    graph has a cycle or a dangling edge.
    """
    violations: list[Violation] = []

    for feature_name, dependency in pipeline.missing_dependencies():
        violations.append(Violation(
            feature=feature_name,
            kind="missing_dependency",
            detail=f"depends on {dependency!r}, which is not in this pipeline",
        ))

    cycle = pipeline.find_cycle()
    if cycle:
        violations.append(Violation(
            feature=cycle[0],
            kind="cycle",
            detail="dependency cycle: " + " -> ".join(cycle),
        ))
        return AuditReport(pipeline=pipeline, violations=tuple(violations))

    direct: dict[str, list[Violation]] = {}
    for feature in pipeline.features:
        found = _own_window_violations(feature)
        if found:
            direct[feature.name] = found
            violations.extend(found)

    # Inheritance. A feature whose own window is impeccable is still uncomputable if
    # anything upstream of it is not ready, and this is the case that survives review:
    # the definition in front of you looks fine.
    for feature in pipeline.features:
        if feature.name in direct:
            continue
        tainted = [a for a in pipeline.ancestors(feature.name) if a in direct]
        if tainted:
            violations.append(Violation(
                feature=feature.name,
                kind="inherited",
                detail=(
                    "own window is legal, but it is computed from "
                    + ", ".join(tainted)
                    + ", which cannot be computed in time"
                ),
            ))

    violations.sort(key=lambda v: (v.severity_rank, v.feature))
    return AuditReport(pipeline=pipeline, violations=tuple(violations))
