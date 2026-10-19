# ADR-001: the audit reads definitions and never touches data

## Status

Accepted. This is the architectural decision the repository exists to demonstrate.

## Context

Temporal leakage is usually found by noticing that a score is too good. That requires
knowing what score would have been reasonable, which is the thing nobody knows in advance,
and it requires a trained model, which means the leak is found after the work rather than
before it.

## Decision

The audit takes a pipeline specification and returns violations. It imports no model
library, builds no panel, and runs in roughly two seconds on a pipeline that has no data
behind it at all.

## Alternatives considered

**Detect leakage by comparing scores.** The approach this repository exists to argue
against. Experiment one measures why: a leaking pipeline scores the same under a
time-ordered splitter as under a shuffled one, so the comparison teams actually run
recovers under one percent of the inflation. There is no score-based test here because
there is no score-based test that works.

**Detect it by recomputing features as-of and diffing the columns.** This does work, and
it is strictly stronger than reading metadata because it catches a window declared as
trailing and implemented as centred. Rejected as the primary mechanism for a practical
reason: it needs the data, the compute and the historical state to recompute against, so
it cannot run in a pre-commit hook on a feature definition that was written an hour ago.
It is the Pivot, with its trigger condition stated.

**Combine both and call the static pass an optimisation.** Rejected because it would blur
the claim. The static pass is not a faster version of the dynamic one; it answers a
question the dynamic one cannot ask before the pipeline exists.

## Consequences

The audit trusts what a feature says about itself, and a declaration that lies audits
clean. That limitation is stated in the README's own section rather than buried, because
it is the first thing a reviewer should press on.

The constraint is enforced rather than documented: a test monkeypatches the data
generation functions to raise, and the audit still completes. If the audit ever grows a
dependency on data, that test fails and the architectural claim fails with it.
