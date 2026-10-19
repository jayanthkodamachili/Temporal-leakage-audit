# ADR-003: one feature, one reason

## Status

Accepted, after the first version did the opposite.

## Context

A feature can fail several checks at once. A rolling target mean that also arrives late
violates the target rule and the ingestion rule, and a leaking feature built from another
leaking feature could be reported both for its own window and for what it inherits.

## Decision

Each feature contributes at most one violation from the availability checks, chosen by
severity, and a feature with its own violation is never also reported as inheriting one.

## What the first version did

It walked ancestors for every feature unconditionally. A chain of three leaking features
produced six entries: three direct and three inherited, describing three problems. The
report doubled in length without containing anything more, and the most valuable line in
it, the one feature whose own window was clean and which only failed through inheritance,
was buried among duplicates of findings already made.

## Alternatives considered

**Report everything and let the reader filter.** Defensible for a machine readable payload
and wrong for a report a human reads before a commit. Padding costs attention, and attention
is the only budget this tool spends.

**Report every violation but deduplicate in the renderer.** Rejected: the structured output
and the printed table would then disagree about how many problems exist, and the count in
the JSON is what a CI gate would threshold on.

## Consequences

A feature failing two checks shows the worse one, so fixing it can reveal a second
violation on the next run. That is an acceptable iteration cost and the severity order
makes it predictable: the target rules outrank the window rules, which outrank ingestion,
which outranks inheritance.

The rule is pinned by a test asserting that a leaking feature built from a leaking feature
appears exactly once in the report.
