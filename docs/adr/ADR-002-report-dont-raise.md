# ADR-002: dangling dependencies are reported, not raised

## Status

Accepted.

## Context

A pipeline can name a dependency that is not in the pipeline. The strict reading is that
such a specification is malformed and should be rejected before anything else happens.

## Decision

Report a dangling edge as a violation of its own kind and continue auditing everything
else.

## Alternatives considered

**Raise on the first missing dependency.** Clean, and it makes the tool useless exactly
where it is needed. A feature pipeline assembled from several repositories almost always
has edges pointing at features defined somewhere the auditor cannot see. Refusing to look
at the other forty features because of one dangling edge means the tool gets run once and
then removed from the hook.

**Silently ignore unknown dependencies.** Rejected because an edge pointing nowhere is
itself a finding: it means the graph the audit walked is not the graph that runs, so the
inheritance results are incomplete, and the reader needs to know that.

## The one exception

A dependency cycle does stop the walk. Availability is inherited through the graph, and
inheritance has no meaning on a graph with no topological order. Reporting a confident set
of downstream findings computed from a cyclic graph would be worse than reporting the
cycle and stopping, so the audit returns immediately after recording it. A test pins that
behaviour: a pipeline with both a cycle and a leaking window reports the cycle and no
inherited violations.

## Consequences

The report can contain two quite different things: features that leak, and reasons the
audit could not fully see the graph. They are separate kinds with separate severity ranks
so a reader can tell them apart, and the counts are published per kind rather than as one
total.
