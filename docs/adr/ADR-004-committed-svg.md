# ADR-004: a committed SVG rather than a Mermaid fence

## Status

Accepted.

## Context

The README needs an architecture diagram. GitHub renders Mermaid fences natively, which is
the path of least effort and keeps the diagram source in the document.

## Decision

Render the diagram with `tools/render_diagram.py` into a committed `docs/architecture.svg`,
embed it with an `img` tag, and have CI regenerate it and fail if the committed copy has
drifted.

## Alternatives considered

**A Mermaid fence.** Rejected on reliability. GitHub's Mermaid support is a lazily loaded
client side renderer, and when it declines to run, the reader sees an error string where the
diagram should be, with nothing fixable in the source. A diagram that sometimes does not
appear is worse than a plain one that always does. The committed SVG also renders in an
editor preview, in a PDF export and offline.

**A Mermaid fence rendered to SVG at build time.** Closer, and rejected on two specifics.
Mermaid with HTML labels emits unclosed tags inside a `foreignObject`, which is not well
formed XML and silently fails to load as an `img` source. It also emits a percentage width
with no height, so an embedded copy has no aspect ratio to work from. Both are fixable by
post processing the output, which is more machinery than drawing the boxes here.

**A diagramming library such as graphviz.** Adds a system binary to every contributor's
machine and to CI, for a diagram whose layout is being chosen by hand anyway.

## Consequences

The diagram is written as Python that emits SVG, which is more tedious to edit than Mermaid
text. The background is painted white explicitly rather than left transparent, because
transparent is not theme neutral: muted label text with no fill behind it comes out nearly
invisible against a dark theme. The layout is hand positioned, so a new box means moving
coordinates. Acceptable for a diagram that is not going to grow.
