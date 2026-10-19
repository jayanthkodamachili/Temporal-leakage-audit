"""Render the committed architecture SVG.

Hand written rather than produced by a diagramming library, for the reasons in ADR-004: a
committed SVG with intrinsic dimensions and an explicit light background renders the same
on GitHub in either theme, which a Mermaid fence does not reliably do.
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "architecture.svg"

W, H = 960, 500
INK = "#1b1f23"
MUTED = "#57606a"
BAD = "#b42318"
GOOD = "#116329"
LINE = "#8c959f"


def box(x, y, w, h, title, subtitle="", stroke=MUTED, fill="#ffffff", bold=False):
    weight = "600" if bold else "400"
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1.5"/>',
        f'<text x="{x + w / 2}" y="{y + (20 if subtitle else h / 2 + 4)}" '
        f'text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="13" '
        f'font-weight="{weight}" fill="{INK}">{title}</text>',
    ]
    if subtitle:
        for index, line in enumerate(subtitle.split("|")):
            parts.append(
                f'<text x="{x + w / 2}" y="{y + 38 + index * 14}" text-anchor="middle" '
                f'font-family="Helvetica,Arial,sans-serif" font-size="11" '
                f'fill="{MUTED}">{line}</text>'
            )
    return "".join(parts)


def arrow(x1, y1, x2, y2, dashed=False):
    dash = ' stroke-dasharray="4 3"' if dashed else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{LINE}" '
        f'stroke-width="1.5" marker-end="url(#a)"{dash}/>'
    )


def label(x, y, text, fill=MUTED, size=11, anchor="middle", weight="400"):
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Helvetica,Arial,sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">{text}</text>'
    )


def render() -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" role="img" '
        f'aria-label="Architecture of temporal-leakage-audit">',
        '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
        f'markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="{LINE}"/>'
        "</marker></defs>",
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        label(W / 2, 26, "One pipeline, two instruments, one of them blind",
              INK, 14, weight="600"),
    ]

    parts.append(box(24, 60, 200, 66, "feature definitions",
                     "window, ingestion lag,|dependency graph", bold=True))

    # Upper path: everything that needs data, and sees nothing.
    parts.append(box(300, 58, 268, 50, "materialise the features",
                     "the future goes into the column"))
    parts.append(box(300, 128, 268, 50, "split the rows",
                     "shuffled or walk-forward"))
    parts.append(box(300, 198, 268, 50, "train and score",
                     "every fold sees the same leak"))
    parts.append(arrow(224, 86, 300, 83))
    parts.append(arrow(434, 108, 434, 128))
    parts.append(arrow(434, 178, 434, 198))

    parts.append(box(640, 148, 190, 60, "a high score",
                     "no diagnostic objects", stroke=BAD, fill="#fff5f4", bold=True))
    parts.append(arrow(568, 223, 640, 196))

    # Lower path: no data at all.
    parts.append(box(300, 300, 268, 64, "read every window against zero",
                     "is this value knowable|when the prediction is made?",
                     stroke=GOOD, fill="#f2fbf4", bold=True))
    parts.append(arrow(224, 110, 300, 318))

    parts.append(box(640, 300, 190, 64, "named violations",
                     "before a model exists", stroke=GOOD, fill="#f2fbf4", bold=True))
    parts.append(arrow(568, 332, 640, 332))

    parts.append(label(434, 282, "no data, no model, no training", MUTED, 10))

    parts.append(
        box(130, 416, 700, 62,
            "the splitter is not the lever; the definition is",
            "changing how the rows are divided recovers almost none of what a leaking "
            "column gave away", bold=True)
    )
    parts.append(arrow(735, 208, 600, 416))
    parts.append(arrow(735, 364, 420, 416))

    parts.append("</svg>")
    return "".join(parts)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render() + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
