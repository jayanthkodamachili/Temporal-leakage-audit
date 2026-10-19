"""Shared plumbing for the four experiments.

Every experiment writes a JSON file into docs/experiments and prints a table. Nothing in a
document is typed by hand, so these two outputs are the only place a published number can
come from.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "experiments"


def write(name: str, payload: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _row(cells: list[str], widths: list[int]) -> str:
    return "".join(
        cell.ljust(width) if i == 0 else cell.rjust(width)
        for i, (cell, width) in enumerate(zip(cells, widths, strict=True))
    )


def table(headers: list[str], rows: list[list[str]], widths: list[int]) -> str:
    out = [_row(headers, widths), "-" * sum(widths)]
    out.extend(_row(row, widths) for row in rows)
    return "\n".join(out)
