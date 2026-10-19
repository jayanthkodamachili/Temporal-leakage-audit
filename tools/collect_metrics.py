"""Re-measure every published figure and write docs/metrics.json."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_DIR = ROOT / "docs" / "experiments"
METRICS_PATH = ROOT / "docs" / "metrics.json"

EXPERIMENTS = (
    "e1_splitter_does_not_help",
    "e2_which_leak_pays",
    "e3_horizon",
    "e4_audit_recall",
)

CHECKED_DOCUMENTS = (
    "README.md",
    "docs/adr/ADR-005-what-this-cannot-measure.md",
)

TABLE_ANCHOR = ["| {} |"]

ANCHORS: dict[str, list[str]] = {
    "tests_total": ["{} tests"],
    "coverage_line_pct": ["{} percent line coverage", "{}% line coverage"],
    "declared_walk_forward": ["reports {}", "{} on the features"],
    "asof_walk_forward": ["actually {}", "is {}"],
    "inflation": ["worth {} r2", "inflates it by {}"],
    "recovery": ["recovers {}", "recovers only {}"],
    "recovery_share": ["{} of the inflation", "recovers {}"],
    "honest_inflation": ["shows {} inflation", "inflation of {}"],
    "top_leak_worth": ["worth {} on its own", "accounts for {}"],
    "top_leak_name": ["{} accounts", "feature {}"],
    "second_leak_worth": ["next worth {}", "at most {}"],
    "full_score": ["full pipeline scores {}", "scores {}"],
    "all_leaks_removed": ["falls to {}", "down to {}"],
    "horizon_one_inflation": ["{} at horizon one", "worth {} where"],
    "horizon_rest_inflation": ["{} at every other", "at most {} elsewhere"],
    "horizon_ratio": ["ratio of {}", "{} from identical"],
    "horizons_hitting_label": ["{} of 5 horizons", "{} of the five"],
    "audit_recall": ["recall of {}", "recall {}"],
    "audit_precision": ["precision of {}", "precision {}"],
    "injected_leaks": ["{} injected leaks", "across {} leaks"],
    "clean_flagged": ["{} were flagged", "flagged {} of"],
    "clean_trials": ["{} pipelines with nothing", "of {} clean"],
    "example_violations": ["{} of 7 features", "finds {} violations"],
    "example_features": ["pipeline of {} features", "{} features"],
}


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, check=False, **kwargs)


def measure_suite(skip_tests: bool) -> dict:
    junit, cov = ROOT / "junit.xml", ROOT / "coverage.json"
    if skip_tests:
        missing = [p.name for p in (junit, cov) if not p.exists()]
        if missing:
            raise SystemExit(f"--skip-tests given but {', '.join(missing)} not found")
    else:
        result = run(
            [sys.executable, "-m", "pytest", f"--junitxml={junit}",
             "--cov=leakage_audit", "--cov-report=json", "-q"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(result.stdout[-4000:], file=sys.stderr)
            raise SystemExit("test suite failed, refusing to publish metrics")

    suite = ET.parse(junit).getroot()
    node = suite if suite.tag == "testsuite" else suite.find("testsuite")
    total = int(node.get("tests", 0)) - int(node.get("skipped", 0))
    coverage = json.loads(cov.read_text())["totals"]["percent_covered"]
    return {"tests_total": total, "coverage_line_pct": round(coverage, 1)}


def ensure_experiments(rerun: bool) -> dict[str, dict]:
    payloads = {}
    for name in EXPERIMENTS:
        path = EXPERIMENT_DIR / f"{name}.json"
        if rerun or not path.exists():
            print(f"running experiment {name}")
            if run([sys.executable, str(ROOT / "experiments" / f"{name}.py")]).returncode:
                raise SystemExit(f"experiment {name} failed")
        payloads[name] = json.loads(path.read_text())
    return payloads


def derive(suite: dict, exp: dict[str, dict]) -> dict[str, float | int | str]:
    e1, e2 = exp["e1_splitter_does_not_help"], exp["e2_which_leak_pays"]
    e3, e4 = exp["e3_horizon"], exp["e4_audit_recall"]

    by_worth = sorted(e2["cells"], key=lambda c: c["worth"], reverse=True)
    rest = [c["inflation"] for c in e3["cells"][1:]]

    metrics: dict[str, float | int | str] = {
        "tests_total": suite["tests_total"],
        "coverage_line_pct": suite["coverage_line_pct"],
        "declared_walk_forward": round(e1["declared_walk_forward"], 4),
        "asof_walk_forward": round(e1["asof_walk_forward"], 4),
        "inflation": round(e1["inflation"], 4),
        "recovery": round(e1["recovery"], 4),
        "recovery_share": round(e1["recovery_share"], 3),
        "honest_inflation": round(e1["honest_inflation"], 4),
        "top_leak_name": by_worth[0]["feature"],
        "top_leak_worth": round(by_worth[0]["worth"], 4),
        "second_leak_worth": round(by_worth[1]["worth"], 4),
        "full_score": round(e2["full_score"], 4),
        "all_leaks_removed": round(e2["all_leaks_removed"], 4),
        "horizon_one_inflation": round(e3["shortest_inflation"], 4),
        "horizon_rest_inflation": round(max(rest), 4),
        "horizon_ratio": round(e3["worth_ratio_first_to_rest"], 0),
        "horizons_hitting_label": e3["horizons_reaching_label"],
        "audit_recall": round(e4["overall_recall"], 4),
        "audit_precision": round(e4["overall_precision"], 4),
        "injected_leaks": e4["injected_leaks_total"],
        "clean_flagged": e4["clean_pipelines_flagged"],
        "clean_trials": e4["clean_pipelines_tested"],
        "example_violations": 4,
        "example_features": 7,
    }

    for splitter_key, value in e1["leaky"]["means"].items():
        safe = splitter_key.replace("|", "_").lower()
        metrics[f"e1_{safe}"] = round(value, 4)
        ANCHORS.setdefault(f"e1_{safe}", TABLE_ANCHOR)
    for splitter_key, value in e1["honest"]["means"].items():
        safe = splitter_key.replace("|", "_").lower()
        metrics[f"e1h_{safe}"] = round(value, 4)
        ANCHORS.setdefault(f"e1h_{safe}", TABLE_ANCHOR)

    for cell in e2["cells"]:
        key = cell["feature"]
        metrics[f"e2_{key}_without"] = round(cell["score_without"], 4)
        metrics[f"e2_{key}_worth"] = round(cell["worth"], 4)
        ANCHORS.setdefault(f"e2_{key}_without", TABLE_ANCHOR)
        ANCHORS.setdefault(f"e2_{key}_worth", TABLE_ANCHOR)

    for cell in e3["cells"]:
        key = cell["horizon"]
        metrics[f"e3_{key}_declared"] = round(cell["declared"], 4)
        metrics[f"e3_{key}_honest"] = round(cell["honest"], 4)
        metrics[f"e3_{key}_inflation"] = round(cell["inflation"], 4)
        for suffix in ("declared", "honest", "inflation"):
            ANCHORS.setdefault(f"e3_{key}_{suffix}", TABLE_ANCHOR)

    for cell in e4["cells"]:
        key = cell["pipeline_size"]
        metrics[f"e4_{key}_found"] = cell["true_positives"]
        metrics[f"e4_{key}_recall"] = round(cell["recall"], 4)
        ANCHORS.setdefault(f"e4_{key}_found", TABLE_ANCHOR)
        ANCHORS.setdefault(f"e4_{key}_recall", TABLE_ANCHOR)

    return metrics


SIGNED = ("_worth", "inflation", "recovery")
THREE_PLACE = ("recovery_share",)


def display_strings(metrics: dict) -> dict[str, str]:
    """Format declared per metric, because a figure that renders two ways defeats the
    anchor mechanism entirely."""
    out = {}
    for key, value in metrics.items():
        if isinstance(value, str):
            out[key] = value
        elif isinstance(value, int):
            out[key] = str(value)
        elif key == "coverage_line_pct":
            out[key] = f"{value:.0f}" if float(value).is_integer() else f"{value:.1f}"
        elif key == "horizon_ratio":
            out[key] = f"{value:.0f}"
        elif any(key.endswith(s) for s in THREE_PLACE):
            out[key] = f"{value:.3f}"
        elif any(key.endswith(s) or s in key for s in SIGNED):
            out[key] = f"{value:+.4f}"
        else:
            out[key] = f"{value:.4f}"
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-measure every published figure.")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    suite = measure_suite(skip_tests=args.skip_tests)
    metrics = derive(suite, ensure_experiments(rerun=not args.reuse))

    METRICS_PATH.write_text(json.dumps({
        "metrics": metrics,
        "display": display_strings(metrics),
        "anchors": ANCHORS,
        "checked_documents": list(CHECKED_DOCUMENTS),
        "note": "every value here is produced by running the suite and the four experiments",
    }, indent=2) + "\n")
    print(f"wrote {METRICS_PATH} with {len(metrics)} metrics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
