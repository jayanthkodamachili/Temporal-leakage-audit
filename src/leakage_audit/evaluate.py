"""Score the same pipeline four ways, to show which of them notices anything.

The four cells are the experiment. Two splitters, shuffled and walk-forward, crossed with
two materialisations, as declared and as-of correct. A team worried about temporal leakage
reaches for the splitter, so the top row against the bottom row is the question: does
changing how you split recover what a leaking definition gave away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score

from leakage_audit.dataset import Panel, materialise
from leakage_audit.features import Pipeline

SHUFFLED = "shuffled_kfold"
WALK_FORWARD = "walk_forward"
SPLITTERS = (SHUFFLED, WALK_FORWARD)


@dataclass(frozen=True)
class Score:
    """One cell: a splitter, a materialisation, and the number it produced."""

    splitter: str
    as_of: bool
    r2: float
    folds: int

    def as_dict(self) -> dict:
        return {
            "splitter": self.splitter,
            "as_of": self.as_of,
            "r2": round(self.r2, 6),
            "folds": self.folds,
        }


@dataclass(frozen=True)
class Comparison:
    """All four cells, and the two differences worth reading."""

    scores: tuple[Score, ...]

    def get(self, splitter: str, as_of: bool) -> Score:
        for score in self.scores:
            if score.splitter == splitter and score.as_of == as_of:
                return score
        raise KeyError(f"no score for {splitter} with as_of={as_of}")

    @property
    def inflation(self) -> float:
        """What the leak is worth under the splitter people trust most.

        Walk-forward on declared features minus walk-forward on as-of features. This is
        the number a team would have shipped believing it.
        """
        return self.get(WALK_FORWARD, False).r2 - self.get(WALK_FORWARD, True).r2

    @property
    def splitter_recovery(self) -> float:
        """How much of the inflation switching to walk-forward recovers.

        Shuffled minus walk-forward, both on the declared features. If changing the
        splitter were the fix, this would be close to the inflation. It is not.
        """
        return self.get(SHUFFLED, False).r2 - self.get(WALK_FORWARD, False).r2

    @property
    def recovery_share(self) -> float:
        """Share of the inflation that changing the splitter removes."""
        if self.inflation == 0:
            return 0.0
        return self.splitter_recovery / self.inflation

    def as_dict(self) -> dict:
        return {
            "scores": [s.as_dict() for s in self.scores],
            "inflation": round(self.inflation, 6),
            "splitter_recovery": round(self.splitter_recovery, 6),
            "recovery_share": round(self.recovery_share, 6),
        }


def _model(seed: int) -> GradientBoostingRegressor:
    """One model, held fixed across all four cells.

    Nothing about the model is the subject here, so it is pinned rather than tuned: a
    tuned model would invite the reading that the inflation is a property of this
    estimator, when it is a property of the columns it was handed.
    """
    return GradientBoostingRegressor(random_state=seed, n_estimators=60, max_depth=3)


def score_one(
    pipeline: Pipeline,
    panel: Panel,
    splitter: str,
    as_of: bool,
    folds: int = 5,
    seed: int = 0,
) -> Score:
    design, target, periods = materialise(pipeline, panel, as_of=as_of)

    if splitter == WALK_FORWARD:
        order = np.argsort(periods, kind="stable")
        design, target = design[order], target[order]
        cv = TimeSeriesSplit(n_splits=folds)
    elif splitter == SHUFFLED:
        cv = KFold(n_splits=folds, shuffle=True, random_state=seed)
    else:
        raise ValueError(f"unreachable: unknown splitter {splitter!r}")

    scores = cross_val_score(_model(seed), design, target, cv=cv, scoring="r2")
    return Score(splitter=splitter, as_of=as_of, r2=float(np.mean(scores)), folds=folds)


def compare(
    pipeline: Pipeline, panel: Panel, folds: int = 5, seed: int = 0
) -> Comparison:
    """Run all four cells against one panel."""
    return Comparison(tuple(
        score_one(pipeline, panel, splitter, as_of, folds=folds, seed=seed)
        for splitter in SPLITTERS
        for as_of in (False, True)
    ))
