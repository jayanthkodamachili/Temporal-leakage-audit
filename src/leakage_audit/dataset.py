"""A seeded panel, and the two ways of materialising features from one specification.

The repository needs to compare what a leaking pipeline reports against what an honest one
would have reported, on the same data, with the same model. That requires building the
features twice: once as the specification literally describes them, and once clamped so no
window reaches past the prediction moment.

Only a generator can support this. A production feature table has already been built one
way, and the as-of correct version of it does not exist to compare against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from leakage_audit.features import TARGET, FeatureSpec, Pipeline


@dataclass(frozen=True)
class Panel:
    """Raw entity-by-time arrays, before any feature is computed."""

    events: np.ndarray
    target: np.ndarray
    n_entities: int
    n_periods: int

    def source(self, name: str) -> np.ndarray:
        return self.target if name == TARGET else self.events


def generate_panel(
    n_entities: int, n_periods: int, noise: float, seed: int
) -> Panel:
    """Draw a panel where the target genuinely depends only on the recent past.

    The dependence is deliberately real and deliberately modest. If the target were pure
    noise every honest model would score zero and the inflation from leakage would be the
    only signal in the table, which would overstate the case. A target with real but
    limited predictability is the situation a forecasting team is actually in.
    """
    rng = np.random.default_rng(seed)

    events = np.cumsum(rng.normal(size=(n_entities, n_periods)), axis=1) * 0.3
    target = np.zeros((n_entities, n_periods))
    disturbance = rng.normal(size=(n_entities, n_periods))

    for period in range(2, n_periods):
        target[:, period] = (
            0.6 * events[:, period - 1]
            - 0.3 * events[:, period - 2]
            + noise * disturbance[:, period]
        )

    return Panel(
        events=events, target=target, n_entities=n_entities, n_periods=n_periods
    )


def _window_mean(
    array: np.ndarray, period: int, start: int, end: int
) -> np.ndarray:
    """Mean over the inclusive window, clipped to the panel's edges."""
    low = max(0, period + start)
    high = min(array.shape[1] - 1, period + end)
    if low > high:
        return np.zeros(array.shape[0])
    return array[:, low : high + 1].mean(axis=1)


def _effective_window(feature: FeatureSpec, as_of: bool) -> tuple[int, int]:
    """The window actually read, under one of the two materialisation rules.

    Under as_of the window is clamped so its newest point is the last offset genuinely
    available: back by the ingestion lag, and never past the prediction moment. A feature
    whose whole window sits in the future collapses to nothing, which is the honest
    representation of a value that cannot be computed.
    """
    if not as_of:
        return feature.window_start, feature.window_end
    latest_allowed = -feature.ingestion_lag
    end = min(feature.window_end, latest_allowed)
    start = min(feature.window_start, end)
    return start, end


def materialise(
    pipeline: Pipeline, panel: Panel, as_of: bool, burn_in: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the design matrix, the target vector, and the period index.

    Rows are stacked entity-major within each period so the period index is sorted, which
    is what a walk-forward splitter needs and what a shuffled splitter ignores.
    """
    horizon = pipeline.horizon
    first = max(burn_in, -min(f.window_start for f in pipeline.features) + 1)
    last = panel.n_periods - max(1, max(f.window_end for f in pipeline.features)) - horizon

    if last <= first:
        raise ValueError(
            "panel is too short for this pipeline's windows and horizon; "
            f"usable periods would run from {first} to {last}"
        )

    blocks, targets, periods = [], [], []
    for period in range(first, last):
        columns = []
        for feature in pipeline.features:
            start, end = _effective_window(feature, as_of)
            columns.append(_window_mean(panel.source(feature.source), period, start, end))
        blocks.append(np.column_stack(columns))
        targets.append(panel.target[:, period + horizon])
        periods.append(np.full(panel.n_entities, period))

    return np.vstack(blocks), np.concatenate(targets), np.concatenate(periods)
