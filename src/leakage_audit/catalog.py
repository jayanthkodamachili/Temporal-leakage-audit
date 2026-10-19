"""The example pipeline, written the way a real one accumulates rather than as a demo.

Every leaking feature here is one somebody would write on purpose, with a reason that
sounds good at the time. That is the point: a catalogue of obviously broken definitions
would prove nothing, because nobody ships those.
"""

from __future__ import annotations

from leakage_audit.features import EVENTS, TARGET, FeatureSpec, Pipeline


def example_pipeline(horizon: int = 1) -> Pipeline:
    """A seven feature pipeline with four problems, three of them plausible."""
    return Pipeline(
        horizon=horizon,
        features=(
            FeatureSpec(
                name="events_mean_7d",
                source=EVENTS,
                window_start=-7,
                window_end=-1,
                description="trailing weekly mean, the feature everyone starts with",
            ),
            FeatureSpec(
                name="events_lag_1",
                source=EVENTS,
                window_start=-1,
                window_end=-1,
                description="yesterday's value",
            ),
            FeatureSpec(
                name="target_lag_1",
                source=TARGET,
                window_start=-1,
                window_end=-1,
                description="yesterday's outcome, a legitimate autoregressive term",
            ),
            FeatureSpec(
                name="events_centered_7d",
                source=EVENTS,
                window_start=-3,
                window_end=3,
                description=(
                    "centred weekly mean, written to smooth the series without a phase "
                    "shift, which is correct for a chart and fatal for a feature"
                ),
            ),
            FeatureSpec(
                name="events_same_day",
                source=EVENTS,
                window_start=0,
                window_end=0,
                ingestion_lag=2,
                description=(
                    "today's value, available in the warehouse at training time and not "
                    "for another two days at serving time"
                ),
            ),
            FeatureSpec(
                name="target_rolling_3d",
                source=TARGET,
                window_start=-1,
                window_end=1,
                description=(
                    "rolling outcome mean, built from a dataframe that already held the "
                    "label column"
                ),
            ),
            FeatureSpec(
                name="momentum_ratio",
                source=EVENTS,
                window_start=-7,
                window_end=-1,
                depends_on=("events_mean_7d", "events_centered_7d"),
                description=(
                    "ratio of the trailing mean to the smoothed mean; its own window is "
                    "impeccable and it inherits a leak from upstream"
                ),
            ),
        ),
    )


def honest_pipeline(horizon: int = 1) -> Pipeline:
    """The same idea with every definition repaired, for the comparison."""
    return Pipeline(
        horizon=horizon,
        features=(
            FeatureSpec(
                name="events_mean_7d", source=EVENTS, window_start=-7, window_end=-1,
                description="trailing weekly mean",
            ),
            FeatureSpec(
                name="events_lag_1", source=EVENTS, window_start=-1, window_end=-1,
                description="yesterday's value",
            ),
            FeatureSpec(
                name="target_lag_1", source=TARGET, window_start=-1, window_end=-1,
                description="yesterday's outcome",
            ),
            FeatureSpec(
                name="events_trailing_7d", source=EVENTS, window_start=-7, window_end=-1,
                description="the centred mean, shifted back so it reads only the past",
            ),
            FeatureSpec(
                name="events_lag_3", source=EVENTS, window_start=-3, window_end=-3,
                ingestion_lag=2,
                description="today's value, lagged past its own ingestion delay",
            ),
            FeatureSpec(
                name="target_mean_3d", source=TARGET, window_start=-3, window_end=-1,
                description="rolling outcome mean, ending before the label",
            ),
            FeatureSpec(
                name="momentum_ratio", source=EVENTS, window_start=-7, window_end=-1,
                depends_on=("events_mean_7d", "events_trailing_7d"),
                description="ratio of two trailing means",
            ),
        ),
    )
