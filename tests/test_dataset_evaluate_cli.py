from __future__ import annotations

import json

import numpy as np
import pytest

from leakage_audit import cli
from leakage_audit.catalog import example_pipeline, honest_pipeline
from leakage_audit.dataset import (
    _effective_window,
    generate_panel,
    materialise,
)
from leakage_audit.evaluate import (
    SHUFFLED,
    WALK_FORWARD,
    Comparison,
    Score,
    compare,
    score_one,
)
from leakage_audit.features import TARGET, FeatureSpec, Pipeline

SMALL = ["--entities", "20", "--periods", "60", "--folds", "3", "--seed", "3"]


# ----- the panel -----

def test_a_panel_has_the_requested_shape(small_panel):
    assert small_panel.events.shape == (20, 60)
    assert small_panel.target.shape == (20, 60)


def test_panel_generation_is_reproducible():
    first = generate_panel(10, 30, 1.0, seed=7)
    second = generate_panel(10, 30, 1.0, seed=7)
    np.testing.assert_array_equal(first.target, second.target)


def test_a_different_seed_produces_a_different_panel():
    first = generate_panel(10, 30, 1.0, seed=7)
    second = generate_panel(10, 30, 1.0, seed=8)
    assert not np.allclose(first.target, second.target)


def test_the_target_really_depends_on_the_recent_past():
    """If it did not, every honest score would be zero and the leak would be the only
    signal in the table, which would overstate the case this repository makes."""
    panel = generate_panel(200, 200, noise=0.5, seed=4)
    correlation = np.corrcoef(
        panel.events[:, 10:-1].ravel(), panel.target[:, 11:].ravel()
    )[0, 1]
    assert abs(correlation) > 0.3


def test_source_dispatches_between_the_two_tables(small_panel):
    assert small_panel.source(TARGET) is small_panel.target
    assert small_panel.source("events") is small_panel.events


# ----- materialisation -----

def test_as_declared_keeps_the_window_the_spec_asked_for():
    feature = FeatureSpec(name="f", window_start=-3, window_end=3)
    assert _effective_window(feature, as_of=False) == (-3, 3)


def test_as_of_clamps_a_future_window_back_to_the_prediction_moment():
    feature = FeatureSpec(name="f", window_start=-3, window_end=3)
    assert _effective_window(feature, as_of=True) == (-3, 0)


def test_as_of_also_clamps_for_the_ingestion_lag():
    feature = FeatureSpec(name="f", window_start=-3, window_end=0, ingestion_lag=2)
    assert _effective_window(feature, as_of=True) == (-3, -2)


def test_as_of_collapses_a_window_that_lies_entirely_in_the_future():
    """The honest representation of a value that simply cannot be computed."""
    feature = FeatureSpec(name="f", window_start=2, window_end=5)
    start, end = _effective_window(feature, as_of=True)
    assert start == end == 0


def test_as_of_leaves_an_already_legal_window_alone():
    feature = FeatureSpec(name="f", window_start=-7, window_end=-1)
    assert _effective_window(feature, as_of=True) == (-7, -1)


def test_materialise_returns_aligned_arrays(small_panel):
    design, target, periods = materialise(example_pipeline(), small_panel, as_of=False)
    assert design.shape[0] == target.shape[0] == periods.shape[0]
    assert design.shape[1] == len(example_pipeline())


def test_the_period_index_is_sorted_so_a_walk_forward_split_means_something(small_panel):
    _, _, periods = materialise(example_pipeline(), small_panel, as_of=False)
    assert np.all(np.diff(periods) >= 0)


def test_the_two_materialisations_differ_for_a_leaking_pipeline(small_panel):
    declared, _, _ = materialise(example_pipeline(), small_panel, as_of=False)
    honest, _, _ = materialise(example_pipeline(), small_panel, as_of=True)
    assert not np.allclose(declared, honest)


def test_the_two_materialisations_agree_for_an_honest_pipeline(small_panel):
    """Clamping a window that never reached past the prediction changes nothing.

    This is what makes the as-of path a correction rather than a different feature set.
    """
    declared, _, _ = materialise(honest_pipeline(), small_panel, as_of=False)
    honest, _, _ = materialise(honest_pipeline(), small_panel, as_of=True)
    np.testing.assert_allclose(declared, honest)


def test_a_panel_too_short_for_the_windows_is_refused():
    tiny = generate_panel(5, 12, 1.0, seed=1)
    with pytest.raises(ValueError, match="too short"):
        materialise(example_pipeline(), tiny, as_of=False)


def test_a_window_reaching_before_the_panel_starts_is_clipped_not_crashed(small_panel):
    pipeline = Pipeline(features=(FeatureSpec(name="long", window_start=-40, window_end=-1),))
    design, _, _ = materialise(pipeline, small_panel, as_of=False)
    assert np.all(np.isfinite(design))


# ----- evaluation -----

def test_a_leaking_pipeline_scores_higher_than_its_honest_materialisation(small_panel):
    declared = score_one(example_pipeline(), small_panel, WALK_FORWARD, as_of=False, folds=3)
    honest = score_one(example_pipeline(), small_panel, WALK_FORWARD, as_of=True, folds=3)
    assert declared.r2 > honest.r2


def test_changing_the_splitter_does_not_recover_the_inflation(small_panel):
    """The finding the repository exists for, as a test rather than only an experiment.

    Walk-forward validation is the standard advice for temporal leakage. It addresses
    leakage through the split and is blind to leakage in the definition, so it recovers
    almost none of what a leaking feature gave away.
    """
    comparison = compare(example_pipeline(), small_panel, folds=3)
    assert comparison.inflation > 0.05
    assert abs(comparison.recovery_share) < 0.25


def test_an_honest_pipeline_shows_no_inflation(small_panel):
    comparison = compare(honest_pipeline(), small_panel, folds=3)
    assert abs(comparison.inflation) < 1e-9


def test_a_comparison_holds_all_four_cells(small_panel):
    comparison = compare(honest_pipeline(), small_panel, folds=3)
    assert len(comparison.scores) == 4


def test_asking_for_a_cell_that_does_not_exist_raises(small_panel):
    comparison = compare(honest_pipeline(), small_panel, folds=3)
    with pytest.raises(KeyError, match="no score for"):
        comparison.get("bogus_splitter", as_of=True)


def test_an_unknown_splitter_is_refused(small_panel):
    with pytest.raises(ValueError, match="unreachable"):
        score_one(honest_pipeline(), small_panel, "vibes", as_of=True, folds=3)


def test_recovery_share_is_zero_when_there_is_no_inflation():
    """Guards the division. An honest pipeline has nothing to recover."""
    scores = tuple(
        Score(splitter=splitter, as_of=as_of, r2=0.5, folds=3)
        for splitter in (SHUFFLED, WALK_FORWARD)
        for as_of in (False, True)
    )
    assert Comparison(scores).recovery_share == 0.0


def test_scoring_is_reproducible(small_panel):
    first = score_one(honest_pipeline(), small_panel, SHUFFLED, as_of=True, folds=3, seed=2)
    second = score_one(honest_pipeline(), small_panel, SHUFFLED, as_of=True, folds=3, seed=2)
    assert first.r2 == second.r2


# ----- cli -----

def test_parser_defaults_to_auditing_without_data():
    args = cli.build_parser().parse_args([])
    assert args.score is False
    assert args.pipeline == "example"


def test_a_leaking_pipeline_exits_one(capsys):
    code = cli.main([])
    capsys.readouterr()
    assert code == cli.EXIT_LEAK_FOUND


def test_a_clean_pipeline_exits_zero(capsys):
    code = cli.main(["--pipeline", "honest"])
    out = capsys.readouterr().out
    assert code == cli.EXIT_CLEAN
    assert "no violations" in out


def test_an_invalid_pipeline_exits_two_without_a_traceback(capsys):
    code = cli.main(["--horizon", "0"])
    captured = capsys.readouterr()
    assert code == cli.EXIT_BAD_PIPELINE
    assert "invalid pipeline" in captured.err
    assert "Traceback" not in captured.err


def test_an_internal_failure_exits_three(monkeypatch, capsys):
    def explode(*_args, **_kwargs):
        raise RuntimeError("sklearn went home")

    monkeypatch.setattr(cli, "audit", explode)
    code = cli.main([])
    assert code == cli.EXIT_INTERNAL_ERROR
    assert "failed to complete" in capsys.readouterr().err


def test_every_exit_code_is_distinct():
    codes = {
        cli.EXIT_CLEAN, cli.EXIT_LEAK_FOUND,
        cli.EXIT_BAD_PIPELINE, cli.EXIT_INTERNAL_ERROR,
    }
    assert len(codes) == 4


def test_the_audit_table_names_every_violating_feature(capsys):
    cli.main([])
    out = capsys.readouterr().out
    for name in ("target_rolling_3d", "events_centered_7d", "events_same_day",
                 "momentum_ratio"):
        assert name in out


def test_the_score_flag_adds_the_four_cell_table(capsys):
    cli.main([*SMALL, "--score"])
    out = capsys.readouterr().out
    assert "as declared" in out
    assert "as-of correct" in out
    assert "switching splitter recovers" in out


def test_json_mode_is_parseable_and_omits_scores_by_default(capsys):
    cli.main(["--json"])
    payload = json.loads(capsys.readouterr().out)
    assert "comparison" not in payload
    assert payload["clean"] is False


def test_json_mode_includes_scores_when_asked(capsys):
    cli.main([*SMALL, "--json", "--score"])
    payload = json.loads(capsys.readouterr().out)
    assert "inflation" in payload["comparison"]


def test_an_unknown_pipeline_name_is_rejected_by_the_parser():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--pipeline", "imaginary"])


def test_a_window_entirely_past_the_end_of_the_panel_yields_zeros(small_panel):
    """The clip can leave no periods at all, and returning zeros beats an empty mean.

    numpy's mean of an empty slice is nan with a warning, and a nan column silently
    poisons every downstream score.
    """
    from leakage_audit.dataset import _window_mean

    column = _window_mean(small_panel.events, period=59, start=5, end=9)
    assert column.shape == (20,)
    assert np.all(column == 0.0)
