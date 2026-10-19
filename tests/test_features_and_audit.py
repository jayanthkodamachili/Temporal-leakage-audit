"""The audit is the product, so this file is the longest in the repository.

Every test names a property of the static analysis. None of them train a model or touch a
panel, which is the same claim the tool makes: these questions are answerable by reading.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from leakage_audit.audit import SEVERITY_ORDER, audit
from leakage_audit.catalog import example_pipeline, honest_pipeline
from leakage_audit.features import EVENTS, TARGET, FeatureSpec, Pipeline, SpecError


def spec(name: str, **kwargs) -> FeatureSpec:
    return FeatureSpec(name=name, **kwargs)


def one(*features: FeatureSpec) -> Pipeline:
    return Pipeline(features=features)


# ----- the spec -----

def test_a_feature_needs_a_name():
    with pytest.raises(SpecError, match="needs a name"):
        FeatureSpec(name="")


def test_an_unknown_source_is_rejected():
    with pytest.raises(SpecError, match="unknown source"):
        spec("f", source="warehouse")


def test_a_backwards_window_is_rejected():
    """A window whose start is after its end describes nothing at all."""
    with pytest.raises(SpecError, match="no window at all"):
        spec("f", window_start=-1, window_end=-7)


def test_a_negative_ingestion_lag_is_rejected():
    with pytest.raises(SpecError, match="ingestion_lag"):
        spec("f", ingestion_lag=-2)


def test_a_feature_cannot_depend_on_itself():
    with pytest.raises(SpecError, match="depend on itself"):
        spec("f", depends_on=("f",))


def test_specs_are_frozen(small_spec):
    with pytest.raises(FrozenInstanceError):
        small_spec.window_end = 5  # type: ignore[misc]


def test_available_from_adds_the_lag_to_the_window_end():
    """The whole availability test is this quantity against zero."""
    assert spec("f", window_end=-1, ingestion_lag=3).available_from == 2
    assert spec("f", window_end=-5, ingestion_lag=2).available_from == -3


def test_a_feature_with_no_lag_is_available_when_its_window_closes():
    assert spec("f", window_end=-1, ingestion_lag=0).available_from == -1


# ----- the pipeline graph -----

def test_duplicate_feature_names_are_rejected():
    with pytest.raises(SpecError, match="duplicate feature names"):
        one(spec("a"), spec("a"))


def test_a_horizon_of_zero_is_rejected():
    """Predicting the current period is not forecasting, it is reporting."""
    with pytest.raises(SpecError, match="horizon"):
        Pipeline(features=(spec("a"),), horizon=0)


def test_getting_an_absent_feature_raises():
    with pytest.raises(SpecError, match="no feature named"):
        one(spec("a")).get("b")


def test_names_are_reported_in_declaration_order():
    assert one(spec("b"), spec("a")).names == ("b", "a")


def test_a_dangling_dependency_is_reported_rather_than_raised():
    """A pipeline assembled from several repositories often has dangling edges.

    Refusing to audit it would mean the tool is unusable exactly where it is needed most,
    so this is a finding rather than a crash.
    """
    pipeline = one(spec("a", depends_on=("ghost",)))
    assert pipeline.missing_dependencies() == (("a", "ghost"),)


def test_a_complete_graph_reports_no_missing_dependencies():
    assert one(spec("a"), spec("b", depends_on=("a",))).missing_dependencies() == ()


def test_a_two_node_cycle_is_found():
    pipeline = Pipeline(features=(
        FeatureSpec(name="a", depends_on=("b",)),
        FeatureSpec(name="b", depends_on=("a",)),
    ))
    assert pipeline.find_cycle() != ()


def test_a_longer_cycle_is_found():
    pipeline = Pipeline(features=(
        FeatureSpec(name="a", depends_on=("b",)),
        FeatureSpec(name="b", depends_on=("c",)),
        FeatureSpec(name="c", depends_on=("a",)),
    ))
    cycle = pipeline.find_cycle()
    assert set(cycle) >= {"a", "b", "c"}


def test_an_acyclic_graph_reports_no_cycle():
    pipeline = Pipeline(features=(
        FeatureSpec(name="a"),
        FeatureSpec(name="b", depends_on=("a",)),
        FeatureSpec(name="c", depends_on=("a", "b")),
    ))
    assert pipeline.find_cycle() == ()


def test_a_diamond_is_not_mistaken_for_a_cycle():
    """Two paths to one ancestor is normal, and a naive traversal calls it a cycle."""
    pipeline = Pipeline(features=(
        FeatureSpec(name="root"),
        FeatureSpec(name="left", depends_on=("root",)),
        FeatureSpec(name="right", depends_on=("root",)),
        FeatureSpec(name="join", depends_on=("left", "right")),
    ))
    assert pipeline.find_cycle() == ()


def test_a_dangling_edge_does_not_confuse_cycle_detection():
    assert one(spec("a", depends_on=("ghost",))).find_cycle() == ()


def test_ancestors_are_transitive():
    pipeline = Pipeline(features=(
        FeatureSpec(name="a"),
        FeatureSpec(name="b", depends_on=("a",)),
        FeatureSpec(name="c", depends_on=("b",)),
    ))
    assert pipeline.ancestors("c") == ("a", "b")


def test_a_feature_with_no_dependencies_has_no_ancestors():
    assert one(spec("a")).ancestors("a") == ()


def test_ancestors_ignore_edges_pointing_outside_the_pipeline():
    assert one(spec("a", depends_on=("ghost",))).ancestors("a") == ()


# ----- the audit -----

def test_a_trailing_window_is_clean():
    assert audit(one(spec("a", window_start=-7, window_end=-1))).clean is True


def test_a_window_ending_after_the_prediction_is_a_future_window_violation():
    report = audit(one(spec("a", window_start=-3, window_end=3)))
    assert report.by_kind("future_window")[0].offending_offset == 3


def test_a_window_ending_exactly_at_the_prediction_moment_is_allowed_for_events():
    """Offset zero is the prediction moment itself, and an event already recorded there
    is legitimately knowable. It is the ingestion lag that usually makes it unusable."""
    assert audit(one(spec("a", window_start=-1, window_end=0))).clean is True


def test_an_ingestion_lag_that_outruns_the_window_is_a_violation():
    report = audit(one(spec("a", window_start=-1, window_end=-1, ingestion_lag=3)))
    violation = report.by_kind("ingestion_lag")[0]
    assert violation.offending_offset == 2


def test_an_ingestion_lag_inside_the_window_is_clean():
    assert audit(one(spec("a", window_end=-3, ingestion_lag=2))).clean is True


def test_a_lag_landing_exactly_at_the_prediction_moment_is_clean():
    """Available at offset zero means available when the prediction is made."""
    assert audit(one(spec("a", window_end=-2, ingestion_lag=2))).clean is True


def test_reading_the_target_at_the_prediction_moment_is_the_worst_kind():
    report = audit(one(spec("a", source=TARGET, window_start=0, window_end=0)))
    assert report.worst_kind == "target_future"


def test_a_lagged_target_is_legitimate_and_passes():
    """Autoregression is not leakage, and a tool that flagged it would be ignored."""
    assert audit(one(spec("a", source=TARGET, window_start=-1, window_end=-1))).clean


def test_a_target_window_straddling_the_label_is_caught():
    report = audit(one(spec("a", source=TARGET, window_start=-1, window_end=1)))
    assert report.by_kind("target_future")[0].offending_offset == 1


def test_an_events_feature_reading_the_future_is_not_called_a_target_leak():
    """The two are different problems and the report must not blur them."""
    report = audit(one(spec("a", source=EVENTS, window_start=0, window_end=2)))
    assert report.by_kind("target_future") == ()
    assert report.by_kind("future_window") != ()


def test_a_clean_feature_built_from_a_leaking_one_inherits_the_violation():
    """The case that survives code review, because the definition in front of you is fine."""
    pipeline = Pipeline(features=(
        FeatureSpec(name="leaky", window_start=-1, window_end=2),
        FeatureSpec(name="derived", window_start=-7, window_end=-1, depends_on=("leaky",)),
    ))
    report = audit(pipeline)
    assert report.by_kind("inherited")[0].feature == "derived"


def test_inheritance_reaches_through_an_intermediate_feature():
    pipeline = Pipeline(features=(
        FeatureSpec(name="leaky", window_start=-1, window_end=2),
        FeatureSpec(name="middle", window_start=-7, window_end=-1, depends_on=("leaky",)),
        FeatureSpec(name="outer", window_start=-7, window_end=-1, depends_on=("middle",)),
    ))
    assert {v.feature for v in audit(pipeline).by_kind("inherited")} == {"middle", "outer"}


def test_a_feature_with_its_own_violation_is_not_also_reported_as_inherited():
    """One feature, one reason, or the report pads itself and the reader stops trusting it."""
    pipeline = Pipeline(features=(
        FeatureSpec(name="leaky", window_start=-1, window_end=2),
        FeatureSpec(name="also_leaky", window_start=-1, window_end=5, depends_on=("leaky",)),
    ))
    report = audit(pipeline)
    assert [v.feature for v in report.by_kind("inherited")] == []
    assert len([v for v in report.violations if v.feature == "also_leaky"]) == 1


def test_a_dangling_dependency_is_reported_by_the_audit():
    report = audit(one(spec("a", depends_on=("ghost",))))
    assert report.by_kind("missing_dependency")[0].detail.endswith("not in this pipeline")


def test_a_cycle_stops_the_availability_walk():
    """Inheritance is meaningless on a graph that has no order, so the audit says so
    and stops rather than reporting a confident set of downstream findings."""
    pipeline = Pipeline(features=(
        FeatureSpec(name="a", window_end=5, depends_on=("b",)),
        FeatureSpec(name="b", depends_on=("a",)),
    ))
    report = audit(pipeline)
    assert report.by_kind("cycle") != ()
    assert report.by_kind("inherited") == ()


def test_violations_are_ordered_worst_first():
    report = audit(example_pipeline())
    ranks = [v.severity_rank for v in report.violations]
    assert ranks == sorted(ranks)


def test_the_severity_order_puts_the_target_leak_first():
    assert SEVERITY_ORDER[0] == "target_future"


def test_safe_and_offending_features_partition_the_pipeline():
    report = audit(example_pipeline())
    assert set(report.safe_features) | set(report.offending_features) == set(
        report.pipeline.names
    )
    assert not set(report.safe_features) & set(report.offending_features)


def test_the_example_pipeline_has_one_violation_of_each_kind():
    """The catalogue exists to exercise every rule, so it is pinned."""
    report = audit(example_pipeline())
    kinds = {v.kind for v in report.violations}
    assert kinds == {"target_future", "future_window", "ingestion_lag", "inherited"}


def test_the_honest_pipeline_audits_clean():
    assert audit(honest_pipeline()).clean is True


def test_the_two_catalogue_pipelines_have_the_same_feature_count():
    """The comparison is only fair if the repair did not also change the model's size."""
    assert len(example_pipeline()) == len(honest_pipeline())


def test_a_clean_report_names_no_offending_features():
    report = audit(honest_pipeline())
    assert report.offending_features == ()
    assert report.worst_kind is None


def test_report_as_dict_counts_every_kind_it_found():
    payload = audit(example_pipeline()).as_dict()
    assert payload["counts_by_kind"]["target_future"] == 1
    assert payload["clean"] is False


def test_the_audit_touches_no_data(monkeypatch):
    """The central architectural claim, enforced rather than asserted in prose.

    Importing numpy is fine; building a panel is not. If the audit ever needs data it
    stops being runnable in a pre-commit hook on a pipeline that has none.
    """
    import leakage_audit.dataset as dataset

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the audit must not generate data")

    monkeypatch.setattr(dataset, "generate_panel", forbidden)
    monkeypatch.setattr(dataset, "materialise", forbidden)
    assert audit(example_pipeline()).clean is False


def test_latest_offset_read_reports_the_window_end():
    """Published in reports so a reader can see the offending moment without arithmetic."""
    assert spec("f", window_start=-7, window_end=-2).latest_offset_read == -2


def test_cycle_detection_skips_nodes_already_settled():
    """A node reached from two roots must not be re-walked.

    Without the early continue, a wide graph is traversed once per root and a large
    pipeline becomes quadratic. 'shared' is declared last on purpose: the outer loop
    reaches it only after both roots have already settled it, which is the path that
    exercises the skip.
    """
    pipeline = Pipeline(features=(
        FeatureSpec(name="root_a", depends_on=("shared",)),
        FeatureSpec(name="root_b", depends_on=("shared",)),
        FeatureSpec(name="shared"),
    ))
    assert pipeline.find_cycle() == ()
    assert pipeline.ancestors("root_b") == ("shared",)
