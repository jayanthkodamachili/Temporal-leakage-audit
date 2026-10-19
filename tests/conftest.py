"""Shared fixtures.

Panels here are small. The published scores come from the experiments; a test only needs
the evaluation path to run and return something sane, and training gradient boosting on a
realistic panel in every test would make the suite unusable.
"""

from __future__ import annotations

import pytest

from leakage_audit.dataset import generate_panel
from leakage_audit.features import FeatureSpec


@pytest.fixture
def small_spec() -> FeatureSpec:
    return FeatureSpec(name="trailing", window_start=-7, window_end=-1)


@pytest.fixture
def small_panel():
    return generate_panel(n_entities=20, n_periods=60, noise=1.0, seed=3)
