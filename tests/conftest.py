"""Shared test configuration for the postgis-fixtures suite.

``pytester`` is pytest's own plugin-testing fixture; it lets the tests run
throwaway pytest sessions in a temporary directory so the plugin's fixture
wiring is genuinely exercised without needing Docker or a database.
"""

from __future__ import annotations

import pytest

from postgis_fixtures.geometry import BoundingBox, GeneratorConfig, UrbanCentre

pytest_plugins = ["pytester"]


@pytest.fixture()
def small_config() -> GeneratorConfig:
    """A tiny, tightly bounded config that makes assertions easy to reason about."""
    return GeneratorConfig(
        seed=7,
        bbox=BoundingBox(-1.0, 51.0, 1.0, 52.0),
        centres=(UrbanCentre("Alpha", 0.0, 51.5, radius_m=2_000.0, weight=1.0),),
        cluster_fraction=0.5,
    )
