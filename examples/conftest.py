"""Example consuming project: how a real spatial test suite wires the plugin.

The plugin is not installed through packaging entry points, so it is enabled
here explicitly. Put the repository root on ``PYTHONPATH`` (or vendor the
``postgis_fixtures`` package into your own tree) and this is all it takes::

    PYTHONPATH=/path/to/postgis-fixtures pytest examples

Everything below the ``pytest_plugins`` line is ordinary project code: a
narrowed dataset, a session-scoped table load, and a skip guard so the suite
degrades to "skipped" rather than "error" on a machine with no database.
"""

from __future__ import annotations

from typing import Iterator

import pytest

from postgis_fixtures import GeneratorConfig, PostgisDB, build_fixtures
from postgis_fixtures.errors import PostgisFixturesError

pytest_plugins = ["postgis_fixtures.plugin"]

#: Geofencing only needs the point and polygon datasets, and only a few hundred
#: rows — but enough that the planner will consider the GiST index.
ROW_COUNTS = {"cities": 2_000, "service_areas": 120}


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip the whole example suite when no PostGIS database can be provided."""
    from postgis_fixtures.plugin import resolve_choice

    try:
        resolve_choice(config)
    except PostgisFixturesError as exc:
        skip = pytest.mark.skip(reason=f"no PostGIS available: {exc}")
        for item in items:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def geofence_data(postgis_seed: int):
    """Generate just the datasets this suite needs, once per session."""
    return build_fixtures(
        GeneratorConfig(seed=postgis_seed),
        names=["cities", "service_areas"],
        row_counts=ROW_COUNTS,
    )


@pytest.fixture()
def geofence_db(postgis_db: PostgisDB, geofence_data) -> Iterator[PostgisDB]:
    """Create and load the geofencing tables inside the test's transaction.

    Because ``postgis_connection`` rolls back after every test, this runs per
    test and leaves nothing behind — no truncation step, no ordering coupling.
    """
    postgis_db.install_extensions(("postgis",))
    for dataset in geofence_data:
        postgis_db.create_table(dataset)
        postgis_db.load(dataset)
    yield postgis_db
