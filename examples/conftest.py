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

#: Geofencing only needs the point and polygon datasets — but the row counts are
#: not arbitrary. An index test is only meaningful when an index scan is genuinely
#: the cheaper plan, and for a table of a hundred-odd narrow rows it is not: the
#: whole heap fits in a page or two, so PostgreSQL correctly prefers a sequential
#: scan and ``assert_uses_index`` fails for an honest reason. Measured against
#: PostGIS 16-3.4, ``service_areas`` flips to a bitmap index scan on the partial
#: index somewhere between 120 and 500 rows; 2,000 leaves comfortable headroom so
#: the suite does not sit on the planner's cost boundary.
ROW_COUNTS = {"cities": 2_000, "service_areas": 2_000}

#: KNN ordering in metres needs a *geography* GiST index — the ``cities`` dataset
#: ships a geometry index, which ``geom::geography <-> ...`` cannot use. Expression
#: indexes are outside what ``IndexSpec`` renders, so the example creates it
#: directly, exactly as a consuming project would in its own migration.
CITIES_GEOGRAPHY_INDEX = (
    "CREATE INDEX IF NOT EXISTS cities_geog_gist ON cities USING gist ((geom::geography))"
)


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
    postgis_db.execute(CITIES_GEOGRAPHY_INDEX)
    # ``load`` ANALYZEs each table, but the expression index is created afterwards
    # and needs its own statistics before the planner will trust it.
    postgis_db.execute("ANALYZE cities")
    yield postgis_db
