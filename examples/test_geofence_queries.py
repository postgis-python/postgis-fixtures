"""A realistic spatial test suite: geofence containment and proximity queries.

This is what the plugin is for. Each test states a spatial claim about the
generated data, runs the query the production code would run, and asserts on
the result with the package's spatial assertion helpers.

Every test here needs a live PostGIS; ``examples/conftest.py`` skips them all
when none is configured.
"""

from __future__ import annotations

import pytest

from postgis_fixtures import (
    BRITISH_NATIONAL_GRID,
    WGS84,
    PostgisDB,
    assert_geometry_valid,
    assert_srid,
    assert_uses_index,
    assert_within_distance,
)

pytestmark = pytest.mark.postgis

#: The production query under test: every settlement inside a service area.
CONTAINMENT_SQL = """
SELECT c.id, c.name, a.name AS zone
FROM cities c
JOIN service_areas a ON ST_Contains(a.geom, c.geom)
WHERE a.tier = %s
ORDER BY c.id
"""

#: Nearest-neighbour lookup using the KNN operator rather than ORDER BY distance.
NEAREST_SQL = """
SELECT id, name, ST_AsText(geom)
FROM cities
ORDER BY geom <-> ST_SetSRID(ST_MakePoint(%s, %s), 4326)
LIMIT %s
"""


def test_the_fixture_tables_are_populated(geofence_db: PostgisDB) -> None:
    assert geofence_db.count("cities") == 2_000
    assert geofence_db.count("service_areas") == 120


def test_every_service_area_polygon_is_valid(geofence_db: PostgisDB) -> None:
    """Invalid polygons make ST_Contains silently wrong, so check up front."""
    rows = geofence_db.fetchall("SELECT id, ST_AsBinary(geom) FROM service_areas")
    for identifier, wkb in rows:
        assert_geometry_valid(bytes(wkb), label=f"service_areas.id={identifier}")


def test_containment_join_returns_only_premium_zones(geofence_db: PostgisDB) -> None:
    rows = geofence_db.fetchall(CONTAINMENT_SQL, ("premium",))
    zones = {row[2] for row in rows}
    premium = {
        row[0]
        for row in geofence_db.fetchall("SELECT name FROM service_areas WHERE tier = 'premium'")
    }
    assert zones <= premium


def test_containment_join_uses_the_gist_index(geofence_db: PostgisDB) -> None:
    """A spatial join must not degrade into a nested loop over sequential scans."""
    plan = geofence_db.explain(
        CONTAINMENT_SQL.replace("%s", "'premium'"), analyze=True, buffers=True
    )
    assert_uses_index(plan)


def test_partial_index_covers_the_premium_predicate(geofence_db: PostgisDB) -> None:
    plan = geofence_db.explain(
        "SELECT id FROM service_areas "
        "WHERE tier = 'premium' AND geom && ST_MakeEnvelope(-1, 51, 0.5, 52, 4326)",
        analyze=True,
    )
    assert_uses_index(plan)


def test_knn_returns_progressively_more_distant_neighbours(geofence_db: PostgisDB) -> None:
    rows = geofence_db.fetchall(NEAREST_SQL, (-0.1276, 51.5072, 5))
    assert len(rows) == 5
    previous = 0.0
    for _, name, wkt in rows:
        distance = assert_within_distance(
            wkt,
            "POINT (-0.1276 51.5072)",
            300_000.0,
            label_a=name,
            label_b="Charing Cross",
        )
        assert distance >= previous
        previous = distance


def test_dwithin_radius_matches_the_geodesic_measurement(geofence_db: PostgisDB) -> None:
    """ST_DWithin on geography measures metres; verify the boundary is honest."""
    rows = geofence_db.fetchall(
        "SELECT name, ST_AsText(geom) FROM cities "
        "WHERE ST_DWithin(geom::geography, "
        "ST_SetSRID(ST_MakePoint(-0.1276, 51.5072), 4326)::geography, 20000)"
    )
    assert rows
    for name, wkt in rows:
        assert_within_distance(wkt, "POINT (-0.1276 51.5072)", 20_000.0, label_a=name)


def test_geometry_round_trips_through_the_database_unchanged(
    geofence_db: PostgisDB, geofence_data
) -> None:
    expected = geofence_data["cities"].rows[0]
    stored = geofence_db.scalar(
        "SELECT ST_AsBinary(geom) FROM cities WHERE id = %s", (expected["id"],)
    )
    assert_srid(str(expected["geom"]), WGS84)
    assert_within_distance(bytes(stored), str(expected["geom"]), 0.001)


def test_reprojection_into_the_national_grid_keeps_distances(
    geofence_db: PostgisDB,
) -> None:
    """A 27700 round trip must not move anything more than a millimetre."""
    rows = geofence_db.fetchall(
        "SELECT ST_AsBinary(geom), "
        "ST_AsBinary(ST_Transform(ST_Transform(geom, %s), 4326)) "
        "FROM cities LIMIT 25",
        (BRITISH_NATIONAL_GRID,),
    )
    for original, round_tripped in rows:
        assert_within_distance(bytes(original), bytes(round_tripped), 0.01)


def test_null_and_empty_geometry_do_not_break_the_query(
    geofence_db: PostgisDB, spatial_fixtures
) -> None:
    """The edge-case table is where your error handling gets exercised."""
    edge_cases = spatial_fixtures["edge_cases"]
    geofence_db.create_table(edge_cases)
    geofence_db.load(edge_cases)
    nulls = geofence_db.count("edge_cases", where="geom IS NULL")
    empties = geofence_db.count("edge_cases", where="geom IS NOT NULL AND ST_IsEmpty(geom)")
    assert nulls == 1
    assert empties == 1
    assert geofence_db.scalar(
        "SELECT count(*) FROM edge_cases WHERE ST_Intersects(geom, "
        "ST_MakeEnvelope(-1, 51, 0.5, 52, 4326))"
    ) >= 1
