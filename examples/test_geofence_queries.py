"""A realistic spatial test suite: geofence containment and proximity queries.

This is what the plugin is for. Each test states a spatial claim about the
generated data, runs the query the production code would run, and asserts on
the result with the package's spatial assertion helpers.

Every test here needs a live PostGIS; ``examples/conftest.py`` skips them all
when none is configured.
"""

from __future__ import annotations

import psycopg
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
#:
#: Both sides are cast to ``geography`` deliberately. Written the obvious way —
#: ``geom <-> ST_SetSRID(ST_MakePoint(...), 4326)`` — the ``<->`` operator on
#: *geometry* in EPSG:4326 orders by planar distance in **degrees**, and a degree
#: of longitude at London's latitude is only ~0.62 of a degree of latitude. So
#: "nearest in degrees" and "nearest in metres" genuinely disagree: against this
#: fixture the geometry form returns Rabdale (652 m) ahead of Cabford (578 m).
#: Any test that orders in degrees and then measures in metres will flap.
#: ``geography``'s ``<->`` is metre-based and still KNN-index-assisted, so
#: ordering and measurement finally agree on the same units.
NEAREST_SQL = """
SELECT id, name, ST_AsText(geom)
FROM cities
ORDER BY geom::geography <-> ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography
LIMIT %s
"""


def _load_edge_cases(db: PostgisDB, spatial_fixtures):
    """Create and populate the ``edge_cases`` table, returning its schema."""
    dataset = spatial_fixtures["edge_cases"]
    db.create_table(dataset)
    db.load(dataset)
    return dataset.table


def test_the_fixture_tables_are_populated(geofence_db: PostgisDB) -> None:
    assert geofence_db.count("cities") == 2_000
    assert geofence_db.count("service_areas") == 2_000


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
    """A predicate matching the index's WHERE clause must reach the partial index.

    The index is named explicitly rather than asserting "some index was used":
    ``service_areas`` also carries a full GiST index over the same column, so an
    unnamed assertion would pass on the wrong one and prove nothing about the
    partial index at all.
    """
    plan = geofence_db.explain(
        "SELECT id FROM service_areas "
        "WHERE tier = 'premium' AND geom && ST_MakeEnvelope(-1, 51, 0.5, 52, 4326)",
        analyze=True,
    )
    assert_uses_index(plan, "service_areas_premium_gist")


def test_knn_orders_by_metres_and_still_uses_the_index(geofence_db: PostgisDB) -> None:
    """Geography KNN must stay index-assisted, not degrade into a sort.

    Casting to ``geography`` fixes the units but costs the geometry GiST index:
    ``geom::geography <-> ...`` cannot use an index on ``geom``. Without the
    matching expression index this plan is a full ``Seq Scan`` plus a top-N
    ``Sort`` — correct, but no longer a KNN lookup. That index is created in
    ``conftest.py``; this asserts the planner actually reaches for it.
    """
    plan = geofence_db.explain(NEAREST_SQL, (-0.1276, 51.5072, 5), analyze=True)
    assert_uses_index(plan, "cities_geog_gist")


def test_knn_returns_progressively_more_distant_neighbours(geofence_db: PostgisDB) -> None:
    """Each successive neighbour must be at least as far away as the last.

    This is the assertion that catches a units mix-up: it only holds when the
    ``ORDER BY`` and the measurement agree on what "distance" means. See
    ``NEAREST_SQL`` for why ordering on plain geometry breaks it.
    """
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
    """NULL and empty geometry pass through spatial predicates without erroring.

    Neither is an error case in PostGIS: ``ST_Intersects`` returns NULL for a
    NULL geometry and false for an empty one, so both are simply filtered out.
    The row that *does* raise is a different problem — see the test below.
    """
    edge_cases = _load_edge_cases(geofence_db, spatial_fixtures)
    assert geofence_db.count(edge_cases, where="geom IS NULL") == 1
    assert (
        geofence_db.count(edge_cases, where="geom IS NOT NULL AND ST_IsEmpty(geom)") == 1
    )


def test_a_wrong_srid_row_breaks_an_unguarded_predicate(
    geofence_db: PostgisDB, spatial_fixtures
) -> None:
    """The planted EPSG:3857 row makes an unguarded spatial predicate fail loudly.

    ``edge_cases`` deliberately ships one geometry tagged EPSG:3857 in a column
    with no SRID constraint. PostGIS refuses to compare it against a 4326
    envelope rather than silently returning a wrong answer — which is the good
    outcome, and the reason the row exists. A query that scans a table it does
    not control has to say which SRID it means.
    """
    edge_cases = _load_edge_cases(geofence_db, spatial_fixtures)
    unguarded = (
        f"SELECT count(*) FROM {edge_cases.qualified_name} "
        "WHERE ST_Intersects(geom, ST_MakeEnvelope(-1, 51, 0.5, 52, 4326))"
    )
    with pytest.raises(psycopg.errors.InternalError_, match="mixed SRID"):
        geofence_db.scalar(unguarded)


def test_filtering_by_srid_makes_the_predicate_safe(
    geofence_db: PostgisDB, spatial_fixtures
) -> None:
    """The guarded form returns the right answer despite the wrong-SRID row.

    Two guards work, and they mean different things. ``ST_SRID(geom) = 4326``
    *excludes* anything not already in the query's CRS; ``ST_Transform`` instead
    *converts* it. Transforming is right only when the SRID tag is truthful —
    here it is not (the row holds degrees mislabelled as EPSG:3857), so
    transforming it would fling the point into the Gulf of Guinea. Excluding is
    the honest choice for data you do not trust.
    """
    edge_cases = _load_edge_cases(geofence_db, spatial_fixtures)
    matched = geofence_db.scalar(
        f"SELECT count(*) FROM {edge_cases.qualified_name} "
        "WHERE ST_SRID(geom) = 4326 "
        "AND ST_Intersects(geom, ST_MakeEnvelope(-1, 51, 0.5, 52, 4326))"
    )
    # polygon_with_hole and duplicate_points; the antimeridian and degenerate
    # rows sit elsewhere, and NULL/empty/wrong-SRID are all excluded.
    assert matched == 2
