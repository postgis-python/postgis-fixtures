"""Tests for the spatial assertion helpers and their failure messages."""

from __future__ import annotations

import pytest
import shapely
from shapely.geometry import LineString, Point, Polygon

from postgis_fixtures.assertions import (
    DOCS_URLS,
    as_geometry,
    assert_geometries_equal,
    assert_geometry_valid,
    assert_srid,
    assert_uses_index,
    assert_within_distance,
    measure_distance,
    parse_index_names,
    srid_of,
    uses_index_scan,
)
from postgis_fixtures.crs import BRITISH_NATIONAL_GRID, WEB_MERCATOR, WGS84
from postgis_fixtures.errors import PostgisFixturesError
from postgis_fixtures.geometry import to_ewkb_hex

BOWTIE = "POLYGON ((0 0, 1 1, 1 0, 0 1, 0 0))"

SEQ_SCAN_PLAN = """Seq Scan on cities  (cost=0.00..24.50 rows=2 width=40)
  Filter: (geom && '0101...'::geometry)"""

GIST_PLAN = """Index Scan using cities_geom_gist on cities  (cost=0.15..8.42 rows=1 width=40)
  Index Cond: (geom && '0101...'::geometry)"""

BITMAP_PLAN = """Bitmap Heap Scan on service_areas  (cost=4.20..18.30 rows=6 width=64)
  Recheck Cond: (geom && '0103...'::geometry)
  ->  Bitmap Index Scan on service_areas_geom_gist  (cost=0.00..4.19 rows=6 width=0)"""

INDEX_ONLY_PLAN = """Index Only Scan using "Cities_Geom_Idx" on cities  (cost=0.15..4.20 rows=1 width=32)
  Heap Fetches: 0"""


class TestAsGeometry:
    def test_accepts_shapely_objects_unchanged(self) -> None:
        point = Point(1, 2)
        assert as_geometry(point) is point

    def test_accepts_wkt(self) -> None:
        assert as_geometry("POINT (1 2)").equals(Point(1, 2))

    def test_accepts_hex_ewkb_and_keeps_the_srid(self) -> None:
        hex_value = to_ewkb_hex(Point(1, 2), WEB_MERCATOR)
        assert srid_of(hex_value) == WEB_MERCATOR

    def test_accepts_raw_wkb_bytes(self) -> None:
        assert as_geometry(shapely.to_wkb(Point(3, 4))).equals(Point(3, 4))

    def test_rejects_an_empty_string(self) -> None:
        with pytest.raises(PostgisFixturesError, match="empty string"):
            as_geometry("   ")

    def test_rejects_unparseable_text(self) -> None:
        with pytest.raises(PostgisFixturesError, match="Cannot parse geometry"):
            as_geometry("PINT (1 2)")

    def test_rejects_unsupported_types(self) -> None:
        with pytest.raises(PostgisFixturesError, match="got int"):
            as_geometry(42)

    def test_untagged_geometry_reports_srid_zero(self) -> None:
        assert srid_of("POINT (1 2)") == 0


class TestAssertGeometryValid:
    def test_valid_geometry_is_returned(self) -> None:
        assert assert_geometry_valid("POINT (1 2)").equals(Point(1, 2))

    def test_self_intersection_is_reported_with_the_reason(self) -> None:
        with pytest.raises(AssertionError) as excinfo:
            assert_geometry_valid(BOWTIE, label="service area")
        message = str(excinfo.value)
        assert "service area is not OGC-valid" in message
        assert "Self-intersection" in message
        assert "ST_MakeValid()" in message
        assert "POLYGON ((0 0" in message

    def test_empty_geometry_passes_by_default(self) -> None:
        assert assert_geometry_valid("POLYGON EMPTY").is_empty

    def test_empty_geometry_can_be_rejected(self) -> None:
        with pytest.raises(AssertionError, match="is empty"):
            assert_geometry_valid("POLYGON EMPTY", allow_empty=False)

    def test_long_geometry_is_truncated_in_the_message(self) -> None:
        ring = [(i * 0.01, 0.0) for i in range(40)]
        bad = Polygon(ring + [(0.2, 1.0), (0.1, -1.0)])
        with pytest.raises(AssertionError) as excinfo:
            assert_geometry_valid(bad)
        assert "..." in str(excinfo.value)


class TestAssertSrid:
    def test_matching_srid_returns_it(self) -> None:
        assert assert_srid(to_ewkb_hex(Point(1, 2), 4326), 4326) == 4326

    def test_mismatch_names_both_crss_and_their_units(self) -> None:
        with pytest.raises(AssertionError) as excinfo:
            assert_srid(to_ewkb_hex(Point(1, 2), WGS84), WEB_MERCATOR, label="route")
        message = str(excinfo.value)
        assert "route has SRID EPSG:4326, expected EPSG:3857" in message
        assert "expected units: metre" in message
        assert "degree" in message
        assert "differ in kind" in message
        assert DOCS_URLS["srid"] in message

    def test_untagged_geometry_is_called_out(self) -> None:
        with pytest.raises(AssertionError) as excinfo:
            assert_srid("POINT (1 2)", 4326)
        assert "untagged (SRID 0)" in str(excinfo.value)

    def test_two_projected_crss_do_not_claim_a_kind_mismatch(self) -> None:
        with pytest.raises(AssertionError) as excinfo:
            assert_srid(to_ewkb_hex(Point(1, 2), WEB_MERCATOR), BRITISH_NATIONAL_GRID)
        assert "differ in kind" not in str(excinfo.value)


class TestDistance:
    def test_geographic_distance_is_in_metres(self) -> None:
        london = "POINT (-0.1276 51.5072)"
        greenwich = "POINT (-0.0005 51.4779)"
        distance, unit = measure_distance(london, greenwich, WGS84)
        assert unit == "metre"
        assert 8_000 < distance < 10_000

    def test_projected_distance_uses_planar_maths(self) -> None:
        distance, unit = measure_distance(
            "POINT (0 0)", "POINT (300 400)", BRITISH_NATIONAL_GRID
        )
        assert unit == "metre"
        assert distance == pytest.approx(500.0)

    def test_distance_to_a_line_uses_the_nearest_point(self) -> None:
        distance, _ = measure_distance(
            Point(0.0, 51.0), LineString([(0.0, 51.5), (1.0, 51.5)]), WGS84
        )
        assert 55_000 < distance < 56_000

    def test_empty_geometry_is_rejected(self) -> None:
        with pytest.raises(PostgisFixturesError, match="empty geometry"):
            measure_distance("POLYGON EMPTY", "POINT (0 0)")

    def test_within_distance_passes_and_returns_the_distance(self) -> None:
        distance = assert_within_distance("POINT (0 51)", "POINT (0 51.001)", 200.0)
        assert 100 < distance < 120

    def test_failure_message_quantifies_the_overshoot(self) -> None:
        with pytest.raises(AssertionError) as excinfo:
            assert_within_distance(
                "POINT (0 51)", "POINT (1 51)", 1_000.0, label_a="depot", label_b="drop"
            )
        message = str(excinfo.value)
        assert "depot is 70,197.140 metre from drop" in message
        assert "exceeds the limit of 1,000.000 metre" in message
        assert DOCS_URLS["query_patterns"] in message

    def test_negative_limit_is_a_programming_error(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            assert_within_distance("POINT (0 0)", "POINT (0 0)", -1)


class TestAssertGeometriesEqual:
    def test_identical_geometries_pass(self) -> None:
        assert_geometries_equal("POINT (1 2)", "POINT (1 2)")

    def test_tiny_differences_are_tolerated(self) -> None:
        assert_geometries_equal("POINT (1 2)", "POINT (1.00000005 2)", tolerance=1e-6)

    def test_two_empties_are_equal(self) -> None:
        assert_geometries_equal("POLYGON EMPTY", "POLYGON EMPTY")

    def test_type_mismatch_is_reported_first(self) -> None:
        with pytest.raises(AssertionError) as excinfo:
            assert_geometries_equal("POINT (1 2)", "LINESTRING (1 2, 3 4)")
        assert "Point vs LineString" in str(excinfo.value)

    def test_reordered_vertices_are_reported_as_structural(self) -> None:
        with pytest.raises(AssertionError) as excinfo:
            assert_geometries_equal(
                "POLYGON ((0 0, 1 0, 1 1, 0 0))", "POLYGON ((1 0, 1 1, 0 0, 1 0))"
            )
        assert "different vertex structure" in str(excinfo.value)

    def test_real_differences_report_the_hausdorff_distance(self) -> None:
        with pytest.raises(AssertionError) as excinfo:
            assert_geometries_equal("POINT (0 0)", "POINT (3 4)")
        message = str(excinfo.value)
        assert "Hausdorff 5" in message
        assert "left:  POINT (0 0)" in message
        assert "right: POINT (3 4)" in message

    def test_negative_tolerance_is_a_programming_error(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            assert_geometries_equal("POINT (0 0)", "POINT (0 0)", tolerance=-1)


class TestPlanParsing:
    def test_index_scan_name_is_extracted(self) -> None:
        assert parse_index_names(GIST_PLAN) == ("cities_geom_gist",)

    def test_bitmap_plans_expose_the_underlying_index(self) -> None:
        assert parse_index_names(BITMAP_PLAN) == ("service_areas_geom_gist",)

    def test_quoted_index_names_are_unquoted(self) -> None:
        assert parse_index_names(INDEX_ONLY_PLAN) == ("Cities_Geom_Idx",)

    def test_sequential_plans_name_no_index(self) -> None:
        assert parse_index_names(SEQ_SCAN_PLAN) == ()
        assert not uses_index_scan(SEQ_SCAN_PLAN)

    def test_duplicate_index_references_are_collapsed(self) -> None:
        plan = GIST_PLAN + "\n" + GIST_PLAN
        assert parse_index_names(plan) == ("cities_geom_gist",)


class TestAssertUsesIndex:
    def test_any_index_is_enough_without_a_name(self) -> None:
        assert assert_uses_index(GIST_PLAN) == ("cities_geom_gist",)

    def test_named_index_must_appear(self) -> None:
        assert assert_uses_index(BITMAP_PLAN, "service_areas_geom_gist")

    def test_sequential_scan_fails_with_advice(self) -> None:
        with pytest.raises(AssertionError) as excinfo:
            assert_uses_index(SEQ_SCAN_PLAN, "cities_geom_gist")
        message = str(excinfo.value)
        assert "did not use index 'cities_geom_gist'" in message
        assert "used a sequential scan" in message
        assert "Seq Scan on cities" in message
        assert "ANALYZE" in message
        assert DOCS_URLS["index_usage"] in message

    def test_wrong_index_names_what_was_used_instead(self) -> None:
        with pytest.raises(AssertionError) as excinfo:
            assert_uses_index(GIST_PLAN, "cities_population_btree")
        assert "used cities_geom_gist instead" in str(excinfo.value)

    def test_unnamed_assertion_fails_on_a_sequential_scan(self) -> None:
        with pytest.raises(AssertionError, match="did not use any index"):
            assert_uses_index(SEQ_SCAN_PLAN)

    def test_unparseable_index_scan_is_described_honestly(self) -> None:
        plan = "Index Scan on cities  (cost=0.15..8.42 rows=1 width=40)"
        with pytest.raises(AssertionError) as excinfo:
            assert_uses_index(plan, "cities_geom_gist")
        assert "could not be parsed" in str(excinfo.value)

    def test_long_plans_are_truncated_to_twelve_lines(self) -> None:
        plan = "\n".join(f"Seq Scan step {i}" for i in range(40))
        with pytest.raises(AssertionError) as excinfo:
            assert_uses_index(plan)
        assert "Seq Scan step 11" in str(excinfo.value)
        assert "Seq Scan step 12" not in str(excinfo.value)

    def test_empty_plan_is_a_usage_error(self) -> None:
        with pytest.raises(PostgisFixturesError, match="EXPLAIN output was empty"):
            assert_uses_index("   ")
