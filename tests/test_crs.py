"""Tests for the CRS helpers."""

from __future__ import annotations

import pytest
import shapely
from shapely.geometry import LineString, Point

from postgis_fixtures.crs import (
    BRITISH_NATIONAL_GRID,
    SUPPORTED_SRIDS,
    UTM_33N,
    WEB_MERCATOR,
    WGS84,
    geodesic_distance,
    is_projected,
    metres_to_degrees,
    reproject,
    transformer_for,
    units_for,
)
from postgis_fixtures.errors import PostgisFixturesError


class TestCrsMetadata:
    def test_supported_srids_are_the_documented_four(self) -> None:
        assert SUPPORTED_SRIDS == (WGS84, WEB_MERCATOR, BRITISH_NATIONAL_GRID, UTM_33N)

    @pytest.mark.parametrize(
        ("srid", "projected"),
        [(WGS84, False), (WEB_MERCATOR, True), (BRITISH_NATIONAL_GRID, True), (UTM_33N, True)],
    )
    def test_projected_flag(self, srid: int, projected: bool) -> None:
        assert is_projected(srid) is projected

    def test_units(self) -> None:
        assert units_for(WGS84) == "degree"
        assert units_for(BRITISH_NATIONAL_GRID) == "metre"


class TestTransformer:
    def test_transformers_are_cached(self) -> None:
        assert transformer_for(WGS84, WEB_MERCATOR) is transformer_for(WGS84, WEB_MERCATOR)

    def test_unknown_epsg_code_is_reported_clearly(self) -> None:
        with pytest.raises(PostgisFixturesError, match="Cannot build a transformer"):
            transformer_for(WGS84, 999_999)


class TestReproject:
    def test_same_srid_returns_the_input_object(self) -> None:
        point = Point(1, 2)
        assert reproject(point, WGS84, WGS84) is point

    def test_empty_geometry_passes_through(self) -> None:
        empty = shapely.from_wkt("POLYGON EMPTY")
        assert reproject(empty, WGS84, WEB_MERCATOR) is empty

    def test_axis_order_is_lon_lat(self) -> None:
        """always_xy=True: x stays longitude even for EPSG:4326 -> EPSG:27700."""
        projected = reproject(Point(-0.1276, 51.5072), WGS84, BRITISH_NATIONAL_GRID)
        assert 528_000 < projected.x < 532_000
        assert 179_000 < projected.y < 183_000

    def test_linestrings_are_transformed_vertex_by_vertex(self) -> None:
        line = LineString([(-0.1, 51.5), (-0.2, 51.6)])
        projected = reproject(line, WGS84, WEB_MERCATOR)
        assert len(projected.coords) == 2
        assert all(abs(y) > 1_000_000 for _, y in projected.coords)


class TestDistances:
    def test_geodesic_distance_is_symmetric_and_positive(self) -> None:
        forward = geodesic_distance(-0.1276, 51.5072, -3.1883, 55.9533)
        backward = geodesic_distance(-3.1883, 55.9533, -0.1276, 51.5072)
        assert forward == pytest.approx(backward)
        assert 520_000 < forward < 540_000

    def test_zero_distance(self) -> None:
        assert geodesic_distance(1.0, 2.0, 1.0, 2.0) == pytest.approx(0.0)

    def test_metres_to_degrees_widens_towards_the_poles(self) -> None:
        assert metres_to_degrees(1_000, 0.0) < metres_to_degrees(1_000, 60.0)
        assert metres_to_degrees(111_320, 0.0) == pytest.approx(1.0, rel=1e-6)

    def test_metres_to_degrees_does_not_divide_by_zero_at_the_pole(self) -> None:
        assert metres_to_degrees(1_000, 90.0) > 0
