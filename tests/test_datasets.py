"""Tests for the named dataset builders."""

from __future__ import annotations

from datetime import timedelta

import pytest
import shapely

from postgis_fixtures.crs import BRITISH_NATIONAL_GRID, WEB_MERCATOR, WGS84
from postgis_fixtures.datasets import (
    DEFAULT_ROW_COUNTS,
    EPOCH,
    build_dataset,
    build_delivery_routes,
    build_edge_cases,
    build_fixtures,
    build_sensor_readings,
    dataset_names,
    edge_case_specs,
)
from postgis_fixtures.errors import DatasetError
from postgis_fixtures.geometry import GeneratorConfig


def geometry_of(row: dict) -> shapely.Geometry | None:
    """Decode a row's hex EWKB geometry."""
    raw = row.get("geom")
    return None if raw is None else shapely.from_wkb(bytes.fromhex(str(raw)))


class TestCatalogue:
    def test_every_builder_is_reachable_by_name(self) -> None:
        assert set(dataset_names()) == {
            "cities",
            "delivery_routes",
            "service_areas",
            "sensor_readings",
            "edge_cases",
        }

    def test_unknown_dataset_lists_the_alternatives(self) -> None:
        with pytest.raises(DatasetError, match="Available: cities"):
            build_dataset("towns")

    def test_negative_row_count_is_rejected(self) -> None:
        with pytest.raises(DatasetError, match="non-negative"):
            build_dataset("cities", rows=-5)

    def test_default_row_counts_are_applied(self) -> None:
        assert len(build_dataset("cities")) == DEFAULT_ROW_COUNTS["cities"]

    def test_edge_cases_ignores_the_row_count(self) -> None:
        assert len(build_dataset("edge_cases", rows=3)) == len(edge_case_specs())


class TestCities:
    def test_schema_and_rows(self, small_config: GeneratorConfig) -> None:
        dataset = build_dataset("cities", small_config, rows=30)
        assert dataset.column_names == ("id", "name", "population", "cluster", "geom")
        assert [row["id"] for row in dataset.rows] == list(range(1, 31))
        assert all(geometry_of(row).geom_type == "Point" for row in dataset.rows)

    def test_all_geometry_is_valid_and_tagged(self, small_config: GeneratorConfig) -> None:
        dataset = build_dataset("cities", small_config, rows=40)
        for row in dataset.rows:
            geometry = geometry_of(row)
            assert geometry.is_valid
            assert shapely.get_srid(geometry) == WGS84

    def test_ddl_declares_a_gist_index(self, small_config: GeneratorConfig) -> None:
        ddl = build_dataset("cities", small_config, rows=5).ddl()
        assert "USING gist (geom)" in ddl
        assert "geom geometry(POINT, 4326) NOT NULL" in ddl

    def test_populations_are_plausible(self, small_config: GeneratorConfig) -> None:
        dataset = build_dataset("cities", small_config, rows=100)
        populations = [row["population"] for row in dataset.rows]
        assert min(populations) >= 800
        assert max(populations) <= 2_400_000

    def test_dataset_is_iterable_and_sized(self, small_config: GeneratorConfig) -> None:
        dataset = build_dataset("cities", small_config, rows=4)
        assert len(dataset) == 4
        assert [row["id"] for row in dataset] == [1, 2, 3, 4]

    def test_values_match_column_order(self, small_config: GeneratorConfig) -> None:
        dataset = build_dataset("cities", small_config, rows=3)
        first = dataset.values()[0]
        assert len(first) == len(dataset.column_names)
        assert first[0] == 1


class TestDeliveryRoutes:
    def test_routes_are_distinct_linestrings(self, small_config: GeneratorConfig) -> None:
        dataset = build_dataset("delivery_routes", small_config, rows=10)
        wkts = {geometry_of(row).wkt for row in dataset.rows}
        assert len(wkts) == 10
        assert all(geometry_of(row).geom_type == "LineString" for row in dataset.rows)

    def test_codes_are_zero_padded(self, small_config: GeneratorConfig) -> None:
        dataset = build_dataset("delivery_routes", small_config, rows=3)
        assert [row["code"] for row in dataset.rows] == ["RT-0001", "RT-0002", "RT-0003"]

    def test_vertex_count_is_configurable(self, small_config: GeneratorConfig) -> None:
        dataset = build_delivery_routes(2, small_config, vertices=6)
        assert all(len(geometry_of(row).coords) == 6 for row in dataset.rows)

    def test_too_few_vertices_is_rejected(self) -> None:
        with pytest.raises(DatasetError, match="at least 2 vertices"):
            build_delivery_routes(1, vertices=1)


class TestServiceAreas:
    def test_polygons_are_valid_and_simple(self, small_config: GeneratorConfig) -> None:
        dataset = build_dataset("service_areas", small_config, rows=15)
        for row in dataset.rows:
            geometry = geometry_of(row)
            assert geometry.geom_type == "Polygon"
            assert geometry.is_valid
            assert geometry.is_simple
            assert geometry.area > 0

    def test_both_shape_families_appear(self, small_config: GeneratorConfig) -> None:
        dataset = build_dataset("service_areas", small_config, rows=15)
        assert {row["shape"] for row in dataset.rows} == {"hull", "corridor"}

    def test_partial_index_is_declared(self, small_config: GeneratorConfig) -> None:
        assert "WHERE tier = 'premium'" in build_dataset("service_areas", small_config, rows=3).ddl()


class TestSensorReadings:
    def test_timestamps_ascend_from_the_epoch(self, small_config: GeneratorConfig) -> None:
        dataset = build_sensor_readings(20, small_config, sensors=4, interval=timedelta(minutes=15))
        timestamps = [row["recorded_at"] for row in dataset.rows]
        assert timestamps[0] == EPOCH
        assert timestamps == sorted(timestamps)
        assert timestamps[1] - timestamps[0] == timedelta(minutes=15)

    def test_each_sensor_keeps_a_fixed_location(self, small_config: GeneratorConfig) -> None:
        dataset = build_sensor_readings(40, small_config, sensors=5)
        by_sensor: dict[int, set[str]] = {}
        for row in dataset.rows:
            by_sensor.setdefault(row["sensor_id"], set()).add(str(row["geom"]))
        assert len(by_sensor) == 5
        assert all(len(locations) == 1 for locations in by_sensor.values())

    def test_brin_index_on_the_timestamp(self, small_config: GeneratorConfig) -> None:
        ddl = build_sensor_readings(5, small_config, sensors=2).ddl()
        assert "USING brin (recorded_at)" in ddl
        assert "USING btree (sensor_id, recorded_at)" in ddl

    def test_temperatures_stay_in_a_believable_band(self, small_config: GeneratorConfig) -> None:
        dataset = build_sensor_readings(300, small_config, sensors=10)
        temperatures = [row["temperature_c"] for row in dataset.rows]
        assert -5 < min(temperatures) < max(temperatures) < 30

    def test_invalid_parameters_are_rejected(self) -> None:
        with pytest.raises(DatasetError, match="at least one sensor"):
            build_sensor_readings(10, sensors=0)
        with pytest.raises(DatasetError, match="interval must be positive"):
            build_sensor_readings(10, interval=timedelta(0))


class TestEdgeCases:
    def test_catalogue_covers_the_documented_nasties(self) -> None:
        labels = {spec.label for spec in edge_case_specs()}
        assert labels == {
            "antimeridian_linestring",
            "polygon_with_hole",
            "zero_length_linestring",
            "duplicate_points",
            "empty_geometry",
            "null_geometry",
            "wrong_srid",
        }

    def test_geometry_column_is_unconstrained_and_nullable(self) -> None:
        dataset = build_edge_cases()
        assert dataset.table.geometry.srid is None
        assert dataset.table.geometry.nullable is True
        assert "geom geometry\n" in dataset.ddl()

    def test_null_geometry_row_stores_none(self) -> None:
        rows = {row["label"]: row for row in build_edge_cases().rows}
        assert rows["null_geometry"]["geom"] is None
        assert rows["null_geometry"]["srid"] is None

    def test_wrong_srid_row_is_tagged_with_the_wrong_srid(self) -> None:
        rows = {row["label"]: row for row in build_edge_cases().rows}
        geometry = geometry_of(rows["wrong_srid"])
        assert shapely.get_srid(geometry) == WEB_MERCATOR
        assert abs(geometry.x) < 180  # still degree coordinates: that is the bug

    def test_empty_geometry_is_empty_but_not_null(self) -> None:
        rows = {row["label"]: row for row in build_edge_cases().rows}
        geometry = geometry_of(rows["empty_geometry"])
        assert geometry is not None
        assert geometry.is_empty

    def test_polygon_with_hole_has_an_interior_ring(self) -> None:
        rows = {row["label"]: row for row in build_edge_cases().rows}
        assert len(list(geometry_of(rows["polygon_with_hole"]).interiors)) == 1

    def test_antimeridian_linestring_spans_the_dateline(self) -> None:
        rows = {row["label"]: row for row in build_edge_cases().rows}
        minx, _, maxx, _ = geometry_of(rows["antimeridian_linestring"]).bounds
        assert minx < -179 and maxx > 179

    def test_zero_length_linestring_has_no_length(self) -> None:
        rows = {row["label"]: row for row in build_edge_cases().rows}
        assert geometry_of(rows["zero_length_linestring"]).length == 0

    def test_duplicate_points_repeat_a_coordinate(self) -> None:
        rows = {row["label"]: row for row in build_edge_cases().rows}
        coords = [tuple(point.coords)[0] for point in geometry_of(rows["duplicate_points"]).geoms]
        assert len(coords) != len(set(coords))

    def test_edge_cases_are_not_seed_dependent(self) -> None:
        first = build_edge_cases(GeneratorConfig(seed=1))
        second = build_edge_cases(GeneratorConfig(seed=2))
        assert first.rows == second.rows


class TestDeterminismAcrossDatasets:
    @pytest.mark.parametrize("name", dataset_names())
    def test_same_seed_gives_identical_rows(self, name: str) -> None:
        config = GeneratorConfig(seed=1234)
        assert build_dataset(name, config, rows=12).rows == build_dataset(
            name, config, rows=12
        ).rows

    @pytest.mark.parametrize("name", ["cities", "delivery_routes", "service_areas"])
    def test_different_seed_changes_geometry(self, name: str) -> None:
        first = build_dataset(name, GeneratorConfig(seed=1), rows=8)
        second = build_dataset(name, GeneratorConfig(seed=2), rows=8)
        assert [row["geom"] for row in first.rows] != [row["geom"] for row in second.rows]

    def test_adding_a_dataset_does_not_shift_the_others(self) -> None:
        """Salted RNGs mean each dataset's stream is independent."""
        config = GeneratorConfig(seed=88)
        alone = build_fixtures(config, names=["cities"])
        together = build_fixtures(config, names=["delivery_routes", "cities"])
        assert alone["cities"].rows == together["cities"].rows


class TestSpatialFixtures:
    def test_attribute_and_item_access(self) -> None:
        fixtures = build_fixtures(GeneratorConfig(seed=5), row_counts={"cities": 10})
        assert fixtures.cities is fixtures["cities"]
        assert "cities" in fixtures
        assert len(fixtures) == len(dataset_names())
        assert len(fixtures.cities) == 10

    def test_unknown_names_raise_clearly(self) -> None:
        fixtures = build_fixtures(names=["cities"], row_counts={"cities": 2})
        with pytest.raises(DatasetError, match="was not built"):
            fixtures["service_areas"]
        with pytest.raises(AttributeError):
            fixtures.service_areas

    def test_iteration_and_names(self) -> None:
        fixtures = build_fixtures(names=["cities", "edge_cases"], row_counts={"cities": 3})
        assert fixtures.names() == ("cities", "edge_cases")
        assert [dataset.name for dataset in fixtures] == ["cities", "edge_cases"]

    def test_ddl_covers_every_built_dataset(self) -> None:
        fixtures = build_fixtures(names=["cities", "service_areas"], row_counts={"cities": 2, "service_areas": 2})
        ddl = fixtures.ddl()
        assert "CREATE TABLE IF NOT EXISTS cities" in ddl
        assert "CREATE TABLE IF NOT EXISTS service_areas" in ddl

    def test_extra_points_share_the_seed_lineage(self) -> None:
        fixtures = build_fixtures(GeneratorConfig(seed=17), names=["cities"], row_counts={"cities": 2})
        first = fixtures.extra_points(3, salt="probe")
        second = fixtures.extra_points(3, salt="probe")
        assert [f.wkt() for f in first] == [f.wkt() for f in second]

    def test_reproject_returns_metric_coordinates(self) -> None:
        fixtures = build_fixtures(names=["cities"], row_counts={"cities": 5})
        projected = fixtures.reproject("cities", BRITISH_NATIONAL_GRID)
        assert len(projected) == 5
        assert all(feature.srid == BRITISH_NATIONAL_GRID for feature in projected)
        assert all(feature.geometry.x > 1000 for feature in projected)

    def test_reproject_skips_null_geometry(self) -> None:
        fixtures = build_fixtures(names=["edge_cases"])
        projected = fixtures.reproject("edge_cases", WGS84)
        assert len(projected) == len(edge_case_specs()) - 1
