"""Tests for the deterministic geometry generators."""

from __future__ import annotations

import math

import pytest
import shapely
from shapely.geometry import LineString, Point

from postgis_fixtures.crs import BRITISH_NATIONAL_GRID, WEB_MERCATOR, WGS84
from postgis_fixtures.geometry import (
    DEFAULT_BBOX,
    BoundingBox,
    Feature,
    GeneratorConfig,
    UrbanCentre,
    buffer_route,
    generate_hull_polygon,
    generate_points,
    generate_route,
    precision_for,
    project_feature,
    round_geometry,
    to_ewkb_hex,
    to_wkt,
)


class TestBoundingBox:
    def test_rejects_degenerate_box(self) -> None:
        with pytest.raises(ValueError, match="Degenerate bounding box"):
            BoundingBox(1.0, 0.0, 1.0, 1.0)

    def test_contains_and_clamp(self) -> None:
        box = BoundingBox(-1.0, 50.0, 1.0, 52.0)
        assert box.contains(0.0, 51.0)
        assert not box.contains(2.0, 51.0)
        assert box.clamp(5.0, 49.0) == (1.0, 50.0)


class TestGeneratorConfig:
    def test_rejects_out_of_range_cluster_fraction(self) -> None:
        with pytest.raises(ValueError, match="cluster_fraction"):
            GeneratorConfig(cluster_fraction=1.5)

    def test_rejects_clustering_without_centres(self) -> None:
        with pytest.raises(ValueError, match="at least one urban centre"):
            GeneratorConfig(centres=(), cluster_fraction=0.5)

    def test_pure_background_scatter_is_allowed(self) -> None:
        config = GeneratorConfig(centres=(), cluster_fraction=0.0)
        features = generate_points(20, config)
        assert {f.properties["cluster"] for f in features} == {"background"}

    def test_salted_rngs_are_independent(self) -> None:
        config = GeneratorConfig(seed=42)
        assert config.rng("a").random() != config.rng("b").random()
        assert config.rng("a").random() == config.rng("a").random()

    def test_with_seed_returns_a_copy(self) -> None:
        config = GeneratorConfig(seed=1)
        other = config.with_seed(2)
        assert config.seed == 1 and other.seed == 2
        assert other.bbox is config.bbox


class TestDeterminism:
    def test_same_seed_gives_byte_identical_wkt(self, small_config: GeneratorConfig) -> None:
        first = [f.wkt() for f in generate_points(50, small_config)]
        second = [f.wkt() for f in generate_points(50, small_config)]
        assert first == second

    def test_different_seed_gives_different_wkt(self, small_config: GeneratorConfig) -> None:
        first = [f.wkt() for f in generate_points(20, small_config)]
        second = [f.wkt() for f in generate_points(20, small_config.with_seed(99))]
        assert first != second

    def test_wkt_is_stable_across_process_state(self, small_config: GeneratorConfig) -> None:
        import random

        random.seed(1)
        first = generate_route(10, small_config).wkt()
        random.seed(999)
        [random.random() for _ in range(100)]
        assert generate_route(10, small_config).wkt() == first

    def test_ewkb_round_trips_and_carries_srid(self, small_config: GeneratorConfig) -> None:
        feature = generate_points(1, small_config)[0]
        restored = shapely.from_wkb(bytes.fromhex(feature.ewkb_hex()))
        assert shapely.get_srid(restored) == WGS84
        assert restored.equals_exact(feature.geometry, 1e-9)


class TestPointGeneration:
    def test_count_and_bounds(self, small_config: GeneratorConfig) -> None:
        features = generate_points(200, small_config)
        assert len(features) == 200
        assert all(small_config.bbox.contains(f.geometry.x, f.geometry.y) for f in features)

    def test_rejects_negative_count(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            generate_points(-1)

    def test_zero_count_is_empty(self) -> None:
        assert generate_points(0) == []

    def test_clustering_beats_uniform_scatter(self) -> None:
        """Clustered points must concentrate far more tightly than uniform ones."""
        centre = UrbanCentre("Alpha", 0.0, 51.5, radius_m=3_000.0)
        box = BoundingBox(-2.0, 50.5, 2.0, 52.5)
        clustered = generate_points(
            400, GeneratorConfig(seed=3, bbox=box, centres=(centre,), cluster_fraction=1.0)
        )
        uniform = generate_points(
            400, GeneratorConfig(seed=3, bbox=box, centres=(), cluster_fraction=0.0)
        )

        def spread(features: list[Feature]) -> float:
            return sum(
                math.dist((f.geometry.x, f.geometry.y), (centre.longitude, centre.latitude))
                for f in features
            ) / len(features)

        assert spread(clustered) < spread(uniform) / 5

    def test_weights_shift_the_cluster_mix(self) -> None:
        heavy = UrbanCentre("Heavy", -0.1, 51.5, weight=9.0)
        light = UrbanCentre("Light", -3.0, 55.0, weight=1.0)
        config = GeneratorConfig(
            seed=11, centres=(heavy, light), cluster_fraction=1.0, bbox=DEFAULT_BBOX
        )
        labels = [f.properties["cluster"] for f in generate_points(500, config)]
        assert labels.count("Heavy") > labels.count("Light") * 3


class TestRouteGeneration:
    def test_vertex_count_and_validity(self, small_config: GeneratorConfig) -> None:
        route = generate_route(30, small_config)
        assert isinstance(route.geometry, LineString)
        assert len(route.geometry.coords) == 30
        assert route.geometry.is_valid
        assert route.properties["vertices"] == 30

    def test_rejects_too_few_vertices(self) -> None:
        with pytest.raises(ValueError, match="at least 2 vertices"):
            generate_route(1)

    def test_vertex_spacing_is_plausible(self) -> None:
        """Consecutive vertices sit near the requested step, not scattered randomly."""
        from postgis_fixtures.crs import geodesic_distance

        route = generate_route(
            60,
            GeneratorConfig(seed=5, bbox=BoundingBox(-3.0, 50.0, 3.0, 56.0)),
            start=(0.0, 52.0),
            step_m=500.0,
        )
        coords = list(route.geometry.coords)
        gaps = [
            geodesic_distance(a[0], a[1], b[0], b[1])
            for a, b in zip(coords, coords[1:])
        ]
        assert min(gaps) > 400
        assert max(gaps) < 600

    def test_route_without_centres_starts_in_the_background_scatter(self) -> None:
        config = GeneratorConfig(seed=6, centres=(), cluster_fraction=0.0)
        route = generate_route(6, config)
        assert config.bbox.contains(*route.geometry.coords[0])

    def test_explicit_start_is_honoured(self, small_config: GeneratorConfig) -> None:
        route = generate_route(5, small_config, start=(0.25, 51.25))
        assert route.geometry.coords[0] == pytest.approx((0.25, 51.25))


class TestPolygonGeneration:
    def test_hull_is_valid_and_simple(self, small_config: GeneratorConfig) -> None:
        polygon = generate_hull_polygon(small_config)
        assert polygon.geometry.is_valid
        assert polygon.geometry.is_simple
        assert polygon.geometry.area > 0
        assert not list(polygon.geometry.interiors)

    def test_hull_without_centres_is_anchored_to_the_background(self) -> None:
        polygon = generate_hull_polygon(
            GeneratorConfig(seed=6, centres=(), cluster_fraction=0.0)
        )
        assert polygon.properties["anchor"] == "background"
        assert polygon.geometry.is_valid

    def test_rejects_tiny_cloud(self) -> None:
        with pytest.raises(ValueError, match="at least 3 points"):
            generate_hull_polygon(cloud_size=2)

    def test_collapsed_cloud_still_has_area(self) -> None:
        """A zero-spread cloud collapses to a point; the fallback buffers it."""
        polygon = generate_hull_polygon(
            GeneratorConfig(seed=1, cluster_fraction=1.0), cloud_size=4, spread_m=0.0
        )
        assert polygon.geometry.is_valid
        assert polygon.geometry.area > 0

    def test_buffer_route_produces_a_corridor(self, small_config: GeneratorConfig) -> None:
        route = generate_route(12, small_config)
        corridor = buffer_route(route, 800.0)
        assert corridor.geometry.is_valid
        assert corridor.geometry.contains(route.geometry)
        assert corridor.properties["corridor_width_m"] == 800.0

    def test_buffer_route_rejects_non_positive_width(self, small_config: GeneratorConfig) -> None:
        with pytest.raises(ValueError, match="width_m must be positive"):
            buffer_route(generate_route(3, small_config), 0.0)

    def test_buffer_in_projected_crs_uses_metres_directly(self) -> None:
        route = Feature(LineString([(0.0, 0.0), (1000.0, 0.0)]), BRITISH_NATIONAL_GRID)
        corridor = buffer_route(route, 100.0)
        minx, miny, maxx, maxy = corridor.geometry.bounds
        assert maxy - miny == pytest.approx(200.0, abs=1.0)


class TestSerialisation:
    def test_precision_depends_on_crs_kind(self) -> None:
        assert precision_for(WGS84) == 7
        assert precision_for(WEB_MERCATOR) == 3

    def test_wkt_rounds_to_crs_precision(self) -> None:
        point = Point(1.123456789012, 2.0)
        assert to_wkt(point, WGS84) == "POINT (1.1234568 2)"
        assert to_wkt(Point(1.12345, 2.0), WEB_MERCATOR) == "POINT (1.123 2)"

    def test_round_geometry_leaves_empty_alone(self) -> None:
        empty = shapely.from_wkt("POLYGON EMPTY")
        assert round_geometry(empty, WGS84).is_empty

    def test_ewkb_includes_the_srid(self) -> None:
        hex_value = to_ewkb_hex(Point(1.0, 2.0), WEB_MERCATOR)
        assert shapely.get_srid(shapely.from_wkb(bytes.fromhex(hex_value))) == WEB_MERCATOR


class TestProjection:
    def test_project_to_web_mercator_moves_into_metres(self, small_config: GeneratorConfig) -> None:
        feature = generate_points(1, small_config)[0]
        projected = project_feature(feature, WEB_MERCATOR)
        assert projected.srid == WEB_MERCATOR
        assert abs(projected.geometry.y) > 1_000_000
        assert projected.properties == feature.properties

    def test_projection_round_trips_within_a_millimetre(self, small_config: GeneratorConfig) -> None:
        from postgis_fixtures.crs import geodesic_distance

        feature = generate_points(1, small_config)[0]
        there = project_feature(feature, BRITISH_NATIONAL_GRID)
        back = project_feature(there, WGS84)
        assert (
            geodesic_distance(
                feature.geometry.x, feature.geometry.y, back.geometry.x, back.geometry.y
            )
            < 0.01
        )

    def test_projecting_to_the_same_srid_is_a_no_op(self, small_config: GeneratorConfig) -> None:
        feature = generate_points(1, small_config)[0]
        assert project_feature(feature, WGS84) is feature

    def test_config_srid_projects_every_generator(self) -> None:
        config = GeneratorConfig(seed=4, srid=BRITISH_NATIONAL_GRID)
        assert generate_points(3, config)[0].srid == BRITISH_NATIONAL_GRID
        assert generate_route(4, config).srid == BRITISH_NATIONAL_GRID
        assert generate_hull_polygon(config).srid == BRITISH_NATIONAL_GRID
