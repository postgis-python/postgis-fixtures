"""Deterministic generation of realistic spatial features.

Everything here is a pure function of an explicit :class:`random.Random`
instance, so the same seed always yields byte-identical WKT. That property is
what makes generated fixture data usable in assertions rather than just as
filler: a test can hard-code an expected WKT string and it will keep passing.

The generators aim for *plausible* rather than uniform data:

* points cluster around configurable urban centres with a uniform background
  scatter, so a GiST index over the result has a realistic selectivity profile;
* linestrings are produced by a bounded random walk with a heading that changes
  slowly, giving road/route-like vertex spacing instead of spaghetti;
* polygons are convex hulls of a jittered cloud, or buffered routes, both of
  which are guaranteed valid and non-self-intersecting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from random import Random

import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry

from .crs import WGS84, is_projected, metres_to_degrees, reproject

#: Decimal places kept for geographic coordinates (~1cm at the equator).
GEOGRAPHIC_PRECISION = 7
#: Decimal places kept for projected coordinates (millimetres).
PROJECTED_PRECISION = 3


@dataclass(frozen=True)
class UrbanCentre:
    """A weighted cluster centre used when scattering points.

    Attributes:
        name: Human-readable label, copied onto generated rows.
        longitude: Centre longitude in degrees (EPSG:4326).
        latitude: Centre latitude in degrees (EPSG:4326).
        radius_m: One standard deviation of the cluster, in metres.
        weight: Relative share of clustered points drawn from this centre.
    """

    name: str
    longitude: float
    latitude: float
    radius_m: float = 8_000.0
    weight: float = 1.0


#: Default centres: a spread of real cities across Great Britain and Europe,
#: chosen so that both EPSG:27700 and EPSG:32633 are meaningful targets.
DEFAULT_CENTRES: tuple[UrbanCentre, ...] = (
    UrbanCentre("London", -0.1276, 51.5072, radius_m=12_000, weight=3.0),
    UrbanCentre("Manchester", -2.2426, 53.4808, radius_m=8_000, weight=1.6),
    UrbanCentre("Edinburgh", -3.1883, 55.9533, radius_m=6_000, weight=1.2),
    UrbanCentre("Bristol", -2.5879, 51.4545, radius_m=5_000, weight=1.0),
    UrbanCentre("Leeds", -1.5491, 53.8008, radius_m=6_000, weight=1.0),
)


@dataclass(frozen=True)
class BoundingBox:
    """An axis-aligned lon/lat box used for the uniform background scatter."""

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def __post_init__(self) -> None:
        if self.min_lon >= self.max_lon or self.min_lat >= self.max_lat:
            raise ValueError(
                f"Degenerate bounding box: {self.min_lon},{self.min_lat} "
                f"-> {self.max_lon},{self.max_lat}"
            )

    def contains(self, longitude: float, latitude: float) -> bool:
        """Return ``True`` when the coordinate falls inside the box."""
        return (
            self.min_lon <= longitude <= self.max_lon
            and self.min_lat <= latitude <= self.max_lat
        )

    def clamp(self, longitude: float, latitude: float) -> tuple[float, float]:
        """Clamp a coordinate into the box."""
        return (
            min(max(longitude, self.min_lon), self.max_lon),
            min(max(latitude, self.min_lat), self.max_lat),
        )


#: Great Britain plus a margin of sea, the default generation extent.
DEFAULT_BBOX = BoundingBox(min_lon=-8.2, min_lat=49.9, max_lon=1.8, max_lat=58.7)


@dataclass(frozen=True)
class GeneratorConfig:
    """Knobs controlling how fixture geometry is generated.

    Attributes:
        seed: Seed for the :class:`random.Random` used by every generator.
        bbox: Extent for the uniform background scatter.
        centres: Urban centres used for clustering.
        cluster_fraction: Share of points drawn from a centre rather than the
            uniform background. ``0.0`` gives pure noise, ``1.0`` gives pure
            clusters.
        srid: SRID the emitted geometry should be expressed in.
    """

    seed: int = 20260719
    bbox: BoundingBox = DEFAULT_BBOX
    centres: tuple[UrbanCentre, ...] = DEFAULT_CENTRES
    cluster_fraction: float = 0.75
    srid: int = WGS84

    def __post_init__(self) -> None:
        if not 0.0 <= self.cluster_fraction <= 1.0:
            raise ValueError(
                f"cluster_fraction must be in [0, 1], got {self.cluster_fraction}"
            )
        if self.cluster_fraction > 0 and not self.centres:
            raise ValueError("cluster_fraction > 0 requires at least one urban centre")

    def rng(self, salt: str = "") -> Random:
        """Return a fresh RNG derived from the seed and an optional salt.

        Salting means adding a new dataset to a suite does not shift the values
        produced for the datasets that were already there.
        """
        return Random(f"{self.seed}:{salt}")

    def with_seed(self, seed: int) -> "GeneratorConfig":
        """Return a copy of this config with a different seed."""
        return replace(self, seed=seed)


@dataclass(frozen=True)
class Feature:
    """A generated geometry plus the attributes that travel with it."""

    geometry: BaseGeometry
    srid: int
    properties: dict[str, object] = field(default_factory=dict)

    def wkt(self) -> str:
        """Return canonical, precision-stable WKT for this feature."""
        return to_wkt(self.geometry, self.srid)

    def ewkb_hex(self) -> str:
        """Return hex EWKB (geometry plus SRID), the format PostGIS ingests."""
        return to_ewkb_hex(self.geometry, self.srid)


def precision_for(srid: int) -> int:
    """Return the rounding precision used for coordinates in ``srid``."""
    return PROJECTED_PRECISION if is_projected(srid) else GEOGRAPHIC_PRECISION


def to_wkt(geometry: BaseGeometry, srid: int) -> str:
    """Serialise ``geometry`` to WKT at the precision appropriate for ``srid``.

    Fixing the precision is what turns "deterministic floats" into
    "byte-identical strings" across platforms and shapely versions.
    """
    return shapely.to_wkt(
        geometry, rounding_precision=precision_for(srid), trim=True, output_dimension=2
    )


def to_ewkb_hex(geometry: BaseGeometry, srid: int) -> str:
    """Serialise ``geometry`` to hex EWKB carrying ``srid``."""
    tagged = shapely.set_srid(geometry, srid)
    return shapely.to_wkb(tagged, hex=True, include_srid=True, output_dimension=2)


def round_geometry(geometry: BaseGeometry, srid: int) -> BaseGeometry:
    """Return ``geometry`` with coordinates rounded to ``srid`` precision."""
    if geometry.is_empty:
        return geometry
    ndigits = precision_for(srid)
    return shapely.transform(
        geometry, lambda coords: coords.round(ndigits), include_z=False
    )


def project_feature(feature: Feature, target_srid: int) -> Feature:
    """Reproject a feature, rounding to the target CRS's precision."""
    if feature.srid == target_srid:
        return feature
    projected = reproject(feature.geometry, feature.srid, target_srid)
    return Feature(
        geometry=round_geometry(projected, target_srid),
        srid=target_srid,
        properties=dict(feature.properties),
    )


def _weighted_centre(rng: Random, centres: tuple[UrbanCentre, ...]) -> UrbanCentre:
    """Pick a centre with probability proportional to its weight."""
    total = sum(centre.weight for centre in centres)
    target = rng.random() * total
    cumulative = 0.0
    for centre in centres:
        cumulative += centre.weight
        if target <= cumulative:
            return centre
    return centres[-1]  # pragma: no cover - float guard


def _jitter_around(rng: Random, centre: UrbanCentre) -> tuple[float, float]:
    """Draw a normally distributed offset from an urban centre."""
    lat_sigma = centre.radius_m / 111_320.0
    lon_sigma = metres_to_degrees(centre.radius_m, centre.latitude)
    return (
        centre.longitude + rng.gauss(0.0, lon_sigma),
        centre.latitude + rng.gauss(0.0, lat_sigma),
    )


def generate_points(
    count: int, config: GeneratorConfig | None = None, *, salt: str = "points"
) -> list[Feature]:
    """Generate ``count`` clustered points.

    Each returned feature carries a ``cluster`` property naming the urban centre
    it was drawn from, or ``"background"`` for the uniform scatter. Tests that
    care about selectivity can filter on it.
    """
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    cfg = config or GeneratorConfig()
    rng = cfg.rng(salt)
    features: list[Feature] = []
    for _ in range(count):
        if cfg.centres and rng.random() < cfg.cluster_fraction:
            centre = _weighted_centre(rng, cfg.centres)
            lon, lat = cfg.bbox.clamp(*_jitter_around(rng, centre))
            cluster = centre.name
        else:
            lon = rng.uniform(cfg.bbox.min_lon, cfg.bbox.max_lon)
            lat = rng.uniform(cfg.bbox.min_lat, cfg.bbox.max_lat)
            cluster = "background"
        geom = round_geometry(Point(lon, lat), WGS84)
        features.append(Feature(geom, WGS84, {"cluster": cluster}))
    if cfg.srid != WGS84:
        return [project_feature(feature, cfg.srid) for feature in features]
    return features


def generate_route(
    vertices: int,
    config: GeneratorConfig | None = None,
    *,
    start: tuple[float, float] | None = None,
    step_m: float = 450.0,
    turn_sigma_deg: float = 22.0,
    salt: str = "route",
) -> Feature:
    """Generate one route-like linestring.

    The walk keeps a heading that drifts by a normally distributed turn at each
    step, which produces the gently curving, evenly spaced vertices you get from
    a real road centreline or GPS trace rather than random zig-zags.

    Args:
        vertices: Number of vertices; must be at least two.
        config: Generation configuration.
        start: Optional explicit start coordinate; defaults to a clustered point.
        step_m: Nominal spacing between consecutive vertices, in metres.
        turn_sigma_deg: Standard deviation of the per-step heading change.
        salt: RNG salt.
    """
    if vertices < 2:
        raise ValueError(f"a linestring needs at least 2 vertices, got {vertices}")
    cfg = config or GeneratorConfig()
    rng = cfg.rng(salt)
    if start is None:
        centre = _weighted_centre(rng, cfg.centres) if cfg.centres else None
        if centre is not None:
            lon, lat = cfg.bbox.clamp(*_jitter_around(rng, centre))
        else:
            lon = rng.uniform(cfg.bbox.min_lon, cfg.bbox.max_lon)
            lat = rng.uniform(cfg.bbox.min_lat, cfg.bbox.max_lat)
    else:
        lon, lat = start

    heading = rng.uniform(0.0, 360.0)
    coords: list[tuple[float, float]] = [(lon, lat)]
    for _ in range(vertices - 1):
        heading += rng.gauss(0.0, turn_sigma_deg)
        radians = math.radians(heading)
        step = step_m * rng.uniform(0.85, 1.15)
        lat += (step * math.cos(radians)) / 111_320.0
        lon += metres_to_degrees(step * math.sin(radians), lat)
        lon, lat = cfg.bbox.clamp(lon, lat)
        coords.append((lon, lat))

    geom = round_geometry(LineString(coords), WGS84)
    feature = Feature(geom, WGS84, {"vertices": len(coords), "step_m": step_m})
    return project_feature(feature, cfg.srid) if cfg.srid != WGS84 else feature


def generate_hull_polygon(
    config: GeneratorConfig | None = None,
    *,
    cloud_size: int = 14,
    spread_m: float = 3_000.0,
    salt: str = "polygon",
) -> Feature:
    """Generate a convex-hull polygon around a jittered point cloud.

    Convex hulls are valid and non-self-intersecting by construction, which is
    what you want for the "happy path" fixtures — the deliberately broken shapes
    live in the edge-case dataset instead.
    """
    if cloud_size < 3:
        raise ValueError(f"a hull needs at least 3 points, got {cloud_size}")
    cfg = config or GeneratorConfig()
    rng = cfg.rng(salt)
    if cfg.centres:
        centre = _weighted_centre(rng, cfg.centres)
        origin_lon, origin_lat = centre.longitude, centre.latitude
        label = centre.name
    else:
        origin_lon = rng.uniform(cfg.bbox.min_lon, cfg.bbox.max_lon)
        origin_lat = rng.uniform(cfg.bbox.min_lat, cfg.bbox.max_lat)
        label = "background"

    lat_sigma = spread_m / 111_320.0
    lon_sigma = metres_to_degrees(spread_m, origin_lat)
    cloud = [
        cfg.bbox.clamp(
            origin_lon + rng.gauss(0.0, lon_sigma),
            origin_lat + rng.gauss(0.0, lat_sigma),
        )
        for _ in range(cloud_size)
    ]
    hull = shapely.MultiPoint(cloud).convex_hull
    if not isinstance(hull, Polygon):
        # A degenerate cloud collapsed to a line or point; give it area back.
        # The floor keeps a zero-spread cloud from buffering by zero.
        fallback = max(metres_to_degrees(spread_m / 10, origin_lat), 1e-5)
        hull = hull.buffer(fallback)
    geom = round_geometry(shapely.force_2d(hull), WGS84)
    feature = Feature(geom, WGS84, {"anchor": label})
    return project_feature(feature, cfg.srid) if cfg.srid != WGS84 else feature


def buffer_route(
    route: Feature, width_m: float, *, quad_segs: int = 4
) -> Feature:
    """Buffer a route into a corridor polygon.

    The buffer is computed in the route's own CRS when that CRS is projected,
    and in degrees approximated at the route's latitude when it is geographic —
    the same trade-off you make when buffering in the database without casting
    to ``geography``.
    """
    if width_m <= 0:
        raise ValueError(f"width_m must be positive, got {width_m}")
    if is_projected(route.srid):
        distance = width_m
    else:
        centroid = route.geometry.centroid
        distance = metres_to_degrees(width_m, centroid.y)
    polygon = route.geometry.buffer(distance, quad_segs=quad_segs)
    properties = dict(route.properties)
    properties["corridor_width_m"] = width_m
    return Feature(round_geometry(polygon, route.srid), route.srid, properties)
