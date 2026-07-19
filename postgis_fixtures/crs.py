"""Coordinate reference system helpers.

Fixture data is always *generated* in WGS84 (EPSG:4326) and then reprojected on
demand, which is what makes CRS-handling bugs catchable: a test can ask for the
same feature in 4326, 3857 and a projected national grid and compare results.

Reprojection uses :mod:`pyproj` with ``always_xy=True`` so that coordinates are
consistently ``(longitude, latitude)`` / ``(easting, northing)`` regardless of
the authority-defined axis order.
"""

from __future__ import annotations

from functools import lru_cache

from pyproj import CRS, Geod, Transformer
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

from .errors import PostgisFixturesError

#: Geographic CRS used for all generation.
WGS84 = 4326
#: Spherical Mercator, the CRS of web tile schemes.
WEB_MERCATOR = 3857
#: OSGB36 / British National Grid — a projected national grid in metres.
BRITISH_NATIONAL_GRID = 27700
#: WGS 84 / UTM zone 33N — a metric grid covering central Europe.
UTM_33N = 32633

#: SRIDs this package knows how to reproject into out of the box.
SUPPORTED_SRIDS: tuple[int, ...] = (WGS84, WEB_MERCATOR, BRITISH_NATIONAL_GRID, UTM_33N)

_WGS84_GEOD = Geod(ellps="WGS84")


@lru_cache(maxsize=64)
def transformer_for(source_srid: int, target_srid: int) -> Transformer:
    """Return a cached :class:`~pyproj.Transformer` between two SRIDs.

    Transformers are expensive to build and entirely stateless once built, so
    they are memoised. ``always_xy=True`` keeps the coordinate order lon/lat.
    """
    try:
        return Transformer.from_crs(
            CRS.from_epsg(source_srid),
            CRS.from_epsg(target_srid),
            always_xy=True,
        )
    except Exception as exc:  # pragma: no cover - pyproj raises several types
        raise PostgisFixturesError(
            f"Cannot build a transformer from EPSG:{source_srid} to EPSG:{target_srid}: {exc}"
        ) from exc


def is_projected(srid: int) -> bool:
    """Return ``True`` when ``srid`` is a projected (metric) CRS."""
    return bool(CRS.from_epsg(srid).is_projected)


def units_for(srid: int) -> str:
    """Return the name of the linear/angular unit used by ``srid``.

    Useful when formatting distance assertions: ``ST_DWithin`` on a geometry in
    4326 measures degrees, which is a classic source of wrong-by-100km bugs.
    """
    axis = CRS.from_epsg(srid).axis_info
    if not axis:  # pragma: no cover - every EPSG code in use here has axes
        return "unknown"
    return axis[0].unit_name


def reproject(geometry: BaseGeometry, source_srid: int, target_srid: int) -> BaseGeometry:
    """Reproject ``geometry`` between two SRIDs.

    Returns the geometry unchanged when source and target match, so callers can
    use this unconditionally in a loop over target CRSs.
    """
    if source_srid == target_srid:
        return geometry
    if geometry.is_empty:
        return geometry
    transformer = transformer_for(source_srid, target_srid)
    return shapely_transform(transformer.transform, geometry)


def geodesic_distance(
    lon_a: float, lat_a: float, lon_b: float, lat_b: float
) -> float:
    """Return the WGS84 ellipsoidal distance between two points, in metres."""
    _, _, distance = _WGS84_GEOD.inv(lon_a, lat_a, lon_b, lat_b)
    return abs(distance)


def metres_to_degrees(metres: float, latitude: float) -> float:
    """Approximate a metre distance as degrees of longitude at ``latitude``.

    This is a deliberately crude helper used only to size generated features;
    it is never used for assertions, where :func:`geodesic_distance` is used
    instead.
    """
    from math import cos, radians

    metres_per_degree = 111_320.0 * max(cos(radians(latitude)), 1e-6)
    return metres / metres_per_degree
