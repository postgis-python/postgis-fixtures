"""Assertion helpers for spatial tests.

Each helper builds its failure message through a pure ``_message_*`` function so
the wording can be unit-tested without a database and without provoking a
failure. The messages aim to answer the question a failing spatial assertion
actually raises — *by how much, in what units, and in which CRS* — because
"assert False" tells you nothing when the bug is a 100 km SRID mix-up.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

import shapely
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points
from shapely.validation import explain_validity

from .crs import WGS84, geodesic_distance, is_projected, units_for
from .errors import PostgisFixturesError

#: Deep links to background reading, attached to the failures where the fix is
#: a technique rather than a typo.
DOCS_URLS: dict[str, str] = {
    "index_usage": "https://www.postgis-python.com/advanced-gist-indexing-optimization/",
    "query_patterns": "https://www.postgis-python.com/mastering-core-spatial-query-patterns/",
    "srid": "https://www.postgis-python.com/spatial-schema-migrations-and-evolution/",
}

#: Plan node names that count as "used the index".
INDEX_SCAN_NODES: tuple[str, ...] = (
    "Index Scan",
    "Index Only Scan",
    "Bitmap Index Scan",
    "Bitmap Heap Scan",
)

# ``Index Scan``/``Index Only Scan`` name the index after "using"; ``Bitmap
# Index Scan`` names it after "on". Both forms have to be recognised.
_INDEX_NAME_RE = re.compile(
    r"(?:Index Only Scan|Index Scan)[^\n]*?\busing\s+(?P<using>[A-Za-z0-9_\".]+)"
    r"|Bitmap Index Scan\s+on\s+(?P<on>[A-Za-z0-9_\".]+)"
)


def as_geometry(value: Any) -> BaseGeometry:
    """Coerce WKT, hex EWKB, WKB bytes or a shapely object to a geometry.

    Raises:
        PostgisFixturesError: when the value cannot be interpreted.
    """
    if isinstance(value, BaseGeometry):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return shapely.from_wkb(bytes(value))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise PostgisFixturesError("Cannot parse an empty string as geometry")
        try:
            if _looks_like_hex(text):
                return shapely.from_wkb(bytes.fromhex(text))
            return shapely.from_wkt(text)
        except Exception as exc:
            raise PostgisFixturesError(f"Cannot parse geometry from {text[:60]!r}: {exc}") from exc
    raise PostgisFixturesError(
        f"Expected WKT, WKB or a shapely geometry, got {type(value).__name__}"
    )


def _looks_like_hex(text: str) -> bool:
    """Return ``True`` when ``text`` looks like a hex WKB payload."""
    return len(text) >= 10 and len(text) % 2 == 0 and all(c in "0123456789abcdefABCDEF" for c in text)


def srid_of(value: Any) -> int:
    """Return the SRID carried by a geometry or hex EWKB value (0 if untagged)."""
    geometry = as_geometry(value)
    return int(shapely.get_srid(geometry))


def _describe(geometry: BaseGeometry, limit: int = 120) -> str:
    """Return truncated WKT for use in a failure message."""
    text = shapely.to_wkt(geometry, rounding_precision=6, trim=True)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _message_invalid(geometry: BaseGeometry, reason: str, label: str) -> str:
    """Build the failure message for an invalid geometry."""
    return (
        f"{label} is not OGC-valid: {reason}\n"
        f"  geometry: {_describe(geometry)}\n"
        f"  hint: ST_MakeValid() repairs most self-intersections, but check whether the "
        f"input should have been valid in the first place."
    )


def assert_geometry_valid(value: Any, *, label: str = "geometry", allow_empty: bool = True) -> BaseGeometry:
    """Assert a geometry is OGC-valid, returning it for chaining.

    Args:
        value: WKT, hex EWKB, WKB bytes or a shapely geometry.
        label: Name used in the failure message.
        allow_empty: When ``False``, an empty geometry also fails.
    """
    geometry = as_geometry(value)
    if geometry.is_empty:
        if allow_empty:
            return geometry
        raise AssertionError(f"{label} is empty, which this assertion does not allow")
    if not geometry.is_valid:
        raise AssertionError(_message_invalid(geometry, explain_validity(geometry), label))
    return geometry


def _message_srid(actual: int, expected: int, label: str) -> str:
    """Build the failure message for an SRID mismatch."""
    actual_text = f"EPSG:{actual}" if actual else "untagged (SRID 0)"
    lines = [
        f"{label} has SRID {actual_text}, expected EPSG:{expected}",
        f"  expected units: {units_for(expected)}",
    ]
    if actual and actual != expected:
        lines.append(f"  actual units:   {units_for(actual)}")
        if is_projected(actual) != is_projected(expected):
            lines.append(
                "  these CRSs differ in kind (one geographic, one projected), so distance "
                "and area comparisons between them are meaningless, not merely imprecise."
            )
    lines.append(f"  see: {DOCS_URLS['srid']}")
    return "\n".join(lines)


def assert_srid(value: Any, expected: int, *, label: str = "geometry") -> int:
    """Assert a geometry carries ``expected`` as its SRID."""
    actual = srid_of(value)
    if actual != expected:
        raise AssertionError(_message_srid(actual, expected, label))
    return actual


def _message_distance(
    actual: float, maximum: float, unit: str, label_a: str, label_b: str
) -> str:
    """Build the failure message for a distance assertion."""
    overshoot = actual - maximum
    return (
        f"{label_a} is {actual:,.3f} {unit} from {label_b}, "
        f"which exceeds the limit of {maximum:,.3f} {unit} by {overshoot:,.3f} {unit}\n"
        f"  hint: if the limit looks off by a factor of ~100,000 the query is probably "
        f"measuring degrees; ST_DWithin on geometry uses the column's own units.\n"
        f"  see: {DOCS_URLS['query_patterns']}"
    )


def measure_distance(a: Any, b: Any, srid: int = WGS84) -> tuple[float, str]:
    """Return the distance between two geometries and its unit name.

    In a geographic CRS the ellipsoidal (geodesic) distance between the nearest
    points is returned in metres — which is what a reader means by "distance" —
    rather than a meaningless number of degrees.
    """
    geom_a = as_geometry(a)
    geom_b = as_geometry(b)
    if geom_a.is_empty or geom_b.is_empty:
        raise PostgisFixturesError("Cannot measure a distance involving an empty geometry")
    if is_projected(srid):
        return geom_a.distance(geom_b), units_for(srid)
    point_a, point_b = nearest_points(geom_a, geom_b)
    return geodesic_distance(point_a.x, point_a.y, point_b.x, point_b.y), "metre"


def assert_within_distance(
    a: Any,
    b: Any,
    maximum: float,
    *,
    srid: int = WGS84,
    label_a: str = "geometry A",
    label_b: str = "geometry B",
) -> float:
    """Assert two geometries lie within ``maximum`` of each other.

    Args:
        a: First geometry.
        b: Second geometry.
        maximum: Distance limit, in the CRS's units (metres for geographic CRSs,
            where the geodesic distance is used).
        srid: CRS the coordinates are expressed in.
    """
    if maximum < 0:
        raise ValueError(f"maximum must be non-negative, got {maximum}")
    distance, unit = measure_distance(a, b, srid)
    if distance > maximum:
        raise AssertionError(_message_distance(distance, maximum, unit, label_a, label_b))
    return distance


def _message_not_equal(
    left: BaseGeometry, right: BaseGeometry, tolerance: float, detail: str
) -> str:
    """Build the failure message for a geometry equality assertion."""
    return (
        f"geometries differ beyond a tolerance of {tolerance:g}: {detail}\n"
        f"  left:  {_describe(left)}\n"
        f"  right: {_describe(right)}"
    )


def assert_geometries_equal(a: Any, b: Any, *, tolerance: float = 1e-7) -> None:
    """Assert two geometries are equal to within a coordinate tolerance.

    Uses ``equals_exact`` semantics with a tolerance, so vertex order and
    structure matter — the strictness you want when checking that a round-trip
    through the database did not silently reorder or simplify anything.
    """
    if tolerance < 0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")
    left = as_geometry(a)
    right = as_geometry(b)
    if left.is_empty and right.is_empty:
        return
    if left.geom_type != right.geom_type:
        raise AssertionError(
            _message_not_equal(
                left, right, tolerance, f"{left.geom_type} vs {right.geom_type}"
            )
        )
    if left.equals_exact(right, tolerance):
        return
    if left.equals(right):
        detail = "same coverage but different vertex structure"
    else:
        detail = f"maximum vertex offset exceeds the tolerance (Hausdorff {left.hausdorff_distance(right):g})"
    raise AssertionError(_message_not_equal(left, right, tolerance, detail))


def parse_index_names(plan: str) -> tuple[str, ...]:
    """Return every index name referenced by an ``EXPLAIN`` plan, in order."""
    names: list[str] = []
    for match in _INDEX_NAME_RE.finditer(plan):
        name = (match.group("using") or match.group("on")).strip('"')
        if name not in names:
            names.append(name)
    return tuple(names)


def uses_index_scan(plan: str) -> bool:
    """Return ``True`` when the plan contains any index-based scan node."""
    return any(node in plan for node in INDEX_SCAN_NODES)


def _message_no_index(plan: str, index_name: str | None, found: Iterable[str]) -> str:
    """Build the failure message for :func:`assert_uses_index`."""
    found = tuple(found)
    target = f"index {index_name!r}" if index_name else "any index"
    if found:
        observed = "used " + ", ".join(sorted(found)) + " instead"
    elif uses_index_scan(plan):
        observed = "used an index scan whose name could not be parsed"
    else:
        observed = "used a sequential scan"
    plan_excerpt = "\n".join(f"    {line}" for line in plan.strip().splitlines()[:12])
    return (
        f"the planner did not use {target}; it {observed}.\n"
        f"  plan:\n{plan_excerpt}\n"
        f"  hint: a fixture table is often too small for the planner to bother, and "
        f"un-ANALYZEd statistics make that worse. Load enough rows, ANALYZE, and check "
        f"that the operator in the predicate is one the index supports.\n"
        f"  see: {DOCS_URLS['index_usage']}"
    )


def assert_uses_index(plan: str, index_name: str | None = None) -> tuple[str, ...]:
    """Assert an ``EXPLAIN`` plan uses an index — optionally a specific one.

    Args:
        plan: The text output of ``EXPLAIN``, as returned by
            :meth:`~postgis_fixtures.db.PostgisDB.explain`.
        index_name: When given, that exact index must appear in the plan.

    Returns:
        The index names found in the plan.
    """
    if not plan.strip():
        raise PostgisFixturesError("EXPLAIN output was empty; nothing to assert on")
    found = parse_index_names(plan)
    if index_name is None:
        if not uses_index_scan(plan):
            raise AssertionError(_message_no_index(plan, None, found))
        return found
    if index_name not in found:
        raise AssertionError(_message_no_index(plan, index_name, found))
    return found
