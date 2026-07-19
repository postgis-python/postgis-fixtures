"""Named, ready-made spatial datasets.

Each dataset bundles three things: a :class:`~postgis_fixtures.ddl.TableSchema`
(so the plugin can create it), a deterministic row generator, and a description
of what the data is shaped like. Row counts are parameters, so a unit test can
ask for 50 rows and a benchmark for 500,000 without changing anything else.

Datasets are built entirely offline; loading them into PostGIS is a separate
step handled by :mod:`postgis_fixtures.db`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterator, Mapping, Sequence

import shapely
from shapely.geometry import LineString, MultiPoint, Point, Polygon

from .crs import WGS84
from .ddl import Column, GeometryColumn, IndexSpec, TableSchema, schema_sql
from .errors import DatasetError
from .geometry import (
    Feature,
    GeneratorConfig,
    buffer_route,
    generate_hull_polygon,
    generate_points,
    generate_route,
    project_feature,
    to_ewkb_hex,
    to_wkt,
)

#: Fixed epoch for time-series datasets, so timestamps are reproducible too.
EPOCH = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)

_CITY_SUFFIXES = ("ford", "bury", "ton", "wick", "combe", "dale", "mouth", "cester")
_VEHICLES = ("van-01", "van-02", "van-03", "bike-01", "hgv-01")
_TIERS = ("premium", "standard", "standard", "trial")


@dataclass(frozen=True)
class Dataset:
    """A generated table's schema plus its rows.

    Attributes:
        name: Dataset name as used by ``spatial_fixtures["cities"]``.
        description: One-line summary of the data's shape.
        table: The table definition, including indexes.
        rows: Rows in ``column name -> value`` form. Geometry values are hex
            EWKB strings, or ``None`` for a NULL geometry.
        srid: SRID the geometry was emitted in.
    """

    name: str
    description: str
    table: TableSchema
    rows: tuple[Mapping[str, object], ...]
    srid: int | None

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[Mapping[str, object]]:
        return iter(self.rows)

    @property
    def column_names(self) -> tuple[str, ...]:
        """Return the column order used for loading."""
        return self.table.column_names

    @property
    def geometry_column(self) -> str:
        """Return the name of the geometry column."""
        return self.table.geometry.name

    def ddl(self, *, include_indexes: bool = True) -> str:
        """Return the full DDL for this dataset's table."""
        return schema_sql(self.table, include_indexes=include_indexes)

    def values(self) -> list[tuple[object, ...]]:
        """Return rows as tuples ordered to match :attr:`column_names`."""
        names = self.column_names
        return [tuple(row.get(name) for name in names) for row in self.rows]


def _city_name(index: int) -> str:
    """Return a stable pseudo place name for row ``index``."""
    stem = f"{chr(ord('A') + index % 26)}{'ab' if index % 3 else 'or'}"
    return f"{stem}{_CITY_SUFFIXES[index % len(_CITY_SUFFIXES)]}"


def _feature_geometry(feature: Feature) -> str:
    """Return the storage representation (hex EWKB) of a feature."""
    return feature.ewkb_hex()


def build_cities(
    rows: int = 200, config: GeneratorConfig | None = None
) -> Dataset:
    """Build the ``cities`` dataset: clustered points with a population column.

    Schema:

    ===============  ==========================  ==================================
    Column           Type                        Notes
    ===============  ==========================  ==================================
    ``id``           ``integer``                 Primary key, 1-based
    ``name``         ``text``                    Stable pseudo place name
    ``population``   ``integer``                 Log-ish distribution, 800..2.4M
    ``cluster``      ``text``                    Urban centre name or ``background``
    ``geom``         ``geometry(POINT, srid)``   GiST-indexed
    ===============  ==========================  ==================================
    """
    cfg = config or GeneratorConfig()
    features = generate_points(rows, cfg, salt="cities")
    rng = cfg.rng("cities-attributes")
    records: list[Mapping[str, object]] = []
    for index, feature in enumerate(features, start=1):
        clustered = feature.properties.get("cluster") != "background"
        population = int(rng.triangular(800, 2_400_000, 40_000 if clustered else 4_000))
        records.append(
            {
                "id": index,
                "name": _city_name(index),
                "population": population,
                "cluster": feature.properties.get("cluster"),
                "geom": _feature_geometry(feature),
            }
        )
    table = TableSchema(
        name="cities",
        columns=(
            Column("id", "integer", nullable=False, primary_key=True),
            Column("name", "text", nullable=False),
            Column("population", "integer"),
            Column("cluster", "text"),
        ),
        geometry=GeometryColumn("geom", "POINT", cfg.srid, nullable=False),
        comment="Clustered settlement points for spatial-join and KNN tests.",
        indexes=(
            IndexSpec("cities_geom_gist", ("geom",), "gist"),
            IndexSpec("cities_population_btree", ("population",), "btree"),
        ),
    )
    return Dataset(
        name="cities",
        description="Point settlements clustered around urban centres, with a uniform background scatter.",
        table=table,
        rows=tuple(records),
        srid=cfg.srid,
    )


def build_delivery_routes(
    rows: int = 40, config: GeneratorConfig | None = None, *, vertices: int = 24
) -> Dataset:
    """Build the ``delivery_routes`` dataset: route-like linestrings.

    Schema:

    ==================  ================================  =========================
    Column              Type                              Notes
    ==================  ================================  =========================
    ``id``              ``integer``                       Primary key
    ``code``            ``text``                          ``RT-0001`` style
    ``vehicle``         ``text``                          One of five vehicle ids
    ``stops``           ``integer``                       2..12
    ``geom``            ``geometry(LINESTRING, srid)``    GiST-indexed
    ==================  ================================  =========================
    """
    if vertices < 2:
        raise DatasetError(f"delivery_routes needs at least 2 vertices, got {vertices}")
    cfg = config or GeneratorConfig()
    rng = cfg.rng("routes-attributes")
    records: list[Mapping[str, object]] = []
    for index in range(1, rows + 1):
        feature = generate_route(vertices, cfg, salt=f"route-{index}")
        records.append(
            {
                "id": index,
                "code": f"RT-{index:04d}",
                "vehicle": _VEHICLES[index % len(_VEHICLES)],
                "stops": rng.randint(2, 12),
                "geom": _feature_geometry(feature),
            }
        )
    table = TableSchema(
        name="delivery_routes",
        columns=(
            Column("id", "integer", nullable=False, primary_key=True),
            Column("code", "text", nullable=False),
            Column("vehicle", "text", nullable=False),
            Column("stops", "integer"),
        ),
        geometry=GeometryColumn("geom", "LINESTRING", cfg.srid, nullable=False),
        comment="Vehicle routes with plausible vertex spacing, for ST_DWithin and length tests.",
        indexes=(IndexSpec("delivery_routes_geom_gist", ("geom",), "gist"),),
    )
    return Dataset(
        name="delivery_routes",
        description="Route-like linestrings generated by a bounded random walk with slowly drifting heading.",
        table=table,
        rows=tuple(records),
        srid=cfg.srid,
    )


def build_service_areas(
    rows: int = 25, config: GeneratorConfig | None = None
) -> Dataset:
    """Build the ``service_areas`` dataset: valid coverage polygons.

    Roughly one polygon in three is a buffered route corridor rather than a
    convex hull, so the table contains both compact and elongated shapes — a
    mix that exercises bounding-box selectivity rather than flattering it.

    Schema:

    ==============  ==============================  ============================
    Column          Type                            Notes
    ==============  ==============================  ============================
    ``id``          ``integer``                     Primary key
    ``name``        ``text``                        ``Zone 01`` style
    ``tier``        ``text``                        premium / standard / trial
    ``shape``       ``text``                        ``hull`` or ``corridor``
    ``geom``        ``geometry(POLYGON, srid)``     GiST-indexed, plus a partial
                                                    GiST index on premium rows
    ==============  ==============================  ============================
    """
    cfg = config or GeneratorConfig()
    records: list[Mapping[str, object]] = []
    for index in range(1, rows + 1):
        if index % 3 == 0:
            route = generate_route(8, cfg, step_m=900.0, salt=f"corridor-{index}")
            feature = buffer_route(route, 1_200.0)
            shape = "corridor"
        else:
            feature = generate_hull_polygon(cfg, salt=f"area-{index}")
            shape = "hull"
        records.append(
            {
                "id": index,
                "name": f"Zone {index:02d}",
                "tier": _TIERS[index % len(_TIERS)],
                "shape": shape,
                "geom": _feature_geometry(feature),
            }
        )
    table = TableSchema(
        name="service_areas",
        columns=(
            Column("id", "integer", nullable=False, primary_key=True),
            Column("name", "text", nullable=False),
            Column("tier", "text", nullable=False),
            Column("shape", "text", nullable=False),
        ),
        geometry=GeometryColumn("geom", "POLYGON", cfg.srid, nullable=False),
        comment="Coverage polygons: convex hulls plus buffered route corridors.",
        indexes=(
            IndexSpec("service_areas_geom_gist", ("geom",), "gist"),
            IndexSpec(
                "service_areas_premium_gist",
                ("geom",),
                "gist",
                where="tier = 'premium'",
            ),
        ),
    )
    return Dataset(
        name="service_areas",
        description="Valid, non-self-intersecting coverage polygons in two shape families.",
        table=table,
        rows=tuple(records),
        srid=cfg.srid,
    )


def build_sensor_readings(
    rows: int = 500,
    config: GeneratorConfig | None = None,
    *,
    sensors: int = 25,
    interval: timedelta = timedelta(minutes=5),
) -> Dataset:
    """Build the ``sensor_readings`` dataset: a time-ordered point series.

    Rows are emitted in ascending ``recorded_at`` order and the timestamp is
    correlated with physical insertion order, which is exactly the condition a
    BRIN index needs to be useful — so a test can compare BRIN and GiST plans on
    honest data.

    Schema:

    =================  ==========================  ==============================
    Column             Type                        Notes
    =================  ==========================  ==============================
    ``id``             ``integer``                 Primary key
    ``sensor_id``      ``integer``                 1..``sensors``
    ``recorded_at``    ``timestamptz``             Ascending from 2026-01-05Z
    ``temperature_c``  ``numeric(5,2)``            Diurnal curve plus noise
    ``geom``           ``geometry(POINT, srid)``   Sensor's fixed location
    =================  ==========================  ==============================
    """
    if sensors < 1:
        raise DatasetError(f"sensor_readings needs at least one sensor, got {sensors}")
    if interval <= timedelta(0):
        raise DatasetError(f"interval must be positive, got {interval}")
    cfg = config or GeneratorConfig()
    stations = generate_points(sensors, cfg, salt="sensors")
    rng = cfg.rng("sensor-readings")
    records: list[Mapping[str, object]] = []
    for index in range(rows):
        station_index = index % sensors
        timestamp = EPOCH + interval * index
        hour = timestamp.hour + timestamp.minute / 60.0
        temperature = round(
            9.0 + 6.0 * _diurnal(hour) + rng.gauss(0.0, 0.8) - station_index * 0.05, 2
        )
        records.append(
            {
                "id": index + 1,
                "sensor_id": station_index + 1,
                "recorded_at": timestamp,
                "temperature_c": temperature,
                "geom": _feature_geometry(stations[station_index]),
            }
        )
    table = TableSchema(
        name="sensor_readings",
        columns=(
            Column("id", "integer", nullable=False, primary_key=True),
            Column("sensor_id", "integer", nullable=False),
            Column("recorded_at", "timestamptz", nullable=False),
            Column("temperature_c", "numeric(5,2)"),
        ),
        geometry=GeometryColumn("geom", "POINT", cfg.srid, nullable=False),
        comment="Append-only sensor time series at fixed locations.",
        indexes=(
            IndexSpec("sensor_readings_geom_gist", ("geom",), "gist"),
            IndexSpec("sensor_readings_recorded_at_brin", ("recorded_at",), "brin"),
            IndexSpec(
                "sensor_readings_sensor_time_btree",
                ("sensor_id", "recorded_at"),
                "btree",
            ),
        ),
    )
    return Dataset(
        name="sensor_readings",
        description="Append-only readings at fixed sensor locations, timestamp-correlated for BRIN tests.",
        table=table,
        rows=tuple(records),
        srid=cfg.srid,
    )


def _diurnal(hour: float) -> float:
    """Return a -1..1 diurnal factor peaking mid-afternoon."""
    return math.sin((hour - 9.0) / 24.0 * 2 * math.pi)


@dataclass(frozen=True)
class EdgeCase:
    """One deliberately awkward geometry, with the reason it is awkward."""

    label: str
    note: str
    wkt: str | None
    srid: int | None


def edge_case_specs() -> tuple[EdgeCase, ...]:
    """Return the edge-case catalogue as pure data.

    These exist so downstream code can test its *error handling*: every one of
    them is something PostGIS will happily store and then behave surprisingly
    with.
    """
    hole = Polygon(
        [(-1.0, 51.0), (-0.6, 51.0), (-0.6, 51.3), (-1.0, 51.3)],
        [[(-0.9, 51.05), (-0.7, 51.05), (-0.7, 51.2), (-0.9, 51.2)]],
    )
    return (
        EdgeCase(
            "antimeridian_linestring",
            "Crosses the 180th meridian; naive ST_Envelope spans the whole globe.",
            to_wkt(LineString([(179.4, -16.5), (-179.6, -16.62)]), WGS84),
            WGS84,
        ),
        EdgeCase(
            "polygon_with_hole",
            "Interior ring: ST_Area and point-in-polygon must respect the hole.",
            to_wkt(hole, WGS84),
            WGS84,
        ),
        EdgeCase(
            "zero_length_linestring",
            "Both vertices identical; ST_Length is 0 and ST_Azimuth errors.",
            to_wkt(LineString([(-2.5, 53.4), (-2.5, 53.4)]), WGS84),
            WGS84,
        ),
        EdgeCase(
            "duplicate_points",
            "MultiPoint with repeated coordinates; distinct-count logic often breaks.",
            to_wkt(
                MultiPoint([(-0.1276, 51.5072), (-0.1276, 51.5072), (-0.1280, 51.5075)]),
                WGS84,
            ),
            WGS84,
        ),
        EdgeCase(
            "empty_geometry",
            "Non-NULL but empty; ST_IsEmpty is true and ST_Centroid returns empty.",
            "POLYGON EMPTY",
            WGS84,
        ),
        EdgeCase(
            "null_geometry",
            "SQL NULL geometry; every ST_* function returns NULL, silently.",
            None,
            None,
        ),
        EdgeCase(
            "wrong_srid",
            "Degree coordinates tagged as EPSG:3857; distances come out ~0 metres.",
            to_wkt(Point(-0.1276, 51.5072), WGS84),
            3857,
        ),
    )


def build_edge_cases(config: GeneratorConfig | None = None) -> Dataset:
    """Build the ``edge_cases`` dataset from :func:`edge_case_specs`.

    The geometry column is deliberately unconstrained (``geometry``, no type or
    SRID modifier) — a constrained column would reject the wrong-SRID row, which
    is the row most worth testing against.

    Schema:

    ============  ===================  ==========================================
    Column        Type                 Notes
    ============  ===================  ==========================================
    ``id``        ``integer``          Primary key
    ``label``     ``text``             Stable identifier, e.g. ``wrong_srid``
    ``note``      ``text``             Why this input is awkward
    ``srid``      ``integer``          Declared SRID, NULL for the NULL geometry
    ``geom``      ``geometry``         Unconstrained and nullable
    ============  ===================  ==========================================
    """
    del config  # edge cases are fixed data, not seed-dependent
    records: list[Mapping[str, object]] = []
    for index, spec in enumerate(edge_case_specs(), start=1):
        if spec.wkt is None:
            geom_value: str | None = None
        else:
            geom_value = to_ewkb_hex(shapely.from_wkt(spec.wkt), spec.srid or WGS84)
        records.append(
            {
                "id": index,
                "label": spec.label,
                "note": spec.note,
                "srid": spec.srid,
                "geom": geom_value,
            }
        )
    table = TableSchema(
        name="edge_cases",
        columns=(
            Column("id", "integer", nullable=False, primary_key=True),
            Column("label", "text", nullable=False),
            Column("note", "text", nullable=False),
            Column("srid", "integer"),
        ),
        geometry=GeometryColumn("geom", "GEOMETRY", None, nullable=True),
        comment="Deliberately awkward geometry for error-handling tests.",
        indexes=(IndexSpec("edge_cases_geom_gist", ("geom",), "gist"),),
    )
    return Dataset(
        name="edge_cases",
        description="Antimeridian, hole, degenerate, duplicate, empty, NULL and wrong-SRID geometry.",
        table=table,
        rows=tuple(records),
        srid=None,
    )


#: Builder registry. Every builder takes ``(rows, config)`` except
#: ``edge_cases``, whose row set is fixed.
DATASET_BUILDERS: Mapping[str, Callable[..., Dataset]] = {
    "cities": build_cities,
    "delivery_routes": build_delivery_routes,
    "service_areas": build_service_areas,
    "sensor_readings": build_sensor_readings,
    "edge_cases": build_edge_cases,
}

#: Default row counts, sized so a full build stays well under a second.
DEFAULT_ROW_COUNTS: Mapping[str, int] = {
    "cities": 200,
    "delivery_routes": 40,
    "service_areas": 25,
    "sensor_readings": 500,
}


def dataset_names() -> tuple[str, ...]:
    """Return every known dataset name, in build order."""
    return tuple(DATASET_BUILDERS)


def build_dataset(
    name: str, config: GeneratorConfig | None = None, *, rows: int | None = None
) -> Dataset:
    """Build a single dataset by name.

    Raises:
        DatasetError: if ``name`` is unknown or ``rows`` is negative.
    """
    try:
        builder = DATASET_BUILDERS[name]
    except KeyError:
        raise DatasetError(
            f"Unknown dataset {name!r}. Available: {', '.join(dataset_names())}"
        ) from None
    if name == "edge_cases":
        return builder(config)
    if rows is None:
        rows = DEFAULT_ROW_COUNTS[name]
    if rows < 0:
        raise DatasetError(f"rows must be non-negative for {name!r}, got {rows}")
    return builder(rows, config)


@dataclass(frozen=True)
class SpatialFixtures:
    """The full set of generated datasets, keyed by name.

    Supports both ``fixtures["cities"]`` and ``fixtures.cities``, and exposes
    the :class:`~postgis_fixtures.geometry.GeneratorConfig` that produced it so
    tests can regenerate extra features with the same seed.
    """

    config: GeneratorConfig
    datasets: Mapping[str, Dataset] = field(default_factory=dict)

    def __getitem__(self, name: str) -> Dataset:
        try:
            return self.datasets[name]
        except KeyError:
            raise DatasetError(
                f"Dataset {name!r} was not built. Built: {', '.join(self.datasets)}"
            ) from None

    def __getattr__(self, name: str) -> Dataset:
        datasets = object.__getattribute__(self, "datasets")
        if name in datasets:
            return datasets[name]
        raise AttributeError(name)

    def __contains__(self, name: object) -> bool:
        return name in self.datasets

    def __iter__(self) -> Iterator[Dataset]:
        return iter(self.datasets.values())

    def __len__(self) -> int:
        return len(self.datasets)

    def names(self) -> tuple[str, ...]:
        """Return the names of the datasets that were built."""
        return tuple(self.datasets)

    def ddl(self) -> str:
        """Return the DDL for every built dataset, in build order."""
        return "\n\n".join(dataset.ddl() for dataset in self.datasets.values())

    def extra_points(self, count: int, *, salt: str) -> list[Feature]:
        """Generate additional points with the same seed lineage.

        Useful for building a query geometry that is guaranteed to sit inside
        the same clusters as the loaded data.
        """
        return generate_points(count, self.config, salt=salt)

    def reproject(self, name: str, target_srid: int) -> list[Feature]:
        """Return one dataset's geometry reprojected into ``target_srid``."""
        dataset = self[name]
        features: list[Feature] = []
        for row in dataset.rows:
            raw = row.get(dataset.geometry_column)
            if raw is None:
                continue
            geometry = shapely.from_wkb(bytes.fromhex(str(raw)))
            source = row.get("srid") if dataset.srid is None else dataset.srid
            features.append(
                project_feature(Feature(geometry, int(source or WGS84)), target_srid)
            )
        return features


def build_fixtures(
    config: GeneratorConfig | None = None,
    *,
    names: Sequence[str] | None = None,
    row_counts: Mapping[str, int] | None = None,
) -> SpatialFixtures:
    """Build several datasets at once.

    Args:
        config: Generation configuration; a default-seeded one is used if omitted.
        names: Dataset names to build. Defaults to all of them.
        row_counts: Per-dataset row-count overrides.
    """
    cfg = config or GeneratorConfig()
    selected = tuple(names) if names is not None else dataset_names()
    counts = dict(row_counts or {})
    built = {
        name: build_dataset(name, cfg, rows=counts.get(name))
        for name in selected
    }
    return SpatialFixtures(config=cfg, datasets=built)
