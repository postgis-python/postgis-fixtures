"""Ephemeral PostGIS databases and deterministic spatial fixture data for pytest.

The package has two halves that can be used independently:

* a **generation layer** (:mod:`~postgis_fixtures.geometry`,
  :mod:`~postgis_fixtures.datasets`, :mod:`~postgis_fixtures.ddl`) that is pure,
  offline and seeded — the same seed always produces byte-identical WKT;
* a **pytest plugin** (:mod:`~postgis_fixtures.plugin`) that supplies a database
  to run it against, either an ephemeral container or a DSN you configured.

Enable the plugin from your own ``conftest.py``::

    pytest_plugins = ["postgis_fixtures.plugin"]
"""

from __future__ import annotations

from .assertions import (
    assert_geometries_equal,
    assert_geometry_valid,
    assert_srid,
    assert_uses_index,
    assert_within_distance,
    measure_distance,
    parse_index_names,
)
from .crs import (
    BRITISH_NATIONAL_GRID,
    SUPPORTED_SRIDS,
    UTM_33N,
    WEB_MERCATOR,
    WGS84,
    reproject,
)
from .datasets import (
    Dataset,
    SpatialFixtures,
    build_dataset,
    build_fixtures,
    dataset_names,
    edge_case_specs,
)
from .db import PostgisDB
from .ddl import Column, GeometryColumn, IndexSpec, TableSchema, schema_sql
from .errors import (
    DatasetError,
    PostgisFixturesError,
    ProviderError,
    ReadinessTimeout,
    SchemaError,
)
from .geometry import (
    BoundingBox,
    Feature,
    GeneratorConfig,
    UrbanCentre,
    generate_hull_polygon,
    generate_points,
    generate_route,
)
from .provider import ProviderChoice, select_provider

__version__ = "1.0.0"

__all__ = [
    "BRITISH_NATIONAL_GRID",
    "BoundingBox",
    "Column",
    "Dataset",
    "DatasetError",
    "Feature",
    "GeneratorConfig",
    "GeometryColumn",
    "IndexSpec",
    "PostgisDB",
    "PostgisFixturesError",
    "ProviderChoice",
    "ProviderError",
    "ReadinessTimeout",
    "SUPPORTED_SRIDS",
    "SchemaError",
    "SpatialFixtures",
    "TableSchema",
    "UTM_33N",
    "UrbanCentre",
    "WEB_MERCATOR",
    "WGS84",
    "__version__",
    "assert_geometries_equal",
    "assert_geometry_valid",
    "assert_srid",
    "assert_uses_index",
    "assert_within_distance",
    "build_dataset",
    "build_fixtures",
    "dataset_names",
    "edge_case_specs",
    "generate_hull_polygon",
    "generate_points",
    "generate_route",
    "measure_distance",
    "parse_index_names",
    "reproject",
    "schema_sql",
    "select_provider",
]
