"""Tests for the PostgisDB facade, driven by a fake DB-API connection."""

from __future__ import annotations

from typing import Any, Sequence

import pytest

from postgis_fixtures.datasets import build_dataset
from postgis_fixtures.db import PostgisDB
from postgis_fixtures.ddl import Column, GeometryColumn, IndexSpec, TableSchema
from postgis_fixtures.errors import PostgisFixturesError
from postgis_fixtures.geometry import GeneratorConfig


class FakeCopy:
    """Records the rows written through a COPY stream."""

    def __init__(self, statement: str, sink: list[Any]) -> None:
        self.statement = statement
        self._sink = sink

    def __enter__(self) -> "FakeCopy":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def write_row(self, row: Sequence[Any]) -> None:
        self._sink.append(tuple(row))


class FakeCursor:
    """A minimal psycopg-shaped cursor that records SQL and replays results."""

    def __init__(self, connection: "FakeConnection") -> None:
        self._connection = connection
        self.rowcount = 0
        self._rows: list[tuple[Any, ...]] = []

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> None:
        self._connection.statements.append((sql, params))
        self._rows = list(self._connection.results.get(sql, []))
        self.rowcount = len(self._rows) or self._connection.default_rowcount

    def executemany(self, sql: str, rows: Sequence[Sequence[Any]]) -> None:
        self._connection.statements.append((sql, None))
        self._connection.inserted.extend(tuple(row) for row in rows)
        self.rowcount = len(rows)

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows

    def copy(self, statement: str) -> FakeCopy:
        return FakeCopy(statement, self._connection.copied)


class FakeConnection:
    """A fake connection whose cursors can optionally lack COPY support."""

    def __init__(self, *, default_rowcount: int = 0) -> None:
        self.statements: list[tuple[str, Any]] = []
        self.results: dict[str, list[tuple[Any, ...]]] = {}
        self.copied: list[tuple[Any, ...]] = []
        self.inserted: list[tuple[Any, ...]] = []
        self.default_rowcount = default_rowcount

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    @property
    def sql(self) -> list[str]:
        return [statement for statement, _ in self.statements]


class NoCopyCursor(FakeCursor):
    """A cursor without a ``copy`` method, like a plain DB-API 2.0 driver."""

    def __getattribute__(self, name: str) -> Any:
        if name == "copy":
            raise AttributeError(name)
        return super().__getattribute__(name)


class NoCopyConnection(FakeConnection):
    def cursor(self) -> FakeCursor:
        return NoCopyCursor(self)


@pytest.fixture()
def connection() -> FakeConnection:
    return FakeConnection()


@pytest.fixture()
def db(connection: FakeConnection) -> PostgisDB:
    return PostgisDB(connection, dsn="postgresql://localhost/gis")


class TestQueryHelpers:
    def test_execute_returns_rowcount(self, connection: FakeConnection, db: PostgisDB) -> None:
        connection.default_rowcount = 7
        assert db.execute("DELETE FROM cities") == 7
        assert connection.statements == [("DELETE FROM cities", None)]

    def test_execute_passes_parameters_through(self, connection: FakeConnection, db: PostgisDB) -> None:
        db.execute("DELETE FROM cities WHERE id = %s", (4,))
        assert connection.statements[-1] == ("DELETE FROM cities WHERE id = %s", (4,))

    def test_fetchall_and_fetchone(self, connection: FakeConnection, db: PostgisDB) -> None:
        connection.results["SELECT id FROM cities"] = [(1,), (2,)]
        assert db.fetchall("SELECT id FROM cities") == [(1,), (2,)]
        assert db.fetchone("SELECT id FROM cities") == (1,)

    def test_fetchone_returns_none_when_empty(self, db: PostgisDB) -> None:
        assert db.fetchone("SELECT id FROM cities") is None

    def test_scalar_unwraps_the_first_column(self, connection: FakeConnection, db: PostgisDB) -> None:
        connection.results["SELECT count(*) FROM cities"] = [(42,)]
        assert db.scalar("SELECT count(*) FROM cities") == 42

    def test_scalar_on_no_rows_is_an_error(self, db: PostgisDB) -> None:
        with pytest.raises(PostgisFixturesError, match="expected one"):
            db.scalar("SELECT 1 WHERE false")

    def test_connection_is_exposed(self, connection: FakeConnection, db: PostgisDB) -> None:
        assert db.connection is connection

    def test_postgis_version(self, connection: FakeConnection, db: PostgisDB) -> None:
        connection.results["SELECT postgis_lib_version()"] = [("3.4.2",)]
        assert db.postgis_version() == "3.4.2"


class TestSchemaOperations:
    def test_create_table_drops_then_creates_then_indexes(
        self, connection: FakeConnection, db: PostgisDB
    ) -> None:
        dataset = build_dataset("cities", GeneratorConfig(seed=1), rows=2)
        db.create_table(dataset)
        assert connection.sql[0].startswith("DROP TABLE IF EXISTS cities")
        assert connection.sql[1].startswith("CREATE TABLE IF NOT EXISTS cities")
        assert any("USING gist (geom)" in sql for sql in connection.sql)

    def test_create_table_can_skip_the_drop_and_indexes(
        self, connection: FakeConnection, db: PostgisDB
    ) -> None:
        dataset = build_dataset("cities", GeneratorConfig(seed=1), rows=2)
        db.create_table(dataset, drop_first=False, include_indexes=False)
        assert len(connection.sql) == 1
        assert connection.sql[0].startswith("CREATE TABLE")

    def test_create_table_accepts_a_bare_schema(self, connection: FakeConnection, db: PostgisDB) -> None:
        table = TableSchema(
            name="zones",
            columns=(Column("id", "integer", nullable=False, primary_key=True),),
            geometry=GeometryColumn("geom", "POLYGON", 4326),
            indexes=(IndexSpec("zones_geom_gist", ("geom",)),),
        )
        assert db.create_table(table) is table
        assert any("zones_geom_gist" in sql for sql in connection.sql)

    def test_drop(self, connection: FakeConnection, db: PostgisDB) -> None:
        db.drop(build_dataset("cities", rows=1))
        assert connection.sql == ["DROP TABLE IF EXISTS cities CASCADE;"]

    def test_install_extensions_quotes_names(self, connection: FakeConnection, db: PostgisDB) -> None:
        db.install_extensions(("postgis", "postgis_raster"))
        assert connection.sql == [
            "CREATE EXTENSION IF NOT EXISTS postgis",
            "CREATE EXTENSION IF NOT EXISTS postgis_raster",
        ]


class TestLoading:
    def test_copy_path_writes_every_row_in_column_order(
        self, connection: FakeConnection, db: PostgisDB
    ) -> None:
        dataset = build_dataset("cities", GeneratorConfig(seed=2), rows=5)
        assert db.load(dataset) == 5
        assert len(connection.copied) == 5
        assert connection.copied[0] == dataset.values()[0]
        assert connection.sql[-1] == "ANALYZE cities;"

    def test_analyze_can_be_suppressed(self, connection: FakeConnection, db: PostgisDB) -> None:
        db.load(build_dataset("cities", rows=2), analyze=False)
        assert not any(sql.startswith("ANALYZE") for sql in connection.sql)

    def test_empty_dataset_loads_nothing(self, connection: FakeConnection, db: PostgisDB) -> None:
        assert db.load(build_dataset("cities", rows=0)) == 0
        assert connection.sql == []

    def test_insert_path_is_used_when_copy_is_disabled(self) -> None:
        connection = FakeConnection()
        db = PostgisDB(connection)
        dataset = build_dataset("cities", GeneratorConfig(seed=3), rows=4)
        assert db.load(dataset, use_copy=False) == 4
        assert connection.inserted == dataset.values()
        assert not connection.copied

    def test_insert_path_is_used_when_the_driver_lacks_copy(self) -> None:
        connection = NoCopyConnection()
        db = PostgisDB(connection)
        dataset = build_dataset("cities", GeneratorConfig(seed=3), rows=3)
        assert db.load(dataset) == 3
        assert len(connection.inserted) == 3

    def test_inserts_are_batched(self) -> None:
        connection = FakeConnection()
        db = PostgisDB(connection)
        dataset = build_dataset("sensor_readings", rows=1200)
        db.load(dataset, use_copy=False, analyze=False)
        insert_calls = [sql for sql in connection.sql if sql.startswith("INSERT")]
        assert len(insert_calls) == 3
        assert len(connection.inserted) == 1200

    def test_null_geometry_survives_the_load(self, connection: FakeConnection, db: PostgisDB) -> None:
        dataset = build_dataset("edge_cases")
        db.load(dataset)
        geometry_index = dataset.column_names.index("geom")
        assert any(row[geometry_index] is None for row in connection.copied)


class TestCountAndExplain:
    def test_count_accepts_datasets_schemas_and_names(
        self, connection: FakeConnection, db: PostgisDB
    ) -> None:
        connection.results["SELECT count(*) FROM cities"] = [(11,)]
        dataset = build_dataset("cities", rows=1)
        assert db.count(dataset) == 11
        assert db.count(dataset.table) == 11
        assert db.count("cities") == 11

    def test_count_with_a_predicate(self, connection: FakeConnection, db: PostgisDB) -> None:
        sql = "SELECT count(*) FROM cities WHERE population > 1000"
        connection.results[sql] = [(3,)]
        assert db.count("cities", where="population > 1000") == 3

    def test_explain_joins_plan_lines(self, connection: FakeConnection, db: PostgisDB) -> None:
        connection.results["EXPLAIN SELECT 1"] = [("Result  (cost=0.00..0.01)",), ("  more",)]
        assert db.explain("SELECT 1") == "Result  (cost=0.00..0.01)\n  more"

    def test_explain_analyze_buffers_prefix(self, connection: FakeConnection, db: PostgisDB) -> None:
        db.explain("SELECT 1", analyze=True, buffers=True)
        assert connection.sql[-1] == "EXPLAIN (ANALYZE, BUFFERS) SELECT 1"

    def test_explain_costs_off(self, connection: FakeConnection, db: PostgisDB) -> None:
        db.explain("SELECT 1", costs=False)
        assert connection.sql[-1] == "EXPLAIN (COSTS OFF) SELECT 1"

    def test_buffers_without_analyze_is_rejected(self, db: PostgisDB) -> None:
        with pytest.raises(PostgisFixturesError, match="requires ANALYZE"):
            db.explain("SELECT 1", buffers=True)

    def test_explain_forwards_parameters(self, connection: FakeConnection, db: PostgisDB) -> None:
        db.explain("SELECT * FROM cities WHERE id = %s", (1,))
        assert connection.statements[-1][1] == (1,)
