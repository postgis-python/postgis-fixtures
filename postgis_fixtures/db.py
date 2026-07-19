"""A small facade over a DB-API connection, tuned for spatial test suites.

The facade is deliberately thin: it owns no connection lifecycle, opens no
transactions of its own, and holds no state beyond the connection it was handed.
That is what lets the plugin wrap every test in a transaction and roll it back
afterwards while still handing the test a fully featured helper object.

It is duck-typed against psycopg 3, so a fake connection in a unit test only has
to implement ``cursor()`` (plus ``copy()`` on the cursor if bulk loading is
exercised).
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from .datasets import Dataset
from .ddl import (
    TableSchema,
    analyze_statement,
    copy_statement,
    create_index_sql,
    create_table_sql,
    drop_table_sql,
    insert_statement,
    quote_ident,
)
from .errors import PostgisFixturesError

#: Rows per ``executemany`` batch when COPY is unavailable.
INSERT_BATCH_SIZE = 500


class PostgisDB:
    """Query helpers bound to one open connection.

    Args:
        connection: An open DB-API connection (psycopg 3 in normal use).
        dsn: The DSN the connection was opened from, for diagnostics.
    """

    def __init__(self, connection: Any, dsn: str = "") -> None:
        self._connection = connection
        self.dsn = dsn

    @property
    def connection(self) -> Any:
        """Return the underlying connection."""
        return self._connection

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> int:
        """Execute a statement and return its ``rowcount``.

        Multi-statement strings (such as generated DDL) are executed as-is;
        PostgreSQL accepts them when no parameters are bound.
        """
        with self._connection.cursor() as cursor:
            if params is None:
                cursor.execute(sql)
            else:
                cursor.execute(sql, params)
            rowcount = getattr(cursor, "rowcount", -1)
        return int(rowcount if rowcount is not None else -1)

    def fetchall(self, sql: str, params: Sequence[Any] | None = None) -> list[tuple[Any, ...]]:
        """Execute a query and return every row."""
        with self._connection.cursor() as cursor:
            if params is None:
                cursor.execute(sql)
            else:
                cursor.execute(sql, params)
            return list(cursor.fetchall())

    def fetchone(self, sql: str, params: Sequence[Any] | None = None) -> tuple[Any, ...] | None:
        """Execute a query and return the first row, or ``None``."""
        rows = self.fetchall(sql, params)
        return rows[0] if rows else None

    def scalar(self, sql: str, params: Sequence[Any] | None = None) -> Any:
        """Execute a query and return the first column of the first row.

        Raises:
            PostgisFixturesError: when the query returned no rows.
        """
        row = self.fetchone(sql, params)
        if row is None:
            raise PostgisFixturesError(f"Query returned no rows, expected one: {sql}")
        return row[0]

    def create_table(
        self, dataset: Dataset | TableSchema, *, include_indexes: bool = True, drop_first: bool = True
    ) -> TableSchema:
        """Create a fixture table (and its indexes), returning its schema."""
        table = dataset.table if isinstance(dataset, Dataset) else dataset
        if drop_first:
            self.execute(drop_table_sql(table))
        self.execute(create_table_sql(table))
        if include_indexes:
            for index in table.indexes:
                self.execute(create_index_sql(table, index))
        return table

    def load(self, dataset: Dataset, *, use_copy: bool = True, analyze: bool = True) -> int:
        """Bulk-load a dataset's rows and return the number loaded.

        ``COPY`` is used when the driver exposes it, because inserting a few
        hundred thousand geometries row-by-row dominates test runtime. The
        ``executemany`` path is kept for drivers and fakes without COPY.
        """
        table = dataset.table
        columns = dataset.column_names
        rows = dataset.values()
        if not rows:
            return 0
        loaded = self._copy_rows(table, columns, rows) if use_copy else -1
        if loaded < 0:
            loaded = self._insert_rows(table, columns, rows)
        if analyze:
            self.execute(analyze_statement(table))
        return loaded

    def _copy_rows(
        self, table: TableSchema, columns: tuple[str, ...], rows: Sequence[tuple[Any, ...]]
    ) -> int:
        """Load rows with ``COPY``; return ``-1`` if the driver cannot do it."""
        with self._connection.cursor() as cursor:
            if not hasattr(cursor, "copy"):
                return -1
            statement = copy_statement(table, columns)
            with cursor.copy(statement) as copy:
                for row in rows:
                    copy.write_row(row)
        return len(rows)

    def _insert_rows(
        self, table: TableSchema, columns: tuple[str, ...], rows: Sequence[tuple[Any, ...]]
    ) -> int:
        """Load rows with batched ``executemany``."""
        statement = insert_statement(table, columns)
        with self._connection.cursor() as cursor:
            for start in range(0, len(rows), INSERT_BATCH_SIZE):
                cursor.executemany(statement, rows[start : start + INSERT_BATCH_SIZE])
        return len(rows)

    def count(self, table: Dataset | TableSchema | str, where: str | None = None) -> int:
        """Return ``COUNT(*)`` for a table, optionally filtered.

        ``where`` is interpolated verbatim, so pass only test-authored SQL.
        """
        name = _qualified_name(table)
        sql = f"SELECT count(*) FROM {name}"
        if where:
            sql += f" WHERE {where}"
        return int(self.scalar(sql))

    def explain(
        self,
        sql: str,
        params: Sequence[Any] | None = None,
        *,
        analyze: bool = False,
        buffers: bool = False,
        costs: bool = True,
    ) -> str:
        """Return the plan for ``sql`` as text.

        ``EXPLAIN (ANALYZE, BUFFERS)`` is the form worth reaching for when a
        test is about index usage rather than shape: it reports the rows the
        plan actually touched and the pages it read to do so.
        """
        options = []
        if analyze:
            options.append("ANALYZE")
        if buffers:
            if not analyze:
                raise PostgisFixturesError("EXPLAIN BUFFERS requires ANALYZE")
            options.append("BUFFERS")
        if not costs:
            options.append("COSTS OFF")
        prefix = f"EXPLAIN ({', '.join(options)}) " if options else "EXPLAIN "
        rows = self.fetchall(prefix + sql, params)
        return "\n".join(str(row[0]) for row in rows)

    def drop(self, dataset: Dataset | TableSchema) -> None:
        """Drop a fixture table if it exists."""
        table = dataset.table if isinstance(dataset, Dataset) else dataset
        self.execute(drop_table_sql(table))

    def install_extensions(self, names: Iterable[str] = ("postgis",)) -> None:
        """Ensure the given extensions exist. Requires a superuser connection."""
        for name in names:
            self.execute(f"CREATE EXTENSION IF NOT EXISTS {quote_ident(name)}")

    def postgis_version(self) -> str:
        """Return the server's PostGIS version string."""
        return str(self.scalar("SELECT postgis_lib_version()"))


def _qualified_name(table: Dataset | TableSchema | str) -> str:
    """Return a quoted, optionally schema-qualified table name."""
    if isinstance(table, str):
        return quote_ident(table)
    schema = table.table if isinstance(table, Dataset) else table
    return schema.qualified_name
