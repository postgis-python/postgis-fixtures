"""Pure SQL/DDL generation.

Nothing in this module touches a database. Every function takes plain data and
returns a string, which means the whole schema layer is unit-testable offline
and diffable in review.

Identifiers are quoted with the PostgreSQL rules (double quotes, doubled inner
quotes) and validated, so a dataset name can never smuggle SQL into DDL.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .errors import SchemaError

#: PostGIS geometry subtypes accepted in a typed geometry column.
GEOMETRY_TYPES: frozenset[str] = frozenset(
    {
        "POINT",
        "LINESTRING",
        "POLYGON",
        "MULTIPOINT",
        "MULTILINESTRING",
        "MULTIPOLYGON",
        "GEOMETRYCOLLECTION",
        "GEOMETRY",
    }
)

#: Index access methods that make sense over a geometry column.
SPATIAL_INDEX_METHODS: frozenset[str] = frozenset({"gist", "spgist", "brin"})

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


def quote_ident(identifier: str) -> str:
    """Quote a SQL identifier, rejecting anything that cannot be an identifier.

    Simple lower-case identifiers are returned unquoted for readable DDL;
    everything else is double-quoted.
    """
    if not identifier or "\x00" in identifier:
        raise SchemaError(f"Invalid SQL identifier: {identifier!r}")
    if _IDENT_RE.match(identifier) and identifier == identifier.lower():
        return identifier
    return '"' + identifier.replace('"', '""') + '"'


def quote_literal(value: str) -> str:
    """Quote a string literal for inclusion in generated SQL."""
    return "'" + value.replace("'", "''") + "'"


@dataclass(frozen=True)
class Column:
    """A non-spatial column in a fixture table."""

    name: str
    type: str
    nullable: bool = True
    primary_key: bool = False

    def definition(self) -> str:
        """Return the ``CREATE TABLE`` fragment for this column."""
        if not self.type.strip():
            raise SchemaError(f"Column {self.name!r} has no type")
        parts = [quote_ident(self.name), self.type.strip()]
        if self.primary_key:
            parts.append("PRIMARY KEY")
        elif not self.nullable:
            parts.append("NOT NULL")
        return " ".join(parts)


@dataclass(frozen=True)
class GeometryColumn:
    """A typed PostGIS geometry column.

    Attributes:
        name: Column name.
        geometry_type: One of :data:`GEOMETRY_TYPES`.
        srid: SRID constraint applied by the ``geometry(type, srid)`` modifier,
            or ``None`` for an unconstrained ``geometry`` column. The edge-case
            dataset needs ``None`` so it can hold a wrong-SRID row.
        nullable: Whether NULL geometry is allowed. The edge-case dataset needs
            this to be ``True``; production-shaped tables usually want ``False``.
    """

    name: str
    geometry_type: str = "POINT"
    srid: int | None = 4326
    nullable: bool = False

    def __post_init__(self) -> None:
        if self.geometry_type.upper() not in GEOMETRY_TYPES:
            raise SchemaError(
                f"Unknown geometry type {self.geometry_type!r}; "
                f"expected one of {', '.join(sorted(GEOMETRY_TYPES))}"
            )
        if self.srid is not None and self.srid <= 0:
            raise SchemaError(f"SRID must be positive, got {self.srid}")

    def definition(self) -> str:
        """Return the ``CREATE TABLE`` fragment for this geometry column."""
        geometry_type = self.geometry_type.upper()
        if self.srid is None:
            sql_type = "geometry" if geometry_type == "GEOMETRY" else f"geometry({geometry_type})"
        else:
            sql_type = f"geometry({geometry_type}, {self.srid})"
        suffix = "" if self.nullable else " NOT NULL"
        return f"{quote_ident(self.name)} {sql_type}{suffix}"


@dataclass(frozen=True)
class TableSchema:
    """The full definition of a fixture table."""

    name: str
    columns: tuple[Column, ...]
    geometry: GeometryColumn
    schema: str | None = None
    comment: str | None = None
    indexes: tuple["IndexSpec", ...] = field(default_factory=tuple)

    @property
    def qualified_name(self) -> str:
        """Return the optionally schema-qualified, quoted table name."""
        table = quote_ident(self.name)
        return f"{quote_ident(self.schema)}.{table}" if self.schema else table

    @property
    def column_names(self) -> tuple[str, ...]:
        """Return every column name in load order, geometry last."""
        return tuple(column.name for column in self.columns) + (self.geometry.name,)


@dataclass(frozen=True)
class IndexSpec:
    """A single index over a fixture table."""

    name: str
    columns: tuple[str, ...]
    method: str = "gist"
    where: str | None = None
    include: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.columns:
            raise SchemaError(f"Index {self.name!r} needs at least one column")


def create_table_sql(table: TableSchema, *, if_not_exists: bool = True) -> str:
    """Render ``CREATE TABLE`` for a fixture table.

    The geometry column is always emitted last so the column order matches
    :meth:`TableSchema.column_names` and therefore the ``COPY`` payload.
    """
    if not table.columns and table.geometry is None:  # pragma: no cover - guard
        raise SchemaError(f"Table {table.name!r} has no columns")
    exists = "IF NOT EXISTS " if if_not_exists else ""
    body = ",\n".join(
        f"    {column.definition()}" for column in table.columns
    )
    body = f"{body},\n    {table.geometry.definition()}" if body else f"    {table.geometry.definition()}"
    return f"CREATE TABLE {exists}{table.qualified_name} (\n{body}\n);"


def create_index_sql(
    table: TableSchema, index: IndexSpec, *, if_not_exists: bool = True, concurrently: bool = False
) -> str:
    """Render ``CREATE INDEX`` for one :class:`IndexSpec`.

    Args:
        table: The table the index belongs to.
        index: The index definition.
        if_not_exists: Emit ``IF NOT EXISTS``.
        concurrently: Emit ``CONCURRENTLY``. This cannot run inside a
            transaction block, so the fixtures never use it by default; it is
            available for tests that reproduce a zero-downtime index build.
    """
    method = index.method.lower()
    if method not in SPATIAL_INDEX_METHODS and method not in {"btree", "hash"}:
        raise SchemaError(
            f"Unsupported index method {index.method!r} for index {index.name!r}"
        )
    parts = ["CREATE INDEX"]
    if concurrently:
        parts.append("CONCURRENTLY")
    if if_not_exists:
        parts.append("IF NOT EXISTS")
    parts.append(quote_ident(index.name))
    parts.append(f"ON {table.qualified_name}")
    parts.append(f"USING {method}")
    parts.append("(" + ", ".join(quote_ident(c) for c in index.columns) + ")")
    if index.include:
        parts.append("INCLUDE (" + ", ".join(quote_ident(c) for c in index.include) + ")")
    if index.where:
        parts.append(f"WHERE {index.where}")
    return " ".join(parts) + ";"


def drop_table_sql(table: TableSchema, *, cascade: bool = True) -> str:
    """Render ``DROP TABLE IF EXISTS`` for a fixture table."""
    suffix = " CASCADE" if cascade else ""
    return f"DROP TABLE IF EXISTS {table.qualified_name}{suffix};"


def copy_statement(table: TableSchema, columns: tuple[str, ...] | None = None) -> str:
    """Render the ``COPY ... FROM STDIN`` statement used by the bulk loader.

    Text format is used rather than binary so the payload stays inspectable;
    the geometry column is fed hex EWKB, which PostGIS casts on input.
    """
    names = table.column_names if columns is None else columns
    if not names:
        raise SchemaError(f"COPY into {table.name!r} needs at least one column")
    column_list = ", ".join(quote_ident(name) for name in names)
    return f"COPY {table.qualified_name} ({column_list}) FROM STDIN"


def insert_statement(table: TableSchema, columns: tuple[str, ...] | None = None) -> str:
    """Render a parameterised ``INSERT`` used for small or partial loads."""
    names = table.column_names if columns is None else columns
    column_list = ", ".join(quote_ident(name) for name in names)
    placeholders = ", ".join("%s" for _ in names)
    return f"INSERT INTO {table.qualified_name} ({column_list}) VALUES ({placeholders})"


def analyze_statement(table: TableSchema) -> str:
    """Render ``ANALYZE`` for a table.

    Loading fixtures without analysing them is the single most common reason a
    test sees a sequential scan where production sees an index scan.
    """
    return f"ANALYZE {table.qualified_name};"


def schema_sql(table: TableSchema, *, include_indexes: bool = True) -> str:
    """Render the full DDL for a table: create, indexes, analyze."""
    statements = [create_table_sql(table)]
    if table.comment:
        statements.append(
            f"COMMENT ON TABLE {table.qualified_name} IS {quote_literal(table.comment)};"
        )
    if include_indexes:
        statements.extend(create_index_sql(table, index) for index in table.indexes)
    statements.append(analyze_statement(table))
    return "\n".join(statements)
