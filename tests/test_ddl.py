"""Tests for the pure SQL/DDL generation layer."""

from __future__ import annotations

import pytest

from postgis_fixtures.ddl import (
    Column,
    GeometryColumn,
    IndexSpec,
    TableSchema,
    analyze_statement,
    copy_statement,
    create_index_sql,
    create_table_sql,
    drop_table_sql,
    insert_statement,
    quote_ident,
    quote_literal,
    schema_sql,
)
from postgis_fixtures.errors import SchemaError


@pytest.fixture()
def table() -> TableSchema:
    return TableSchema(
        name="parcels",
        columns=(
            Column("id", "integer", nullable=False, primary_key=True),
            Column("owner", "text", nullable=False),
            Column("valued_at", "numeric(12,2)"),
        ),
        geometry=GeometryColumn("boundary", "POLYGON", 27700, nullable=False),
        comment="Land parcels",
        indexes=(
            IndexSpec("parcels_boundary_gist", ("boundary",), "gist"),
            IndexSpec("parcels_owner_btree", ("owner",), "btree", where="valued_at > 0"),
        ),
    )


class TestQuoting:
    @pytest.mark.parametrize("name", ["cities", "geom_2", "_private"])
    def test_plain_identifiers_are_left_bare(self, name: str) -> None:
        assert quote_ident(name) == name

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("Mixed", '"Mixed"'),
            ("has space", '"has space"'),
            ('quote"inside', '"quote""inside"'),
            ("drop table x;--", '"drop table x;--"'),
        ],
    )
    def test_unsafe_identifiers_are_quoted(self, name: str, expected: str) -> None:
        assert quote_ident(name) == expected

    @pytest.mark.parametrize("name", ["", "nul\x00byte"])
    def test_impossible_identifiers_are_rejected(self, name: str) -> None:
        with pytest.raises(SchemaError, match="Invalid SQL identifier"):
            quote_ident(name)

    def test_literals_escape_single_quotes(self) -> None:
        assert quote_literal("it's") == "'it''s'"


class TestColumns:
    def test_primary_key_wins_over_not_null(self) -> None:
        column = Column("id", "integer", nullable=False, primary_key=True)
        assert column.definition() == "id integer PRIMARY KEY"

    def test_not_null_is_emitted(self) -> None:
        assert Column("name", "text", nullable=False).definition() == "name text NOT NULL"

    def test_nullable_columns_carry_no_suffix(self) -> None:
        assert Column("note", "text").definition() == "note text"

    def test_missing_type_is_rejected(self) -> None:
        with pytest.raises(SchemaError, match="has no type"):
            Column("bad", "   ").definition()


class TestGeometryColumn:
    def test_typed_and_constrained(self) -> None:
        assert (
            GeometryColumn("geom", "point", 4326, nullable=False).definition()
            == "geom geometry(POINT, 4326) NOT NULL"
        )

    def test_unconstrained_column_drops_the_modifier(self) -> None:
        assert GeometryColumn("geom", "GEOMETRY", None, nullable=True).definition() == "geom geometry"

    def test_typed_but_srid_free_column(self) -> None:
        assert (
            GeometryColumn("geom", "LINESTRING", None, nullable=True).definition()
            == "geom geometry(LINESTRING)"
        )

    def test_unknown_geometry_type_is_rejected(self) -> None:
        with pytest.raises(SchemaError, match="Unknown geometry type"):
            GeometryColumn("geom", "BLOB")

    def test_non_positive_srid_is_rejected(self) -> None:
        with pytest.raises(SchemaError, match="SRID must be positive"):
            GeometryColumn("geom", "POINT", 0)


class TestCreateTable:
    def test_renders_columns_with_geometry_last(self, table: TableSchema) -> None:
        sql = create_table_sql(table)
        assert sql.startswith("CREATE TABLE IF NOT EXISTS parcels (")
        assert sql.index("boundary geometry(POLYGON, 27700)") > sql.index("valued_at")
        assert sql.rstrip().endswith(");")

    def test_if_not_exists_can_be_suppressed(self, table: TableSchema) -> None:
        assert create_table_sql(table, if_not_exists=False).startswith("CREATE TABLE parcels")

    def test_schema_qualification(self, table: TableSchema) -> None:
        qualified = TableSchema(
            name="parcels", columns=table.columns, geometry=table.geometry, schema="gis"
        )
        assert "gis.parcels" in create_table_sql(qualified)

    def test_geometry_only_table(self) -> None:
        minimal = TableSchema(name="shapes", columns=(), geometry=GeometryColumn("geom"))
        assert create_table_sql(minimal) == (
            "CREATE TABLE IF NOT EXISTS shapes (\n"
            "    geom geometry(POINT, 4326) NOT NULL\n"
            ");"
        )

    def test_column_names_put_geometry_last(self, table: TableSchema) -> None:
        assert table.column_names == ("id", "owner", "valued_at", "boundary")


class TestCreateIndex:
    def test_gist_index(self, table: TableSchema) -> None:
        assert create_index_sql(table, table.indexes[0]) == (
            "CREATE INDEX IF NOT EXISTS parcels_boundary_gist ON parcels USING gist (boundary);"
        )

    def test_partial_index_keeps_the_predicate(self, table: TableSchema) -> None:
        assert create_index_sql(table, table.indexes[1]).endswith("WHERE valued_at > 0;")

    def test_concurrently_and_no_if_not_exists(self, table: TableSchema) -> None:
        sql = create_index_sql(table, table.indexes[0], concurrently=True, if_not_exists=False)
        assert sql.startswith("CREATE INDEX CONCURRENTLY parcels_boundary_gist")

    def test_covering_index(self, table: TableSchema) -> None:
        index = IndexSpec("parcels_cover", ("owner",), "btree", include=("valued_at",))
        assert "INCLUDE (valued_at)" in create_index_sql(table, index)

    def test_composite_columns(self, table: TableSchema) -> None:
        index = IndexSpec("parcels_multi", ("owner", "valued_at"), "btree")
        assert "(owner, valued_at)" in create_index_sql(table, index)

    def test_unsupported_method_is_rejected(self, table: TableSchema) -> None:
        with pytest.raises(SchemaError, match="Unsupported index method"):
            create_index_sql(table, IndexSpec("bad", ("owner",), "rtree"))

    def test_index_needs_columns(self) -> None:
        with pytest.raises(SchemaError, match="at least one column"):
            IndexSpec("empty", ())


class TestStatements:
    def test_copy_statement(self, table: TableSchema) -> None:
        assert copy_statement(table) == (
            "COPY parcels (id, owner, valued_at, boundary) FROM STDIN"
        )

    def test_copy_statement_with_explicit_columns(self, table: TableSchema) -> None:
        assert copy_statement(table, ("id",)) == "COPY parcels (id) FROM STDIN"

    def test_copy_statement_needs_columns(self, table: TableSchema) -> None:
        with pytest.raises(SchemaError, match="at least one column"):
            copy_statement(table, ())

    def test_insert_statement_placeholders_match_columns(self, table: TableSchema) -> None:
        sql = insert_statement(table)
        assert sql.endswith("VALUES (%s, %s, %s, %s)")

    def test_drop_and_analyze(self, table: TableSchema) -> None:
        assert drop_table_sql(table) == "DROP TABLE IF EXISTS parcels CASCADE;"
        assert drop_table_sql(table, cascade=False) == "DROP TABLE IF EXISTS parcels;"
        assert analyze_statement(table) == "ANALYZE parcels;"


class TestSchemaSql:
    def test_full_script_order(self, table: TableSchema) -> None:
        script = schema_sql(table)
        lines = script.splitlines()
        assert lines[0].startswith("CREATE TABLE")
        assert "COMMENT ON TABLE parcels IS 'Land parcels';" in script
        assert script.index("CREATE INDEX") < script.index("ANALYZE")
        assert lines[-1] == "ANALYZE parcels;"

    def test_indexes_can_be_omitted(self, table: TableSchema) -> None:
        assert "CREATE INDEX" not in schema_sql(table, include_indexes=False)
