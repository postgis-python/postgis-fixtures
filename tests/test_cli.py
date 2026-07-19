"""Tests for the offline inspection CLI."""

from __future__ import annotations

import io
import json

import pytest

from postgis_fixtures.cli import main
from postgis_fixtures.crs import BRITISH_NATIONAL_GRID


def run(*argv: str) -> str:
    """Run the CLI and return everything it wrote to stdout."""
    buffer = io.StringIO()
    assert main(list(argv), out=buffer) == 0
    return buffer.getvalue()


class TestList:
    def test_names_every_dataset_with_its_description(self) -> None:
        output = run("list")
        assert "cities" in output
        assert "edge_cases" in output
        assert "clustered around urban centres" in output
        assert len(output.strip().splitlines()) == 5


class TestDdl:
    def test_prints_every_table_by_default(self) -> None:
        output = run("ddl")
        assert output.count("CREATE TABLE IF NOT EXISTS") == 5

    def test_restricts_to_the_requested_dataset(self) -> None:
        output = run("ddl", "--dataset", "cities")
        assert "CREATE TABLE IF NOT EXISTS cities" in output
        assert "service_areas" not in output

    def test_indexes_can_be_omitted(self) -> None:
        assert "CREATE INDEX" not in run("ddl", "--dataset", "cities", "--no-indexes")

    def test_srid_flag_changes_the_column_type(self) -> None:
        output = run("--srid", str(BRITISH_NATIONAL_GRID), "ddl", "--dataset", "cities")
        assert "geometry(POINT, 27700)" in output


class TestSample:
    def test_prints_the_requested_number_of_rows(self) -> None:
        output = run("sample", "cities", "--rows", "3")
        assert output.count("POINT (") == 3

    def test_json_output_is_machine_readable(self) -> None:
        records = json.loads(run("sample", "delivery_routes", "--rows", "2", "--json"))
        assert len(records) == 2
        assert records[0]["code"] == "RT-0001"
        assert records[0]["wkt"].startswith("LINESTRING (")
        assert records[0]["srid"] == 4326

    def test_seed_changes_the_geometry(self) -> None:
        first = run("--seed", "1", "sample", "cities", "--rows", "1")
        second = run("--seed", "2", "sample", "cities", "--rows", "1")
        assert first != second

    def test_same_seed_reproduces_output_exactly(self) -> None:
        assert run("--seed", "7", "sample", "cities", "--rows", "4") == run(
            "--seed", "7", "sample", "cities", "--rows", "4"
        )

    def test_null_geometry_is_rendered_as_none(self) -> None:
        records = json.loads(run("sample", "edge_cases", "--rows", "7", "--json"))
        assert any(record["wkt"] is None for record in records)

    def test_zero_rows_is_rejected(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["sample", "cities", "--rows", "0"], out=io.StringIO()) == 2
        assert "--rows must be at least 1" in capsys.readouterr().err


class TestEdgeCases:
    def test_describes_every_case_with_its_reason(self) -> None:
        output = run("edge-cases")
        assert "wrong_srid (SRID 3857)" in output
        assert "null_geometry (SRID -)" in output
        assert "NULL" in output
        assert "Crosses the 180th meridian" in output


class TestArgumentParsing:
    def test_a_missing_subcommand_is_an_error(self) -> None:
        with pytest.raises(SystemExit):
            main([])

    def test_an_unknown_dataset_is_rejected_by_argparse(self) -> None:
        with pytest.raises(SystemExit):
            main(["sample", "towns"])

    def test_an_unsupported_srid_is_rejected_by_argparse(self) -> None:
        with pytest.raises(SystemExit):
            main(["--srid", "1234", "ddl"])
