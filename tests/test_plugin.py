"""Tests for the pytest plugin.

The option/ini parsing helpers are tested directly; the fixture wiring is tested
through ``pytester``, which runs a real (throwaway) pytest session against a
generated test file. A fake provider and a fake psycopg connection are injected
from the generated ``conftest.py``, so these tests exercise the genuine fixture
graph — scopes, ordering, teardown — with no Docker and no network.
"""

from __future__ import annotations

import pytest

from postgis_fixtures.errors import PostgisFixturesError
from postgis_fixtures.geometry import GeneratorConfig
from postgis_fixtures.plugin import parse_row_counts, resolve_seed, sqlalchemy_url

#: Injected into every generated project: a provider and connection that record
#: what the plugin did to them without touching a database.
FAKE_CONFTEST = '''
import psycopg

import postgis_fixtures.plugin as plugin

pytest_plugins = ["postgis_fixtures.plugin"]

EVENTS = []


class FakeProvider:
    def __init__(self, choice):
        self.choice = choice
        self.description = "fake provider"

    def start(self):
        EVENTS.append("start")
        return "postgresql://fake:pw@localhost:5432/gis"

    def stop(self):
        EVENTS.append("stop")

    def probe(self, dsn):
        EVENTS.append(f"probe:{dsn}")


class FakeCursor:
    def __init__(self, connection):
        self._connection = connection
        self.rowcount = 0
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def execute(self, sql, params=None):
        self._connection.statements.append(sql)
        self._rows = [(len(self._connection.copied),)] if "count(*)" in sql else []

    def executemany(self, sql, rows):
        self._connection.statements.append(sql)

    def fetchall(self):
        return self._rows

    def copy(self, statement):
        return FakeCopy(self._connection)


class FakeCopy:
    def __init__(self, connection):
        self._connection = connection

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def write_row(self, row):
        self._connection.copied.append(tuple(row))


class FakeConnection:
    def __init__(self, dsn, **kwargs):
        self.dsn = dsn
        self.statements = []
        self.copied = []
        EVENTS.append("connect")

    def cursor(self):
        return FakeCursor(self)

    def rollback(self):
        EVENTS.append("rollback")

    def close(self):
        EVENTS.append("close")


def pytest_configure(config):
    plugin.PROVIDER_FACTORY = FakeProvider
    plugin.testcontainers_available = lambda: True
    psycopg.connect = FakeConnection
'''


class TestResolveSeed:
    def test_cli_wins(self) -> None:
        assert resolve_seed(99, "123") == 99

    def test_ini_is_used_when_no_flag(self) -> None:
        assert resolve_seed(None, " 123 ") == 123

    def test_default_when_neither_is_set(self) -> None:
        assert resolve_seed(None, "") == GeneratorConfig().seed

    def test_non_integer_ini_is_reported_clearly(self) -> None:
        with pytest.raises(PostgisFixturesError, match="must be an integer"):
            resolve_seed(None, "spring")


class TestParseRowCounts:
    def test_parses_pairs_and_ignores_blanks_and_comments(self) -> None:
        assert parse_row_counts(["cities = 10", "", "# skip me", "service_areas=2"]) == {
            "cities": 10,
            "service_areas": 2,
        }

    def test_missing_equals_is_rejected(self) -> None:
        with pytest.raises(PostgisFixturesError, match="name=count form"):
            parse_row_counts(["cities 10"])

    def test_non_integer_count_is_rejected(self) -> None:
        with pytest.raises(PostgisFixturesError, match="non-integer count"):
            parse_row_counts(["cities=lots"])

    def test_negative_count_is_rejected(self) -> None:
        with pytest.raises(PostgisFixturesError, match="negative count"):
            parse_row_counts(["cities=-1"])


class TestOfflineFixtures:
    """``spatial_fixtures`` must work with no database and no provider at all."""

    def test_spatial_fixtures_needs_no_database(self, pytester: pytest.Pytester) -> None:
        pytester.makeconftest('pytest_plugins = ["postgis_fixtures.plugin"]')
        pytester.makepyfile(
            """
            def test_datasets_are_generated(spatial_fixtures):
                assert len(spatial_fixtures.cities) == 200
                assert "edge_cases" in spatial_fixtures
            """
        )
        pytester.runpytest().assert_outcomes(passed=1)

    def test_seed_flag_reaches_the_generator(self, pytester: pytest.Pytester) -> None:
        pytester.makeconftest('pytest_plugins = ["postgis_fixtures.plugin"]')
        pytester.makepyfile(
            """
            def test_seed(spatial_fixtures, postgis_seed):
                assert postgis_seed == 4242
                assert spatial_fixtures.config.seed == 4242
            """
        )
        pytester.runpytest("--postgis-seed", "4242").assert_outcomes(passed=1)

    def test_row_counts_ini_is_honoured(self, pytester: pytest.Pytester) -> None:
        pytester.makeconftest('pytest_plugins = ["postgis_fixtures.plugin"]')
        pytester.makeini(
            """
            [pytest]
            postgis_row_counts =
                cities = 12
                delivery_routes = 3
            """
        )
        pytester.makepyfile(
            """
            def test_counts(spatial_fixtures):
                assert len(spatial_fixtures.cities) == 12
                assert len(spatial_fixtures.delivery_routes) == 3
            """
        )
        pytester.runpytest().assert_outcomes(passed=1)

    def test_bad_seed_ini_fails_loudly(self, pytester: pytest.Pytester) -> None:
        pytester.makeconftest('pytest_plugins = ["postgis_fixtures.plugin"]')
        pytester.makeini("[pytest]\npostgis_seed = later\n")
        pytester.makepyfile("def test_seed(postgis_seed): pass")
        result = pytester.runpytest()
        result.assert_outcomes(errors=1)
        result.stdout.fnmatch_lines(["*postgis_seed must be an integer*"])

    def test_marker_is_registered(self, pytester: pytest.Pytester) -> None:
        pytester.makeconftest('pytest_plugins = ["postgis_fixtures.plugin"]')
        result = pytester.runpytest("--markers")
        result.stdout.fnmatch_lines(["*@pytest.mark.postgis*"])


class TestDatabaseFixtures:
    """The database fixtures, wired to a fake provider and fake driver."""

    @pytest.fixture(autouse=True)
    def project(self, pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("POSTGIS_FIXTURES_DSN", raising=False)
        pytester.makeconftest(FAKE_CONFTEST)

    def test_dsn_is_probed_before_use(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile(
            """
            from conftest import EVENTS

            def test_dsn(postgis_dsn):
                assert postgis_dsn == "postgresql://fake:pw@localhost:5432/gis"
                assert EVENTS[0] == "start"
                assert EVENTS[1].startswith("probe:")
            """
        )
        pytester.runpytest().assert_outcomes(passed=1)

    def test_provider_starts_once_and_stops_at_session_end(
        self, pytester: pytest.Pytester
    ) -> None:
        pytester.makepyfile(
            """
            from conftest import EVENTS

            def test_one(postgis_dsn): pass

            def test_two(postgis_dsn): pass

            def test_started_once(postgis_dsn):
                assert EVENTS.count("start") == 1
                assert "stop" not in EVENTS
            """
        )
        pytester.runpytest().assert_outcomes(passed=3)

    def test_connection_is_rolled_back_and_closed_per_test(
        self, pytester: pytest.Pytester
    ) -> None:
        pytester.makepyfile(
            """
            from conftest import EVENTS

            def test_first(postgis_connection):
                assert EVENTS.count("connect") == 1

            def test_second(postgis_connection):
                assert EVENTS.count("connect") == 2
                assert EVENTS.count("rollback") == 1
                assert EVENTS.count("close") == 1
            """
        )
        pytester.runpytest().assert_outcomes(passed=2)

    def test_postgis_db_creates_and_loads_a_dataset(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile(
            """
            def test_load(postgis_db, spatial_fixtures):
                cities = spatial_fixtures["cities"]
                postgis_db.create_table(cities)
                assert postgis_db.load(cities) == len(cities)
                statements = postgis_db.connection.statements
                assert any(s.startswith("CREATE TABLE") for s in statements)
                assert any("USING gist (geom)" in s for s in statements)
                assert postgis_db.count("cities") == len(cities)
            """
        )
        pytester.runpytest().assert_outcomes(passed=1)

    def test_postgis_db_carries_the_dsn(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile(
            """
            def test_dsn(postgis_db, postgis_dsn):
                assert postgis_db.dsn == postgis_dsn
            """
        )
        pytester.runpytest().assert_outcomes(passed=1)

    def test_keep_flag_skips_teardown_and_reports_the_dsn(
        self, pytester: pytest.Pytester
    ) -> None:
        pytester.makepyfile(
            """
            def test_dsn(postgis_dsn): pass
            """
        )
        result = pytester.runpytest("--postgis-keep")
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(["*kept fake provider at postgresql://fake:***@localhost*"])

    def test_image_flag_reaches_provider_selection(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile(
            """
            def test_image(postgis_provider):
                choice = postgis_provider.provider.choice
                assert choice.kind == "container"
                assert choice.image == "postgis/postgis:15-3.3"
            """
        )
        pytester.runpytest("--postgis-image", "postgis/postgis:15-3.3").assert_outcomes(passed=1)

    def test_ini_dsn_is_preferred_over_a_container(self, pytester: pytest.Pytester) -> None:
        pytester.makeini(
            """
            [pytest]
            postgis_dsn = postgresql://ci-service:5432/gis
            """
        )
        pytester.makepyfile(
            """
            def test_choice(postgis_provider):
                choice = postgis_provider.provider.choice
                assert choice.kind == "dsn"
                assert choice.dsn == "postgresql://ci-service:5432/gis"
            """
        )
        pytester.runpytest().assert_outcomes(passed=1)


class TestProviderUnavailable:
    def test_tests_are_skipped_with_an_actionable_message(
        self, pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("POSTGIS_FIXTURES_DSN", raising=False)
        pytester.makeconftest(
            """
            import postgis_fixtures.plugin as plugin

            pytest_plugins = ["postgis_fixtures.plugin"]

            def pytest_configure(config):
                plugin.testcontainers_available = lambda: False
            """
        )
        pytester.makepyfile("def test_needs_db(postgis_dsn): pass")
        result = pytester.runpytest("-rs")
        result.assert_outcomes(skipped=1)
        result.stdout.fnmatch_lines(["*POSTGIS_FIXTURES_DSN*"])


class TestReadinessProbe:
    def test_probe_opens_a_connection_and_asks_for_the_postgis_version(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import psycopg

        from postgis_fixtures.plugin import _probe

        asked: list[str] = []

        class Cursor:
            def __enter__(self) -> "Cursor":
                return self

            def __exit__(self, *exc: object) -> None:
                return None

            def execute(self, sql: str) -> None:
                asked.append(sql)

            def fetchone(self) -> tuple[str, ...]:
                return ("3.4.2",)

        class Connection:
            def __enter__(self) -> "Connection":
                return self

            def __exit__(self, *exc: object) -> None:
                return None

            def cursor(self) -> Cursor:
                return Cursor()

        monkeypatch.setattr(psycopg, "connect", lambda dsn, **kwargs: Connection())
        _probe("postgresql://localhost/gis")
        assert asked == ["SELECT postgis_lib_version()"]

    def test_probe_propagates_driver_failures(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import psycopg

        from postgis_fixtures.plugin import _probe

        def refuse(dsn: str, **kwargs: object) -> None:
            raise ConnectionRefusedError("no listener")

        monkeypatch.setattr(psycopg, "connect", refuse)
        with pytest.raises(ConnectionRefusedError):
            _probe("postgresql://localhost/gis")


class TestEngineFixture:
    def test_engine_is_built_from_the_dsn_and_disposed(
        self, pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("POSTGIS_FIXTURES_DSN", raising=False)
        pytester.makeconftest(FAKE_CONFTEST)
        pytester.makepyfile(
            """
            import sqlalchemy

            def test_engine(postgis_engine):
                assert isinstance(postgis_engine, sqlalchemy.Engine)
                assert postgis_engine.url.database == "gis"
                assert postgis_engine.url.drivername == "postgresql+psycopg"
            """
        )
        pytester.runpytest().assert_outcomes(passed=1)

    @pytest.mark.parametrize(
        ("dsn", "expected"),
        [
            ("postgres://gis:pw@host/gis", "postgresql+psycopg://gis:pw@host/gis"),
            ("postgresql://gis@host/gis", "postgresql+psycopg://gis@host/gis"),
            ("postgresql+asyncpg://gis@host/gis", "postgresql+asyncpg://gis@host/gis"),
            ("host=localhost dbname=gis", "host=localhost dbname=gis"),
        ],
    )
    def test_dsn_is_bound_to_the_psycopg_driver(self, dsn: str, expected: str) -> None:
        assert sqlalchemy_url(dsn) == expected
