"""The pytest plugin: options, session lifecycle and user-facing fixtures.

This module is not registered through packaging entry points. Enable it from the
consuming project's ``conftest.py``::

    pytest_plugins = ["postgis_fixtures.plugin"]

or on the command line with ``-p postgis_fixtures.plugin``.

Fixture layout, session-scoped unless noted:

``postgis_provider``
    The resolved :class:`~postgis_fixtures.provider.Provider`. Monkeypatching
    :data:`PROVIDER_FACTORY` replaces it, which is how the plugin's own tests
    exercise the wiring without Docker.
``postgis_dsn``
    The DSN of a database that has answered a readiness probe.
``postgis_engine``
    A SQLAlchemy ``Engine``, if SQLAlchemy is installed.
``postgis_connection`` (function scope)
    A psycopg connection inside a transaction that is rolled back after the test.
``postgis_db`` (function scope)
    A :class:`~postgis_fixtures.db.PostgisDB` over ``postgis_connection``.
``spatial_fixtures`` (session scope)
    The generated :class:`~postgis_fixtures.datasets.SpatialFixtures`, built
    offline — it needs no database at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import pytest

from .datasets import SpatialFixtures, build_fixtures
from .db import PostgisDB
from .errors import PostgisFixturesError, ProviderError
from .geometry import GeneratorConfig
from .provider import (
    DSN_ENV_VAR,
    DEFAULT_IMAGE,
    Provider,
    ProviderChoice,
    build_provider,
    redact,
    select_provider,
    testcontainers_available,
    wait_for_ready,
)

#: Indirection point for tests: swap this to inject a fake provider.
PROVIDER_FACTORY: Callable[[ProviderChoice], Provider] = build_provider

#: Default readiness budget: 30 attempts, one second apart.
READY_ATTEMPTS = 30
READY_DELAY = 1.0


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register command-line flags and their ini equivalents."""
    group = parser.getgroup("postgis", "ephemeral PostGIS fixtures")
    group.addoption(
        "--postgis-image",
        action="store",
        default=None,
        help=f"container image to start when no DSN is configured (default: {DEFAULT_IMAGE})",
    )
    group.addoption(
        "--postgis-keep",
        action="store_true",
        default=False,
        help="do not tear the database down after the session; print its DSN for debugging",
    )
    group.addoption(
        "--postgis-seed",
        action="store",
        type=int,
        default=None,
        help="seed for spatial data generation (default: the postgis_seed ini value)",
    )
    parser.addini("postgis_dsn", "DSN of an existing PostGIS instance to use", default="")
    parser.addini("postgis_image", "container image to start when no DSN is configured", default="")
    parser.addini("postgis_seed", "seed for spatial data generation", default="")
    parser.addini(
        "postgis_row_counts",
        "per-dataset row counts, as name=count pairs",
        type="linelist",
        default=[],
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the markers the plugin and its examples rely on."""
    config.addinivalue_line(
        "markers", "postgis: test requires a live PostGIS database provided by postgis-fixtures"
    )


def resolve_seed(cli_seed: int | None, ini_seed: str) -> int:
    """Resolve the generation seed from CLI and ini values.

    Raises:
        PostgisFixturesError: when the ini value is not an integer.
    """
    if cli_seed is not None:
        return cli_seed
    text = (ini_seed or "").strip()
    if not text:
        return GeneratorConfig().seed
    try:
        return int(text)
    except ValueError:
        raise PostgisFixturesError(
            f"postgis_seed must be an integer, got {text!r}"
        ) from None


def parse_row_counts(entries: list[str]) -> dict[str, int]:
    """Parse ``name=count`` ini lines into a mapping.

    Raises:
        PostgisFixturesError: on a malformed entry or a negative count.
    """
    counts: dict[str, int] = {}
    for entry in entries:
        text = entry.strip()
        if not text or text.startswith("#"):
            continue
        name, separator, value = text.partition("=")
        if not separator:
            raise PostgisFixturesError(
                f"postgis_row_counts entry {entry!r} must be in name=count form"
            )
        try:
            count = int(value.strip())
        except ValueError:
            raise PostgisFixturesError(
                f"postgis_row_counts entry {entry!r} has a non-integer count"
            ) from None
        if count < 0:
            raise PostgisFixturesError(
                f"postgis_row_counts entry {entry!r} has a negative count"
            )
        counts[name.strip()] = count
    return counts


def resolve_choice(config: pytest.Config) -> ProviderChoice:
    """Resolve the provider choice from the environment, ini file and CLI."""
    return select_provider(
        env_dsn=os.environ.get(DSN_ENV_VAR),
        ini_dsn=config.getini("postgis_dsn") or None,
        image=config.getoption("--postgis-image") or config.getini("postgis_image") or None,
        has_testcontainers=testcontainers_available(),
    )


def sqlalchemy_url(dsn: str) -> str:
    """Normalise a libpq DSN into a SQLAlchemy URL bound to the psycopg 3 driver.

    Bare ``postgresql://`` URLs make SQLAlchemy reach for ``psycopg2``, which is
    not the driver the rest of this package uses. Any DSN that already names a
    driver (``postgresql+asyncpg://`` and friends) is left alone.
    """
    for prefix in ("postgres://", "postgresql://"):
        if dsn.startswith(prefix):
            return "postgresql+psycopg://" + dsn[len(prefix) :]
    return dsn


def _probe(dsn: str) -> None:
    """Open and close a connection, raising if the server is not usable yet."""
    import psycopg

    with psycopg.connect(dsn, connect_timeout=3) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT postgis_lib_version()")
            if cursor.fetchone() is None:  # pragma: no cover - server always answers
                raise PostgisFixturesError("postgis_lib_version() returned no rows")


@dataclass
class ProviderSession:
    """A started provider together with the DSN it produced."""

    provider: Provider
    dsn: str


@pytest.fixture(scope="session")
def postgis_provider(request: pytest.FixtureRequest) -> Iterator[ProviderSession]:
    """Resolve, start and eventually tear down the session's database.

    The provider is started exactly once. ``--postgis-keep`` suppresses teardown
    and prints the (password-redacted) DSN so you can attach ``psql`` to the
    database a failing test left behind.
    """
    try:
        choice = resolve_choice(request.config)
    except ProviderError as exc:
        pytest.skip(str(exc))
    provider = PROVIDER_FACTORY(choice)
    dsn = provider.start()
    keep = request.config.getoption("--postgis-keep")
    try:
        yield ProviderSession(provider=provider, dsn=dsn)
    finally:
        if keep:
            request.config.get_terminal_writer().line(
                f"\n[postgis-fixtures] kept {provider.description} at {redact(dsn)} "
                f"({choice.reason})",
                bold=True,
            )
        else:
            provider.stop()


@pytest.fixture(scope="session")
def postgis_dsn(postgis_provider: ProviderSession) -> str:
    """Return the DSN of a database that has passed a readiness probe."""
    probe = getattr(postgis_provider.provider, "probe", _probe)
    wait_for_ready(
        postgis_provider.dsn, probe, attempts=READY_ATTEMPTS, delay=READY_DELAY
    )
    return postgis_provider.dsn


@pytest.fixture(scope="session")
def postgis_engine(postgis_dsn: str) -> Iterator[Any]:
    """Return a SQLAlchemy engine over the session database.

    Skipped when SQLAlchemy is not installed, so the rest of the plugin stays
    usable without it.
    """
    sqlalchemy = pytest.importorskip("sqlalchemy", reason="postgis_engine requires SQLAlchemy")
    engine = sqlalchemy.create_engine(sqlalchemy_url(postgis_dsn), future=True)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture()
def postgis_connection(postgis_dsn: str) -> Iterator[Any]:
    """Yield a connection wrapped in a transaction that is always rolled back.

    Tests therefore see the fixture data, can create and drop tables freely, and
    leave the database exactly as they found it — including sequences, because
    the whole transaction is discarded.
    """
    import psycopg

    connection = psycopg.connect(postgis_dsn, autocommit=False)
    try:
        yield connection
    finally:
        connection.rollback()
        connection.close()


@pytest.fixture()
def postgis_db(postgis_connection: Any, postgis_dsn: str) -> PostgisDB:
    """Return the query facade bound to the per-test connection."""
    return PostgisDB(postgis_connection, dsn=postgis_dsn)


@pytest.fixture(scope="session")
def postgis_seed(request: pytest.FixtureRequest) -> int:
    """Return the seed used for spatial data generation."""
    return resolve_seed(
        request.config.getoption("--postgis-seed"), request.config.getini("postgis_seed")
    )


@pytest.fixture(scope="session")
def spatial_config(postgis_seed: int) -> GeneratorConfig:
    """Return the generation config for the session."""
    return GeneratorConfig(seed=postgis_seed)


@pytest.fixture(scope="session")
def spatial_fixtures(
    request: pytest.FixtureRequest, spatial_config: GeneratorConfig
) -> SpatialFixtures:
    """Return the generated datasets. Requires no database."""
    row_counts = parse_row_counts(list(request.config.getini("postgis_row_counts")))
    return build_fixtures(spatial_config, row_counts=row_counts)
