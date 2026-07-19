"""Exception types raised by the postgis-fixtures plugin.

All errors carry a human-readable diagnostic message; the plugin never fails a
test run with a bare ``AssertionError`` or an opaque driver exception.
"""

from __future__ import annotations


class PostgisFixturesError(Exception):
    """Base class for every error raised by this package."""


class ProviderError(PostgisFixturesError):
    """Raised when no usable PostGIS database can be provided."""


class ReadinessTimeout(ProviderError):
    """Raised when a database never became ready within the retry budget."""

    def __init__(self, dsn: str, attempts: int, elapsed: float, last_error: BaseException | None) -> None:
        self.dsn = dsn
        self.attempts = attempts
        self.elapsed = elapsed
        self.last_error = last_error
        detail = f"{type(last_error).__name__}: {last_error}" if last_error is not None else "no error recorded"
        super().__init__(
            f"PostGIS at {dsn} was not ready after {attempts} attempt(s) over {elapsed:.1f}s. "
            f"Last failure was {detail}. "
            "If you are running against a pre-provisioned service container, check that "
            "POSTGIS_FIXTURES_DSN points at it and that the postgis extension is installed."
        )


class DatasetError(PostgisFixturesError):
    """Raised for unknown dataset names or invalid dataset parameters."""


class SchemaError(PostgisFixturesError):
    """Raised when a table/column definition cannot be turned into valid DDL."""
