"""Deciding where the PostGIS under test comes from, and waiting for it.

There are exactly two ways to get a database:

``dsn``
    An explicitly configured DSN — a service container in CI, or a local
    development database. Chosen whenever one is configured, because an
    explicit choice should always beat an implicit one.

``container``
    An ephemeral ``postgis/postgis`` container started through
    :mod:`testcontainers`. Chosen when no DSN is configured and testcontainers
    is importable.

:func:`select_provider` implements that decision as a pure function of plain
values, so the policy can be unit-tested without Docker, without environment
variables, and without a pytest session.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Callable, Literal, Protocol

from .errors import ProviderError, ReadinessTimeout

#: Environment variable read for an explicitly configured DSN.
DSN_ENV_VAR = "POSTGIS_FIXTURES_DSN"
#: Container image used when none is configured.
DEFAULT_IMAGE = "postgis/postgis:16-3.4"

ProviderKind = Literal["dsn", "container"]


@dataclass(frozen=True)
class ProviderChoice:
    """The outcome of provider selection.

    Attributes:
        kind: ``"dsn"`` or ``"container"``.
        dsn: The DSN to use, for ``kind == "dsn"``; ``None`` otherwise.
        image: The image to start, for ``kind == "container"``; ``None`` otherwise.
        reason: Human-readable explanation, reported by ``--postgis-keep`` and
            included in error messages.
    """

    kind: ProviderKind
    dsn: str | None
    image: str | None
    reason: str


def testcontainers_available() -> bool:
    """Return ``True`` when the optional ``testcontainers`` package is importable."""
    return find_spec("testcontainers") is not None


def select_provider(
    *,
    env_dsn: str | None = None,
    ini_dsn: str | None = None,
    image: str | None = None,
    has_testcontainers: bool = False,
) -> ProviderChoice:
    """Choose how to obtain a PostGIS database.

    Precedence is environment DSN, then ini DSN, then a container. Blank and
    whitespace-only DSNs are treated as unset, which is what you want when CI
    exports ``POSTGIS_FIXTURES_DSN=""`` for a job that does not need it.

    Raises:
        ProviderError: when no DSN is configured and testcontainers is missing.
    """
    for value, source in ((env_dsn, DSN_ENV_VAR), (ini_dsn, "postgis_dsn ini option")):
        if value is not None and value.strip():
            return ProviderChoice(
                kind="dsn",
                dsn=value.strip(),
                image=None,
                reason=f"using the DSN configured via {source}",
            )
    if has_testcontainers:
        chosen = (image or DEFAULT_IMAGE).strip() or DEFAULT_IMAGE
        return ProviderChoice(
            kind="container",
            dsn=None,
            image=chosen,
            reason=f"no DSN configured; starting an ephemeral {chosen} container",
        )
    raise ProviderError(
        "No PostGIS database available. Either set "
        f"{DSN_ENV_VAR} (or the postgis_dsn ini option) to a running PostGIS instance, "
        "or install testcontainers so an ephemeral container can be started."
    )


class Provider(Protocol):
    """The interface the plugin needs from whatever supplies a database."""

    def start(self) -> str:
        """Start (or validate) the database and return its DSN."""

    def stop(self) -> None:
        """Release the database. Must be idempotent."""

    @property
    def description(self) -> str:
        """Return a short description used in diagnostics."""


@dataclass
class DsnProvider:
    """Provider that hands back an already-running database."""

    dsn: str

    def start(self) -> str:
        """Return the configured DSN unchanged."""
        if not self.dsn.strip():
            raise ProviderError("Configured DSN is empty")
        return self.dsn

    def stop(self) -> None:
        """No-op: this provider does not own the database."""

    @property
    def description(self) -> str:
        """Return a short description used in diagnostics."""
        return f"external database at {redact(self.dsn)}"


class ContainerProvider:
    """Provider that starts an ephemeral PostGIS container via testcontainers.

    The import is deferred to :meth:`start` so that merely constructing the
    provider — which the unit tests do — never requires the optional dependency.
    """

    def __init__(self, image: str = DEFAULT_IMAGE) -> None:
        self.image = image
        self._container: object | None = None

    def start(self) -> str:
        """Start the container and return the DSN it exposes."""
        try:
            from testcontainers.postgres import PostgresContainer
        except ImportError as exc:  # pragma: no cover - requires the dep to be absent
            raise ProviderError(
                "testcontainers is not installed, so no ephemeral PostGIS container "
                f"can be started. Install it, or set {DSN_ENV_VAR}."
            ) from exc
        container = PostgresContainer(self.image, driver=None)
        try:
            container.start()
        except Exception as exc:
            raise ProviderError(
                f"Failed to start the {self.image} container: {exc}. "
                "Check that a container runtime is running and the image can be pulled."
            ) from exc
        self._container = container
        return container.get_connection_url()

    def stop(self) -> None:
        """Stop the container if one is running."""
        container = self._container
        self._container = None
        if container is not None:
            container.stop()  # type: ignore[attr-defined]

    @property
    def description(self) -> str:
        """Return a short description used in diagnostics."""
        return f"ephemeral {self.image} container"


def build_provider(choice: ProviderChoice) -> Provider:
    """Instantiate the provider described by a :class:`ProviderChoice`."""
    if choice.kind == "dsn":
        if not choice.dsn:  # pragma: no cover - select_provider guarantees this
            raise ProviderError("DSN provider selected without a DSN")
        return DsnProvider(choice.dsn)
    return ContainerProvider(choice.image or DEFAULT_IMAGE)


def redact(dsn: str) -> str:
    """Return ``dsn`` with any password replaced, safe for logs and reports."""
    if "://" not in dsn:
        return dsn
    scheme, _, rest = dsn.partition("://")
    if "@" not in rest:
        return dsn
    credentials, _, host = rest.partition("@")
    if ":" not in credentials:
        return dsn
    user, _, _password = credentials.partition(":")
    return f"{scheme}://{user}:***@{host}"


def wait_for_ready(
    dsn: str,
    probe: Callable[[str], None],
    *,
    attempts: int = 30,
    delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Poll ``probe`` until the database answers, or give up with diagnostics.

    ``probe`` is expected to raise on failure and return ``None`` on success.
    Injecting ``sleep`` and ``clock`` keeps the retry loop testable in
    microseconds rather than minutes.

    Raises:
        ValueError: if ``attempts`` is not positive.
        ReadinessTimeout: if every attempt failed.
    """
    if attempts < 1:
        raise ValueError(f"attempts must be at least 1, got {attempts}")
    started = clock()
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            probe(dsn)
        except Exception as exc:  # noqa: BLE001 - any driver error means "not ready yet"
            last_error = exc
            if attempt < attempts:
                sleep(delay)
        else:
            return
    raise ReadinessTimeout(redact(dsn), attempts, clock() - started, last_error)
