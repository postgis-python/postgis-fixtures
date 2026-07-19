"""Tests for provider selection and the readiness loop."""

from __future__ import annotations

import pytest

from postgis_fixtures.errors import ProviderError, ReadinessTimeout
from postgis_fixtures.provider import (
    DEFAULT_IMAGE,
    DSN_ENV_VAR,
    ContainerProvider,
    DsnProvider,
    ProviderChoice,
    build_provider,
    redact,
    select_provider,
    testcontainers_available as _testcontainers_available,
    wait_for_ready,
)

DSN = "postgresql://gis:secret@localhost:5432/gis"


class TestSelectProvider:
    def test_environment_dsn_wins(self) -> None:
        choice = select_provider(env_dsn=DSN, ini_dsn="postgresql://other/db", has_testcontainers=True)
        assert choice.kind == "dsn"
        assert choice.dsn == DSN
        assert DSN_ENV_VAR in choice.reason

    def test_ini_dsn_is_used_when_the_environment_is_unset(self) -> None:
        choice = select_provider(ini_dsn=DSN, has_testcontainers=True)
        assert choice.kind == "dsn"
        assert "ini option" in choice.reason

    @pytest.mark.parametrize("blank", ["", "   ", "\t"])
    def test_blank_dsns_are_treated_as_unset(self, blank: str) -> None:
        choice = select_provider(env_dsn=blank, ini_dsn=blank, has_testcontainers=True)
        assert choice.kind == "container"

    def test_dsn_is_stripped(self) -> None:
        assert select_provider(env_dsn=f"  {DSN}  ").dsn == DSN

    def test_container_uses_the_default_image(self) -> None:
        choice = select_provider(has_testcontainers=True)
        assert choice.kind == "container"
        assert choice.image == DEFAULT_IMAGE
        assert choice.dsn is None

    def test_container_honours_an_explicit_image(self) -> None:
        choice = select_provider(image="postgis/postgis:15-3.3", has_testcontainers=True)
        assert choice.image == "postgis/postgis:15-3.3"
        assert "postgis/postgis:15-3.3" in choice.reason

    def test_blank_image_falls_back_to_the_default(self) -> None:
        assert select_provider(image="   ", has_testcontainers=True).image == DEFAULT_IMAGE

    def test_no_dsn_and_no_testcontainers_is_an_error(self) -> None:
        with pytest.raises(ProviderError) as excinfo:
            select_provider(has_testcontainers=False)
        message = str(excinfo.value)
        assert DSN_ENV_VAR in message
        assert "install testcontainers" in message

    def test_selection_is_pure(self) -> None:
        """Same inputs, same output — no environment or filesystem reads."""
        first = select_provider(env_dsn=DSN)
        second = select_provider(env_dsn=DSN)
        assert first == second

    def test_availability_probe_returns_a_bool(self) -> None:
        assert isinstance(_testcontainers_available(), bool)


class TestProviders:
    def test_dsn_provider_returns_its_dsn_and_stops_cleanly(self) -> None:
        provider = DsnProvider(DSN)
        assert provider.start() == DSN
        provider.stop()
        provider.stop()
        assert "external database" in provider.description
        assert "secret" not in provider.description

    def test_dsn_provider_rejects_an_empty_dsn(self) -> None:
        with pytest.raises(ProviderError, match="empty"):
            DsnProvider("  ").start()

    def test_build_provider_dispatches_on_kind(self) -> None:
        assert isinstance(build_provider(select_provider(env_dsn=DSN)), DsnProvider)
        assert isinstance(
            build_provider(select_provider(has_testcontainers=True)), ContainerProvider
        )

    def test_container_provider_describes_its_image_without_starting(self) -> None:
        provider = ContainerProvider("postgis/postgis:16-3.4")
        assert provider.description == "ephemeral postgis/postgis:16-3.4 container"

    def test_container_provider_stop_is_idempotent(self) -> None:
        class FakeContainer:
            def __init__(self) -> None:
                self.stops = 0

            def stop(self) -> None:
                self.stops += 1

        provider = ContainerProvider()
        container = FakeContainer()
        provider._container = container
        provider.stop()
        provider.stop()
        assert container.stops == 1

    def test_dsn_provider_from_choice_without_dsn_is_rejected(self) -> None:
        with pytest.raises(ProviderError, match="without a DSN"):
            build_provider(ProviderChoice(kind="dsn", dsn=None, image=None, reason="forced"))


class TestRedact:
    @pytest.mark.parametrize(
        ("dsn", "expected"),
        [
            (DSN, "postgresql://gis:***@localhost:5432/gis"),
            ("postgresql://gis@localhost/gis", "postgresql://gis@localhost/gis"),
            ("host=localhost dbname=gis", "host=localhost dbname=gis"),
            ("postgresql://localhost/gis", "postgresql://localhost/gis"),
        ],
    )
    def test_password_is_hidden_and_nothing_else_changes(self, dsn: str, expected: str) -> None:
        assert redact(dsn) == expected


class TestWaitForReady:
    def test_returns_on_the_first_success(self) -> None:
        calls: list[str] = []
        wait_for_ready(DSN, calls.append, attempts=3, sleep=lambda _: None)
        assert calls == [DSN]

    def test_retries_until_the_probe_succeeds(self) -> None:
        attempts = {"n": 0}
        slept: list[float] = []

        def probe(_: str) -> None:
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise ConnectionRefusedError("not up yet")

        wait_for_ready(DSN, probe, attempts=5, delay=0.25, sleep=slept.append)
        assert attempts["n"] == 3
        assert slept == [0.25, 0.25]

    def test_gives_up_with_a_diagnostic_message(self) -> None:
        def probe(_: str) -> None:
            raise ConnectionRefusedError("connection refused")

        ticks = iter([0.0, 12.5])
        with pytest.raises(ReadinessTimeout) as excinfo:
            wait_for_ready(
                DSN, probe, attempts=4, sleep=lambda _: None, clock=lambda: next(ticks)
            )
        message = str(excinfo.value)
        assert "4 attempt(s)" in message
        assert "12.5s" in message
        assert "ConnectionRefusedError: connection refused" in message
        assert "POSTGIS_FIXTURES_DSN" in message
        assert "secret" not in message
        assert excinfo.value.attempts == 4
        assert isinstance(excinfo.value.last_error, ConnectionRefusedError)

    def test_does_not_sleep_after_the_final_attempt(self) -> None:
        slept: list[float] = []

        def probe(_: str) -> None:
            raise TimeoutError("nope")

        with pytest.raises(ReadinessTimeout):
            wait_for_ready(DSN, probe, attempts=3, sleep=slept.append)
        assert len(slept) == 2

    def test_zero_attempts_is_a_programming_error(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            wait_for_ready(DSN, lambda _: None, attempts=0)


class TestContainerProviderStart:
    """Exercise the container path with a stub ``testcontainers`` module."""

    @staticmethod
    def _install_stub(monkeypatch: pytest.MonkeyPatch, container_factory: object) -> None:
        import sys
        import types

        package = types.ModuleType("testcontainers")
        module = types.ModuleType("testcontainers.postgres")
        module.PostgresContainer = container_factory  # type: ignore[attr-defined]
        package.postgres = module  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "testcontainers", package)
        monkeypatch.setitem(sys.modules, "testcontainers.postgres", module)

    def test_start_returns_the_container_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class StubContainer:
            def __init__(self, image: str, driver: object = None) -> None:
                self.image = image
                self.started = False

            def start(self) -> None:
                self.started = True

            def stop(self) -> None:
                self.started = False

            def get_connection_url(self) -> str:
                return f"postgresql://test:test@localhost:5432/{self.image}"

        self._install_stub(monkeypatch, StubContainer)
        provider = ContainerProvider("postgis")
        assert provider.start() == "postgresql://test:test@localhost:5432/postgis"
        provider.stop()

    def test_start_failure_explains_what_to_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class BrokenContainer:
            def __init__(self, image: str, driver: object = None) -> None:
                self.image = image

            def start(self) -> None:
                raise RuntimeError("docker daemon not reachable")

        self._install_stub(monkeypatch, BrokenContainer)
        with pytest.raises(ProviderError) as excinfo:
            ContainerProvider().start()
        message = str(excinfo.value)
        assert "docker daemon not reachable" in message
        assert "container runtime is running" in message
