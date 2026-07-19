"""The bundled example project must stay runnable and must degrade to skips.

Running the examples in a subprocess keeps their ``pytest_plugins`` declaration
and their own ``pytest.ini`` rootdir intact, which is exactly how a consumer
would run them.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"


def run_examples(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the example suite in a subprocess with no database configured."""
    env = dict(os.environ)
    env.pop("POSTGIS_FIXTURES_DSN", None)
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(EXAMPLES), "-q", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        check=False,
    )


@pytest.fixture(scope="module")
def collected() -> subprocess.CompletedProcess[str]:
    return run_examples("--collect-only")


def test_example_suite_collects_cleanly(collected: subprocess.CompletedProcess[str]) -> None:
    assert collected.returncode == 0, collected.stdout + collected.stderr
    assert "test_geofence_queries.py::test_containment_join_uses_the_gist_index" in collected.stdout


#: The two legitimate ways the example suite reports "no database here". Which one
#: fires depends on the machine: with no container runtime *reachable* the
#: collection hook in ``examples/conftest.py`` skips everything up front, but when
#: testcontainers is installed and a runtime is merely broken (or absent), provider
#: selection succeeds and the skip comes later, from the ``postgis_provider``
#: fixture. Both are skips, which is the property under test; asserting on only one
#: of them makes this test pass or fail on the developer's Docker setup.
NO_DATABASE_REASONS = (
    "no PostGIS available",
    "Check that a container runtime is running",
)


def test_example_suite_skips_without_a_database() -> None:
    result = run_examples("-rs")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "skipped" in result.stdout
    assert "failed" not in result.stdout
    assert "error" not in result.stdout
    assert any(reason in result.stdout for reason in NO_DATABASE_REASONS), result.stdout


def test_example_tests_are_marked_for_selection() -> None:
    result = run_examples("-m", "postgis", "--collect-only")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "deselected" not in result.stdout
