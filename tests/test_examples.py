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


def test_example_suite_skips_without_a_database() -> None:
    result = run_examples("-rs")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "skipped" in result.stdout
    assert "failed" not in result.stdout
    assert "no PostGIS available" in result.stdout


def test_example_tests_are_marked_for_selection() -> None:
    result = run_examples("-m", "postgis", "--collect-only")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "deselected" not in result.stdout
