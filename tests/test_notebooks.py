"""
Tests that auto-run the example notebooks using nbconvert.

This addresses issue #106 — we needed a way to automatically run all
example notebooks and detect breakage in CI.

Approach (as suggested in the issue):
  - A Python script discovers all .ipynb files in docs/examples/
  - Each notebook is executed via nbconvert in a subprocess
  - Notebooks requiring network access (OPeNDAP, THREDDS, S3) are marked
    with @pytest.mark.online and are skipped unless --online is passed,
    matching the existing pattern used in the rest of the test suite.
  - Notebooks using only local example_data run in the default offline suite.

Usage:
  # Run offline notebooks only (default):
  pytest tests/test_notebooks.py -v

  # Run all notebooks including network ones:
  pytest tests/test_notebooks.py -v --online
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Root of the examples directory
EXAMPLES_DIR = Path(__file__).parent.parent / "docs" / "examples"

# Notebooks that use ONLY local data from docs/examples/example_data/
# These run in the default (offline) test suite.
OFFLINE_NOTEBOOKS = [
    # No notebooks currently use only local data.
    # Add notebook filenames here when local-only examples are created.
    # e.g. "my_local_example.ipynb"
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_notebooks():
    """Return all .ipynb files in docs/examples/, sorted."""
    return sorted(EXAMPLES_DIR.glob("*.ipynb"))


def _online_notebooks():
    """Return notebooks that require network access."""
    offline = set(OFFLINE_NOTEBOOKS)
    return [nb for nb in _all_notebooks() if nb.name not in offline]


def _offline_notebooks():
    """Return notebooks that use only local data."""
    return [
        EXAMPLES_DIR / name
        for name in OFFLINE_NOTEBOOKS
        if (EXAMPLES_DIR / name).exists()
    ]


def _run_notebook(notebook_path: Path) -> tuple[bool, str, str]:
    """Execute a notebook via nbconvert and return (success, stdout, stderr).

    The notebook is executed in a temporary directory so no output files
    are left behind in the repository.
    The subprocess runs with the notebook's own directory as cwd so that
    relative paths inside the notebook (e.g. to example_data/) resolve
    correctly.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                sys.executable, "-m", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--ExecutePreprocessor.timeout=300",
                "--output-dir", tmpdir,
                str(notebook_path),
            ],
            capture_output=True,
            text=True,
            cwd=str(notebook_path.parent),  # resolve relative paths correctly
        )
    return result.returncode == 0, result.stdout, result.stderr


# ---------------------------------------------------------------------------
# Offline notebook tests (no --online flag required)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "notebook",
    _offline_notebooks(),
    ids=[nb.name for nb in _offline_notebooks()],
)
def test_notebook_offline(notebook):
    """Execute notebooks that use only local data.

    These run as part of the default test suite so CI catches regressions
    without needing network access.
    """
    success, stdout, stderr = _run_notebook(notebook)
    if not success:
        pytest.fail(
            f"Notebook '{notebook.name}' failed to execute.\n"
            f"--- stdout ---\n{stdout}\n"
            f"--- stderr ---\n{stderr}"
        )


# ---------------------------------------------------------------------------
# Online notebook tests (requires --online flag)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "notebook",
    _online_notebooks(),
    ids=[nb.name for nb in _online_notebooks()],
)
@pytest.mark.online
def test_notebook_online(notebook):
    """Execute notebooks that require network access (OPeNDAP, THREDDS, S3).

    Skipped by default. Run with:
        pytest tests/test_notebooks.py --online
    """
    success, stdout, stderr = _run_notebook(notebook)
    if not success:
        pytest.fail(
            f"Notebook '{notebook.name}' failed to execute.\n"
            f"--- stdout ---\n{stdout}\n"
            f"--- stderr ---\n{stderr}"
        )