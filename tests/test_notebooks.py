"""
Pytest integration for auto-running example notebooks (issue #106).

Each notebook is converted to a Python script via nbconvert and then
executed in a subprocess, mirroring the approach described in the issue.

Offline notebooks (local data only) run unconditionally.
Network notebooks are decorated with @pytest.mark.online and are skipped
unless pytest is invoked with --online (matching the existing convention
in conftest.py).

To run offline notebook tests:
    pytest tests/test_notebooks.py -v

To run all notebook tests (needs internet):
    pytest tests/test_notebooks.py -v --online
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent / "docs" / "examples"

# ── Offline: use only local files in docs/examples/example_data/ ────────────
OFFLINE_NOTEBOOKS = [
    "local_subset_example.ipynb",
]

# ── Online: require OPeNDAP / THREDDS / S3 / AWS access ─────────────────────
ONLINE_NOTEBOOKS = [
    "regular_grid_2d.ipynb",
    "RegularGridTHREDDS.ipynb",
    "fvcom.ipynb",
    "fvcom_3d.ipynb",
    "gfs_opendap.ipynb",
    "nam_opendap.ipynb",
    "roms.ipynb",
    "roms_3d.ipynb",
    "roms-compare.ipynb",
    "rtofs.ipynb",
    "selfe.ipynb",
    "sscofs.ipynb",
    "stofs_2d.ipynb",
    "stofs_3d.ipynb",
    "subset_from_ncfile.ipynb",
]


# ── Helper ───────────────────────────────────────────────────────────────────

def _run_notebook(nb_path: Path) -> tuple[int, str]:
    """
    Convert *nb_path* to a Python script with nbconvert, execute the
    script in a subprocess, and return (returncode, combined_output).

    The notebook's own directory is used as the working directory so that
    relative paths to example_data resolve correctly.
    A non-interactive matplotlib backend is set via MPLBACKEND=Agg so
    that plt.show() calls do not open windows or hang the process.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Step 1: notebook → Python script
        convert = subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "script",
                str(nb_path),
                "--output-dir", str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )
        if convert.returncode != 0:
            return convert.returncode, convert.stdout + "\n" + convert.stderr

        scripts = list(tmp_path.glob("*.py")) or list(tmp_path.glob("*.txt"))
        if not scripts:
            return 1, "nbconvert produced no output file (.py or .txt)"
        # Step 2: execute the script
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"          # non-interactive backend

        run = subprocess.run(
            [sys.executable, str(scripts[0])],
            capture_output=True,
            text=True,
            cwd=nb_path.parent,            # resolve relative data paths
            env=env,
        )

    output = run.stdout
    if run.stderr:
        output += "\nSTDERR:\n" + run.stderr
    return run.returncode, output


# ── Tests ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("notebook", OFFLINE_NOTEBOOKS)
def test_offline_notebook(notebook):
    """
    Notebooks that require only local data.
    These always run as part of the normal test suite.
    """
    nb_path = EXAMPLES_DIR / notebook
    if not nb_path.exists():
        pytest.skip(f"Notebook not found: {nb_path}")

    returncode, output = _run_notebook(nb_path)

    if returncode != 0:
        pytest.fail(
            f"Notebook '{notebook}' failed (exit code {returncode}).\n"
            f"Output:\n{output}"
        )


@pytest.mark.online
@pytest.mark.parametrize("notebook", ONLINE_NOTEBOOKS)
def test_online_notebook(notebook):
    """
    Notebooks that require network access (OPeNDAP, THREDDS, S3, AWS).
    Skipped unless pytest is run with --online.
    """
    nb_path = EXAMPLES_DIR / notebook
    if not nb_path.exists():
        pytest.skip(f"Notebook not found: {nb_path}")

    returncode, output = _run_notebook(nb_path)

    if returncode != 0:
        pytest.fail(
            f"Notebook '{notebook}' failed (exit code {returncode}).\n"
            f"Output:\n{output}"
        )