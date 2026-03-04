#!/usr/bin/env python
"""
Custom runner script for xarray-subset-grid example notebooks.

Finds all example notebooks in docs/examples/, converts each to a Python
script using nbconvert (as suggested in issue #106), runs the script in a
subprocess, and produces a pass/fail summary report.

Usage
-----
Run offline notebooks only (default):
    python scripts/run_notebooks.py

Run all notebooks including those that need network access:
    python scripts/run_notebooks.py --online
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent.parent / "docs" / "examples"

# Notebooks that use only local data from docs/examples/example_data/
# These can run without any network access.
OFFLINE_NOTEBOOKS = [
    "local_subset_example.ipynb",
]

# Notebooks that require network access (OPeNDAP, THREDDS, S3, AWS).
# Only run these when --online is passed.
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


def run_notebook(nb_path: Path) -> tuple[bool, float, str]:
    """
    Convert a notebook to a Python script using nbconvert, then run
    the script in a subprocess.

    Parameters
    ----------
    nb_path : Path
        Absolute path to the .ipynb file.

    Returns
    -------
    success : bool
    elapsed : float   seconds taken
    output  : str     combined stdout + stderr from the run
    """
    start = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Step 1 — Convert notebook → Python script
        convert_result = subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "script",
                str(nb_path),
                "--output-dir", str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )

        if convert_result.returncode != 0:
            elapsed = time.time() - start
            return False, elapsed, (
                convert_result.stdout + "\n" + convert_result.stderr
            )

        # Step 2 — Locate the generated .py file
        scripts = list(tmp_path.glob("*.py")) or list(tmp_path.glob("*.txt"))
        if not scripts:
            return 1, "nbconvert produced no output file (.py or .txt)"

        script_path = scripts[0]

        # Use a non-interactive matplotlib backend so plots don't block
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"

        # Step 3 — Run the script; cwd = notebook's own directory so that
        # relative paths to example_data work correctly
        run_result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            cwd=nb_path.parent,
            env=env,
        )

    elapsed = time.time() - start
    output = run_result.stdout
    if run_result.stderr:
        output += "\nSTDERR:\n" + run_result.stderr

    return run_result.returncode == 0, elapsed, output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run xarray-subset-grid example notebooks via nbconvert."
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Also run notebooks that need network access (OPeNDAP / S3 / THREDDS).",
    )
    args = parser.parse_args()

    to_run = list(OFFLINE_NOTEBOOKS)
    if args.online:
        to_run += ONLINE_NOTEBOOKS

    results: list[tuple[str, str, float, str]] = []

    print(f"\nRunning {'all' if args.online else 'offline'} example notebooks")
    print("=" * 60)

    for nb_name in to_run:
        nb_path = EXAMPLES_DIR / nb_name
        tag = "[online] " if nb_name in ONLINE_NOTEBOOKS else "[offline]"

        if not nb_path.exists():
            print(f"  SKIP   {tag} {nb_name}  (file not found)")
            results.append((nb_name, "SKIP", 0.0, ""))
            continue

        print(f"  RUN    {tag} {nb_name} ...", end="", flush=True)
        success, elapsed, output = run_notebook(nb_path)
        status = "PASS" if success else "FAIL"
        print(f"  {status}  ({elapsed:.1f}s)")

        if not success:
            # Indent the output so it's easy to read in the report
            indented = "\n".join("    " + line for line in output.splitlines())
            print(indented)

        results.append((nb_name, status, elapsed, output))

    # ── Summary report ──────────────────────────────────────────────────────
    passed  = sum(1 for _, s, _, _ in results if s == "PASS")
    failed  = sum(1 for _, s, _, _ in results if s == "FAIL")
    skipped = sum(1 for _, s, _, _ in results if s == "SKIP")

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed | {failed} failed | {skipped} skipped")

    if failed:
        print("\nFailed notebooks:")
        for name, status, _, _ in results:
            if status == "FAIL":
                print(f"  ✗  {name}")
        sys.exit(1)
    else:
        print("All notebooks ran successfully! ✓")


if __name__ == "__main__":
    main()