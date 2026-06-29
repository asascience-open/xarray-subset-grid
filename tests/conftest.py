# conftest: some configuration for the tests

import zipfile
from pathlib import Path

import pooch
import pytest


@pytest.fixture(scope="session", autouse=True)
def download_data():
    """Download support data for tests and documentation."""
    url = "https://github.com/ioos/xarray-subset-grid/releases/download"
    version = "2026.06.24"

    fname = pooch.retrieve(
        url=f"{url}/{version}/test_data.zip",
        known_hash="sha256:525e09f4d478c7484692ce4c30f5dd0f81115209e54d94d03831cca6bbfaceb0",
    )

    here = Path(__file__).resolve().parent.parent
    with zipfile.ZipFile(fname, "r") as zip_ref:
        zip_ref.extractall(here)


def pytest_addoption(parser):
    # put a @pytest.mark.online decorator on tests that require net access
    parser.addoption(
        "--online",
        action="store_true",  # what is this?
        default=False,
        help="run tests that access AWS resources - have to be online",
    )


def pytest_configure(config):
    # register an additional marker
    config.addinivalue_line(
        "markers", "online: mark test to run only when online (using AWS resources)"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--online"):
        # --online not given in cli: skip tests that require online access
        skip_online = pytest.mark.skip(reason="need --online option to run")
        for item in items:
            if "online" in item.keywords:
                item.add_marker(skip_online)


# # example from docs
# def pytest_runtest_setup(item):
#     envnames = [mark.args[0] for mark in item.iter_markers(name="env")]
#     if envnames:
#         if item.config.getoption("-E") not in envnames:
#             pytest.skip(f"test requires env in {envnames!r}")

EXAMPLE_DATA = Path(__file__).parent / "example_data"

UGRID_FILES = [
    EXAMPLE_DATA / "SFBOFS_subset1.nc",
    EXAMPLE_DATA / "small_ugrid_zero_based.nc",
    EXAMPLE_DATA / "tris_and_bounds.nc",
]

SGRID_FILES = [
    EXAMPLE_DATA / "arakawa_c_test_grid.nc",
]

RGRID_FILES = [
    EXAMPLE_DATA / "2D-rectangular_grid_wind.nc",
    EXAMPLE_DATA / "rectangular_grid_decreasing.nc",
    EXAMPLE_DATA / "AMSEAS-subset.nc",
]
