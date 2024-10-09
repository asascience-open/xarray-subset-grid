"""
tests for the mpl_plotting routines

hard to test plotting, but at least they don't barf.
"""

from pathlib import Path

import pytest
import xarray as xr

from xarray_subset_grid.grids import ugrid

try:
    import matplotlib.pyplot as plt  # noqa

    from xarray_subset_grid.visualization.mpl_plotting import (  # plot_sgrid,
        plot_ugrid,
    )
except ImportError:
    pytestmark = pytest.mark.skip(reason="matplotlib is not installed")


EXAMPLE_DATA = Path(__file__).parent.parent.parent / "docs" / "examples" / "example_data"
OUTPUT_DIR = Path(__file__).parent / "output"


def test_plot_ugrid_no_numbers():
    ds = xr.open_dataset(EXAMPLE_DATA / "small_ugrid_zero_based.nc")

    fig, axis = plt.subplots()

    plot_ugrid(axis, ds, nodes=True)

    fig.savefig(OUTPUT_DIR / "ugrid_plot_no_numbers")


def test_plot_ugrid_face_numbers():
    ds = xr.open_dataset(EXAMPLE_DATA / "small_ugrid_zero_based.nc")

    fig, axis = plt.subplots()

    plot_ugrid(axis, ds, face_numbers=True)

    fig.savefig(OUTPUT_DIR / "ugrid_plot_face_numbers")


def test_plot_ugrid_node_numbers():
    ds = xr.open_dataset(EXAMPLE_DATA / "small_ugrid_zero_based.nc")

    fig, axis = plt.subplots()

    plot_ugrid(axis, ds, node_numbers=True)

    fig.savefig(OUTPUT_DIR / "ugrid_plot_node_numbers")


def test_plot_ugrid_start_index_1():
    ds = xr.open_dataset(EXAMPLE_DATA / "SFBOFS_subset1.nc")
    ds = ugrid.assign_ugrid_topology(ds, face_node_connectivity="nv")

    fig, axis = plt.subplots()

    plot_ugrid(axis, ds)

    fig.savefig(OUTPUT_DIR / "ugrid_plot_start_index_1")


#############
# SGRID tests
#############

# def test_plot_sgrid_and_nodes():
#     ds = xr.open_dataset(EXAMPLE_DATA / "wcofs_small_subset.nc", decode_times=False)

#     fig, axis = plt.subplots()

#     plot_sgrid(axis, ds, nodes=True)

#     fig.savefig(OUTPUT_DIR / "sgrid_nodes")



