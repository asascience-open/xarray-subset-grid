import numpy as np
import pytest
import xarray as xr

import xarray_subset_grid.accessor  # noqa: F401 -- register accessor


def test_accessor_warns_when_no_grid_recognized():
    ds = xr.Dataset()
    with pytest.warns(UserWarning, match="no grid type"):
        accessor = ds.xsg
    assert accessor.grid is None


def test_subset_polygon_and_bbox_return_none_without_grid():
    ds = xr.Dataset()
    poly = np.array(
        [
            [-72.0, 41.0],
            [-70.0, 41.0],
            [-71.0, 39.0],
            [-72.0, 41.0],
        ]
    )
    with pytest.warns(UserWarning, match="no grid type"):
        assert ds.xsg.subset_polygon(poly) is None
        assert ds.xsg.subset_bbox((-72, 39, -70, 41)) is None


def test_subset_vars_passthrough_without_grid():
    ds = xr.Dataset({"a": (("x",), [1, 2, 3])})
    with pytest.warns(UserWarning, match="no grid type"):
        out = ds.xsg.subset_vars(["a"])
    # Without a recognized grid, subset_vars returns the dataset unchanged.
    assert "a" in out.data_vars


def test_has_vertical_levels_false_without_grid():
    ds = xr.Dataset()
    with pytest.warns(UserWarning, match="no grid type"):
        assert ds.xsg.has_vertical_levels is False
