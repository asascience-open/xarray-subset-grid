import numpy as np
import pytest
import xarray as xr

import xarray_subset_grid.accessor  # noqa: F401 -- register accessor


def test_accessor_warns_when_no_grid_recognized():
    ds = xr.Dataset()
    with pytest.raises(ValueError, match="Cannot find grid or coords for"):
        ds.xsg


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
    with pytest.raises(ValueError, match="Cannot find grid or coords for"):
        ds.xsg.subset_polygon(poly)
    with pytest.raises(ValueError, match="Cannot find grid or coords for"):
        ds.xsg.subset_bbox((-72, 39, -70, 41))


def test_subset_vars_raises_without_grid():
    ds = xr.Dataset({"a": (("x",), [1, 2, 3])})
    with pytest.raises(ValueError, match="Cannot find grid or coords for"):
        ds.xsg.subset_vars(["a"])


def test_has_vertical_levels_false_without_grid():
    ds = xr.Dataset()
    with pytest.raises(ValueError, match="Cannot find grid or coords for"):
        ds.xsg.has_vertical_levels
