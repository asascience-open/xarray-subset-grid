import pytest
import xarray as xr

import xarray_subset_grid.accessor  # noqa: F401


def test_data_vars_returns_empty_set_when_grid_not_recognized():
    ds = xr.Dataset(
        data_vars={"foo": ("x", [1, 2, 3])},
        coords={"x": [0, 1, 2]},
    )

    with pytest.warns(UserWarning, match="no grid type found in this dataset"):
        accessor = ds.xsg

    assert accessor.grid is None
    assert accessor.data_vars == set()
