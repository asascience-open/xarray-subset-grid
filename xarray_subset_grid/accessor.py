from typing import Optional

import xarray as xr

from xarray_subset_grid.grid import Grid
from xarray_subset_grid.grids import SGrid, UGrid

_grid_impls = [UGrid, SGrid]


def register_grid_impl(grid_impl: Grid, priority: int = 0):
    """
    Register a new grid implementation.
    :param grid_impl: The grid implementation to register
    :param priority: The priority of the implementation. Highest priority is 0. Default is 0.
    """
    _grid_impls.insert(priority, grid_impl)


def grid_factory(ds: xr.Dataset) -> Optional[Grid]:
    """
    Get the grid implementation for the given dataset.
    :param ds: The dataset to get the grid implementation for
    :return: The grid implementation or None if no implementation is found
    """
    for grid_impl in _grid_impls:
        if grid_impl.recognize(ds):
            return grid_impl()

    return None


@xr.register_dataset_accessor("subset_grid")
class GridDatasetAccessor:
    """Accessor for grid operations on datasets"""

    _ds: xr.Dataset
    _grid: Optional[Grid]

    def __init__(self, ds: xr.Dataset):
        """
        Create a new grid dataset accessor.
        :param ds: The dataset to create the accessor for
        """
        self._ds = ds
        self._grid = grid_factory(ds)

    @property
    def grid(self) -> Optional[Grid]:
        """The recognized grid implementation for the given dataset
        :return: The grid implementation or None if no implementation is found
        """
        return self._grid
