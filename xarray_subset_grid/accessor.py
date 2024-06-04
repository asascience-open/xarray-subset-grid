from typing import Optional
import warnings

import xarray as xr
from numpy import ndarray

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
    warnings.warn("no grid type found in this dataset")
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

    @property
    def data_vars(self) -> set[str]:
        """List of data variables

        These variables exist on the grid and are available to used for
        data analysis. These can be discarded when subsetting the dataset
        when they are not needed.
        """
        if self._ds:
            return set(self._ds.coords)
        return set()

    @property
    def coords(self) -> set[str]:
        if self._ds:
            return self._ds.coords
        return set()

    @property
    def grid_vars(self) -> set[str]:
        """List of grid variables

        These variables are used to define the grid and thus should be kept
        when subsetting the dataset
        """
        if self._grid:
            return self._grid.grid_vars(self._ds)
        return set()

    @property
    def extra_vars(self) -> set[str]:
        if self._grid:
            return self._grid.extra_vars(self._ds)
        return set()

    def subset_vars(self, vars: list[str]) -> xr.Dataset:
        """Subset the dataset to the given variables, keeping the grid variables as well

        :param vars: The variables to keep
        :return: The subsetted dataset
        """
        if self._grid:
            return self._grid.subset_vars(self._ds, vars)
        return self._ds

    def subset_polygon(
        self, polygon: list[tuple[float, float]] | ndarray
    ) -> Optional[xr.Dataset]:
        """Subset the dataset to the grid.

        This call is forwarded to the grid implementation with the loaded dataset.

        :param ds: The dataset to subset
        :param polygon: The polygon to subset to
        :return: The subsetted dataset
        """
        if self._grid:
            return self._grid.subset_polygon(self._ds, polygon)
        return None

    def subset_bbox(
        self, bbox: tuple[float, float, float, float]
    ) -> Optional[xr.Dataset]:
        """Subset the dataset to the bounding box

        This call is forwarded to the grid implementation with the loaded dataset.

        :param ds: The dataset to subset
        :param bbox: The bounding box to subset to
        :return: The subsetted dataset
        """
        if self._grid:
            return self._grid.subset_bbox(self._ds, bbox)
        return None
