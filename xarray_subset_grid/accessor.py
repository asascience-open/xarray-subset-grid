import warnings

# from typing import Optional, Union
import numpy as np
import xarray as xr

from xarray_subset_grid.grid import Grid
from xarray_subset_grid.grids import FVCOMGrid, RegularGrid, RegularGrid2d, SELFEGrid, SGrid, UGrid

_grid_impls = [FVCOMGrid, SELFEGrid, UGrid, SGrid, RegularGrid2d, RegularGrid]


def register_grid_impl(grid_impl: Grid, priority: int = 0):
    """Register a new grid implementation.

    :param grid_impl: The grid implementation to register
    :param priority: The priority of the implementation. Highest
        priority is 0. Default is 0.
    """
    _grid_impls.insert(priority, grid_impl)


def grid_factory(ds: xr.Dataset) -> Grid | None:
    """Get the grid implementation for the given dataset.

    :param ds: The dataset to get the grid implementation for
    :return: The grid implementation or None if no implementation is
        found
    """
    for grid_impl in _grid_impls:
        if grid_impl.recognize(ds):
            return grid_impl()
    warnings.warn("no grid type found in this dataset")
    return None


@xr.register_dataset_accessor("xsg")
class GridDatasetAccessor:
    """Accessor for grid operations on datasets."""

    _ds: xr.Dataset
    _grid: Grid | None

    def __init__(self, ds: xr.Dataset):
        """Create a new grid dataset accessor.

        :param ds: The dataset to create the accessor for
        """
        self._ds = ds
        self._grid = grid_factory(ds)

    @property
    def grid(self) -> Grid | None:
        """The recognized grid implementation for the given dataset :return:
        The grid implementation or None if no implementation is found."""
        return self._grid

    @property
    def data_vars(self) -> set[str]:
        """List of data variables.

        These variables exist on the grid and are available to used for
        data analysis. These can be discarded when subsetting the
        dataset when they are not needed.
        """
        if self._ds:
            return self._grid.data_vars(self._ds)
        return set()

    @property
    def coords(self) -> set[str]:
        if self._ds:
            return self._ds.coords
        return set()

    @property
    def grid_vars(self) -> set[str]:
        """List of grid variables.

        These variables are used to define the grid and thus should be
        kept when subsetting the dataset
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
        """Subset the dataset to the given variables, keeping the grid
        variables as well.

        :param vars: The variables to keep
        :return: The subsetted dataset
        """
        if self._grid:
            return self._grid.subset_vars(self._ds, vars)
        return self._ds

    @property
    def has_vertical_levels(self) -> bool:
        """Check if the dataset has vertical coordinates."""
        if self._grid:
            return self._grid.has_vertical_levels(self._ds)
        return False

    def subset_surface_level(self, method: str | None) -> xr.Dataset:
        """Subset the dataset to the surface level."""
        if self._grid:
            return self._grid.subset_surface_level(self._ds, method)
        return self._ds

    def subset_bottom_level(self) -> xr.Dataset:
        """Subset the dataset to the bottom level according to the datasets CF
        metadata and available vertical coordinates using nearest neighbor
        selection."""
        if self._grid:
            return self._grid.subset_bottom_level(self._ds)
        return self._ds

    def subset_top_level(self) -> xr.Dataset:
        """Subset the dataset to the top level according to the datasets CF
        metadata and available vertical coordinates using nearest neighbor
        selection."""
        if self._grid:
            return self._grid.subset_top_level(self._ds)
        return self._ds

    def subset_vertical_level(self, level: float, method: str | None = None) -> xr.Dataset:
        """Subset the dataset to the vertical level.

        :param level: The vertical level to subset to
        :param method: The method to use for the selection, this is the
            same as the method in xarray.Dataset.sel
        :return: The subsetted dataset
        """
        if self._grid:
            return self._grid.subset_vertical_level(self._ds, level, method)
        return self._ds

    def subset_vertical_levels(
        self, levels: tuple[float, float], method: str | None = None
    ) -> xr.Dataset:
        """Subset the dataset to the vertical level.

        :param levels: The vertical levels to subset to
        :param method: The method to use for the selection, this is the
            same as the method in xarray.Dataset.sel
        :return: The subsetted dataset
        """
        if self._grid:
            return self._grid.subset_vertical_levels(self._ds, levels, method)
        return self._ds

    def subset_polygon(self, polygon: list[tuple[float, float]] | np.ndarray) -> xr.Dataset | None:
        """Subset the dataset to the grid.

        This call is forwarded to the grid implementation with the
        loaded dataset.

        :param ds: The dataset to subset
        :param polygon: The polygon to subset to
        :return: The subsetted dataset
        """
        if self._grid:
            return self._grid.subset_polygon(self._ds, polygon)
        return None

    def subset_bbox(self, bbox: tuple[float, float, float, float]) -> xr.Dataset | None:
        """Subset the dataset to the bounding box.

        This call is forwarded to the grid implementation with the
        loaded dataset.

        :param ds: The dataset to subset
        :param bbox: The bounding box to subset to
        :return: The subsetted dataset
        """
        if self._grid:
            return self._grid.subset_bbox(self._ds, bbox)
        return None
