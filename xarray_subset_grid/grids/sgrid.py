import xarray as xr
from numpy import ndarray

from xarray_subset_grid.grid import Grid


class SGrid(Grid):
    '''Grid implementation for SGRID datasets'''

    @staticmethod
    def recognize(ds: xr.Dataset) -> bool:
        """Recognize if the dataset matches the given grid"""
        try:
            _mesh = ds.cf['grid_topology']
        except KeyError:
            return False

        # For now, if the dataset has a grid topology and not a mesh topology, we assume it's a SGRID
        return True

    @property
    def name(self) -> str:
        """Name of the grid type"""
        return "sgrid"

    def subset_polygon(self, ds: xr.Dataset, polygon: list[list[float, float]] | ndarray) -> xr.Dataset:
        """Subset the dataset to the grid
        :param ds: The dataset to subset
        :param polygon: The polygon to subset to
        :return: The subsetted dataset
        """
        return ds
