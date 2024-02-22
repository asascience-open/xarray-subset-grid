import xarray as xr

from xarray_subset_grid.grid import Grid


class ROMSGrid(Grid):
    """ROMS grid implementation"""

    @staticmethod
    def recognize(ds: xr.Dataset) -> bool:
        """Recognize if the dataset matches the given grid"""
        return "grid_topology" in ds.cf.cf_roles

    @property
    def name(self) -> str:
        """Name of the grid type"""
        return "roms"

    def subset(self, ds: xr.Dataset, **kwargs) -> xr.Dataset:
        """Subset the dataset to the grid
        :param ds: The dataset to subset
        :param kwargs: Additional arguments for the subset operation
        :return: The subsetted dataset
        """
        return ds
