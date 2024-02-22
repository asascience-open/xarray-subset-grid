import xarray as xr

from xarray_subset_grid.grid import Grid


class FVCOMGrid(Grid):
    """FVCOM grid implementation"""

    @staticmethod
    def recognize(ds: xr.Dataset) -> bool:
        """Recognize if the dataset matches the given grid"""
        return ds.attrs.get("source", "").lower().startswith("fvcom")

    @property
    def name(self) -> str:
        """Name of the grid type"""
        return "fvcom"

    def subset(self, ds: xr.Dataset, **kwargs) -> xr.Dataset:
        """Subset the dataset to the grid
        :param ds: The dataset to subset
        :param kwargs: Additional arguments for the subset operation
        :return: The subsetted dataset
        """
        return ds
