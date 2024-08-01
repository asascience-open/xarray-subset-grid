import numpy as np
import xarray as xr

from xarray_subset_grid.grids.ugrid import UGrid


class SELFEGrid(UGrid):
    """Grid implementation for SELFE datasets, extending the UGrid
    implementation."""

    @staticmethod
    def recognize(ds: xr.Dataset) -> bool:
        """Recognize if the dataset matches the given grid."""
        is_ugrid = UGrid.recognize(ds)
        if not is_ugrid:
            return False

        # For now, the only reason for this subclass is to add support
        # for subsetting SELFE's non standard vertical levels
        return "sigma" in ds and "nv" in ds.dims and "nv" not in ds["sigma"].dims

    @property
    def name(self) -> str:
        return "selfe"

    def subset_bottom_level(self, ds: xr.Dataset) -> xr.Dataset:
        if not self.has_vertical_levels(ds):
            return ds

        positive_direction = self.vertical_positive_direction(ds)

        if positive_direction == "up":
            index = ds.cf["sigma"].argmin().values
        else:
            index = ds.cf["sigma"].argmax().values

        return ds.isel(sigma=index, nv=index)

    def subset_top_level(self, ds: xr.Dataset) -> xr.Dataset:
        if not self.has_vertical_levels(ds):
            return ds

        positive_direction = self.vertical_positive_direction(ds)

        if positive_direction == "up":
            index = ds.cf["sigma"].argmax().values
        else:
            index = ds.cf["sigma"].argmin().values

        return ds.isel(sigma=index, nv=index)

    def subset_vertical_level(
        self, ds: xr.Dataset, level: float, method: str | None = "nearest"
    ) -> xr.Dataset:
        """Subset the dataset to the vertical level.

        :param ds: The dataset to subset
        :param level: The vertical level to subset to
        :param method: The method to use for the selection, this is the
            same as the method in xarray.Dataset.sel
        """
        if not self.has_vertical_levels(ds):
            return ds

        index = int(np.absolute(ds.cf["sigma"] - level).argmin().values)
        return ds.isel(sigma=index, nv=index)

    def subset_vertical_levels(
        self, ds: xr.Dataset, levels: tuple[float, float], method: str | None = "nearest"
    ) -> xr.Dataset:
        """Subset the dataset to the vertical level.

        :param ds: The dataset to subset
        :param levels: The vertical levels to subset to. This is a tuple
            with the minimum and maximum level. The minimum must be
            smaller than the maximum.
        :param method: The method to use for the selection, this is the
            same as the method in xarray.Dataset.sel
        """
        if not self.has_vertical_levels(ds):
            return ds

        if levels[0] >= levels[1]:
            raise ValueError("The minimum level must be smaller than the maximum level")

        indexes = [int(np.absolute(ds.cf["sigma"] - level).argmin().values) for level in levels]
        return ds.isel(sigma=slice(indexes[0], indexes[1]), nv=slice(indexes[0], indexes[1]))
