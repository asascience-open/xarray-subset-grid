import numpy as np
import xarray as xr

from xarray_subset_grid.grids.ugrid import UGrid


class FVCOMGrid(UGrid):
    """Grid implementation for FVCOM datasets, extending the UGrid
    implementation."""

    @staticmethod
    def recognize(ds: xr.Dataset) -> bool:
        """Recognize if the dataset matches the given grid."""
        is_ugrid = UGrid.recognize(ds)
        if not is_ugrid:
            return False

        # For now, the only reason for this subclass is to add support
        # for subsetting FVCOMs non standard vertical levels
        is_recognized = False
        coords = ["siglay", "siglev"]
        for coord in coords:
            if coord in ds:
                if len(ds[coord].shape) > 1 and "node" in ds[coord].dims:
                    is_recognized = True
                    break

        return is_recognized

    @property
    def name(self) -> str:
        """Name of the grid type."""
        return "fvcom"

    def subset_bottom_level(self, ds: xr.Dataset) -> xr.Dataset:
        if not self.has_vertical_levels(ds):
            return ds

        positive_direction = self.vertical_positive_direction(ds)

        selections = {}
        if "siglay" in ds.dims:
            siglay = ds["siglay"].isel(node=0)
            if positive_direction == "up":
                elevation_index = siglay.argmin().values
            else:
                elevation_index = siglay.argmax().values
            selections["siglay"] = elevation_index
        if "siglev" in ds.dims:
            siglev = ds["siglev"].isel(node=0)
            if positive_direction == "up":
                elevation_index = siglev.argmin().values
            else:
                elevation_index = siglev.argmax().values
            selections["siglev"] = elevation_index

        return ds.isel(selections)

    def subset_top_level(self, ds: xr.Dataset) -> xr.Dataset:
        if not self.has_vertical_levels(ds):
            return ds

        positive_direction = self.vertical_positive_direction(ds)

        selections = {}
        if "siglay" in ds.dims:
            siglay = ds["siglay"].isel(node=0)
            if positive_direction == "up":
                elevation_index = siglay.argmax().values
            else:
                elevation_index = siglay.argmin().values
            selections["siglay"] = elevation_index
        if "siglev" in ds.dims:
            siglev = ds["siglev"].isel(node=0)
            if positive_direction == "up":
                elevation_index = siglev.argmax().values
            else:
                elevation_index = siglev.argmin().values
            selections["siglev"] = elevation_index

        return ds.isel(selections)

    def subset_vertical_level(
        self, ds: xr.Dataset, level: float, method: str | None = None
    ) -> xr.Dataset:
        """Subset the dataset to the vertical level.

        :param ds: The dataset to subset
        :param level: The vertical level to subset to
        :param method: The method to use for the selection, this is the
            same as the method in xarray.Dataset.sel
        """
        if not self.has_vertical_levels(ds):
            return ds

        selections = {}
        if "siglay" in ds.dims:
            siglay = ds["siglay"].isel(node=0)
            elevation_index = int(np.absolute(siglay - level).argmin().values)
            selections["siglay"] = elevation_index
        if "siglev" in ds.dims:
            siglev = ds["siglev"].isel(node=0)
            elevation_index = int(np.absolute(siglev - level).argmin().values)
            selections["siglev"] = elevation_index

        return ds.isel(selections)

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

        selections = {}
        if "siglay" in ds.dims:
            siglay = ds["siglay"].isel(node=0)
            elevation_indexes = [
                int(np.absolute(siglay - level).argmin().values) for level in levels
            ]
            selections["siglay"] = slice(elevation_indexes[0], elevation_indexes[1])
        if "siglev" in ds.dims:
            siglev = ds["siglev"].isel(node=0)
            elevation_index = [int(np.absolute(siglev - level).argmin().values) for level in levels]
            selections["siglev"] = slice(elevation_index[0], elevation_index[1])

        return ds.isel(selections)
