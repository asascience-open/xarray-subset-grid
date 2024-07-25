import xarray as xr

from xarray_subset_grid.grids.ugrid import UGrid


class FVCOMGrid(UGrid):
    """Grid implementation for FVCOM datasets, extending the UGrid implementation"""

    @staticmethod
    def recognize(ds: xr.Dataset) -> bool:
        """Recognize if the dataset matches the given grid"""
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
        """Name of the grid type"""
        return "fvcom"

    def subset_vertical_level(
        self, ds: xr.Dataset, level: float, method: str | None = None
    ) -> xr.Dataset:
        """Subset the dataset to the vertical level

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
            elevation_index = int((siglay - level).argmin().values)
            selections["siglay"] = elevation_index
        if "siglev" in ds.dims:
            siglev = ds["siglev"].isel(node=0)
            elevation_index = int((siglev - level).argmin().values)
            selections["siglev"] = elevation_index

        return ds.isel(selections)
