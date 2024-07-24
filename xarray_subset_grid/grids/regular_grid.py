import numpy as np
import xarray as xr

from xarray_subset_grid.grid import Grid
from xarray_subset_grid.utils import (
    normalize_bbox_x_coords,
    normalize_polygon_x_coords,
    ray_tracing_numpy,
)


class RegularGrid(Grid):
    """Grid implementation for regular lat/lng grids"""

    @staticmethod
    def recognize(ds: xr.Dataset) -> bool:
        """Recognize if the dataset matches the given grid"""
        lat = ds.cf.coordinates.get("latitude", None)
        lon = ds.cf.coordinates.get("longitude", None)
        if lat is None or lon is None:
            return False

        # Make sure the coordinates are 1D and match
        lat_ndim = ds[lat[0]].ndim
        lon_ndim = ds[lon[0]].ndim
        return lat_ndim == lon_ndim and lon_ndim == 1

    @property
    def name(self) -> str:
        """Name of the grid type"""
        return "regular_grid"

    def grid_vars(self, ds: xr.Dataset) -> set[str]:
        """Set of grid variables

        These variables are used to define the grid and thus should be kept
        when subsetting the dataset
        """
        lat = ds.cf.coordinates["latitude"][0]
        lon = ds.cf.coordinates["longitude"][0]
        return {lat, lon}

    def data_vars(self, ds: xr.Dataset) -> set[str]:
        """Set of data variables

        These variables exist on the grid and are available to used for
        data analysis. These can be discarded when subsetting the dataset
        when they are not needed.
        """
        lat = ds.cf.coordinates["latitude"][0]
        lon = ds.cf.coordinates["longitude"][0]
        return {
            var
            for var in ds.data_vars
            if var not in {lat, lon}
            and "latitude" in var.cf.coordinates
            and "longitude" in var.cf.coordinates
        }

    def subset_polygon(
        self, ds: xr.Dataset, polygon: list[tuple[float, float]] | np.ndarray
    ) -> xr.Dataset:
        """Subset the dataset to the grid
        :param ds: The dataset to subset
        :param polygon: The polygon to subset to
        :return: The subsetted dataset
        """
        lat = ds.cf["latitude"]
        lon = ds.cf["longitude"]

        x = np.array(lon.flat)
        polygon = normalize_polygon_x_coords(x, polygon)
        polygon_mask = ray_tracing_numpy(x, lat.flat, polygon).reshape(lon.shape)

        ds_subset = ds.cf.isel(
            lon=polygon_mask,
            lat=polygon_mask,
        )
        return ds_subset

    def subset_bbox(self, ds: xr.Dataset, bbox: tuple[float, float, float, float]) -> xr.Dataset:
        """Subset the dataset to the bounding box
        :param ds: The dataset to subset
        :param bbox: The bounding box to subset to
        :return: The subsetted dataset
        """
        bbox = normalize_bbox_x_coords(ds.cf["longitude"].values, bbox)
        return ds.cf.sel(lon=slice(bbox[0], bbox[2]), lat=slice(bbox[1], bbox[3]))
