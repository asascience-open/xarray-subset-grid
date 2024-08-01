import numpy as np
import xarray as xr

from xarray_subset_grid.grid import Grid
from xarray_subset_grid.selector import Selector
from xarray_subset_grid.utils import (
    normalize_bbox_x_coords,
    normalize_polygon_x_coords,
    ray_tracing_numpy,
)


class RegularGridPolygonSelector(Selector):
    """Selector for regular lat/lng grids."""

    polygon: list[tuple[float, float]] | np.ndarray
    _polygon_mask: xr.DataArray

    def __init__(self, polygon: list[tuple[float, float]] | np.ndarray, mask: xr.DataArray):
        super().__init__()
        self.polygon_mask = mask

    def select(self, ds: xr.Dataset) -> xr.Dataset:
        """Perform the selection on the dataset."""
        ds_subset = ds.cf.isel(
            lon=self._polygon_mask,
            lat=self._polygon_mask,
        )
        return ds_subset


class RegularGridBBoxSelector(Selector):
    """Selector for regular lat/lng grids."""

    bbox: tuple[float, float, float, float]
    _longitude_selection: slice
    _latitude_selection: slice

    def __init__(self, bbox: tuple[float, float, float, float]):
        super().__init__()
        self.bbox = bbox
        self._longitude_selection = slice(bbox[0], bbox[2])
        self._latitude_selection = slice(bbox[1], bbox[3])

    def select(self, ds: xr.Dataset) -> xr.Dataset:
        """Perform the selection on the dataset."""
        ds.cf.sel(lon=self._longitude_selection, lat=self._latitude_selection)


class RegularGrid(Grid):
    """Grid implementation for regular lat/lng grids."""

    @staticmethod
    def recognize(ds: xr.Dataset) -> bool:
        """Recognize if the dataset matches the given grid."""
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
        """Name of the grid type."""
        return "regular_grid"

    def grid_vars(self, ds: xr.Dataset) -> set[str]:
        """Set of grid variables.

        These variables are used to define the grid and thus should be
        kept when subsetting the dataset
        """
        lat = ds.cf.coordinates["latitude"][0]
        lon = ds.cf.coordinates["longitude"][0]
        return {lat, lon}

    def data_vars(self, ds: xr.Dataset) -> set[str]:
        """Set of data variables.

        These variables exist on the grid and are available to used for
        data analysis. These can be discarded when subsetting the
        dataset when they are not needed.
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

    def compute_polygon_subset_selector(
        self, ds: xr.Dataset, polygon: list[tuple[float, float]]
    ) -> Selector:
        lat = ds.cf["latitude"]
        lon = ds.cf["longitude"]

        x = np.array(lon.flat)
        polygon = normalize_polygon_x_coords(x, polygon)
        polygon_mask = ray_tracing_numpy(x, lat.flat, polygon).reshape(lon.shape)

        selector = RegularGridPolygonSelector(polygon, polygon_mask)
        return selector

    def compute_bbox_subset_selector(
        self,
        ds: xr.Dataset,
        bbox: tuple[float, float, float, float],
    ) -> Selector:
        bbox = normalize_bbox_x_coords(ds.cf["longitude"].values, bbox)
        selector = RegularGridBBoxSelector(bbox)
        return selector
