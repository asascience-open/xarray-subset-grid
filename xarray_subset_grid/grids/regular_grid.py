"""
Implementation for Rectangular grids

NOTE: it's called "regular", but I think this will work for
any (grid-aligned) rectangular grid:

The grid is defined by two 1-d arrays.

* delta_lat and delta_lon do not have to be constant.
"""

import numpy as np
import xarray as xr

from xarray_subset_grid.grid import Grid
from xarray_subset_grid.selector import Selector
from xarray_subset_grid.utils import (
    normalize_bbox_x_coords,
    normalize_polygon_x_coords,
)

# class RegularGridPolygonSelector(Selector):
#     """Polygon Selector for regular lat/lon grids."""
#     # with a regular grid, you have to select the full boudning box anyway
#     # this this simply computes the bounding box, and used that

#     polygon: list[tuple[float, float]] | np.ndarray
#     _polygon_mask: xr.DataArray

#     def __init__(self, polygon: list[tuple[float, float]] | np.ndarray, mask: xr.DataArray,
#                  name: str):
#         super().__init__()
#         self.name = name
#         self.polygon = polygon
#         self.polygon_mask = mask

#     def select(self, ds: xr.Dataset) -> xr.Dataset:
#         """Perform the selection on the dataset."""
#         ds_subset = ds.cf.isel(
#             lon=self._polygon_mask,
#             lat=self._polygon_mask,
#         )
#         return ds_subset



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
        """
        Perform the selection on the dataset.
        """
        lat = ds[ds.cf.coordinates.get("latitude")[0]]
        lon = ds[ds.cf.coordinates.get("longitude")[0]]
        if np.all(np.diff(lat) < 0):
            # swap the slice if the latitudes are decending
            self._latitude_selection = slice(self._latitude_selection.stop,
                                             self._latitude_selection.start)
         # and np.all(np.diff(lon) > 0):
        if np.all(np.diff(lon) < 0):
            # swap the slice if the longitudes are decending
            self._longitude_selection = slice(self._longitude_selection.stop,
                                              self._longitude_selection.start)

        return ds.cf.sel(lon=self._longitude_selection, lat=self._latitude_selection)

class RegularGridPolygonSelector(RegularGridBBoxSelector):
    """Polygon Selector for regular lat/lon grids."""
    # with a regular grid, you have to select the full bounding box anyway
    # this this simply computes the bounding box, and uses the same code.

    def __init__(self, polygon: list[tuple[float, float]] | np.ndarray):
        polygon = np.asarray(polygon)
        bbox = (polygon[:,0].min(),
                polygon[:,1].min(),
                polygon[:,0].max(),
                polygon[:,1].max(),
                )
        super().__init__(bbox=bbox)


class RegularGrid(Grid):
    """Grid implementation for regular lat/lng grids."""
    @staticmethod
    def recognize(ds: xr.Dataset) -> bool:
        """
        Recognize if the dataset matches the given grid.
        """
        lat = ds.cf.coordinates.get("latitude", None)
        lon = ds.cf.coordinates.get("longitude", None)
        if lat is None or lon is None:
            return False

        # choose first one -- valid assumption??
        lat = lat[0]
        lon = lon[0]
        # Make sure the coordinates are 1D and match
        if not (1 == ds[lat].ndim == ds[lon].ndim):
            return False

        # make sure that at least one variable is using both the
        #   latitude and longitude dimensions
        #   (ugrids have both coordinates, but not both dimensions)
        for var_name, var in ds.data_vars.items():
            if (lon in var.dims) and (lat in var.dims):
                return True
        return False

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
        data_vars = {var.name for var in ds.data_vars.values()
                         if var.name not in {lat, lon}
                         and "latitude" in var.cf.coordinates
                         and "longitude" in var.cf.coordinates
        }
        return data_vars

    def compute_polygon_subset_selector(self,
                                        ds: xr.Dataset,
                                        polygon: list[tuple[float, float]],
                                        ) -> Selector:

        polygon = np.asarray(polygon)
        lon = ds.cf["longitude"].data

        polygon = normalize_polygon_x_coords(lon, polygon)

        selector = RegularGridPolygonSelector(polygon=polygon)
        return selector

    def compute_bbox_subset_selector(self,
                                     ds: xr.Dataset,
                                     bbox: tuple[float, float, float, float],
                                     ) -> Selector:
        bbox = normalize_bbox_x_coords(ds.cf["longitude"].values, bbox)
        selector = RegularGridBBoxSelector(bbox)
        return selector
