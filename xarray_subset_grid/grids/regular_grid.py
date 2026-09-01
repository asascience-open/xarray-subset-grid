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
#     # with a regular grid, you have to select the full bounding box anyway
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
    _longitude_bounds: tuple[float, float]
    _latitude_selection: slice

    def __init__(self, bbox: tuple[float, float, float, float]):
        super().__init__()
        self.bbox = bbox
        self._longitude_bounds = (bbox[0], bbox[2])
        self._latitude_selection = slice(bbox[1], bbox[3])

    def _longitude_span(self) -> float:
        west, east = self._longitude_bounds
        return (east - west) % 360

    def _validate_longitude_span(self):
        span = self._longitude_span()
        if np.isclose(span, 0.0):
            raise ValueError(
                "Invalid longitude bounds: west and east bounds "
                "cannot define a zero-width selection"
            )
        if span >= 180.0 and not np.isclose(span, 180.0):
            raise ValueError(
                "Invalid longitude bounds: subsetting bounds "
                "must span less than half-way around the earth"
            )
        if np.isclose(span, 180.0):
            raise ValueError(
                "Invalid longitude bounds: subsetting bounds "
                "must span less than half-way around the earth"
            )

    def _build_longitude_slices(self, lon: xr.DataArray) -> list[slice]:
        west, east = self._longitude_bounds
        lon_min = float(lon.min().values)
        lon_max = float(lon.max().values)

        if west <= east:
            longitude_slices = [slice(west, east)]
        else:
            longitude_slices = [slice(west, lon_max), slice(lon_min, east)]

        if np.all(np.diff(lon) < 0):
            longitude_slices = [slice(sl.stop, sl.start) for sl in longitude_slices]

        return longitude_slices

    def select(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Perform the selection on the dataset.
        """
        self._validate_longitude_span()

        lat = ds[ds.cf.coordinates.get("latitude")[0]]
        lon = ds[ds.cf.coordinates.get("longitude")[0]]

        latitude_selection = self._latitude_selection
        if np.all(np.diff(lat) < 0):
            # swap the slice if the latitudes are descending
            latitude_selection = slice(latitude_selection.stop, latitude_selection.start)

        longitude_selections = self._build_longitude_slices(lon)
        selections = [
            ds.cf.sel(lon=lon_sel, lat=latitude_selection) for lon_sel in longitude_selections
        ]

        if len(selections) == 1:
            return selections[0]

        lon_dim = lon.dims[0]
        return xr.concat(selections, dim=lon_dim)


class RegularGridPolygonSelector(RegularGridBBoxSelector):
    """Polygon Selector for regular lat/lon grids."""

    # with a regular grid, you have to select the full bounding box anyway
    # this this simply computes the bounding box, and uses the same code.

    def __init__(self, polygon: list[tuple[float, float]] | np.ndarray):
        polygon = np.asarray(polygon)
        bbox = (
            polygon[:, 0].min(),
            polygon[:, 1].min(),
            polygon[:, 0].max(),
            polygon[:, 1].max(),
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
        data_vars = {
            var.name
            for var in ds.data_vars.values()
            if var.name not in {lat, lon}
            and "latitude" in var.cf.coordinates
            and "longitude" in var.cf.coordinates
        }
        return data_vars

    def compute_polygon_subset_selector(
        self,
        ds: xr.Dataset,
        polygon: list[tuple[float, float]] | np.ndarray,
        name: str | None = None,
    ) -> Selector:

        polygon = np.asarray(polygon)
        lon = ds.cf["longitude"].data

        polygon = normalize_polygon_x_coords(lon, polygon)

        selector = RegularGridPolygonSelector(polygon=polygon)
        return selector

    def compute_bbox_subset_selector(
        self,
        ds: xr.Dataset,
        bbox: tuple[float, float, float, float],
        name: str | None = None,
    ) -> Selector:
        bbox = normalize_bbox_x_coords(ds.cf["longitude"].values, bbox)
        selector = RegularGridBBoxSelector(bbox)
        return selector
