"""
Implementation for any unknown 1D and 2D grids

"""

import numpy as np
import xarray as xr

from xarray_subset_grid.grid import Grid
from xarray_subset_grid.selector import Selector
from xarray_subset_grid.utils import (
    normalize_bbox_x_coords,
    normalize_polygon_x_coords,
)


class RegularGridBBoxSelector(Selector):
    """Selector for regular lat/lng grids."""

    bbox: tuple[float, float, float, float]
    _longitude_selection: slice
    _latitude_selection: slice

    def __init__(self, bbox: tuple[float, float, float, float]):
        super().__init__()
        self.bbox = bbox

    def select(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Perform the selection on the dataset.
        """
        lat = ds[ds.cf.coordinates.get("latitude")[0]]
        lon = ds[ds.cf.coordinates.get("longitude")[0]]

        xmin, xmax = self.bbox[0], self.bbox[2]
        ymin, ymax = self.bbox[1], self.bbox[3]

        return ds.where(
            (xmin <= lon) & (lon <= xmax) & (ymin <= lat) & (lat <= ymax),
            drop=True,
        )


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
        # Short-circut to known grids.
        grid = ds.variables.get("grid", None)
        if grid is not None:
            return False

        # Are coords available?
        lat = ds.cf.coordinates.get("latitude", None)
        lon = ds.cf.coordinates.get("longitude", None)
        if lat is None or lon is None:
            return False

        # Must have only one lon, lat!
        if (len(lat) != len(lon)) or len(lat) > 1:
            return False

        # If lat, lon are consistent and not 3D, we have a grid!
        lat, lon = lat[0], lon[0]
        if (ds[lon].ndim == ds[lat].ndim) or ds[lon].ndim < 3:
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
