"""
Implementation of grids with data at the nodes only:

RectilinearGrid - North-East aligned
QuadGrid (curvilinear grid)

These can be defined with just CF coordinates
-- no need for a grid/mesh variable
"""

import numpy as np
import xarray as xr

from xarray_subset_grid.grid import Grid
from xarray_subset_grid.selector import Selector
from xarray_subset_grid.utils import (
    normalize_bbox_x_coords,
    normalize_polygon_x_coords,
)


class JustNodesBBoxSelector(Selector):
    """Selector for rectilinear lat/long grids."""

    # FixMe: it is worth caching this result?
    #        like is done for UGrids?

    bbox: tuple[float, float, float, float]
    # _longitude_selection: slice
    # _latitude_selection: slice

    def __init__(self,
                 name: str,
                 bbox: tuple[float, float, float, float],
                 ):
        super().__init__()
        self.name = name
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


class JustNodesPolygonSelector(JustNodesBBoxSelector):
    """Polygon Selector for regular lat/lon grids."""

    # with a regular grid, you have to select the full bounding box anyway
    # this this simply computes the bounding box, and uses the same code.

    def __init__(self,
                 name: str | None,
                 polygon: list[tuple[float, float]] | np.ndarray,
):
        polygon = np.asarray(polygon)
        bbox = (
            polygon[:, 0].min(),
            polygon[:, 1].min(),
            polygon[:, 0].max(),
            polygon[:, 1].max(),
        )
        super().__init__(name=name, bbox=bbox)


class JustNodesGrid(Grid):
    """Grid implementation for regular lat/long grids."""

    @staticmethod
    def recognize(ds: xr.Dataset) -> bool:
        # must be defined by subclasses
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Name of the grid type."""
        raise NotImplementedError

    def grid_vars(self, ds: xr.Dataset) -> set[str]:
        """
        Set of grid variables.

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

        selector = JustNodesPolygonSelector(polygon=polygon, name=name)
        return selector

    ## fixme: is this ever called
    ##   the other grids don't have it.
    def compute_bbox_subset_selector(
        self,
        ds: xr.Dataset,
        bbox: tuple[float, float, float, float],
        name: str | None = None,
    ) -> Selector:
        bbox = normalize_bbox_x_coords(ds[ds.cf.coordinates["latitude"][0]].values, bbox)
        selector = JustNodesBBoxSelector(name, bbox)
        return selector

class RectilinearGrid(JustNodesGrid):
    """
    Grid implementation for rectilinear lat/long grids.

    North aligned, 1D lat and lon coordinates
    """
    @property
    def name(self) -> str:
        """Name of the grid type."""
        return "rectilinear grid"

    def recognize(ds: xr.Dataset) -> bool:
        """
        Recognize if the Dataset matches a rectilinear grid.
        """
        # Short-circuit to defined grids (UGRID or SGRID)
        mesh_vars = (ds.cf.cf_roles.get("mesh_topology"),
                    ds.cf.cf_roles.get("grid_topology"),
                    )
        if mesh_vars != (None, None):  # it's an SGRID or UGRID
            return False

        lat = ds.cf.coordinates.get("latitude", None)
        lon = ds.cf.coordinates.get("longitude", None)
        if (lat is None) or (lon is None):
            return False

        # Must have only one lon, lat!
        if (len(lat) != len(lon)) or len(lat) > 1:
            return False

        # Make sure the coordinates are 1D and don't match
        lat_dims = ds[lat[0]].dims
        ndims_lat = ds[lat[0]].ndim
        lon_dims = ds[lon[0]].dims
        ndims_lon = ds[lat[0]].ndim
        if ((lat_dims == lon_dims)
            or (ndims_lat > 1)
            or (ndims_lon > 1)
            ):
            return False

        # make sure that at least one variable is using both the
        #   latitude and longitude dimensions
        #   (ugrids have both coordinates, but not both dimensions)
        for var_name, var in ds.data_vars.items():
            if ((lat_dims[0] in var.dims)
                and (lon_dims[0] in var.dims)):
                return True
        return False

class QuadGrid(JustNodesGrid):
    """
    Grid implementation for quadrilateral grids.
    (curvilinear)

    Not North -- East aligned.

    2D lat and lon coordinates.
    Data only on the nodes
    """
    @property
    def name(self) -> str:
        """Name of the grid type."""
        return "quadrilateral grid"

    def recognize(ds: xr.Dataset) -> bool:
        """
        Recognize if the Dataset matches a rectilinear grid.
        """
        # Short-circuit to defined grids (UGRID or SGRID)
        mesh_vars = (ds.cf.cf_roles.get("mesh_topology"),
                    ds.cf.cf_roles.get("grid_topology"),
                    )
        if mesh_vars != (None, None):  # it's an SGRID or UGRID
            return False

        lat = ds.cf.coordinates.get("latitude", None)
        lon = ds.cf.coordinates.get("longitude", None)
        if (lat is None) or (lon is None):
            return False

        # Must have only one lon, lat!
        if (len(lat) != len(lon)) or len(lat) > 1:
            return False

        # Make sure the coordinates are 2D and match
        lat_dims = ds[lat[0]].dims
        ndims_lat = ds[lat[0]].ndim
        lon_dims = ds[lon[0]].dims
        ndims_lon = ds[lat[0]].ndim
        if not ((lat_dims == lon_dims)
            and (ndims_lat == 2)
            and (ndims_lon == 2)
            ):
            return False

        # make sure that at least one variable is using both the
        #   latitude and longitude dimensions
        #   (ugrids have both coordinates, but not both dimensions)
        for var_name, var in ds.data_vars.items():
            if ((lat_dims[0] in var.dims)
                and (lon_dims[0] in var.dims)):
                return True
        return False


