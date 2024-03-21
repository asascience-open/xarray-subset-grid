import numpy as np
import xarray as xr
from numpy import ndarray

from xarray_subset_grid.grid import Grid
from xarray_subset_grid.utils import ray_tracing_numpy


class SGrid(Grid):
    """Grid implementation for SGRID datasets"""

    @staticmethod
    def recognize(ds: xr.Dataset) -> bool:
        """Recognize if the dataset matches the given grid"""
        try:
            _grid_topology_keys = ds.cf.cf_roles["grid_topology"]
        except KeyError:
            return False

        # For now, if the dataset has a grid topology and not a mesh topology, we assume it's a SGRID
        return len(_grid_topology_keys) > 0 and _grid_topology_keys[0] in ds

    @property
    def name(self) -> str:
        """Name of the grid type"""
        return "sgrid"

    def subset_polygon(
        self, ds: xr.Dataset, polygon: list[tuple[float, float]] | ndarray
    ) -> xr.Dataset:
        """Subset the dataset to the grid
        :param ds: The dataset to subset
        :param polygon: The polygon to subset to
        :return: The subsetted dataset
        """
        grid_topology_key = ds.cf.cf_roles["grid_topology"][0]
        grid_topology = ds[grid_topology_key]
        dims = _get_sgrid_dim_coord_names(grid_topology)

        ds_out = []

        for dim, coord in dims:
            # Get the variables that have the dimensions
            unique_dims = set(dim)
            vars = [k for k in ds.variables if unique_dims.issubset(set(ds[k].dims))]

            # If the dataset has already been subset and there are no variables with
            # the dimensions, we can skip this dimension set
            if len(vars) == 0:
                continue

            # Get the coordinates for the dimension
            lon = np.array([])
            lat = np.array([])
            mask_dims = ("", "")
            for c in coord:
                if "lon" in ds[c].attrs.get("standard_name", ""):
                    lon = ds[c].values
                    mask_dims = ds[c].dims
                elif "lat" in ds[c].attrs.get("standard_name", ""):
                    lat = ds[c].values
                    mask_dims = ds[c].dims

            # Find the subset of the coordinates that are inside the polygon and reshape
            # to match the original dimension shape
            polygon_mask = ray_tracing_numpy(lon.flat, lat.flat, polygon).reshape(
                lon.shape
            )

            # Adjust the mask to only mask the rows and columns that are completely
            # outside the polygon. If the row and column both touch the target polygon
            # then we want to keep them
            polygon_mask = np.where(polygon_mask, 1, 0)
            polygon_row_mask = np.all(polygon_mask is False, axis=0)
            polygon_col_mask = np.all(polygon_mask is False, axis=1)
            polygon_mask[:, ~polygon_row_mask] += 1
            polygon_mask[~polygon_col_mask, :] += 1
            polygon_mask = np.where(polygon_mask > 1, True, False)

            # First, we need to add the mask as a variable in the dataset
            # so that we can use it to mask and drop via xr.where, which requires that
            # the mask and data have the same shape and both are DataArrays with matching
            # dimensions
            ds_subset = ds.assign(
                subset_mask=xr.DataArray(polygon_mask, dims=mask_dims)
            )

            # Now we can use the mask to subset the data
            ds_subset = ds_subset[vars].where(ds_subset.subset_mask, drop=True)

            # Add the subsetted dataset to the list for merging
            ds_out.append(ds_subset)

        # Merge the subsetted datasets
        ds_out = xr.merge(ds_out)

        ds_out = ds_out.assign({grid_topology_key: grid_topology})

        return ds_out


def _get_sgrid_dim_coord_names(
    grid_topology: xr.DataArray,
) -> list[tuple[list[str], list[str]]]:
    """Get the names of the dimensions that are coordinates

    This is really hacky and possibly not a long term solution, but it is our generic best start
    """
    dims = []
    coords = []
    for k, v in grid_topology.attrs.items():
        if "_dimensions" in k:
            # Remove padding for now
            d = " ".join([v for v in v.split(" ") if "(" not in v and ")" not in v])
            if ":" in d:
                d = [d.replace(":", "") for d in d.split(" ") if ":" in d]
            else:
                d = d.split(" ")
            dims.append(d)
        elif "_coordinates" in k:
            coords.append(v.split(" "))

    return list(zip(dims, coords))
