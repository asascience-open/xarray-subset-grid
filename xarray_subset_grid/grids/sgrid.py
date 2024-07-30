import numpy as np
import xarray as xr

from xarray_subset_grid.grid import Grid
from xarray_subset_grid.selector import Selector
from xarray_subset_grid.utils import compute_2d_subset_mask


class SGridSelector(Selector):
    polygon: list[tuple[float, float]] | np.ndarray

    _grid_topology_key: str
    _grid_topology: xr.DataArray
    _subset_masks: list[tuple[list[str], xr.DataArray]]

    def __init__(
        self,
        polygon: list[tuple[float, float]] | np.ndarray,
        subset_masks: list[tuple[list[str], xr.DataArray]],
    ):
        super().__init__()
        self.polygon = polygon
        self._subset_masks = subset_masks

    def select(self, ds: xr.Dataset) -> xr.Dataset:
        ds_out = []
        for mask in self._subset_masks:
            # First, we need to add the mask as a variable in the dataset
            # so that we can use it to mask and drop via xr.where, which requires that
            # the mask and data have the same shape and both are DataArrays with matching
            # dimensions
            ds_subset = ds.assign(subset_mask=mask[1])

            # Now we can use the mask to subset the data
            ds_subset = ds_subset[mask[0]].where(ds_subset.subset_mask, drop=True).drop_encoding()
            ds_subset = ds_subset.drop_vars("subset_mask")

            # Add the subsetted dataset to the list for merging
            ds_out.append(ds_subset)

        # Merge the subsetted datasets
        ds_out = xr.merge(ds_out)

        ds_out = ds_out.assign({self._grid_topology_key: self._grid_topology})
        return ds_out


class SGrid(Grid):
    """Grid implementation for SGRID datasets"""

    @staticmethod
    def recognize(ds: xr.Dataset) -> bool:
        """Recognize if the dataset matches the given grid"""
        try:
            _grid_topology_keys = ds.cf.cf_roles["grid_topology"]
        except KeyError:
            return False

        # For now, if the dataset has a grid topology and not a mesh topology,
        # we assume it's a SGRID
        return len(_grid_topology_keys) > 0 and _grid_topology_keys[0] in ds

    @property
    def name(self) -> str:
        """Name of the grid type"""
        return "sgrid"

    def grid_vars(self, ds: xr.Dataset) -> set[str]:
        """Set of grid variables

        These variables are used to define the grid and thus should be kept
        when subsetting the dataset
        """
        grid_topology_key = ds.cf.cf_roles["grid_topology"][0]
        grid_topology = ds[grid_topology_key]
        grid_coords = [grid_topology_key]
        for _dims, coords in _get_sgrid_dim_coord_names(grid_topology):
            grid_coords.extend(coords)
        return set(grid_coords)

    def data_vars(self, ds: xr.Dataset) -> set[str]:
        """Set of data variables

        These variables exist on the grid and are available to used for
        data analysis. These can be discarded when subsetting the dataset
        when they are not needed.
        """
        grid_topology_key = ds.cf.cf_roles["grid_topology"][0]
        grid_topology = ds[grid_topology_key]
        dims = []
        for dims, _coords in _get_sgrid_dim_coord_names(grid_topology):
            dims.extend(dims)
        dims = set(dims)

        return {var for var in ds.data_vars if not set(ds[var].dims).isdisjoint(dims)}

    def subset_polygon(
        self, ds: xr.Dataset, polygon: list[tuple[float, float]] | np.ndarray
    ) -> xr.Dataset:
        """Subset the dataset to the grid
        :param ds: The dataset to subset
        :param polygon: The polygon to subset to
        :return: The subsetted dataset
        """
        grid_topology_key = ds.cf.cf_roles["grid_topology"][0]
        grid_topology = ds[grid_topology_key]
        dims = _get_sgrid_dim_coord_names(grid_topology)

        subset_masks: list[tuple[list[str], xr.DataArray]] = []
        for dim, coord in dims:
            # Get the variables that have the dimensions
            unique_dims = set(dim)
            vars = [k for k in ds.variables if unique_dims.issubset(set(ds[k].dims))]

            # If the dataset has already been subset and there are no variables with
            # the dimensions, we can skip this dimension set
            if len(vars) == 0:
                continue

            # Get the coordinates for the dimension
            lon: xr.DataArray | None = None
            lat: xr.DataArray | None = None
            for c in coord:
                if "lon" in ds[c].attrs.get("standard_name", ""):
                    lon = ds[c]
                elif "lat" in ds[c].attrs.get("standard_name", ""):
                    lat = ds[c]

            if lon is None or lat is None:
                raise ValueError(f"Could not find lon and lat for dimension {dim}")

            subset_mask = compute_2d_subset_mask(lat=lat, lon=lon, polygon=polygon)

            subset_masks.append((vars, subset_mask))

        selector = SGridSelector(polygon=polygon, subset_masks=subset_masks)
        return selector.select(ds)


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
