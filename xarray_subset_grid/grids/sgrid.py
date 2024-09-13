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
        name: str,
        polygon: list[tuple[float, float]] | np.ndarray,
        grid_topology_key: str,
        grid_topology: xr.DataArray,
        subset_masks: list[tuple[list[str], xr.DataArray]],
    ):
        super().__init__()
        self.name = name
        self.polygon = polygon
        self._grid_topology_key = grid_topology_key
        self._grid_topology = grid_topology
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

            # Add the subsetted dataset to the list for merging
            ds_out.append(ds_subset)

        # Merge the subsetted datasets
        ds_out = xr.merge(ds_out)

        ds_out = ds_out.assign({self._grid_topology_key: self._grid_topology})
        return ds_out


class SGrid(Grid):
    """Grid implementation for SGRID datasets."""

    @staticmethod
    def recognize(ds: xr.Dataset) -> bool:
        """Recognize if the dataset matches the given grid."""
        try:
            _grid_topology_keys = ds.cf.cf_roles["grid_topology"]
        except KeyError:
            return False

        # For now, if the dataset has a grid topology and not a mesh topology,
        # we assume it's a SGRID
        return len(_grid_topology_keys) > 0 and _grid_topology_keys[0] in ds

    @property
    def name(self) -> str:
        """Name of the grid type."""
        return "sgrid"

    def grid_vars(self, ds: xr.Dataset) -> set[str]:
        """Set of grid variables.

        These variables are used to define the grid and thus should be
        kept when subsetting the dataset
        """
        grid_topology_key = ds.cf.cf_roles["grid_topology"][0]
        grid_topology = ds[grid_topology_key]
        grid_coords = [grid_topology_key]
        for _dims, coords in _get_sgrid_dim_coord_names(grid_topology):
            grid_coords.extend(coords)
        return set(grid_coords)

    def data_vars(self, ds: xr.Dataset) -> set[str]:
        """Set of data variables.

        These variables exist on the grid and are available to used for
        data analysis. These can be discarded when subsetting the
        dataset when they are not needed.
        """
        grid_topology_key = ds.cf.cf_roles["grid_topology"][0]
        grid_topology = ds[grid_topology_key]
        dims = []
        for dims, _coords in _get_sgrid_dim_coord_names(grid_topology):
            dims.extend(dims)
        dims = set(dims)

        return {var for var in ds.data_vars if not set(ds[var].dims).isdisjoint(dims)}

    def compute_polygon_subset_selector(
        self, ds: xr.Dataset, polygon: list[tuple[float, float]], name: str = None
    ) -> Selector:

        grid_topology_key = ds.cf.cf_roles["grid_topology"][0]
        grid_topology = ds[grid_topology_key]
        dims = _get_sgrid_dim_coord_names(grid_topology)
        subset_masks: list[tuple[list[str], xr.DataArray]] = []
        
        node_dims = grid_topology.attrs["node_dimensions"].split()
        node_coords = grid_topology.attrs["node_coordinates"].split()

        node_lon: xr.DataArray | None = None
        node_lat: xr.DataArray | None = None
        for c in node_coords:
            if 'lon' in c:
                node_lon = ds[c]
            elif 'lat' in c:
                node_lat = ds[c]
        if node_lon is None or node_lat is None:
            raise ValueError(f"Could not find lon and lat for dimension {node_dims}")
        
        node_mask = compute_2d_subset_mask(lat=node_lat, lon=node_lon, polygon=polygon)
        msk = np.where(node_mask)
        index_bounding_box = np.array([[msk[0].min(), msk[0].max()+1], [msk[1].min(), msk[1].max()+1]])
        # quick and dirty add a 1 node pad around the psi mask. This is to ensure the entire polygon is covered.
        index_bounding_box[0,0] = max(0, index_bounding_box[0,0] - 1)
        index_bounding_box[0,1] = min(node_lon.shape[0], index_bounding_box[0,1] + 1)
        index_bounding_box[1,0] = max(0, index_bounding_box[1,0] - 1)
        index_bounding_box[1,1] = min(node_lon.shape[1], index_bounding_box[1,1] + 1)
        node_mask[index_bounding_box[0][0]:index_bounding_box[0][1],
                  index_bounding_box[1][0]:index_bounding_box[1][1]] = True
        
        subset_masks.append(([node_coords[0], node_coords[1]], node_mask))
        for s in ('face', 'edge1', 'edge2'):
            dims = grid_topology.attrs.get(f"{s}_dimensions", None)
            coords = grid_topology.attrs.get(f"{s}_coordinates", None).split()
            
            lon: xr.DataArray | None = None
            lat: xr.DataArray | None = None
            for c in coords:
                if 'lon' in c:
                    lon = ds[c]
                elif 'lat' in c:
                    lat = ds[c]
            if lon is None or lat is None:
                raise ValueError(f"Could not find lon and lat for dimension {dims}")
            padding = parse_padding_string(dims)
            arranged_padding = [padding[d] for d in lon.dims]
            arranged_padding = [0 if p == 'none' or p == 'low' else 1 for p in arranged_padding]
            mask = np.zeros(lon.shape, dtype=bool)
            mask[index_bounding_box[0][0]:index_bounding_box[0][1] + arranged_padding[0],
                 index_bounding_box[1][0]:index_bounding_box[1][1] + arranged_padding[1]] = True
            xr_mask = xr.DataArray(mask, dims=lon.dims)
            subset_masks.append(([coords[0], coords[1]], xr_mask))
            
        return SGridSelector(
            name=name or 'selector',
            polygon=polygon,
            grid_topology_key=grid_topology_key,
            grid_topology=grid_topology,
            subset_masks=subset_masks,
        )
        

def _get_sgrid_dim_coord_names(
    grid_topology: xr.DataArray,
) -> list[tuple[list[str], list[str]]]:
    """Get the names of the dimensions that are coordinates.

    This is really hacky and possibly not a long term solution, but it
    is our generic best start
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

def parse_padding_string(dim_string):
    '''
    Given a grid_topology dimension string, parse the padding for each dimension.
    Returns a dict of {dim0name: padding,
                       dim1name: padding
                       }
    valid values of padding are: 'none', 'low', 'high', 'both'
    '''
    parsed_string = dim_string.replace('(padding: ', '').replace(')', '').replace(':', '')
    split_parsed_string = parsed_string.split(' ')
    if len(split_parsed_string) == 6:
        return {split_parsed_string[0]:split_parsed_string[2], split_parsed_string[3]:split_parsed_string[5]}
    elif len(split_parsed_string) == 5:
        if split_parsed_string[4] in {'none', 'low', 'high', 'both'}:
            #2nd dim has padding, and with len 5 that means first does not
            split_parsed_string.insert(2, 'none')
        else:
            split_parsed_string.insert(5, 'none')
        return {split_parsed_string[0]:split_parsed_string[2], split_parsed_string[3]:split_parsed_string[5]}
    elif len(split_parsed_string) == 2:
        #node dimensions string could look like this: 'node_dimensions: xi_psi eta_psi'
        return {split_parsed_string[0]: 'none', split_parsed_string[1]: 'none'}
    else:
        raise ValueError(f"Padding parsing failure: {dim_string}")
