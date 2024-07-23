from xarray_subset_grid.grid import Grid
from xarray_subset_grid.utils import compute_2d_subset_mask


class RegularGrid2d(Grid):
    """Grid implementation for 2D regular grids"""

    @staticmethod
    def recognize(ds):
        """Recognize if the dataset matches the given grid"""
        lat = ds.cf.coordinates.get("latitude", None)
        lon = ds.cf.coordinates.get("longitude", None)
        if lat is None or lon is None:
            return False

        # Make sure the coordinates are 1D and match
        lat_dim = ds[lat[0]].dims
        ndim = ds[lat[0]].ndim
        lon_dim = ds[lon[0]].dims
        return lat_dim == lon_dim and ndim == 2

    @property
    def name(self):
        """Name of the grid type"""
        return "regular_grid_2d"

    def grid_vars(self, ds):
        """Set of grid variables

        These variables are used to define the grid and thus should be kept
        when subsetting the dataset
        """
        lat = ds.cf.coordinates["latitude"][0]
        lon = ds.cf.coordinates["longitude"][0]
        return {lat, lon}

    def data_vars(self, ds):
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

    def subset_polygon(self, ds, polygon):
        """Subset the dataset to the grid
        :param ds: The dataset to subset
        :param polygon: The polygon to subset to
        :return: The subsetted dataset
        """
        lat = ds.cf["latitude"]
        lon = ds.cf["longitude"]
        subset_mask = compute_2d_subset_mask(lat=lat, lon=lon, polygon=polygon)

        # First, we need to add the mask as a variable in the dataset
        # so that we can use it to mask and drop via xr.where, which requires that
        # the mask and data have the same shape and both are DataArrays with matching
        # dimensions
        ds_subset = ds.assign(subset_mask=subset_mask)

        # Now we can use the mask to subset the data
        ds_subset = ds_subset.where(ds_subset.subset_mask, drop=True).drop_encoding()
        return ds_subset
