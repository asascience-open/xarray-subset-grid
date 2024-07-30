from abc import abstractmethod

import numpy as np
import xarray as xr


class Selector:
    """A class to select a subset of a grid.

    This is a base class that should be inherited by other classes that
    implement specific selection methods. For example, the UGRID class
    implements a selection method based on the UGRID conventions, and either
    calls it to subset on demand or returns it to the user to save for later
    processing.

    This hides the implementation details of the selection method from the
    user, and allows the user to reuse selection logic across multiple
    datasets, even if the datasets are not the same or are missing metadata,
    as long as they have the same data and coordinate variables. This also
    allows the user cache the selection method and reuse it across multiple
    processes without having to recompute the selection method each time.
    """

    @abstractmethod
    def subset_polygon(
        self, ds: xr.Dataset, polygon: list[tuple[float, float]] | np.ndarray
    ) -> xr.Dataset:
        """Subset the grid based on a polygon.

        Parameters
        ----------
        ds : xr.Dataset
            The dataset to subset. This should be a dataset that follows the same
            conventions as the dataset that the selector was initialized with. If not
            the same dataset, the method will raise an error.
        polygon : list[tuple[float, float]]
            A list of tuples of the form [(lon1, lat1), (lon2, lat2), ...] that
            defines the polygon to subset the grid.

        Returns
        -------
        xr.Dataset
            The subsetted grid.
        """
        raise NotImplementedError

    def subset_bbox(self, ds: xr.Dataset, bbox: tuple[float, float, float, float]) -> xr.Dataset:
        """Subset the grid based on a bounding box.

        Parameters
        ----------
        ds : xr.Dataset
            The dataset to subset. This should be a dataset that follows the same
            conventions as the dataset that the selector was initialized with. If not
            the same dataset, the method will raise an error.
        bbox : tuple[float, float, float, float]
            A tuple of the form (min_lon, min_lat, max_lon, max_lat) that
            defines the bounding box to subset the grid.

        Returns
        -------
        xr.Dataset
            The subsetted grid.
        """
        polygon = np.array(
            [
                [bbox[0], bbox[3]],
                [bbox[0], bbox[1]],
                [bbox[2], bbox[1]],
                [bbox[2], bbox[3]],
                [bbox[0], bbox[3]],
            ]
        )
        return self.subset_polygon(ds, polygon)
