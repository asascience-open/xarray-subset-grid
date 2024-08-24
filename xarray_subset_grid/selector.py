import hashlib
import os
import pickle
from abc import abstractmethod

import numpy.testing as npt
import xarray as xr


class Selector:
    """Selector class to subset datasets.

    This is a base method that should be implemented by a subclass to
    perform selection on a given xarray dataset with whatever context or
    logic is desired by the implementation.

    select should return a new xarray dataset that is a subset of the
    input dataset and must be implemented by the subclass.
    """
    def __init__(self, path=None):
        """Initialize the Selector instance.
        If a filename is provided, attempt to load the selector from that file.
        """
        if path:
            instance = self.load(path)
            self.__dict__.update(instance.__dict__)
            self.__class__ = instance.__class__

    __hash__ = None

    def __eq__(self, other):
        if not isinstance(other, Selector):
            return NotImplemented
        return npt.assert_equal(self.__dict__, other.__dict__) is None

    def __repr__(self):
        return f"{self.__class__} - {self.name}"

    @abstractmethod
    def select(self, ds: xr.Dataset) -> xr.Dataset:
        """Perform the selection on the dataset.

        For example, a selector could hold predifined masks to apply to
        the dataset, and the select method here would apply those masks
        to the dataset and return the result. This workflow is useful
        because computing the masks can be expensive, and we want to
        avoid recomputing them for every dataset that needs to be
        subsetted. It also allows datasets that are non standard to be
        subset using information from manually or otherwise standardized
        datasets..
        """
        raise NotImplementedError()

    def get_cache_filename(self, polygon=None):
        if not polygon:
            polygon = self.polygon
        hashname = hashlib.md5(str(polygon).encode()).hexdigest()
        filename = f"{self.name}_{hashname[:8]}.pkl"
        return filename

    def save(self):
        """Save the selector to the cache file."""
        filename = self.get_cache_filename()
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        return filename

    def load(self, path):
        if os.path.exists(path) and os.path.isfile(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        else:
          raise FileNotFoundError(f"The file '{path}' does not exist.")
