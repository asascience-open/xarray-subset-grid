import hashlib
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

    def __init__(self, bytes=None):
        """Initialize the Selector instance.
        If a bytes object is provided, attempt to load the selector.
        """
        if bytes:
            instance = self.load_from_bytes(bytes)
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

        For example, a selector could hold predefined masks to apply to
        the dataset, and the select method here would apply those masks
        to the dataset and return the result. This workflow is useful
        because computing the masks can be expensive, and we want to
        avoid recomputing them for every dataset that needs to be
        subsetted. It also allows datasets that are non standard to be
        subset using information from manually or otherwise standardized
        datasets..
        """
        raise NotImplementedError()

    def get_hashname(self):
        hash = hashlib.md5(str(self.polygon).encode()).hexdigest()
        hashname = f"{self.name}_{hash[:8]}.pkl"
        return hashname

    def save_to_bytes(self):
        """Return a bytes object representing the serialized selector."""
        return pickle.dumps(self)

    def load_from_bytes(self, bytes):
        """Loads a selector from a bytes object."""
        object = pickle.loads(bytes)
        if isinstance(object, Selector):
            return object
        else:
            raise TypeError("The provided file does not contain a valid Selector.")
