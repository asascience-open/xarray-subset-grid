from abc import abstractmethod

import xarray as xr


class Selector:
    """Selector class to subset datasets

    This is a base method that should be implemented by a subclass
    to perform selection on a given xarray dataset with whatever
    context or logic is desired by the implementation.

    select should return a new xarray dataset that is a subset of the input dataset
    and must be implemented by the subclass.
    """

    @abstractmethod
    def select(self, ds: xr.Dataset) -> xr.Dataset:
        """Perform the selection on the dataset

        For example, a selector could hold predifined masks to apply to the dataset,
        and the select method here would apply those masks to the dataset and return the result.
        This workflow is useful because computing the masks can be expensive, and
        we want to avoid recomputing them for every dataset that needs to be subsetted. It also
        allows datasets that are non standard to be subset using information from manually or
        otherwise standardized datasets..

        """
        raise NotImplementedError()
