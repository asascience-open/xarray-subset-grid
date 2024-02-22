from abc import ABC, abstractmethod
import xarray as xr


class Grid(ABC):
    """Abstract class for grid types"""

    @staticmethod
    @abstractmethod
    def recognize(ds: xr.Dataset) -> bool:
        """Recognize if the dataset matches the given grid"""
        return False

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the grid type"""
        return "grid"

    @abstractmethod
    def subset(self, ds: xr.Dataset, **kwargs) -> xr.Dataset:
        """Subset the dataset to the grid
        :param ds: The dataset to subset
        :param kwargs: Additional arguments for the subset operation
        :return: The subsetted dataset
        """
        return ds
