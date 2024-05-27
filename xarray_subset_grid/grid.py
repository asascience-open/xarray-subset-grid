from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Union

import numpy as np
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
    def grid_vars(self, ds: xr.Dataset) -> list[str]:
        """List of grid variables

        These variables are used to define the grid and thus should be kept
        when subsetting the dataset
        """
        return []

    @abstractmethod
    def data_vars(self, ds: xr.Dataset) -> list[str]:
        """List of data variables

        These variables exist on the grid and are availabel to used for
        data analysis. These can be discarded when subsetting the dataset
        when they are not needed.
        """
        return []

    def subset_vars(self, ds: xr.Dataset, vars: Iterable[str]) -> list[str]:
        """Subset the dataset to the given variables, keeping the grid variables as well"""
        subset = list(set(self.grid_vars(ds) + list(vars)))
        return ds[subset]

    @abstractmethod
    def subset_polygon(
        self, ds: xr.Dataset, polygon: Union[list[tuple[float, float]], np.ndarray]
    ) -> xr.Dataset:
        """Subset the dataset to the grid
        :param ds: The dataset to subset
        :param polygon: The polygon to subset to
        :return: The subsetted dataset
        """
        return ds

    def subset_bbox(
        self, ds: xr.Dataset, bbox: tuple[float, float, float, float]
    ) -> xr.Dataset:
        """Subset the dataset to the bounding box
        :param ds: The dataset to subset
        :param bbox: The bounding box to subset to
        :return: The subsetted dataset
        """
        polygon = np.array(
            [
                [bbox[0], bbox[1]],
                [bbox[2], bbox[1]],
                [bbox[2], bbox[3]],
                [bbox[0], bbox[3]],
            ]
        )
        return self.subset_polygon(ds, polygon)
