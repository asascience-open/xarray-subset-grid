from abc import ABC, abstractmethod

import numpy as np
import xarray as xr
from numpy import ndarray


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
    def subset_polygon(
        self, ds: xr.Dataset, polygon: list[tuple[float, float]] | ndarray
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
