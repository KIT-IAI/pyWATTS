from typing import Dict, Optional

import xarray as xr
from pywatts.core.base import BaseTransformer


class Slicer(BaseTransformer):
    """
    This module slices the input data array starting from the
    start index up to the end index. Similar to the numpy array
    slicing, where we can filter an array with a[start:end].


    :param start: Start index of the slicing operation, defaults to None
    :type start: int, optional
    :param end: End index of the slicing operation, defaults to None
    :type end: int, optional
    """

    def __init__(self, start: Optional[int] = None, end: Optional[int] = None,
                 name: str = "Slicer"):
        super().__init__(name)

        self.start = start
        self.end = end

    def get_params(self) -> Dict[str, object]:
        """
        Return dict of module parameters.

        :return: Dict of module parameters.
        :rtype: Dict[str, object]
        """
        return {
            "start": self.start,
            "end": self.end,
        }

    def set_params(self, start: Optional[int] = None, end: Optional[int] = None):
        """
        Set module parameters.

        :param start: Start index of the slicing operation, defaults to None
        :type start: int, optional
        :param end: End index of the slicing operation, defaults to None
        :type end: int, optional
        """
        self.start = start
        self.end = end

    def transform(self, x: xr.DataArray) -> xr.DataArray:
        """
        Perform the slicing operation on the input array.

        :param x: Input array which should be sliced.
        :type x: xr.DataArray
        :return: Sliced array like in numpy a[start:end].
        :rtype: xr.DataArray
        """
        return x[self.start: self.end]
