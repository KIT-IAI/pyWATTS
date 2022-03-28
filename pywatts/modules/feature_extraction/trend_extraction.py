"""
This module contains the trend extraction
"""

from typing import Dict, List

import xarray as xr

from pywatts.core.base import BaseTransformer
from pywatts.utils._xarray_time_series_utils import _get_time_indexes
import numpy as np


class TrendExtraction(BaseTransformer):
    """
    Module to extract a trend which can be specified through a period and a length,
    where the length indicates the length of the time step going back (ie 7, for a weekly trend with daily data)
    and period indicates the number of times this length is used (ie 10 to extract the last 10 weeks).

    :param period: Length of one period
    :type period: int
    :param length: Number of periods which should be extracted
    :type length: int
    :param indexes: Index over which the trend is extracted (default: all time based indexes)
    :type indexes: List[str]
    :param name: Name of the module
    :type name: str
    """

    def __init__(self, period, length, indexes: List[str] = None, name: str = "trend_extractor"):

        super().__init__(name)
        self.length = length
        self.period = period
        self.indexes = indexes
        if indexes is None:
            self.indexes = []

    def get_params(self) -> Dict[str, object]:
        """
        Get all parameters of the trend extraction

        :return: Dict with params
        :rtype: Dict[str, object]
        """
        return {
            "period": self.period,
            "length": self.length,
            "indexes": self.indexes,
        }

    def set_params(self, period=None, length=None, indexes: List[str] = None):
        """
        Set the parameters

        :param period: Length of one period
        :type period: int
        :param length: Number of periods which should be extracted
        :type length: int
        :param indexes: Time index
        :type indexes: List[str]
        """
        if period:
            self.period = period
        if length:
            self.length = length
        if indexes is not None:
            self.indexes = indexes

    def transform(self, x: xr.DataArray) -> xr.DataArray:
        """
        Extract trend values

        :param x: input xarray DataArray
        :type x: xr.DataArray
        :return: a dataset containing the trend information
        :rtype: xr.DataArray
        """
        indexes = self.indexes
        if not indexes:
            indexes = _get_time_indexes(x)
        trends = [x.shift({index: self.period * i for index in indexes}, fill_value=0) for i in
                  range(1, self.length + 1)]
        trend = xr.DataArray(np.stack(trends, axis=-1), dims=(*x.dims, "length"), coords=x.coords)
        return trend.transpose(_get_time_indexes(x)[0], "length", ...)

    def get_min_data(self):
        return self.period * (self.length + 1)
