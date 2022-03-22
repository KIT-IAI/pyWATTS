from typing import Dict, List

import numpy as np
import pandas as pd
import xarray as xr

from pywatts.core.base import BaseTransformer
from pywatts.core.exceptions.wrong_parameter_exception import WrongParameterException
from pywatts.utils._xarray_time_series_utils import _get_time_indexes


class ClockShift(BaseTransformer):
    """
    This module shifts the data by a certain offset.

    :param lag: The offset for shifting the time series. Please note: The relative time of the shift is determined
    by the current temporal resolution of the arrays in the pipeline.
    :type lag: int
    :param name: The name of the shift module
    :type name: str
    :param indexes: The list of indexes that determine the dimension in which the time should be shifted. If the
    list is None or empty, the time is shifted in all temporal dimensions.
    :type indexes: List
    """

    def __init__(self, lag: int, name: str = "ClockShift", indexes: List[str] = None):
        super().__init__(name)
        self.lag = lag
        self.indexes = indexes

    def get_params(self) -> Dict[str, object]:
        """
        Returns a list of all defined parameters

        :return: List of parameters
        :rtype: Dict
        """
        return {
            "lag": self.lag,
            "indexes": self.indexes
        }

    def set_params(self, lag: int = None, indexes: List[str] = None):
        """
        Sets parameters for clock shifting

        :param lag: The offset for shifting the time series. Please note: The relative time of the shift is determined
        by the current temporal resolution of the arrays in the pipeline.
        :type lag: int
        :param indexes: The list of indexes that determine the dimension in which the time should be shifted. If the
        list is None or empty, the time is shifted in all temporal dimensions.
        :type indexes: List
        """
        if lag:
            self.lag = lag
        if indexes is not None:
            # Do not use if indexes here, since this would be false if indexes is empty.
            self.indexes = indexes

    def transform(self, x: xr.DataArray) -> xr.DataArray:
        """
        Shifts the given time series x by the defined lag

        :param x: the time series to be shifted
        :type x: xr.DataArray
        :return: The shifted time series
        :rtype: xr.DataArray
        :raises WrongParameterException: If not all indexes are part of x
        """
        indexes = self.indexes
        if not indexes:
            indexes = _get_time_indexes(x)
        try:
            return x.shift({index: self.lag for index in indexes}, fill_value=0)
        except ValueError as exc:
            raise WrongParameterException(
                f"Not all indexes ({indexes}) are in the indexes of x ({list(x.indexes.keys())}).",
                "Perhaps you set the wrong indexes with set_params or during the initialization of the ClockShift.",
                module=self.name) from exc

    def get_min_data(self):
        return np.abs(self.lag)
