from typing import Dict, List

import xarray as xr

from pywatts.core.base import BaseTransformer
from pywatts.core.exceptions.wrong_parameter_exception import WrongParameterException
from pywatts.utils._xarray_time_series_utils import _get_time_indeces


class ClockShift(BaseTransformer):
    """
    This module shifts the data by a certain offset.

    :param lag: The offset for shifting the time series. Please note: The relative time of the shift is determined
    by the current temporal resolution of the arrays in the pipeline.
    :type lag: int
    :param name: The name of the shift module
    :type name: str
    :param indices: The list of indices that determine the dimension in which the time should be shifted. If the
    list is None or empty, the time is shifted in all temporal dimensions.
    :type indices: List
    """

    def __init__(self, lag: int, name: str = "ClockShift", indices: List[str] = None):
        super().__init__(name)
        self.lag = lag
        self.indices = indices

    def get_params(self) -> Dict[str, object]:
        """
        Returns a list of all defined parameters

        :return: List of parameters
        :rtype: Dict
        """
        return {
            "lag": self.lag,
            "indices": self.indices
        }

    def set_params(self, lag: int = None, indices: List[str] = None):
        """
        Sets parameters for clock shifting

        :param lag: The offset for shifting the time series. Please note: The relative time of the shift is determined
        by the current temporal resolution of the arrays in the pipeline.
        :type lag: int
        :param indices: The list of indices that determine the dimension in which the time should be shifted. If the
        list is None or empty, the time is shifted in all temporal dimensions.
        :type indices: List
        """
        if lag:
            self.lag = lag
        if indices is not None:
            # Do not use if indices here, since this would be false if indices is empty.
            self.indices = indices

    def transform(self, x: xr.DataArray) -> xr.DataArray:
        """
        Shifts the given time series x by the defined lag

        :param x: the time series to be shifted
        :type x: xr.DataArray
        :return: The shifted time series
        :rtype: xr.DataArray
        :raises WrongParameterException: If not all indices are part of x
        """
        indices = self.indices
        if not indices:
            indices = _get_time_indeces(x)
        try:
            return x.shift({index: self.lag for index in indices}, fill_value=0)
        except ValueError:
            raise WrongParameterException(
                f"Not all indices ({indices}) are in the indices of x ({list(x.indexes.keys())}).",
                "Perhaps you set the wrong indices with set_params or during the initialization of the ClockShift.",
                module=self.name)
