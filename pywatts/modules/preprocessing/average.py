from typing import Dict

import xarray as xr
import numpy as np

from pywatts.core.base import BaseTransformer
from pywatts.utils._xarray_time_series_utils import numpy_to_xarray


class Average(BaseTransformer):
    """
    Aggregation step to average the given time series, ether by simple or weighted averaging.
    By default simple averaging is applied.
    """

    def __init__(self, weights: list = None, name: str = "Average"):
        """ Initialize the averaging step.
        :param weights: List of individual weights of each given time series for weighted averaging.
        :type weights: list, optional
        """
        super().__init__(name)

        self.weights = weights

    def get_params(self) -> Dict[str, object]:
        """ Get parameters for the Average object.
        :return: Parameters as dict object.
        :rtype: Dict[str, object]
        """
        return {
            "weights": self.weights
        }

    def set_params(self, weights: list = None):
        """ Set or change Average object parameters.
        :param weights: List of individual weights of each given time series for weighted averaging.
        :type weights: list, optional
        """
        if weights:
            self.weights = weights

    def transform(self, **kwargs) -> xr.DataArray:
        """ Aggregate the given time series by simple or weighted averaging.
        :return: Xarray dataset aggregated by simple or weighted averaging.
        :rtype: xr.DataArray
        """

        list_of_series = []
        list_of_indexes = []
        for series in kwargs.values():
            list_of_indexes.append(series.indexes)
            list_of_series.append(series.data)

        if not all(all(index) == all(list_of_indexes[0]) for index in list_of_indexes):
            raise ValueError("The indexes of the given time series for averaging do not match")

        result = np.average(list_of_series, axis=0, weights=self.weights)

        return numpy_to_xarray(result, series, self.name)
