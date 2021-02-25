from typing import Dict

import numpy as np
import xarray as xr

from pywatts.core.base import BaseTransformer
from pywatts.utils._xarray_time_series_utils import _get_time_indeces


class RmseCalculator(BaseTransformer):
    """
    Module to calculate the Root Mean Squared Error (RMSE)

    Creates the RmseCalculator

    :param target: Variable to be used as the target to be predicted (actual value)
    :type target: str
    :param predictions: Variable to be used as the predictions
    :type predictions: list.
    """

    def __init__(self, name: str = "RmseCalculator"):
        super().__init__(name)

    def get_params(self) -> Dict[str, object]:
        """
        Returns a list of parameters used in the RMSE Calculator

        :return: Parameters set for the RMSE calculator
        :rtype: Dict[str, object]
        """
        return {}

    def transform(self, y: xr.DataArray, **kwargs: xr.DataArray) -> xr.DataArray:
        """
        Calculates the RMSE based on the predefined target and predictions variables

        :param x: the input dataset
        :type x: Optional[xr.DataArray]

        :return: The calculated RMSE
        :rtype: xr.DataArray
        """
        t = y.values
        rmse = []
        predictions = []
        for key, y_hat in kwargs.items():
            p = y_hat.values
            predictions.append(key)
            rmse.append(np.sqrt(np.mean((p - t) ** 2)))

        time = y.indexes[_get_time_indeces(kwargs)[0]][-1]
        return xr.DataArray(np.array([rmse]), coords={"time": [time], "predictions": predictions},
                            dims=["time", "predictions"])

    def set_params(self, **kwargs):
        """
        No parameters can be set.
        """
        pass
