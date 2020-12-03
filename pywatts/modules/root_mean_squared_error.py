from typing import Optional, Dict

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

    def __init__(self, name: str = "RmseCalculator", target: str = "target", predictions: list = ["predictions"]):
        super().__init__(name)
        self.target = target
        self.predictions = predictions

    def get_params(self) -> Dict[str, object]:
        """
        Returns a list of parameters used in the RMSE Calculator

        :return: Parameters set for the RMSE calculator
        :rtype: Dict[str, object]
        """
        return {"target": self.target,
                "predictions": self.predictions}

    def set_params(self, target: str = None, prediction: list = None):
        """
        Sets the parameters for the linear interpolation

        :param target: Variable to be used as the target (actual value)
        :type target: str

        :param prediction: Variable to be used as the predictions
        :type prediction: list
        """
        if target is not None:
            self.target = target
        if prediction is not None:
            self.predictions = prediction

    def transform(self, y_hat: xr.DataArray, y: xr.DataArray) -> xr.DataArray:
        """
        Calculates the RMSE based on the predefined target and predictions variables

        :param x: the input dataset
        :type x: Optional[xr.Dataset]

        :return: The calculated RMSE
        :rtype: xr.Dataset[str, xr.DataArray]
        # TODO how to handle multiple predictions? -> kwargs argument: y has to be given, and at least one other..
        """
        t = y.values
        rmse = list()
        p = y_hat.values
        rmse.append(np.sqrt(np.mean((p - t) ** 2)))

        dimension = self.predictions
        time = y_hat.indexes[_get_time_indeces(y_hat)[0]][-1]
        return xr.DataArray(np.array([rmse]), coords=[[time], dimension], dims={"time": [time], "Result": dimension, })
