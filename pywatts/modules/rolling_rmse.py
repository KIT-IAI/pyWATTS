from typing import Dict, Optional

import numpy as np
import pandas as pd
import xarray as xr

from pywatts.core.base import BaseTransformer
from pywatts.core.exceptions.input_not_available import InputNotAvailable
from pywatts.utils._xarray_time_series_utils import _get_time_indeces


class RollingRMSE(BaseTransformer):
    """
    Module to calculate the Rolling Root Mean Squared Error (RMSE)
    :param window_size: Determine the window size if a rolling rmse should be calculated. Ignored if rolling is set to
                   False. Default 24
    :type window_size: int
    :param window_size_unit: Determine the unit of the window size. Default Day (d)"
    :type window_size_unit: str

    """

    def __init__(self, name: str = "RollingRMSE", window_size=24, window_size_unit="d"):
        super().__init__(name)
        self.window_size_unit = window_size_unit
        self.window_size = window_size

    def get_params(self) -> Dict[str, object]:
        """
        Returns a dict of parameters used in the RollingRMSE.

        :return: Parameters set for the RollingRMSE
        :rtype: Dict[str, object]
        """
        return {
            "window_size_unit": self.window_size_unit,
            "window_size": self.window_size
        }

    def set_params(self, window_size=None, window_size_unit=None):
        """
        Set the parameter for the RollingRMSE.

        :param window_size: Determine the window size if a rolling rmse should be calculated. Ignored if rolling is set to
                   False. Default 24
        :type window_size: int
        :param window_size_unit: Determine the unit of the window size. Default Day (d)"
        :type window_size_unit: str

        """
        if window_size is not None:
            self.window_size = window_size
        if window_size_unit is not None:
            self.window_size_unit = window_size_unit

    def transform(self, y: xr.DataArray, **kwargs: xr.DataArray) -> xr.DataArray:
        """
        Calculates the RMSE based on the predefined target and predictions variables.

        :param x: the input dataset
        :type x: Optional[xr.DataArray]

        :return: The calculated RMSE
        :rtype: xr.DataArray
        """

        if kwargs == {}:
            error_message = "No predictions are provided as input for the RollingRMSE. You should add the predictions" \
                            " by a seperate key word arguments if you add the RollingRMSE to the pipeline."
            self.logger.error(error_message)
            raise InputNotAvailable(error_message)

        t = y.values
        results = {}
        for key, y_hat in kwargs.items():
            p = y_hat.values
            p_, t_ = p.reshape((len(p), -1)), t.reshape((len(t), -1))
            results[key] = pd.DataFrame(np.mean((p_ - t_) ** 2, axis=-1),
                                        index=y.indexes[_get_time_indeces(kwargs)[0]]).rolling(
                f"{self.window_size}{self.window_size_unit}").apply(
                lambda x: np.sqrt(np.mean(x))).values
        time = y.indexes[_get_time_indeces(y)[0]]

        return xr.DataArray(np.concatenate(list(results.values()), axis=1),
                            coords={"time": time, "predictions": list(results.keys())},
                            dims=["time", "predictions"])
