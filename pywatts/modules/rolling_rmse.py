import logging
from typing import Dict, Callable, Optional

import numpy as np
import xarray as xr
from pywatts.core.exceptions.input_not_available import InputNotAvailable

from pywatts.core.base import BaseTransformer
from pywatts.modules.rolling_base import RollingBase
from pywatts.utils._xarray_time_series_utils import _get_time_indeces
import pandas as pd

logger = logging.getLogger(__name__)


class RollingRMSE(BaseTransformer):
    """
    Module to calculate the Root Mean Squared Error (RMSE)

    TODO

    """

    def __init__(self, name: str="RollingRMSE", window_size=24, window_size_unit="d"):
        super().__init__(name)
        self.window_size_unit = window_size_unit
        self.window_size = window_size

    def get_params(self) -> Dict[str, object]:
        pass

    def set_params(self, *args, **kwargs):
        pass

    def transform(self, y: xr.DataArray, **kwargs: xr.DataArray) -> xr.DataArray:
        """
        Calculates the RMSE based on the predefined target and predictions variables.

        :param x: the input dataset
        :type x: Optional[xr.DataArray]

        :return: The calculated RMSE
        :rtype: xr.DataArray
        """

        if kwargs == {}:
            logger.error("No predictions are provided as input for the RMSE Calculator. "
                         "You should add the predictions by a seperate key word arguments if you add the RMSECalculator "
                         "to the pipeline.")
            raise InputNotAvailable("No predictions are provided as input for the RMSE Calculator. "
                                    "You should add the predictions by a seperate key word arguments if you add the RMSECalculator "
                                    "to the pipeline.")

        t = y.values
        results = {}
        for key, y_hat in kwargs.items():
            p = y_hat.values
            p_, t_ = p.reshape((len(p), -1)), t.reshape((len(t), -1))
            results[key] = pd.DataFrame(np.mean((p_ - t_) ** 2, axis=-1), index=y.indexes[_get_time_indeces(kwargs)[0]]).rolling(
                f"{self.window_size}{self.window_size_unit}").apply(
                lambda x: np.sqrt(np.mean(x))).values
        time = y.indexes[_get_time_indeces(y)[0]]

        return xr.DataArray(np.concatenate(list(results.values()), axis=1), coords={"time": time, "predictions": list(results.keys())},
                            dims=["time", "predictions"])