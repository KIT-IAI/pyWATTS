import logging
from typing import Dict, Callable, Optional

import numpy as np
import xarray as xr
from pywatts.core.exceptions.input_not_available import InputNotAvailable

from pywatts.core.base import BaseTransformer
from pywatts.utils._xarray_time_series_utils import _get_time_indeces
import pandas as pd

logger = logging.getLogger(__name__)


class RmseCalculator(BaseTransformer):
    """
    Module to calculate the Root Mean Squared Error (RMSE)

    :param offset: Offset, which determines the number of ignored values in the beginning for calculating the RMSE.
                   Default 0
    :type offset: int
    :param rolling: Flag that determines if a rolling rmse should be used. Default False
    :type rolling: bool
    :param window: Determine the window size if a rolling rmse should be calculated. Ignored if rolling is set to
                   False. Default 24
    :type window: int

    """

    def __init__(self, name: str = "RmseCalculator", filter:Callable=None, offset:int=0, rolling:bool=False, window:int=24):
        super().__init__(name)
        self.offset = offset
        self.rolling = rolling
        self.window = window
        self.filter = filter
        if not rolling:
            DeprecationWarning("If you do not need a rolling RMSE, you should use the RMSESummary module. Will be removed in 0.3")
        else:
            DeprecationWarning("If you need a rolling RMSE  you should use the RollingRMSE module. Will be removed in 0.3")

    def get_params(self) -> Dict[str, object]:
        """
        Returns a dict of parameters used in the RMSE Calculator.

        :return: Parameters set for the RMSE calculator
        :rtype: Dict[str, object]
        """
        return {"offset": self.offset,
                "rolling": self.rolling,
                "window": self.window}

    def transform(self, y: xr.DataArray, **kwargs: xr.DataArray) -> xr.DataArray:
        """
        Calculates the RMSE based on the predefined target and predictions variables.

        :param x: the input dataset
        :type x: Optional[xr.DataArray]

        :return: The calculated RMSE
        :rtype: xr.DataArray
        """
        t = y.values
        rmse = []
        predictions = []
        if kwargs == {}:
            logger.error("No predictions are provided as input for the RMSE Calculator. "
                         "You should add the predictions by a seperate key word arguments if you add the RMSECalculator "
                         "to the pipeline.")
            raise InputNotAvailable("No predictions are provided as input for the RMSE Calculator. "
                                    "You should add the predictions by a seperate key word arguments if you add the RMSECalculator "
                                    "to the pipeline.")

        for key, y_hat in kwargs.items():
            p = y_hat.values
            predictions.append(key)
            if self.rolling:
                if self.filter:
                    p_, t_ = self.filter(p, t)
                    time = y[_get_time_indeces(y)[0]][-len(p_) + self.offset:]
                else:
                    time = y[_get_time_indeces(y)[0]][self.offset:]
                    p_, t_ = p.reshape((len(p), -1)), t.reshape((len(t), -1))
                _rmse = pd.DataFrame(np.mean((p_[self.offset:] - t_[self.offset:]) ** 2, axis=-1)).rolling(
                    self.window).apply(lambda x: np.sqrt(np.mean(x))).values
            else:
                time = [y.indexes[_get_time_indeces(y)[0]][-1]]
                _rmse = [np.sqrt(np.mean((p[self.offset:] - t[self.offset:]) ** 2))]
            rmse.append(_rmse)
        return xr.DataArray(np.stack(rmse).swapaxes(0, 1).reshape((-1, len(predictions))),
                            coords={"time": time, "predictions": predictions},
                            dims=["time", "predictions"])

    def set_params(self, offset: Optional[int] = None, rolling: Optional[bool] = None, window: Optional[int] = None):
        """
        Set parameters of the RMSECalculator.

        :param offset: Offset, which determines the number of ignored values in the beginning for calculating the RMSE.
        :type offset: int
        :param rolling: Flag that determines if a rolling rmse should be used.
        :type rolling: bool
        :param window: Determine the window size if a rolling rmse should be calculated. Ignored if rolling is set to
                       False.
        :type window: int
        """
        if offset:
            self.offset = offset
        if rolling:
            self.rolling = rolling
        if window:
            self.window = window
