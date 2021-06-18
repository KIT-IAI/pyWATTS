import logging
from typing import Dict, Optional

import xarray as xr
import numpy as np
import pandas as pd

from pywatts.core.base import BaseTransformer
from pywatts.core.exceptions.input_not_available import InputNotAvailable
from pywatts.utils._xarray_time_series_utils import _get_time_indeces

logger = logging.getLogger(__name__)


class MinCalculator(BaseTransformer):
    """
        Module to calculate the min Error

        :param offset: Offset, which determines the number of ignored values in the beginning for calculating the min
                       error, Default 0
        :type offset: int
        :param rolling: Flag that determines if a rolling min should be used. Default False
        :type rolling: bool
        :param window: Determine the window size if a rolling min should be calculated. Ignored if rolling is set to
                       False. Default 24
        :type window: int

        """

    def __init__(self, name: str = "MinCalculator", offset: int = 0, rolling: bool = False, window: int = 24):
        super().__init__(name)
        self.offset = offset
        self.rolling = rolling
        self.window = window

    def get_params(self) -> Dict[str, object]:
        """
        Returns a dict of parameters used in the min calculator.

        :return: Parameters set for the min calculator
        :rtype: Dict[str, object]
        """
        return {"offset": self.offset,
                "rolling": self.rolling,
                "window": self.window}

    def transform(self, y: xr.DataArray, **kwargs: xr.DataArray) -> xr.DataArray:
        """
        Calculates the min based on the predefined target and prediction variables.

        :param y: the input dataset
        :type y: Optional[xr.DataArray]

        :return: The calculated min
        :rtype: xr.DataArray
        """
        t = y.values
        min_err = []
        predictions = []
        if kwargs == {}:
            error_msg = ("No predictions are provided as input for the MinCalculator. " +
                         "You should add the predictions by a seperate key word arguments if you add the " +
                         "MinCalculator to the pipeline.")
            logger.error(error_msg)
            raise InputNotAvailable(error_msg)

        for key, y_hat in kwargs.items():
            p = y_hat.values
            predictions.append(key)

            if self.rolling:
                time = y[_get_time_indeces(y)[0]][self.offset:]
                p_, t_ = p.reshape((len(p), -1)), t.reshape((len(t), -1))
                _min_err = pd.DataFrame(p_[self.offset:] - t_[self.offset:]).rolling(
                    self.window).apply(lambda x: np.min(np.abs(x))).values
            else:
                time = [y.indexes[_get_time_indeces(y)[0]][-1]]
                _min_err = [np.min(np.abs(p[self.offset:] - t[self.offset:]))]
            min_err.append(_min_err)
        return xr.DataArray(np.stack(min_err).swapaxes(0, 1).reshape((-1, len(predictions))),
                            coords={"time": time, "predictions": predictions},
                            dims=["time", "predictions"])

    def set_params(self, offset: Optional[int] = None, rolling: Optional[bool] = None, window: Optional[int] = None):
        """
        Set parameters of the min calculator.

        :param offset: Offset, which determines the number of ignored values in the beginning for calculating the min.
        :type offset: int
        :param rolling: Flag that determines if a rolling min should be used.
        :type rolling: bool
        :param window: Determine the window size if a rolling min should be calculated. Ignored if rolling is set to
                       False.
        :type window: int
        """
        if offset:
            self.offset = offset
        if rolling:
            self.rolling = rolling
        if window:
            self.window = window
