import logging
from typing import Dict, Optional

import xarray as xr
import numpy as np
import pandas as pd

from pywatts.core.base import BaseTransformer
from pywatts.core.exceptions.input_not_available import InputNotAvailable
from pywatts.utils._xarray_time_series_utils import _get_time_indeces

logger = logging.getLogger(__name__)


class MaxCalculator(BaseTransformer):
    """
        Module to calculate the max Error

        :param offset: Offset, which determines the number of ignored values in the beginning for calculating the max
                       error, Default 0
        :type offset: int
        :param rolling: Flag that determines if a rolling max should be used. Default False
        :type rolling: bool
        :param window: Determine the window size if a rolling max should be calculated. Ignored if rolling is set to
                       False. Default 24
        :type window: int

        """

    def __init__(self, name: str = "MaxCalculator", offset: int = 0, rolling: bool = False, window: int = 24):
        super().__init__(name)
        self.offset = offset
        self.rolling = rolling
        self.window = window

    def get_params(self) -> Dict[str, object]:
        """
        Returns a dict of parameters used in the max calculator.

        :return: Parameters set for the max calculator
        :rtype: Dict[str, object]
        """
        return {"offset": self.offset,
                "rolling": self.rolling,
                "window": self.window}

    def transform(self, y: xr.DataArray, **kwargs: xr.DataArray) -> xr.DataArray:
        """
        Calculates the max based on the predefined target and predictions variables.

        :param y: the input dataset
        :type y: Optional[xr.DataArray]

        :return: The calculated max
        :rtype: xr.DataArray
        """
        t = y.values
        max_err = []
        predictions = []
        if kwargs == {}:
            error_msg = ("No predictions are provided as input for the MaxCalculator. " +
                         "You should add the predictions by a seperate key word arguments if you add the " +
                         "MaxCalculator to the pipeline.")
            logger.error(error_msg)
            raise InputNotAvailable(error_msg)

        for key, y_hat in kwargs.items():
            p = y_hat.values
            predictions.append(key)

            if self.rolling:
                time = y[_get_time_indeces(y)[0]][self.offset:]
                p_, t_ = p.reshape((len(p), -1)), t.reshape((len(t), -1))
                _max_err = pd.DataFrame(p_[self.offset:] - t_[self.offset:]).rolling(
                    self.window).apply(lambda x: np.max(np.abs(x))).values
            else:
                time = [y.indexes[_get_time_indeces(y)[0]][-1]]
                _max_err = [np.max(np.abs(p[self.offset:] - t[self.offset:]))]
            max_err.append(_max_err)
        return xr.DataArray(np.stack(max_err).swapaxes(0, 1).reshape((-1, len(predictions))),
                            coords={"time": time, "predictions": predictions},
                            dims=["time", "predictions"])

    def set_params(self, offset: Optional[int] = None, rolling: Optional[bool] = None, window: Optional[int] = None):
        """
        Set parameters of the max calculator.

        :param offset: Offset, which determines the number of ignored values in the beginning for calculating the max.
        :type offset: int
        :param rolling: Flag that determines if a rolling max should be used.
        :type rolling: bool
        :param window: Determine the window size if a rolling max should be calculated. Ignored if rolling is set to
                       False.
        :type window: int
        """
        if offset:
            self.offset = offset
        if rolling:
            self.rolling = rolling
        if window:
            self.window = window
