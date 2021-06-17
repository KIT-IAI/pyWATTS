import logging
from typing import Dict, Optional

import xarray as xr
import numpy as np
import pandas as pd

from pywatts.core.base import BaseTransformer
from pywatts.core.exceptions.input_not_available import InputNotAvailable
from pywatts.utils._xarray_time_series_utils import _get_time_indeces

logger = logging.getLogger(__name__)


class MapeCalculator(BaseTransformer):
    """
        Module to calculate the Mean Absolute Percentage Error (MAPE)

        :param offset: Offset, which determines the number of ignored values in the beginning for calculating the MAPE.
                       Default 0
        :type offset: int
        :param rolling: Flag that determines if a rolling mae should be used. Default False
        :type rolling: bool
        :param window: Determine the window size if a rolling mae should be calculated. Ignored if rolling is set to
                       False. Default 24
        :type window: int

        """

    def __init__(self, name: str = "MapeCalculator", offset: int = 0, rolling: bool = False, window: int = 24):
        super().__init__(name)
        self.offset = offset
        self.rolling = rolling
        self.window = window

    def get_params(self) -> Dict[str, object]:
        """
        Returns a dict of parameters used in the MAPE Calculator.

        :return: Parameters set for the MAPE calculator
        :rtype: Dict[str, object]
        """
        return {"offset": self.offset,
                "rolling": self.rolling,
                "window": self.window}

    def transform(self, y: xr.DataArray, **kwargs: xr.DataArray) -> xr.DataArray:
        """
        Calculates the MAPE based on the predefined target and predictions variables.

        :param y: the input dataset
        :type y: Optional[xr.DataArray]

        :return: The calculated MAPE
        :rtype: xr.DataArray
        """
        usable_size = y.values.nonzero()[0][y.values.nonzero()[0] >= self.offset].size
        t = np.where(y.values != 0, y.values, np.nan)   # we want to divide by t => replace 0s by nan
        mape = []
        predictions = []
        if kwargs == {}:
            error_msg = ("No predictions are provided as input for the MAPE Calculator. " +
                         "You should add the predictions by a seperate key word arguments if you add the " +
                         "MAPE Calculator to the pipeline.")
            logger.error(error_msg)
            raise InputNotAvailable(error_msg)
        if usable_size != t.size:
            percent = (1 - usable_size / (t.size - self.offset)) * 100
            error_msg = "MAPE ignores y zero values ({:.3f}% of given data)".format(percent)
            logger.info(error_msg)

        for key, y_hat in kwargs.items():
            p = y_hat.values
            predictions.append(key)

            if self.rolling:
                time = y[_get_time_indeces(y)[0]][self.offset:]
                p_, t_ = p.reshape((len(p), -1)), t.reshape((len(t), -1))
                _mape = pd.DataFrame((p_[self.offset:] - t_[self.offset:])/t_[self.offset:]).rolling(
                    self.window).apply(lambda x: np.nanmean(np.abs(x))).values
            else:
                time = [y.indexes[_get_time_indeces(y)[0]][-1]]
                _mape = [np.nanmean(np.abs((p[self.offset:] - t[self.offset:])/t[self.offset:]))]
            mape.append(_mape)
        return xr.DataArray(np.stack(mape).swapaxes(0, 1).reshape((-1, len(predictions))),
                            coords={"time": time, "predictions": predictions},
                            dims=["time", "predictions"])

    def set_params(self, offset: Optional[int] = None, rolling: Optional[bool] = None, window: Optional[int] = None):
        """
        Set parameters of the MAPE calculator.

        :param offset: Offset, which determines the number of ignored values in the beginning for calculating the MAPE.
        :type offset: int
        :param rolling: Flag that determines if a rolling MAPE should be used.
        :type rolling: bool
        :param window: Determine the window size if a rolling MAPE should be calculated. Ignored if rolling is set to
                       False.
        :type window: int
        """
        if offset:
            self.offset = offset
        if rolling:
            self.rolling = rolling
        if window:
            self.window = window
