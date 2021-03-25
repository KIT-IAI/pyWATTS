import logging
from typing import Dict

import numpy as np
import xarray as xr
from pywatts.core.exceptions.input_not_available import InputNotAvailable

from pywatts.core.base import BaseTransformer
from pywatts.utils._xarray_time_series_utils import _get_time_indeces

logger = logging.getLogger(__name__)


class RmseCalculator(BaseTransformer):
    """
    Module to calculate the Root Mean Squared Error (RMSE)

    :param offset: Offset, which determines the number of ignored values in the beginning for calculating the RMSE.
    :type offset: int
    """

    def __init__(self, name: str = "RmseCalculator", offset=0):
        super().__init__(name)
        self.offset = offset

    def get_params(self) -> Dict[str, object]:
        """
        Returns a dict of parameters used in the RMSE Calculator.

        :return: Parameters set for the RMSE calculator
        :rtype: Dict[str, object]
        """
        return {"offset": self.offset}

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
            rmse.append(np.sqrt(np.mean((p[self.offset:] - t[self.offset:]) ** 2)))

        time = y.indexes[_get_time_indeces(y)[0]][-1]
        return xr.DataArray(np.array([rmse]), coords={"time": [time], "predictions": predictions},
                            dims=["time", "predictions"])

    def set_params(self, offset=None):
        """
        Set parameters of the RMSECalculator.

        :param offset: Offset, which determines the number of ignored values in the beginning for calculating the RMSE.
        :type offset: int
        """
        if offset:
            self.offset = offset
