import logging
from typing import Dict, Callable, Optional

import numpy as np
import xarray as xr

from pywatts.core.base_summary import BaseSummary
from pywatts.core.exceptions.input_not_available import InputNotAvailable
from pywatts.core.filemanager import FileManager

logger = logging.getLogger(__name__)


class RMSE(BaseSummary):
    """
    Module to calculate the Root Mean Squared Error (RMSE)

    :param offset: Offset, which determines the number of ignored values in the beginning for calculating the RMSE.
                   Default 0
    :type offset: int
    """

    def __init__(self, name: str = "RmseCalculator", filter:Callable=None, offset:int=0):
        super().__init__(name)
        self.offset = offset
        self.filter = filter

    def get_params(self) -> Dict[str, object]:
        """
        Returns a dict of parameters used in the RMSE Calculator.

        :return: Parameters set for the RMSE calculator
        :rtype: Dict[str, object]
        """
        return {"offset": self.offset}

    def transform(self, file_manager:FileManager, y: xr.DataArray, **kwargs: xr.DataArray) -> str:
        """
        Calculates the RMSE based on the predefined target and predictions variables.

        :param x: the input dataset
        :type x: Optional[xr.DataArray]

        :return: The calculated RMSE
        :rtype: xr.DataArray
        """
        t = y.values
        summary = ""
        if kwargs == {}:
            logger.error("No predictions are provided as input for the RMSE Calculator. "
                         "You should add the predictions by a seperate key word arguments if you add the RMSECalculator "
                         "to the pipeline.")
            raise InputNotAvailable("No predictions are provided as input for the RMSE Calculator. "
                                    "You should add the predictions by a seperate key word arguments if you add the RMSECalculator "
                                    "to the pipeline.")

        for key, y_hat in kwargs.items():
            p = y_hat.values
            if self.filter:
                p_, t_ = self.filter(p, t)
                rmse = np.sqrt(np.mean((p_[self.offset:] - t_[self.offset:]) ** 2))

            else:
                rmse = np.sqrt(np.mean((p[self.offset:] - t[self.offset:]) ** 2))
            summary += f"  * {key}: {rmse}\n"
        return summary

    def set_params(self, offset: Optional[int] = None):
        """
        Set parameters of the RMSECalculator.

        :param offset: Offset, which determines the number of ignored values in the beginning for calculating the RMSE.
        :type offset: int
        """
        if offset:
            self.offset = offset
