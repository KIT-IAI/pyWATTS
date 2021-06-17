import logging
from typing import Dict, Callable, Optional, Tuple

import cloudpickle
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
    :param filter_method: Filter which should performed on the data before calculating the RMSE.
    :type filter_method: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
    """

    def __init__(self,
                 name: str = "RMSE",
                 filter_method: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]] = None,
                 offset: int = 0):
        super().__init__(name)
        self.offset = offset
        self.filter_method = filter_method

    def get_params(self) -> Dict[str, object]:
        """
        Returns a dict of parameters used in the RMSE Calculator.

        :return: Parameters set for the RMSE calculator
        :rtype: Dict[str, object]
        """
        return {"offset": self.offset}

    def transform(self, file_manager: FileManager, y: xr.DataArray, **kwargs: xr.DataArray) -> str:
        """
        Calculates the RMSE based on the predefined target and predictions variables.
        :param file_manager: The filemanager, it can be used to store data that corresponds to the summary as a file.
        :type: file_manager: FileManager
        :param y: the input dataset
        :type y: xr.DataArray
        :param kwargs: the predictions
        :type kwargs: xr.DataArray

        :return: The calculated RMSE
        :rtype: xr.DataArray
        """

        t = y.values
        summary = ""
        if kwargs == {}:
            error_message = "No predictions are provided as input for the RMSE.  You should add the predictions by a " \
                            "seperate key word arguments if you add the RMSE to the pipeline."
            logger.error(error_message)
            raise InputNotAvailable(error_message)

        for key, y_hat in kwargs.items():
            p = y_hat.values
            if self.filter_method:
                p_, t_ = self.filter_method(p, t)
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

    def save(self, fm: FileManager) -> Dict:
        json = super().save(fm)
        if self.filter_method is not None:
            filter_path = fm.get_path(f"{self.name}_filter.pickle")
            with open(filter_path, 'wb') as outfile:
                cloudpickle.dump(self.filter_method, outfile)
            json["filter"] = filter_path
        return json

    @classmethod
    def load(cls, load_information: Dict):
        params = load_information["params"]
        name = load_information["name"]
        filter_method = None
        if "filter" in load_information:
            with open(load_information["filter"], 'rb') as pickle_file:
                filter_method = cloudpickle.load(pickle_file)
        return cls(name=name, filter_method=filter_method, **params)
