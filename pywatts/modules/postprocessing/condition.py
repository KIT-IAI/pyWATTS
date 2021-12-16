from typing import Dict, Callable

import xarray as xr
import numpy as np

from pywatts.core.base import BaseTransformer
from pywatts.utils._xarray_time_series_utils import numpy_to_xarray
from pywatts.core.exceptions import InvalidInputException


class Condition(BaseTransformer):
    """
    Condition step to decide which output is applied, if_true or if_false.
    """

    def __init__(self, condition: Callable, name: str = "Condition"):
        """ Initialize the condition step.
        :param condition: A callable that checks if the condition is true or false.
        :type condition: Callable[xr.DataArray, xr.DataArray, bool]
        """
        super().__init__(name)
        self.condition = condition

    def get_params(self) -> Dict[str, object]:
        """ Get parameters for the Condition object.
        :return: Parameters as dict object.
        :rtype: Dict[str, object]
        """
        return {
            "condition": self.condition
        }

    def set_params(self, condition: Callable = None):
        """ Set or change Condition object parameters.
        :param condition: A callable that checks if the condition is true or false.
        :type condition: Callable[xr.DataArray, xr.DataArray, bool]
        """
        if condition:
            self.condition = condition

    def transform(self, dependency, if_true, if_false) -> xr.DataArray:
        """ Decide which input applies depending on the condition.
        :return: Xarray dataset aggregated by simple or weighted averaging.
        :rtype: xr.DataArray
        """

        dependency_data = dependency.data
        if_true_data = if_true.data
        if_false_data = if_false.data

        # check indexes
        list_of_indexes = [
            dependency.indexes,
            if_true.indexes,
            if_false.indexes
        ]
        if not all(all(index) == all(list_of_indexes[0]) for index in list_of_indexes):
            raise ValueError("The indexes of the given time series for conditioning do not match")

        # check shapes
        if if_true_data.shape != dependency_data.shape:
            try:
                if_true_data = if_true_data.reshape(dependency_data.shape)
            except ValueError:
                raise InvalidInputException(
                    f"The if_false does not match to the shape of the dependency in the instance "
                    f"{self.name} of class {self.__class__.__name__}.")
            self.logger.info(f"Reshaped if_false in {self.name}")

        if if_false_data.shape != dependency_data.shape:
            try:
                if_false_data = if_false_data.reshape(dependency_data.shape)
            except ValueError:
                raise InvalidInputException(
                    f"The if_false does not match to the shape of the dependency in the instance "
                    f"{self.name} of class {self.__class__.__name__}.")
            self.logger.info(f"Reshaped if_false in {self.name}")

        result = np.where(self.condition(dependency_data), if_true_data, if_false_data)

        return numpy_to_xarray(result, dependency, self.name)
