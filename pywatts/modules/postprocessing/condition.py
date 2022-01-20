import cloudpickle
import xarray as xr
import numpy as np
from typing import Dict, Callable

from pywatts.core.base import BaseTransformer
from pywatts.core.filemanager import FileManager
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

    def save(self, fm: FileManager):
        """
        Saves the Conditional module to JSON file

        :param fm: A FileManager, from which the path where the JSON file is saved is fetches
        :type fm: FileManager
        :return: Dictionary with name, parameters, related module and class, and path to the file
        :rtype: Dict
        """
        json_module = super().save(fm)
        file_path = fm.get_path(f'{self.name}.pickle')
        with open(file_path, 'wb') as outfile:
            cloudpickle.dump(self, file=outfile)
        json_module["pickled_module"] = file_path
        json_module.pop("params")
        return json_module

    @classmethod
    def load(cls, load_information):
        """
        Loads a condition from a JSON file and creates the corresponding Conditional

        :param load_information: JSON file of the Conditional
        :type load_information: Dict
        :return: conditional module from file
        :rtype: Conditional

        .. warning::
            This method use pickle for loading the module. Note that this is not safe.
            Consequently, load only modules you trust.
            For more details about pickling see https://docs.python.org/3/library/pickle.html

        """
        with open(load_information["pickled_module"], 'rb') as pickle_file:
            module = cloudpickle.load(pickle_file)
        return module
