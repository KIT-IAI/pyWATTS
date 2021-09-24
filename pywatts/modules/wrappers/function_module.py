from typing import Optional, Dict, Callable

import cloudpickle
import xarray as xr

from pywatts.core.base import BaseEstimator
from pywatts.core.filemanager import FileManager


class FunctionModule(BaseEstimator):
    """
    This module calls the function in its transform and optional fit method.
    It can be used, for executing own code in the pipeline.
    Note that the wrapped function is called with the same keyword arguments as the module.

    :param transform_method: The function which should be executed in the transform step.
    :type transform_method: Callable
    :param fit_method: The function which should be executed in the fit step.
    :type fit_method: Callable
    :param name: name of the instance (FunctionModule)
    :type name: str
    """

    def __init__(self, transform_method: Callable, fit_method: Optional[Callable] = None,
                 name: str = "FunctionModule"):
        super().__init__(name)
        if fit_method is None:
            self.is_fitted = True
            self.fit_method = fit_method
        else:
            self.fit_method = fit_method
        self.transform_method = transform_method

    def get_params(self) -> Dict[str, object]:
        """
        Returns an empty dictionary, since this wrappers does not contain any parameters

        :return: Empty dictionary
        :rtype: dict
        """
        return {}

    def set_params(self, *args, **kwargs):
        """
        Does nothing:
        """

    def fit(self, **kwargs: xr.DataArray) -> xr.DataArray:
        """
        Call the fit_method if available wrapped by this module on x.

        :param kwargs: The input arrays
        :type kwargs: xr.DataArray
        """
        if self.fit_method is not None:
            self.fit_method(**kwargs)
            self.is_fitted = True

    def transform(self, **kwargs: xr.DataArray) -> xr.DataArray:
        """
        Call the transform_method wrapped by this module on x.

        :param x: The input arrays
        :type x: xarray.DataArray
        :return: The transformed DataArray
        :rtype: xarray.DataArray
        """
        return self.transform_method(**kwargs)

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
