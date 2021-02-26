from typing import Optional, Dict, Callable

import cloudpickle
import xarray as xr

from pywatts.core.base import BaseTransformer
from pywatts.core.filemanager import FileManager


class FunctionModule(BaseTransformer):
    """
    This module calls the function in its transform. It can be used, for executing own code in the pipeline

    :param name: name of the instance (FunctionModule)
    :type name: str
    :param function: The function which should be executed in the transform step.
    :type function: Callable
    """

    def __init__(self, function: Callable, name: str = "FunctionModule"):
        super().__init__(name)
        self.function = function

    def get_params(self) -> Dict[str, object]:
        """
        Returns an empty dictionary, since this wrapper does not contain any parameters

        :return: Empty dictionary
        :rtype: dict
        """
        return {}

    def set_params(self, *args, **kwargs):
        """
        Does nothing:
        """

    def transform(self, x: Optional[xr.Dataset]) -> xr.Dataset:
        """
        Call the function wrapped by this module on x.

        :param x: The input xrarray dataset
        :type x: xarray.Dataset
        :return: The transformed Dataset
        :rtype: xarray.Dataset
        """
        return self.function(x)

    def save(self, fm: FileManager):
        """
        Saves the Conditional module to JSON file

        :param fm: A FileManager, from which the path where the JSON file is saved is fetcheds
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
