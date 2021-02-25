from abc import ABC, abstractmethod
from typing import Dict

import xarray as xr

from pywatts.core.filemanager import FileManager


class BaseCallback(ABC):
    """
    Base callback class handling filemanager for all child classes.
    All child classes need to implement at least the __call__ method.

    :param ABC: Abstract Base Class
    :type ABC: ABC
    """

    def __init__(self, use_filemanager: bool = True):
        """
        Initialisation method for base callback initialising the filemanager.

        :param use_filemanager: If the pipeline should replace filemanager.
        :type use_filemanager: bool, optional
        """
        self.filemanager = None
        self.use_filemanager = use_filemanager

    @abstractmethod
    def __call__(self, data_dict: Dict[str, xr.DataArray]):
        """
        Abstract call method that need to be implemented by child classes.
        So, callback objects can be called like simple functions
        which are also allowed as callbacks.

        :param data_dict: Dict of DataArrays as output from the pipeline step.
        :type data_dict: Dict[str, xr.DataArray]
        :raises NotImplementedError: Callbacks need to implement __call__ method.
        """
        raise NotImplementedError('Callbacks need to implement __call__ method!')

    def set_filemanager(self, filemanager: FileManager):
        """
        Set or replace filemanager to change save location (e.g. for different runs)
        if the user set the use_filemanager flag to True.

        :param filemanager: [description]
        :type filemanager: [type]
        """
        if self.use_filemanager:
            self.filemanager = filemanager
        else:
            self.filemanager = None

    def get_path(self, filename: str) -> str:
        """
        Get the full path to use as a save location for callback objects.

        :param filename: Suggested name of the file (could be changed if already exist).
        :type filename: str
        :return: Full path to save the file to.
        :rtype: str
        """
        if self.filemanager is None:
            return filename
        else:
            return self.filemanager.get_path(filename)
