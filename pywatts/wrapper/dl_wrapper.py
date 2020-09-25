# pylint: disable=W0223
# Pylint cannot handle abstract subclasses of abstract base classes

from abc import ABC
from typing import Dict

import xarray as xr
from pywatts.wrapper.base_wrapper import BaseWrapper


class DlWrapper(BaseWrapper, ABC):
    """
    Super class for deep learning framework wrappers

    :param model: The deep learning model
    :param name: The name of the wrapper
    :type name: str
    :param fit_kwargs: The fit keyword arguments necessary for fitting the model
    :type fit_kwars: dict
    :param compile_kwargs: The compile keyword arguments necessary for compiling the model.
    :type compile_kwargs: dict
    """

    def __init__(self, model, name, fit_kwargs=None, compile_kwargs=None):
        super().__init__(name)
        self.model = model
        if compile_kwargs is None:
            compile_kwargs = {}
        if fit_kwargs is None:
            fit_kwargs = {}
        self.fit_kwargs = fit_kwargs
        self.compile_kwargs = compile_kwargs
        self.compiled = False

    @staticmethod
    def _to_dl_input(data: xr.Dataset):
        result = {}
        for dv in data.data_vars:
            da = data[dv]
            result[dv] = da.values
        return result

    def get_params(self) -> Dict[str, object]:
        """
        Returns the parameters of deep learning frameworks.
        :return: A dict containing the fit keyword arguments and the compile keyword arguments
        """
        return {
            "fit_kwargs": self.fit_kwargs,
            "compile_kwargs": self.compile_kwargs
        }

    def set_params(self, fit_kwargs=None, compile_kwargs=None):
        """
        Set the parameters of the deep learning wrapper
        :param fit_kwargs: keyword arguments for the fit method.
        :param compile_kwargs: keyword arguments for the compile methods.
        """
        if fit_kwargs:
            self.fit_kwargs = fit_kwargs
        if compile_kwargs:
            self.compile_kwargs = compile_kwargs
