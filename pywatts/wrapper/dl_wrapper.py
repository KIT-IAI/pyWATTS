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

    def __init__(self, model, name, fit_kwargs=None):
        super().__init__(name)
        self.model = model
        if fit_kwargs is None:
            fit_kwargs = {}
        self.fit_kwargs = fit_kwargs
        self.compiled = False

    @staticmethod
    def _to_dl_input(data: xr.Dataset):
        result = {}
        for dv in data.data_vars:
            da = data[dv]
            result[dv] = da.values
        return result