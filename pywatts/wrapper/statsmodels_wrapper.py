import pickle
from typing import Dict

import numpy as np
import xarray as xr

from pywatts.core.filemanager import FileManager
from pywatts.utils._xarray_time_series_utils import _get_time_indeces, numpy_to_xarray
from pywatts.wrapper.base_wrapper import BaseWrapper


class StatsmodelsWrapper(BaseWrapper):
    """
    Wrapper for statsmodels modules.

    :param module: The statsmodels module to wrap
    :type module: sm.tsa.base.tsa_model.TimeSeriesModel
    :param name: The name of the module
    :type name: str
    :param module_kwargs: The module keyword arguments necessary for creating the statsmodel module
    :type module_kwargs: dict
    :param fit_kwargs: The optional fit keyword arguments for fitting the model
    :type fit_kwargs: dict
    :param predict_kwargs: The optional predict keyword arguments for predicting with the model (except start and end)
    :type predict_kwargs: dict
    :param model_params: The fitted model parameters
    :type xr.Dataset
    """

    def __init__(self, module, name: str = None, module_kwargs=None,
                 fit_kwargs=None, predict_kwargs=None, model_params=None):
        if name is None:
            name = module.__class__.__name__
        super().__init__(name)
        if fit_kwargs is None:
            fit_kwargs = {}
        if predict_kwargs is None:
            predict_kwargs = {}
        self.module = module
        self.module_kwargs = module_kwargs
        self.fit_kwargs = fit_kwargs
        self.predict_kwargs = predict_kwargs
        self.model_params = model_params

    def get_params(self) -> Dict[str, object]:
        """
        Returns the parameters of the statsmodels module.

        :return: A dict containing the module keyword arguments, the fit keyword arguments, the predict keyword
        arguments and the fitted model parameters
        :rtype: dict
        """
        return {
            "module_kwargs": self.module_kwargs,
            "fit_kwargs": self.fit_kwargs,
            "predict_kwargs": self.predict_kwargs,
            "model_params": self.model_params
        }

    def set_params(self, module_kwargs=None, fit_kwargs=None, predict_kwargs=None, model_params=None):
        """
        Set the parameters of the statsmodels wrapper

        :param module_kwargs: keyword arguments for the statsmodel module.
        :param fit_kwargs: keyword arguments for the fit method.
        :param predict_kwargs: keyword arguments for the predict method.
        :param model_params: parameters of the fitted model
        """
        if module_kwargs:
            self.module_kwargs = module_kwargs
        if fit_kwargs:
            self.fit_kwargs = fit_kwargs
        if predict_kwargs:
            self.predict_kwargs = predict_kwargs
        if model_params:
            self.model_params = model_params

    def fit(self, **kwargs :xr.DataArray):
        """
        Fits the statsmodels module

        :param kwargs: The input and the target data
        """
        x = []
        y = []
        for key, value in kwargs.items():
            if key.startswith("target"):
                y.append(value.values.reshape(-1))
            else:
                x.append(value.values.reshape(-1))

        self.model = self.module(endog=np.stack(y, axis=-1),exog=np.stack(x, axis=-1), **self.module_kwargs).fit(**self.fit_kwargs)

        self.is_fitted = True

    def transform(self,**kwargs: xr.DataArray) -> xr.DataArray:
        """
        Predicts the result with the wrapped statsmodels module

        :param x: the input dataarray
        :return: the transformed dataarray
        """
        time_data = list(kwargs.values())[0][_get_time_indeces(kwargs)[0]]

        x = []
        for key, value in kwargs.items():
            x.append(value.values.reshape(-1))

        prediction = self.model.forecast(len(time_data), exog=np.stack(x, axis=-1), **self.predict_kwargs)[0]
        return numpy_to_xarray(prediction, list(kwargs.values())[0], self.name)

    def save(self, fm: FileManager):
        """
        Saves the statsmodels wrapper and the containing model

        :param fm: FileManager for getting the path
        :type fm: FileManager
        :return: Dictionary with additional information
        :rtype: Dict
        """
        json = super().save(fm)
        file_path = fm.get_path(f'{self.name}.pickle')
        with open(file_path, 'wb') as outfile:
            pickle.dump(obj=self.module, file=outfile)
        json.update({"statsmodels_module": file_path})
        return json

    @classmethod
    def load(cls, load_information) -> 'StatsmodelsWrapper':
        """
        Loads a statsmodels wrapper

        :param load_information: Information for reloading the StatsmodelsWrapper
        :type load_information: Dict
        :return: The reloaded StatsmodelsWrapper
        :rtype: StatsmodelsWrapper

        .. warning::
            This method use pickle for loading the module. Note that this is not safe.
            Consequently, load only modules you trust.
            For more details about pickling see https://docs.python.org/3/library/pickle.html
        """
        name = load_information["name"]
        with open(load_information["statsmodels_module"], 'rb') as pickle_file:
            module = pickle.load(pickle_file)
        module = cls(module=module, name=name)
        module.is_fitted = load_information["is_fitted"]
        return module
