import inspect
from typing import Dict, Type

import numpy as np
import xarray as xr
from statsmodels.iolib import load_pickle
from statsmodels.tsa.base.tsa_model import TimeSeriesModel

from pywatts.core.filemanager import FileManager
from pywatts.utils._split_kwargs import split_kwargs
from pywatts.utils._xarray_time_series_utils import _get_time_indexes, numpy_to_xarray
from pywatts.modules.wrappers.base_wrapper import BaseWrapper


class SmTimeSeriesModelWrapper(BaseWrapper):
    """
    Wrapper for statsmodels modules. When adding this module to the pipeline, all inputs that starts with target are
    handled as endogenous variables and all other as exogenous variables.

    :param module: The statsmodels module to wrap. Not this module should not be initialised.
    :param name: The name of the module
    :type name: str
    :param module_kwargs: The module keyword arguments necessary for creating the statsmodel module
    :type module_kwargs: dict
    :param fit_kwargs: The optional fit keyword arguments for fitting the model
    :type fit_kwargs: dict
    :param predict_kwargs: The optional predict keyword arguments for predicting with the model (except start and end)
    :type predict_kwargs: dict
    """

    def __init__(self, module: Type[TimeSeriesModel], name: str = None, module_kwargs=None,
                 fit_kwargs=None, predict_kwargs=None, use_exog=True):
        if name is None:
            name = module.__name__
        super().__init__(name)
        if fit_kwargs is None:
            fit_kwargs = {}
        if predict_kwargs is None:
            predict_kwargs = {}
        self.module = module
        if not module_kwargs:
            module_kwargs = {}
        self.module_kwargs = module_kwargs
        self.fit_kwargs = fit_kwargs
        self.predict_kwargs = predict_kwargs
        self.use_exog = use_exog

    def get_params(self) -> Dict[str, object]:
        """
        Returns the parameters of the statsmodels module.

        :return: A dict containing the module keyword arguments, the fit keyword arguments, the predict keyword
        arguments and the fitted model parameters
        :rtype: Dict
        """
        return {
            "module_kwargs": self.module_kwargs,
            "fit_kwargs": self.fit_kwargs,
            "predict_kwargs": self.predict_kwargs,
            "use_exog": self.use_exog
        }

    def set_params(self, module_kwargs=None, fit_kwargs=None, predict_kwargs=None, use_exog=None):
        """
        Set the parameters of the statsmodels wrappers

        :param module_kwargs: keyword arguments for the statsmodel module.
        :type module_kwargs: Dict
        :param fit_kwargs: keyword arguments for the fit method.
        :type fit_kwargs: Dict
        :param predict_kwargs: keyword arguments for the predict method.
        :type predict_kwargs: Dict
        """
        if module_kwargs:
            self.module_kwargs = module_kwargs
        if fit_kwargs:
            self.fit_kwargs = fit_kwargs
        if predict_kwargs:
            self.predict_kwargs = predict_kwargs
        if use_exog is not None:
            self.use_exog = use_exog

    def fit(self, **kwargs: xr.DataArray):
        """
        Fits the statsmodels module

        :param kwargs: A dict of input arrays
        :type kwargs: xr.DataArray
        """
        x, y = split_kwargs(kwargs)
        x = list(map(lambda _x: _x.values, x.values()))
        y = list(map(lambda _y: _y.values.reshape(-1), y.values()))

        # Check if the statsmodel accepts exogenous variables
        if len(x) > 0 and "exog" in inspect.signature(self.module).parameters or "kwargs" in inspect.signature(
                self.module).parameters and self.use_exog:
            self.model = self.module(endog=np.stack(y, axis=-1), exog=np.concatenate(x, axis=-1),
                                     **self.module_kwargs).fit(
                **self.fit_kwargs)
        else:
            self.model = self.module(endog=np.stack(y, axis=-1), **self.module_kwargs).fit(**self.fit_kwargs)

        self.is_fitted = True

    def transform(self, **kwargs: xr.DataArray) -> xr.DataArray:
        """
        Predicts the result with the wrapped statsmodels module

        :param kwargs: A dict of input arrays
        :type kwargs: xr.DataArray
        :return: the transformed dataarray
        :rtype: xr.DataArray
        """
        time_data = list(kwargs.values())[0][_get_time_indexes(kwargs)[0]]

        x = []
        for key, value in kwargs.items():
            x.append(value.values)

        if hasattr(self.model, "forecast"):
            if "exog" in inspect.signature(self.model.forecast).parameters or "kwargs" in inspect.signature(
                    self.model.forecast).parameters and self.use_exog:
                prediction = \
                    self.model.forecast(len(time_data), exog=np.concatenate(x, axis=-1), **self.predict_kwargs)[0]

            else:
                prediction = self.model.forecast(len(time_data), **self.predict_kwargs)[0]
        elif hasattr(self.model, "predict"):
            if "exog" in inspect.signature(self.model.predict).parameters or "kwargs" in inspect.signature(
                    self.model.predict).parameters and self.use_exog:
                prediction = \
                    self.model.predict(len(time_data), exog=np.concatenate(x, axis=-1), **self.predict_kwargs)[0]

            else:
                prediction = self.model.predict(len(time_data), **self.predict_kwargs)[0]
        else:
            raise Exception(f"{self.module.__class__.__name__} has no forecast or predict method...")

        return numpy_to_xarray(prediction, list(kwargs.values())[0])

    def save(self, fm: FileManager):
        """
        Saves the statsmodels wrappers and the containing model

        :param fm: FileManager for getting the path
        :type fm: FileManager
        :return: Dictionary with all information for restoting the module
        :rtype: Dict
        """
        json = super().save(fm)
        if self.is_fitted:
            model_file_path = fm.get_path(f"{self.name}_fitted_model.pickle")
            self.model.save(model_file_path)
            json.update({"statsmodel_model": model_file_path})
        json.update({
            "sm_class": self.module.__name__,
            "sm_module": self.module.__module__
        })
        return json

    @classmethod
    def load(cls, load_information) -> 'SmTimeSeriesModelWrapper':
        """
        Loads a statsmodels wrappers

        :param load_information: Information for reloading the StatsmodelsWrapper
        :type load_information: Dict
        :return: The reloaded StatsmodelsWrapper
        :rtype: SmTimeSeriesModelWrapper

        .. warning::
            This method use pickle for loading the module. Note that this is not safe.
            Consequently, load only modules you trust.
            For more details about pickling see https://docs.python.org/3/library/pickle.html
        """
        name = load_information["name"]
        mod = __import__(load_information["sm_module"], fromlist=load_information["sm_class"])
        module = cls(module=getattr(mod, load_information["sm_class"]), name=name, **load_information["params"])
        module.is_fitted = load_information["is_fitted"]
        if module.is_fitted:
            module.model = load_pickle(load_information["statsmodel_model"])
        return module
