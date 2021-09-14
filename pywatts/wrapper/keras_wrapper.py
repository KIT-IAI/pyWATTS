import logging
from typing import Tuple, Union, Dict

import tensorflow as tf
import xarray as xr

from pywatts.core.filemanager import FileManager
from pywatts.utils._split_kwargs import split_kwargs
from pywatts.utils._xarray_time_series_utils import _get_time_indexes, xarray_to_numpy, numpy_to_xarray
from pywatts.wrapper.dl_wrapper import DlWrapper


class KerasWrapper(DlWrapper):
    """
    Wrapper class for keras models

    :param model: The deep learning model
    :param name: The name of the wrapper
    :type name: str
    :param fit_kwargs: The fit keyword arguments necessary for fitting the model
    :type fit_kwargs: dict
    :param compile_kwargs: The compile keyword arguments necessary for compiling the model.
    :type compile_kwargs: dict
    """

    def __init__(self, model: Union[tf.keras.Model, Tuple[tf.keras.Model, Dict[str, tf.keras.Model]]],
                 name: str = "KerasWrapper", fit_kwargs=None, compile_kwargs=None):
        self.aux_models = {}
        self.targets = []
        if isinstance(model, tuple):
            self.aux_models = model[1]
            model = model[0]
        super().__init__(model, name, fit_kwargs)
        if compile_kwargs is None:
            self.compile_kwargs = {}
        else:
            self.compile_kwargs = compile_kwargs

    def fit(self, **kwargs: xr.DataArray):
        """
        Calls the compile and the fit method of the wrapped keras module.
        :param x: The input data
        :param y: The target data
        """
        x, y = split_kwargs(kwargs)
        x = {name_x: value_x.values for name_x, value_x in x.items()}
        y = {name_y: value_y.values for name_y, value_y in y.items()}
        self.targets = list(y.keys())

        if not self.compiled:
            self.model.compile(**self.compile_kwargs)
            self.compiled = True
        self.model.fit(x=x, y=y, **self.fit_kwargs)
        self.is_fitted = True

    def transform(self, **kwargs: xr.DataArray) -> xr.DataArray:
        """
        Calls predict of the underlying keras Model.
        :param x: The dataset for which a prediction should be performed
        :return:  The prediction. Each output of the keras model is a separate data variable in the returned xarray.
        """
        prediction = self.model.predict({key: da.values for key, da in kwargs.items()})
        if not isinstance(prediction, list):
            prediction = [prediction]

        result = {
            key : numpy_to_xarray(pred, list(kwargs.values())[0], self.name) for key, pred in zip(self.targets, prediction)
        }
        return result

    def save(self, fm: FileManager) -> dict:
        """
        Stores the keras model at the given path
        :param fm: The Filemanager, which contains the path where the model should be stored
        :return: The path where the model is stored.
        """
        json = super().save(fm)
        json["targets"] = self.targets
        model_path = fm.get_path(f"{self.name}.h5")
        self.model.save(filepath=model_path)
        aux_models = []
        for name, aux_model in self.aux_models.items():
            aux_model_path = fm.get_path(f"{self.name}_{name}.h5")
            aux_model.save(filepath=aux_model_path)
            aux_models.append((name, aux_model_path))
        json.update({
            "aux_models": aux_models,
            "model": model_path
        })

        return json

    @classmethod
    def load(cls, load_information) -> "KerasWrapper":
        """
        Load the keras model and instantiate a new keraswrapper class containing the model.
        :param params:  The paramters which should be used for restoring the model.
        (Note: This models should be taken from the pipeline json file)
        :return: A wrapped keras model.
        """
        params = load_information["params"]
        name = load_information["name"]
        try:
            model = tf.keras.models.load_model(filepath=load_information["model"])
        except Exception as exception:
            logging.error("No model found in %s.", load_information['model'])
            raise exception
        aux_models = {}
        if "aux_models" in load_information.keys():
            for aux_name, path in load_information["aux_models"]:
                try:
                    aux_models[aux_name] = tf.keras.models.load_model(filepath=path)
                except Exception as exception:
                    logging.error("No model found in path %s", path)
                    raise exception
            module = cls((model, aux_models), name=name, **params)
        else:
            module = cls(model, name=name, **params)
        module.is_fitted = load_information["is_fitted"]

        module.targets = load_information["targets"]
        return module

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
