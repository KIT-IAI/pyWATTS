import logging
from typing import Tuple, Union, Dict

import cloudpickle
import tensorflow as tf
import xarray as xr

from pywatts.core.filemanager import FileManager
from pywatts.utils._split_kwargs import split_kwargs
from pywatts.utils._xarray_time_series_utils import _get_time_indexes, xarray_to_numpy, numpy_to_xarray
from pywatts.modules.wrappers.dl_wrapper import DlWrapper


class KerasWrapper(DlWrapper):
    """
    Wrapper class for keras models

    :param model: The deep learning model
    :param name: The name of the wrappers
    :type name: str
    :param fit_kwargs: The fit keyword arguments necessary for fitting the model
    :type fit_kwargs: dict
    :param compile_kwargs: The compile keyword arguments necessary for compiling the model.
    :type compile_kwargs: dict
    :param custom_objects: This dict contains all custom objects needed by the keras model. Note,
                           users that uses such customs objects (e.g. Custom Loss) need to specify this to enable
                           the loading of the stored Keras model.
    :type custom_objects: dict
    """

    def __init__(self, model: Union[tf.keras.Model, Tuple[tf.keras.Model, Dict[str, tf.keras.Model]]],
                 name: str = "KerasWrapper", fit_kwargs=None, compile_kwargs=None, custom_objects=None):
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
        if custom_objects is None:
            self.custom_objects = {}
        else:
            self.custom_objects = custom_objects

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
            key: numpy_to_xarray(pred, list(kwargs.values())[0]) for key, pred in zip(self.targets, prediction)
        }
        return result

    def save(self, fm: FileManager) -> dict:
        """
        Stores the keras model at the given path
        :param fm: The Filemanager, which contains the path where the model should be stored
        :return: The path where the model is stored.
        """
        json = {"name": self.name,
                "class": self.__class__.__name__,
                "module": self.__module__}

        params = self.get_params()
        params_path = fm.get_path(f"{self.name}_params.pickle")
        with open(params_path, "wb") as outfile:
            cloudpickle.dump(params, outfile)
        json["params"] = params_path
        json["is_fitted"] = self.is_fitted

        custom_path = fm.get_path(f"{self.name}_custom.pickle")
        with open(custom_path, "wb") as outfile:
            cloudpickle.dump(self.custom_objects, outfile)
        json["custom_objects"] = custom_path

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
        name = load_information["name"]
        params_path = load_information["params"]
        with open(params_path, "rb") as infile:
            params = cloudpickle.load(infile)

        custom_path = load_information["custom_objects"]
        with open(custom_path, "rb") as infile:
            custom_objects = cloudpickle.load(infile)

        try:
            model = tf.keras.models.load_model(filepath=load_information["model"], custom_objects=custom_objects)
        except Exception as exception:
            logging.error("No model found in %s.", load_information['model'])
            raise exception
        aux_models = {}
        if "aux_models" in load_information.keys():
            for aux_name, path in load_information["aux_models"]:
                try:
                    aux_models[aux_name] = tf.keras.models.load_model(filepath=path, custom_objects=custom_objects)
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
            "compile_kwargs": self.compile_kwargs,
            "custom_objects": self.custom_objects
        }

    def set_params(self, fit_kwargs=None, compile_kwargs=None, custom_objects=None):
        """
        Set the parameters of the deep learning wrappers
        :param fit_kwargs: keyword arguments for the fit method.
        :param compile_kwargs: keyword arguments for the compile methods.
        :param custom_objects: This dict contains all custom objects needed by the keras model. Note,
                               users that uses such customs objects (e.g. Custom Loss) need to specify this to enable
                               the loading of the stored Keras model.
        """
        if fit_kwargs:
            self.fit_kwargs = fit_kwargs
        if compile_kwargs:
            self.compile_kwargs = compile_kwargs
        if custom_objects:
            self.custom_objects = custom_objects
