import logging
from typing import Tuple, Union, Dict

import tensorflow as tf
import xarray as xr

from pywatts.core.filemanager import FileManager
from pywatts.utils._xarray_time_series_utils import _get_time_indeces
from pywatts.wrapper.dl_wrapper import DlWrapper


class KerasWrapper(DlWrapper):
    """
    Wrapper for Keras Models.
    """

    def __init__(self, model: Union[tf.keras.Model, Tuple[tf.keras.Model, Dict[str, tf.keras.Model]]],
                 name: str = "KerasWrapper", fit_kwargs=None, compile_kwargs=None):
        self.aux_models = {}
        if isinstance(model, tuple):
            self.aux_models = model[1]
            model = model[0]
        super().__init__(model, name, fit_kwargs, compile_kwargs)

    def fit(self, x: xr.Dataset, y: xr.Dataset):
        """
        Calls the compile and the fit method of the wrapped keras module.
        :param x: The input data
        :param y: The target data
        """
        if not self.compiled:
            self.model.compile(**self.compile_kwargs)
            self.compiled = True
        self.model.fit(x=self._to_dl_input(x), y=self._to_dl_input(y), **self.fit_kwargs)
        self.is_fitted = True

    def transform(self, x: xr.Dataset) -> xr.Dataset:
        """
        Calls predict of the underlying keras Model.
        :param x: The dataset for which a prediction should be performed
        :return:  The prediction. Each output of the keras model is a separate data variable in the returned xarray.
        """
        prediction = self.model.predict(x=self._to_dl_input(x))
        if not isinstance(prediction, list):
            prediction = [prediction]

        result = None
        for i, pred in enumerate(prediction):
            coords = (
                # first dimension is number of batches. We assume that this is the time.
                ("time", x.indexes[_get_time_indeces(x)[0]]),
                *[(f"dim_{j}", list(range(size))) for j, size in enumerate(pred.shape[1:])])
            data = {self.model.outputs[i].name.split("/")[0]: (tuple(map(lambda x: x[0], coords)), pred),
                    "time": x.indexes[_get_time_indeces(x)[0]]}
            if not result:
                result = xr.Dataset(data)
            else:
                result = result.merge(xr.Dataset(data))
        return result

    def save(self, fm: FileManager) -> dict:
        """
        Stores the keras model at the given path
        :param fm: The Filemanager, which contains the path where the model should be stored
        :return: The path where the model is stored.
        """
        json = super().save(fm)
        self.model.save(filepath=fm.get_path(f"{self.name}.h5"))
        aux_models = []
        for name, aux_model in self.aux_models.items():
            aux_model.save(filepath=fm.get_path(f"{self.name}_{name}.h5"))
            aux_models.append((name, fm.get_path(f"{self.name}_{name}.h5")))
        json.update({
            "aux_models": aux_models,
            "model": fm.get_path(f"{self.name}.h5")
        })

        return json

    @classmethod
    def load(cls, load_information) -> "KerasWrapper":
        """
        Load the keras model and instantiate a new keraswrapper class containing the model.
        :param path:  The path where the keras model is stored
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
        return module
