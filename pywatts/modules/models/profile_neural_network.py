import logging
from typing import Dict

import tensorflow
import numpy as np
import xarray as xr
from pywatts.core.base import BaseEstimator
from pywatts.core.filemanager import FileManager
from pywatts.utils._xarray_time_series_utils import numpy_to_xarray

from tensorflow.keras import layers
from tensorflow.keras import activations, optimizers, initializers
from tensorflow import keras


class ProfileNeuralNetwork(BaseEstimator):
    """
    This module implements the profile neural network. It is a model for forecasting short-term electrical load.
    Therefore, it takes into account, trend information, calendar_information, historical input but also the profile
    of the load.
    Note the horizon is extracted from the data
    If you use it please cite:
        Benedikt Heidrich, Marian Turowski, Nicole Ludwig, Ralf Mikut, and Veit Hagenmeyer. 2020.
        Forecasting energy time series with profile neural networks. In Proceedings of the Eleventh ACM International
        Conference on Future Energy Systems (e-Energy ’20). Association for Computing Machinery, New York, NY, USA,
        220–230. DOI:https://doi.org/10.1145/3396851.3397683

    :param name: The name of the module
    :type name: str
    :param epochs: The number of epochs the model should be trained.
    :type epochs: int
    :param offset: The number of samples at the beginning of the dataset that should be **not** considered for training.
    :type offset: int
    :param batch_size: The batch size which should be used for training
    :type batch_size: int
    :param validation_split: The share of data which should be used for validation
    :type validation_split: float
    """

    def __init__(self, name: str = "PNN", epochs=50, offset=0, batch_size=128, validation_split=0.2):
        super().__init__(name)
        self.epochs = epochs
        self.offset = offset
        self.batch_size = batch_size
        self.validation_split = validation_split

    def get_params(self) -> Dict[str, object]:
        """ Get parameter for this object as dict.

        :return: Object parameters as json dict
        """
        return {
            "epochs": self.epochs,
            "offset": self.offset,
            "batch_size": self.batch_size,
            "validation_split": self.validation_split
        }

    def set_params(self, epochs=None, offset=None, batch_size=None, validation_split=None):
        """
        :param epochs: The number of epochs the model should be trained.
        :type epochs: int
        :param offset: The number of samples at the beginning of the dataset that should be **not** considered for training.
        :type offset: int
        :param batch_size: The batch size which should be used for training
        :type batch_size: int
        :param validation_split: The share of data which should be used for validation
        :type validation_split: float
        """
        if batch_size:
            self.batch_size = batch_size
        if epochs:
            self.epochs = epochs
        if offset:
            self.offset = offset
        if validation_split:
            self.validation_split = validation_split

    def transform(self, historical_input, calendar, temperature, humidity, profile, trend) -> xr.DataArray:
        """
        Forecast the electrical load for the given input.

        :param historical_input: The historical input
        :type historical_input: xr.DataArray
        :param calendar: The calendar information of the dates that should be predicted.
        :type calendar: xr.DataArray
        :param temperature: The temperature of the dates that should be predicted
        :type temperature: xr.DataArray
        :param humidity: The humidity of the dates that should be predicted
        :type humidity: xr.DataArray
        :param profile: The profile of the dates that should be predicted
        :type profile: xr.DataArray
        :param trend: The trend information of the dates that should be predicted
        :type trend: xr.DataArray
        :return: The prediction
        :rtype: xr.DataArray
        """
        result = self.pnn.predict({
            "hist_input": historical_input.values,
            "full_trend": trend.values,
            "profile": profile.values,
            "dummy_input": np.concatenate(
                [calendar.values, temperature.values.reshape(-1, self.horizon, 1),
                 humidity.values.reshape(-1, self.horizon, 1)], axis=-1)
        })
        return numpy_to_xarray(result, historical_input)

    def fit(self, historical_input, calendar, temperature, humidity, profile, trend, target):
        """
        Fit the Profile Neural Network.

        :param historical_input: The historical input
        :type historical_input: xr.DataArray
        :param calendar: The calendar information of the dates that should be predicted.
        :type calendar: xr.DataArray
        :param temperature: The temperature of the dates that should be predicted
        :type temperature: xr.DataArray
        :param humidity: The humidity of the dates that should be predicted
        :type humidity: xr.DataArray
        :param profile: The profile of the dates that should be predicted
        :type profile: xr.DataArray
        :param trend: The trend information of the dates that should be predicted
        :type trend: xr.DataArray
        :param target: The ground truth of the desired prediction
        :type target: xr.DataArray
        """
        input_length = historical_input.shape[-1]
        trend_length = trend.shape[-1]
        self.horizon = target.shape[-1]
        self.pnn = _PNN(self.horizon, n_steps_in=input_length, trend_length=trend_length)

        input, t = self._clean_dataset({
            "hist_input": historical_input.values[self.offset:],
            "full_trend": trend.values[self.offset:],
            "profile": profile.values[self.offset:],
            "dummy_input": np.concatenate(
                [calendar.values, temperature.values.reshape(-1, self.horizon, 1),
                 humidity.values.reshape(-1, self.horizon, 1)], axis=-1)[self.offset:]
        }, target.values[self.offset:])
        self.pnn.fit(input, t, epochs=self.epochs, batch_size=self.batch_size, validation_split=self.validation_split)
        self.is_fitted = True

    def save(self, fm: FileManager) -> Dict:
        """
        Stores the PNN at the given path

        :param fm: The Filemanager, which contains the path where the model should be stored
        :return: The path where the model is stored.
        """
        json = super().save(fm)
        if self.is_fitted:
            filepath = fm.get_path(f"{self.name}.h5")
            self.pnn.save(filepath=filepath)
            json.update({
                "pnn": filepath
            })
        return json

    @classmethod
    def load(cls, load_information) -> BaseEstimator:
        """
        Load the PNN model.

        :param params:  The paramters which should be used for restoring the PNN.
        :return: A wrapped keras model.
        """
        pnn_module = ProfileNeuralNetwork(name=load_information["name"], **load_information["params"])
        if load_information["is_fitted"]:
            try:
                pnn = keras.models.load_model(filepath=load_information["pnn"],
                                              custom_objects={"_sum_squared_error": _sum_squared_error,
                                                              "_root_mean_squared_error": _root_mean_squared_error})
            except Exception as exception:
                logging.error("No model found in %s.", load_information['pnn'])
                raise exception
            pnn_module.pnn = pnn
            pnn_module.is_fitted = True
        return pnn_module

    @staticmethod
    def _clean_dataset(X, y, same_values_in_a_row=2):
        """
        Cleans the dataset. The following three rules are applied:
        Arguments:
            X: Input data
            y: Target data
            same_values_in_a_row: parameter which indicates how often the same value in a row is accetable
        """

        def _check_instance(data):
            """
                    Checks if the data is nan, contains more than same_values_in_a_row values which has the same value  and if the value is zero
                Returns: Bool: False if one of the conditions applies
            """
            counter = 0
            d_last = -1
            for d in data:
                if d_last == d:
                    if counter > same_values_in_a_row:
                        return False
                    counter += 1
                else:
                    counter = 0
                d_last = d
            return True

        x_cleaned = {}
        for x in X:
            x_cleaned[x] = []
        y_cleaned = []
        for i in range(len(y)):
            if not np.any(np.isnan(X["hist_input"][i])) and not np.any(np.isnan(y[i])) and not np.any(
                    y[i] == 0) and _check_instance(X["hist_input"][i]) and _check_instance(y[i]) and not np.any(
                np.isnan(X["full_trend"][i])) and not np.any(np.isnan(X["dummy_input"][i])):
                # add to cleaned dataset
                for key, x in X.items():
                    x_cleaned[key].append(x[i])
                y_cleaned.append(y[i, :])
        for key, x in x_cleaned.items():
            x_cleaned[key] = np.array(x)
        return x_cleaned, np.array(y_cleaned)


def _sum_squared_error(y_true, y_pred):
    from tensorflow.keras import backend as K
    return K.sum(K.square(y_true - y_pred), axis=-1)


def _root_mean_squared_error(y_true, y_pred):
    from tensorflow.keras import backend as K
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def _PNN(n_steps_out, n_steps_in=36, trend_length=5) -> tensorflow.keras.Model:
    activation = activations.elu

    def hist_encoder(conv_input):
        conv = layers.Reshape((n_steps_in, 1))(conv_input)
        conv = layers.Conv1D(4, [3], activation=activation, padding='same')(conv)
        conv = layers.MaxPool1D(pool_size=2)(conv)

        conv = layers.Conv1D(1, [7], activation=activation, padding='same')(conv)
        conv = layers.MaxPool1D(pool_size=2)(conv)
        conv = layers.Flatten()(conv)
        conv = layers.Dense(n_steps_out)(conv)
        conv = layers.Reshape((n_steps_out, 1))(conv)
        return conv

    def prediction_network(fc):
        fc = layers.Conv1D(16, [7], padding='same', activation=activation)(fc)
        fc = layers.SpatialDropout1D(rate=0.3)(fc)
        fc = layers.Conv1D(8, [7], padding='same', activation=activation)(fc)
        fc = layers.SpatialDropout1D(rate=0.3)(fc)
        fc = layers.Conv1D(1, [7], padding='same')(fc)
        return fc

    def external_encoder(dummy_input):
        dummy = layers.Conv1D(2, [7], activation=activation, padding='same')(dummy_input)
        dummy = layers.Conv1D(1, [7], activation=activation, padding='same')(dummy)
        dummy = layers.Flatten()(dummy)
        dummy = layers.Reshape((n_steps_out, 1))(dummy)
        return dummy

    def trend_encoder(trend_input):
        trend = layers.Dense(1, activation=activation)(trend_input)
        trend = layers.Dense(4, activation=activation)(trend)
        trend = layers.Conv1D(4, [5], activation=activation, padding='same')(trend)
        trend = layers.Conv1D(1, [5], activation=activation, padding='same')(trend)
        return trend

    conv_input = keras.Input(shape=(n_steps_in,), name="hist_input")
    trend_input = keras.Input(shape=(n_steps_out, trend_length), name="full_trend")
    profile_input = keras.Input(shape=(n_steps_out,), name="profile")
    dummy_input = keras.Input(shape=(n_steps_out, 16), name="dummy_input")

    conv = hist_encoder(conv_input)

    trend = trend_encoder(trend_input)

    dummy = external_encoder(dummy_input)

    fc = layers.concatenate([dummy, conv], axis=2)
    fc = prediction_network(fc)

    profile = layers.Reshape((n_steps_out, 1))(profile_input)

    out = layers.concatenate([fc, profile, trend])
    out = layers.Conv1D(1, [1], padding='same', use_bias=False, activation=activations.linear,
                        kernel_initializer=initializers.Constant(value=1 / 3), name="aggregation_layer")(out)
    pred = layers.Flatten()(out)

    model = keras.Model(inputs=[conv_input, trend_input, dummy_input, profile_input], outputs=pred)

    model.compile(optimizer=optimizers.Adam(), loss=_sum_squared_error,
                  metrics=[_root_mean_squared_error])
    return model
