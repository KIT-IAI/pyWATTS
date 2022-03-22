import os
import unittest
from typing import Optional
from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd
import xarray as xr

from pywatts.modules.models.profile_neural_network import ProfileNeuralNetwork, _sum_squared_error, _root_mean_squared_error
from pywatts.modules import KerasWrapper


class TestPNN(unittest.TestCase):
    def setUp(self) -> None:
        self.pnn = ProfileNeuralNetwork()

    def tearDown(self) -> None:
        self.pnn: Optional[KerasWrapper] = None

    @patch("pywatts.modules.models.profile_neural_network._PNN")
    def test_fit(self, pnn_model):
        keras_pnn = MagicMock()
        pnn_model.return_value = keras_pnn

        time = pd.date_range('2000-01-01', freq='24H', periods=7)
        hist_input = xr.DataArray([[2, 0], [3, 2], [4, 3], [5, 4], [6, 5], [7, 6], [8, 7]],
                                  dims=["time", "horizon"], coords={"time": time, "horizon": [0, 1]})
        cal_input = xr.DataArray([[[2], [0]], [[3], [2]], [[4], [3]], [[5], [4]], [[6], [5]], [[7], [6]], [[8], [7]]],
                                 dims=["time", "horizon", "length"], coords={"time": time, "horizon": [0, 1]})
        trend = xr.DataArray([[[2], [0]], [[3], [2]], [[4], [3]], [[5], [4]], [[6], [5]], [[7], [6]], [[8], [7]]],
                             dims=["time", "horizon", "length"], coords={"time": time, "horizon": [0, 1]})
        temperature = xr.DataArray([[2, 0], [3, 2], [4, 3], [5, 4], [6, 5], [7, 6], [8, 7]],
                                   dims=["time", "horizon"], coords={"time": time, "horizon": [0, 1]})
        humidity = xr.DataArray([[2, 0], [3, 2], [4, 3], [5, 4], [6, 5], [7, 6], [8, 7]],
                                dims=["time", "horizon"], coords={"time": time, "horizon": [0, 1]})
        profile = xr.DataArray([[2, 0], [3, 2], [4, 3], [5, 4], [6, 5], [7, 6], [8, 7]],
                               dims=["time", "horizon"], coords={"time": time, "horizon": [0, 1]})
        target = xr.DataArray([[5, 5], [6, 6], [7, 7], [7, 7], [8, 8], [9, 9], [9, 9]],
                              dims=["time", "horizon"], coords={"time": time, "horizon": [0, 1]})

        self.pnn.fit(historical_input=hist_input, calendar=cal_input, trend=trend, temperature=temperature,
                     profile=profile, humidity=humidity, target=target)

        pnn_model.assert_called_once_with(2, n_steps_in=2, trend_length=1)
        keras_pnn.fit.assert_called_once()
        fit_args = keras_pnn.fit.call_args

        np.testing.assert_equal(fit_args[0][0]["hist_input"], hist_input.values)
        np.testing.assert_equal(fit_args[0][0]["profile"], profile.values)
        np.testing.assert_equal(fit_args[0][0]["full_trend"], trend.values)
        np.testing.assert_equal(fit_args[0][0]["dummy_input"], np.concatenate(
            [cal_input.values, temperature.values.reshape(-1, 2, 1), humidity.values.reshape(-1, 2, 1)], axis=-1))
        np.testing.assert_equal(fit_args[0][1], target.values)
        np.testing.assert_equal(fit_args[1], {'epochs': 50, 'batch_size': 128, 'validation_split': 0.2})
        self.assertTrue(self.pnn.is_fitted)

        self.assertEqual(pnn_model(), self.pnn.pnn)
        self.assertEqual(self.pnn.horizon, 2)

    @patch("pywatts.modules.models.profile_neural_network.numpy_to_xarray")
    def test_transform(self, np2xr_mock):
        keras_pnn = MagicMock()
        keras_result_mock = MagicMock()
        keras_pnn.predict.return_value = keras_result_mock
        self.pnn.pnn = keras_pnn
        self.pnn.horizon = 2
        result_mock = MagicMock()
        np2xr_mock.return_value = result_mock

        time = pd.date_range('2000-01-01', freq='24H', periods=7)
        hist_input = xr.DataArray([[2, 0], [3, 2], [4, 3], [5, 4], [6, 5], [7, 6], [8, 7]],
                                  dims=["time", "horizon"], coords={"time": time, "horizon": [0, 1]})
        cal_input = xr.DataArray([[[2], [0]], [[3], [2]], [[4], [3]], [[5], [4]], [[6], [5]], [[7], [6]], [[8], [7]]],
                                 dims=["time", "horizon", "length"], coords={"time": time, "horizon": [0, 1]})
        trend = xr.DataArray([[[2], [0]], [[3], [2]], [[4], [3]], [[5], [4]], [[6], [5]], [[7], [6]], [[8], [7]]],
                             dims=["time", "horizon", "length"], coords={"time": time, "horizon": [0, 1]})
        temperature = xr.DataArray([[2, 0], [3, 2], [4, 3], [5, 4], [6, 5], [7, 6], [8, 7]],
                                   dims=["time", "horizon"], coords={"time": time, "horizon": [0, 1]})
        humidity = xr.DataArray([[2, 0], [3, 2], [4, 3], [5, 4], [6, 5], [7, 6], [8, 7]],
                                dims=["time", "horizon"], coords={"time": time, "horizon": [0, 1]})
        profile = xr.DataArray([[2, 0], [3, 2], [4, 3], [5, 4], [6, 5], [7, 6], [8, 7]],
                               dims=["time", "horizon"], coords={"time": time, "horizon": [0, 1]})

        result = self.pnn.transform(historical_input=hist_input, calendar=cal_input, trend=trend,
                                    temperature=temperature,
                                    profile=profile, humidity=humidity)

        np2xr_mock.assert_called_once()
        np2xr_args = np2xr_mock.call_args
        self.assertEqual(np2xr_args[0][0], keras_result_mock)
        xr.testing.assert_equal(np2xr_args[0][1], hist_input)

        keras_pnn.predict.assert_called_once()
        transform_args = keras_pnn.predict.call_args

        np.testing.assert_equal(transform_args[0][0]["hist_input"], hist_input.values)
        np.testing.assert_equal(transform_args[0][0]["profile"], profile.values)
        np.testing.assert_equal(transform_args[0][0]["full_trend"], trend.values)
        np.testing.assert_equal(transform_args[0][0]["dummy_input"], np.concatenate(
            [cal_input.values, temperature.values.reshape(-1, 2, 1), humidity.values.reshape(-1, 2, 1)], axis=-1))

        self.assertEqual(result, result_mock)

    def test_get_params(self):
        self.assertEqual(self.pnn.get_params(), {'batch_size': 128, 'epochs': 50, "offset": 0, "validation_split": 0.2})

    def test_set_params(self):
        self.assertEqual(self.pnn.get_params(), {'batch_size': 128, 'epochs': 50, "offset": 0, "validation_split": 0.2})

        self.pnn.set_params(validation_split=0.5, epochs=10, offset=5, batch_size=42)

        self.assertEqual(self.pnn.get_params(), {'batch_size': 42, 'epochs': 10, "offset": 5, "validation_split": 0.5})

    def test_save_if_fitted(self):
        fm_mock = MagicMock()
        fm_mock.get_path.return_value = os.path.join("new_path", "to_somewhere", "pnn.h5")
        keras_pnn = MagicMock()
        self.pnn.is_fitted = True
        self.pnn.pnn = keras_pnn

        json = self.pnn.save(fm_mock)

        keras_pnn.save.assert_called_once_with(
            filepath=os.path.join("new_path", "to_somewhere", "pnn.h5"))
        fm_mock.get_path.assert_called_once_with("PNN.h5")
        self.assertEqual({
            'class': 'ProfileNeuralNetwork',
            'is_fitted': True,
            'pnn': os.path.join("new_path", "to_somewhere", "pnn.h5"),
            'module': 'pywatts.modules.models.profile_neural_network',
            'name': 'PNN',
            'params': {'epochs': 50, "offset": 0, "validation_split": 0.2, "batch_size": 128}}, json)

    def test_save_if_not_fitted(self):
        fm_mock = MagicMock()

        json = self.pnn.save(fm_mock)

        fm_mock.get_path.assert_not_called()
        self.assertEqual({
            'class': 'ProfileNeuralNetwork',
            'is_fitted': False,
            'module': 'pywatts.modules.models.profile_neural_network',
            'name': 'PNN',
            'params': {'epochs': 50, "offset": 0, "validation_split": 0.2, "batch_size": 128}}, json)

    @patch('pywatts.modules.models.profile_neural_network.keras.models.load_model')
    def test_load_if_fitted(self, load_model_mock):
        new_pnn_mock = MagicMock()
        load_model_mock.return_value = new_pnn_mock
        pnn = ProfileNeuralNetwork.load({'class': 'ProfileNeuralNetwork',
                                         'is_fitted': True,
                                         'module': 'pywatts.modules.models.profile_neural_network',
                                         'name': 'PNN',
                                         'params': {'batch_size': 128,
                                                    'epochs': 50,
                                                    'offset': 0,
                                                    'validation_split': 0.2},
                                         'pnn': os.path.join('new_path','to_somewhere','pnn.h5')})
        calls_open = [call(filepath=os.path.join("new_path", "to_somewhere", "pnn.h5"),
                           custom_objects={'_sum_squared_error': _sum_squared_error,
                                           '_root_mean_squared_error': _root_mean_squared_error})]

        load_model_mock.assert_has_calls(calls_open, any_order=True)
        self.assertEqual(load_model_mock.call_count, 1)
        self.assertEqual(new_pnn_mock, pnn.pnn)
        self.assertEqual(pnn.get_params(),
                         {'batch_size': 128,
                          'epochs': 50,
                          'offset': 0,
                          'validation_split': 0.2})

    @patch('pywatts.modules.models.profile_neural_network.keras.models.load_model')
    def test_load_if_not_fitted(self, load_model_mock):
        pnn = ProfileNeuralNetwork.load({'class': 'ProfileNeuralNetwork',
                                         'is_fitted': False,
                                         'name': 'PNN',
                                         'params': {'batch_size': 128,
                                                    'epochs': 50,
                                                    'offset': 0,
                                                    'validation_split': 0.2}})

        load_model_mock.assert_not_called()
        self.assertEqual(pnn.get_params(),
                         {'batch_size': 128,
                          'epochs': 50,
                          'offset': 0,
                          'validation_split': 0.2})
