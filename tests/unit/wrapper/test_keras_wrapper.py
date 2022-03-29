import os
import unittest
from typing import Optional
from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd
import xarray as xr

from pywatts.modules import KerasWrapper

stored_model = {
    "aux_models": [
        [
            "encoder",
            os.path.join("pipe1", "SimpleAE_4encoder.h5")
        ],
        [
            "decoder",
            os.path.join("pipe1", "SimpleAE_4decoder.h5")
        ]
    ],
    "class": "KerasWrapper",
    "model": os.path.join("pipe1", "SimpleAE_4.h5"),
    "module": "pywatts.wrappers.keras_wrapper",
    "name": "SimpleAE",
    'is_fitted': False,
    "params": "params_path",
    "custom_objects" : "custom_path",
    "targets": []
}


class TestKerasWrapper(unittest.TestCase):
    def setUp(self) -> None:
        self.keras_mock: Optional[MagicMock] = MagicMock()
        self.keras_wrapper = KerasWrapper(self.keras_mock, compile_kwargs={"test": "arg1"}, fit_kwargs={"42": 24})

    def tearDown(self) -> None:
        self.keras_wrapper: Optional[KerasWrapper] = None
        self.keras_mock = None

    def test_fit(self):
        self.keras_wrapper.set_params(fit_kwargs={"epochs": 200}, compile_kwargs={"optimizer": "adam"})
        time = pd.date_range('2000-01-01', freq='24H', periods=7)

        da = xr.DataArray([[2, 0], [3, 2], [4, 3], [5, 4], [6, 5], [7, 6], [8, 7]],
                          dims=["time", "horizon"], coords={"time": time, "horizon": [0, 1]})

        target = xr.DataArray([[5, 5], [6, 6], [7, 7], [7, 7], [8, 8], [9, 9], [9, 9]],
                              dims=["time", "horizon"], coords={"time": time, "horizon": [0, 1]})

        self.keras_wrapper.fit(data=da, target=target)
        self.keras_mock.compile.assert_called_once_with(optimizer="adam")
        self.keras_mock.fit.assert_called_once()
        args = self.keras_mock.fit.call_args

        self.assertEqual(type(args[1]["x"]["data"]), np.ndarray)
        self.assertEqual(type(args[1]["y"]["target"]), np.ndarray)

        np.testing.assert_equal(args[1]["x"]["data"],
                                np.array([[2, 0], [3, 2], [4, 3], [5, 4], [6, 5], [7, 6], [8, 7]]))
        np.testing.assert_equal(args[1]["y"]["target"],
                                np.array([[5, 5], [6, 6], [7, 7], [7, 7], [8, 8], [9, 9], [9, 9]])),
        self.assertEqual(len(args[1]["x"]), 1)
        self.assertEqual(len(args[1]["y"]), 1)
        self.assertEqual(args[1]["epochs"], 200)
        self.assertEqual(len(args[1]), 3)

        self.assertTrue(self.keras_wrapper.compiled)

    def test_transform_single_output(self):
        time = pd.date_range('2000-01-01', freq='24H', periods=7)

        da = xr.DataArray([[2, 0], [3, 2], [4, 3], [5, 4], [6, 5], [7, 6], [8, 7]],
                          dims=["time", "horizon"], coords={"time": time, "horizon": [0, 1]})

        target = np.array([[5, 5], [6, 6], [7, 7], [7, 7], [8, 8], [9, 9], [9, 9]])

        self.keras_mock.predict.return_value = target
        self.keras_mock.outputs[0].name = "first/output"
        self.keras_wrapper.targets = ["target"]

        result = self.keras_wrapper.transform(x=da)

        self.keras_mock.predict.assert_called_once()

        np.testing.assert_equal(target,
                                result["target"])

    def test_transform_multiple_output(self):
        time = pd.date_range('2000-01-01', freq='24H', periods=7)

        da = xr.DataArray([[2, 0], [3, 2], [4, 3], [5, 4], [6, 5], [7, 6], [8, 7]],
                          dims=["time", "horizon"], coords={"time": time, "horizon": [0, 1]})

        target = [np.array([[5, 5], [6, 6], [7, 7], [7, 7], [8, 8], [9, 9], [9, 9]]),
                  np.array([[5, 5], [6, 6], [7, 7], [7, 7], [8, 8], [9, 9], [9, 9]])]

        self.keras_mock.predict.return_value = target
        first = MagicMock()
        first.name = "first/output"
        second = MagicMock()
        second.name = "second"
        outputs = [first, second]
        self.keras_mock.outputs = outputs
        self.keras_wrapper.targets = ["target1", "target2"]

        result = self.keras_wrapper.transform(x=da)

        self.keras_mock.predict.assert_called_once()
        args = self.keras_mock.predict.call_args

        np.testing.assert_equal(args[0][0]["x"],
                                np.array([[2, 0], [3, 2], [4, 3], [5, 4], [6, 5], [7, 6], [8, 7]]))

        np.testing.assert_equal(target[0],
                                result["target1"])
        np.testing.assert_equal(target[1],
                                result["target2"])
    def test_get_params(self):
        self.assertEqual(self.keras_wrapper.get_params(),
                         {'compile_kwargs': {'test': 'arg1'},
                          'fit_kwargs': {'42': 24},
                          'custom_objects' : {}
                          })

    def test_set_params(self):
        self.assertEqual(self.keras_wrapper.get_params(),
                         {'compile_kwargs': {'test': 'arg1'},
                          'fit_kwargs': {'42': 24},
                          'custom_objects': {}})
        self.keras_wrapper.set_params(fit_kwargs={"loss": "mse"},
                                      compile_kwargs={"optimizer": "Adam"})
        self.assertEqual(self.keras_wrapper.get_params(),
                         {"fit_kwargs": {"loss": "mse"},
                          "compile_kwargs": {"optimizer": "Adam"},
                          "custom_objects": {}})

    @patch("pywatts.modules.wrappers.keras_wrapper.cloudpickle")
    @patch("builtins.open")
    def test_save(self, open_mock, pickle_mock):
        fm_mock = MagicMock()
        fm_mock.get_path.side_effect = [
            "params_path",
            "custom_path",
            os.path.join("new_path", "to_somewhere", "KerasWrapper.h5")]
        file_mock = MagicMock()
        open_mock().__enter__.return_value = file_mock

        json = self.keras_wrapper.save(fm_mock)
        self.keras_mock.save.assert_called_once_with(
            filepath=os.path.join("new_path", "to_somewhere", "KerasWrapper.h5"))

        open_mock.assert_has_calls([call("params_path", "wb"), call("custom_path", "wb")], any_order=True)

        pickle_mock.dump.assert_has_calls([call(self.keras_wrapper.get_params(), file_mock),
                                      call(self.keras_wrapper.custom_objects, file_mock)])
        fm_mock.get_path.has_calls(call(os.path.join("to_somewhere", "KerasWrapper.h5")),
                                   any_order=True)
        self.assertEqual(json, {'aux_models': [],
                                'class': 'KerasWrapper',
                                'is_fitted': False,
                                'model': os.path.join("new_path", "to_somewhere", "KerasWrapper.h5"),
                                'module': 'pywatts.modules.wrappers.keras_wrapper',
                                'name': 'KerasWrapper',
                                'params': "params_path",
                                "custom_objects" : "custom_path",
                                "targets":[]
                                })

    @patch("pywatts.modules.wrappers.keras_wrapper.cloudpickle")
    @patch("builtins.open")
    @patch('pywatts.modules.wrappers.keras_wrapper.tf.keras.models.load_model')
    def test_load(self, load_model_mock, open_mock, pickle_mock):
        new_keras_mock = MagicMock()
        load_model_mock.return_value = new_keras_mock
        pickle_mock.load.side_effect = [{"compile_kwargs": {"loss": "mse", "metrics": ["mse"], "optimizer": "Adam"},
                                         "fit_kwargs": {"batch_size": 512, "epochs": 1},
                                         "custom_objects": {}, },
                                        "custom_object"]
        new_keras_wrapper = KerasWrapper.load(stored_model)
        calls_open = [call(filepath=os.path.join("pipe1", "SimpleAE_4decoder.h5"), custom_objects="custom_object"),
                      call(filepath=os.path.join("pipe1", "SimpleAE_4encoder.h5"), custom_objects="custom_object"),
                      call(filepath=os.path.join("pipe1", "SimpleAE_4.h5"), custom_objects="custom_object"),
                      ]
        load_model_mock.assert_has_calls(calls_open, any_order=True)
        open_mock.assert_has_calls([call("params_path", "rb"), call("custom_path", "rb")], any_order=True)
        self.assertEqual(load_model_mock.call_count, 3)
        self.assertEqual(new_keras_mock, new_keras_wrapper.model)
        self.assertEqual(new_keras_wrapper.get_params(),
                         {
                             "compile_kwargs": {
                                 "loss": "mse",
                                 "metrics": [
                                     "mse"
                                 ],
                                 "optimizer": "Adam"
                             },
                             "fit_kwargs": {
                                 "batch_size": 512,
                                 "epochs": 1
                             },
                             "custom_objects": {}
                         })
