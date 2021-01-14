import os
import unittest
from typing import Optional
from unittest.mock import MagicMock, call

import xarray as xr
import pandas as pd
import numpy as np

from pywatts.wrapper.statsmodels_wrapper import StatsmodelsWrapper

stored_module = {
    "class": "StatsmodelsWrapper",
    "model": os.path.join("pipe1", "AR.h5"),
    "module": "pywatts.wrapper.statsmodels_wrapper",
    "name": "AR",
    'is_fitted': False,
    "params": {
        "module_kwargs": {
            "lags": [1, 4]
        },
        "fit_kwargs": {
            "cov_type": "nonrobust"
        },
        "predict_kwargs": {
            "dynamic": True
        },
        "model_params": {
            "scale": 2.0
        }
    }
}


class TestStatsmodelsWrapper(unittest.TestCase):

    def setUp(self) -> None:
        self.statsmodels_mock: Optional[MagicMock] = MagicMock()
        self.statsmodels_wrapper = StatsmodelsWrapper(self.statsmodels_mock, module_kwargs={"lags": [1, 2]},
                                                      fit_kwargs={}, predict_kwargs={}, model_params={})

    def tearDown(self) -> None:
        self.statsmodels_wrapper: Optional[StatsmodelsWrapper] = None
        self.statsmodels_mock = None

    def test_get_params(self):
        self.assertEqual(self.statsmodels_wrapper.get_params(),
                         {'module_kwargs': {'lags': [1, 2]},
                          'fit_kwargs': {},
                          'predict_kwargs': {},
                          'model_params': {}
                          })

    def test_set_params(self):
        self.assertEqual(self.statsmodels_wrapper.get_params(),
                         {'module_kwargs': {'lags': [1, 2]},
                          'fit_kwargs': {},
                          'predict_kwargs': {},
                          'model_params': {}
                          })
        self.statsmodels_wrapper.set_params(module_kwargs={"lags": [1, 4]},
                                            fit_kwargs={"cov_type": "nonrobust"},
                                            predict_kwargs={"dynamic": True},
                                            model_params={"scale": 2.0})
        self.assertEqual(self.statsmodels_wrapper.get_params(),
                         {"module_kwargs": {"lags": [1, 4]},
                          "fit_kwargs": {"cov_type": "nonrobust"},
                          "predict_kwargs": {"dynamic": True},
                          "model_params": {"scale": 2.0}
                          })

    def test_fit(self):
        self.statsmodels_wrapper.set_params(module_kwargs={"lags": [1, 2]}, fit_kwargs={}, predict_kwargs={},
                                            model_params={})
        time = pd.date_range('2000-01-01', freq='24H', periods=7)
        data = xr.Dataset(
            {'TestData': (['time', 'measurement'], [[2, 1], [3, 2], [4, 4], [5, 8], [6, 16], [7, 32], [8, 64]]),
             'time': time})
        self.statsmodels_wrapper.fit(data)

        # assert fit is called
        self.statsmodels_mock.module.assert_called_once_with(module_kwargs={"lags": [1, 2]})
        self.statsmodels_mock.fit.assert_called_once()

        # assert correct arguments when calling fit
        args = self.statsmodels_mock.fit.call_args
        np.testing.assert_equal(args[1]["x"]["TestData"],
                                np.array([[2, 1], [3, 2], [4, 4], [5, 8], [6, 16], [7, 32], [8, 64]]))
        self.assertEqual(len(args[1]["x"]), 1)

        # assert is_fitted is set to true
        self.assertTrue(self.statsmodels_wrapper.is_fitted)

    def test_predict(self):
        self.statsmodels_wrapper.set_params(module_kwargs={"lags": [1, 2]}, fit_kwargs={}, predict_kwargs={},
                                            model_params={})
        time = pd.date_range('2000-01-01', freq='24H', periods=7)
        data = xr.Dataset(
            {'TestData': (['time', 'measurement'], [[2, 1], [3, 2], [4, 4], [5, 8], [6, 16], [7, 32], [8, 64]]),
             'time': time})
        self.statsmodels_wrapper.transform(data)

        # assert transform is called
        self.statsmodels_mock.transform.assert_called_once()

        # assert correct arguments when calling transform
        args = self.statsmodels_mock.predict.call_args
        np.testing.assert_equal(args[1], '2000-01-01 00:00:00')
        np.testing.assert_equal(args[2], '2000-01-07 00:00:00')
        np.testing.assert_equal(args[3], {})

    def test_save(self):
        fm_mock = MagicMock()
        fm_mock.get_path.return_value = os.path.join("new_path", "to_somewhere", "StatsmodelsWrapper.pickle")
        json = self.statsmodels_wrapper.save(fm_mock)
        self.statsmodels_mock.save.assert_called_once_with(
            filepath=os.path.join("new_path", "to_somewhere", "StatsmodelsWrapper.pickle"))
        fm_mock.get_path.has_calls(call(os.path.join("to_somewhere", "StatsmodelsWrapper.pickle")),
                                   any_order=True)
        self.assertEqual(json, {'class': 'StatsmodelsWrapper',
                                'is_fitted': False,
                                'model': os.path.join("new_path", "to_somewhere", "StatsmodelsWrapper.pickle"),
                                'module': 'pywatts.wrapper.statsmodels_wrapper',
                                'name': 'StatsmodelsWrapper',
                                })

    def test_load(self, load_module_mock):
        new_statsmodels_mock = MagicMock()
        load_module_mock.return_value = new_statsmodels_mock
        new_statsmodels_wrapper = StatsmodelsWrapper.load(stored_module)
        calls_open = [call(filepath=os.path.join("pipe1", "AR.pickle"))]

        load_module_mock.assert_has_calls(calls_open, any_order=True)
        self.assertEqual(load_module_mock.call_count, 1)
        self.assertEqual(new_statsmodels_mock, new_statsmodels_wrapper.module)
        self.assertEqual(new_statsmodels_wrapper.get_params(),
                         {
                             "module_kwargs": {
                                 "lags": [1, 4]
                             },
                             "fit_kwargs": {
                                 "cov_type": "nonrobust"
                             },
                             "predict_kwargs": {
                                 "dynamic": True
                             },
                             "model_params": {
                                 "scale": 2.0
                             }
                         })
