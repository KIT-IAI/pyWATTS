import unittest
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import xarray as xr
from statsmodels.tsa.arima_model import ARIMA

from pywatts.wrapper.sm_time_series_model_wrapper import SmTimeSeriesModelWrapper

stored_module = {'class': 'SmTimeSeriesModelWrapper',
                 'is_fitted': False,
                 'module': 'pywatts.wrapper.sm_time_series_model_wrapper',
                 'name': 'ARIMA',
                 'params': {
                     "module_kwargs": {
                         "lags": [1, 4]
                     },
                     "fit_kwargs": {
                         "cov_type": "nonrobust"
                     },
                     "predict_kwargs": {
                         "dynamic": True
                     }},
                 'sm_class': 'ARIMA',
                 'sm_module': 'statsmodels.tsa.arima_model'
                 }


class TestSmTimeSeriesModelWrapper(unittest.TestCase):

    def setUp(self) -> None:
        self.statsmodels_mock = MagicMock()
        self.statsmodels_wrapper = SmTimeSeriesModelWrapper(name="wrapper", module=self.statsmodels_mock,
                                                            module_kwargs={"lags": [1, 2]},
                                                            fit_kwargs={}, predict_kwargs={})

        self.model_mock = MagicMock()
        self.fitted_model = MagicMock()
        self.model_mock.fit.return_value = self.fitted_model
        self.statsmodels_mock.return_value = self.model_mock

    def tearDown(self) -> None:
        self.statsmodels_wrapper: Optional[SmTimeSeriesModelWrapper] = None
        self.statsmodels_mock = None

    def test_get_params(self):
        self.assertEqual(self.statsmodels_wrapper.get_params(),
                         {'fit_kwargs': {},
                          'module_kwargs': {'lags': [1, 2]},
                          'predict_kwargs': {}})

    def test_set_params(self):
        self.assertEqual(self.statsmodels_wrapper.get_params(),
                         {
                             'fit_kwargs': {},
                             'predict_kwargs': {},
                             'module_kwargs': {'lags': [1, 2]}
                         })
        self.statsmodels_wrapper.set_params(
            fit_kwargs={"cov_type": "nonrobust"},
            predict_kwargs={"dynamic": True},
            module_kwargs={"scale": 2.0})
        self.assertEqual(self.statsmodels_wrapper.get_params(),
                         {
                             "fit_kwargs": {"cov_type": "nonrobust"},
                             "predict_kwargs": {"dynamic": True},
                             'module_kwargs': {"scale": 2.0}
                         })

    def test_fit(self):
        self.statsmodels_wrapper.set_params(module_kwargs={"lags": [1, 2]}, fit_kwargs={}, predict_kwargs={})
        time = pd.date_range('2000-01-01', freq='24H', periods=7)
        target = xr.DataArray([2, 2, 4, 4, 5, 8, 6], dims=["time"], coords={'time': time})
        exog = xr.DataArray([[1], [2], [3], [4], [5], [8], [9]], dims=["time", "dims"], coords={'time': time})
        self.statsmodels_wrapper.fit(target=target, exog=exog)

        # assert fit is called
        self.statsmodels_mock.assert_called_once()
        self.model_mock.fit.assert_called_once()

        # assert correct arguments when calling fit
        args = self.statsmodels_mock.call_args
        np.testing.assert_equal(args[1]["endog"],
                                np.array([[2], [2], [4], [4], [5], [8], [6]]))
        np.testing.assert_equal(args[1]["exog"],
                                np.array([[1], [2], [3], [4], [5], [8], [9]]))
        # assert is_fitted is set to true
        self.assertTrue(self.statsmodels_wrapper.is_fitted)
        self.assertEqual(self.fitted_model, self.statsmodels_wrapper.model)

    def test_transform(self):
        self.fitted_model.forecast.return_value = np.array([2, 2, 4, 4, 5, 8, 6]),

        self.statsmodels_wrapper.set_params(module_kwargs={"lags": [1, 2]}, fit_kwargs={}, predict_kwargs={})

        self.statsmodels_wrapper.model = self.fitted_model
        time = pd.date_range('2000-01-01', freq='24H', periods=7)
        exog = xr.DataArray([1, 2, 3, 4, 5, 8, 9], dims=["time"], coords={'time': time})
        expected_result = xr.DataArray([2, 2, 4, 4, 5, 8, 6], dims=["time"], coords={'time': time})

        result = self.statsmodels_wrapper.transform(exog=exog)

        # assert transform is called
        self.fitted_model.forecast.assert_called_once()

        # assert correct arguments when calling transform
        args = self.fitted_model.forecast.call_args
        xr.testing.assert_equal(expected_result, result)

    def test_save(self):
        fm_mock = MagicMock()
        self.statsmodels_wrapper = SmTimeSeriesModelWrapper(ARIMA, module_kwargs={"lags": [1, 2]},
                                                            fit_kwargs={}, predict_kwargs={})
        json = self.statsmodels_wrapper.save(fm_mock)

        self.assertEqual(json, {'class': 'SmTimeSeriesModelWrapper',
                                'is_fitted': False,
                                'module': 'pywatts.wrapper.sm_time_series_model_wrapper',
                                'name': 'ARIMA',
                                'params': {'fit_kwargs': {},
                                           'module_kwargs': {'lags': [1, 2]},
                                           'predict_kwargs': {}},
                                'sm_class': 'ARIMA',
                                'sm_module': 'statsmodels.tsa.arima_model'})

    def test_load(self):
        new_statsmodels_wrapper = SmTimeSeriesModelWrapper.load(stored_module)

        self.assertEqual(ARIMA, new_statsmodels_wrapper.module)
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
                             }
                         })
