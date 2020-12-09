import unittest

import xarray as xr
from statsmodels.tsa.ar_model import AutoReg

from pywatts.wrapper.statsmodels_wrapper import StatsmodelsWrapper


class TestStatsmodelsWrapper(unittest.TestCase):

    def test_fit_Regression(self):
        data = xr.Dataset({"test": xr.DataArray([1, 2, 3, 4, 5])})
        ar_model = AutoReg(data, lags=[1, 2])
        wrapper = StatsmodelsWrapper(module=ar_model)
        test_ar_results = wrapper.fit(data)
        expected_ar_results = ar_model.fit()
        self.assertEqual(test_ar_results, expected_ar_results)

    def test_predict_Regression(self):
        data = xr.Dataset({"test": xr.DataArray([1, 2, 3, 4, 5])})
        ar_model = AutoReg(data, lags=[1, 2])
        ar_results = ar_model.fit()
        ar_results_param = ar_results.params
        wrapper = StatsmodelsWrapper(module=ar_model)
        test_prediction = wrapper.transform(ar_results_param)
        expected_prediction = ar_model.predict(ar_results_param)
        self.assertEqual(test_prediction, expected_prediction)
