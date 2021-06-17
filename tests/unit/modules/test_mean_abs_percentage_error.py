import unittest

import pytest
import xarray as xr
import pandas as pd
from pywatts.core.exceptions.input_not_available import InputNotAvailable

from pywatts.modules.mean_abs_percentage_error import MapeCalculator
import numpy as np


class TestMaeCalculator(unittest.TestCase):

    def setUp(self) -> None:
        self.mape_calculator = MapeCalculator()

    def tearDown(self) -> None:
        self.mape_calculator = None

    def test_get_params(self):
        self.assertEqual(self.mape_calculator.get_params(),
                         {'offset': 0, 'rolling': False, 'window': 24})

    def test_set_params(self):
        self.mape_calculator.set_params(offset=24, rolling=True, window=2)
        self.assertEqual(self.mape_calculator.get_params(),
                         {'offset': 24, 'rolling': True, 'window': 2})

    def test_transform_rolling(self):
        self.mape_calculator.set_params(rolling=True, window=2)
        time = pd.to_datetime(['2015-06-03 00:00:00', '2015-06-03 01:00:00',
                               '2015-06-03 02:00:00', '2015-06-03 03:00:00',
                               '2015-06-03 04:00:00'])

        test_data = xr.Dataset({"testCol": ("time", xr.DataArray([-2, -1, 0, 1, 2])),
                                "predictCol1": ("time", xr.DataArray([2, -3, 3, 1, -2])),
                                "predictCol2": ("time", xr.DataArray([4, 4, 3, -2, 1])), "time": time})

        test_result = self.mape_calculator.transform(y=test_data['testCol'], gt=test_data['testCol'],
                                                     pred1=test_data['predictCol1'],
                                                     pred2=test_data['predictCol2'])
        expected_result = xr.DataArray(np.array([[np.nan, np.nan, np.nan],
                                                 [0.0, 2, 4],
                                                 [np.nan, np.nan, np.nan],
                                                 [np.nan, np.nan, np.nan],
                                                 [0.0, 1, 7 / 4]]),
                                       coords={"time": time, "predictions": ["gt", "pred1", "pred2"]},
                                       dims=["time", "predictions"])

        xr.testing.assert_allclose(test_result, expected_result)

    def test_transform(self):
        self.mape_calculator.set_params()

        time = pd.to_datetime(['2015-06-03 00:00:00', '2015-06-03 01:00:00',
                               '2015-06-03 02:00:00', '2015-06-03 03:00:00',
                               '2015-06-03 04:00:00'])

        result_time = pd.to_datetime(['2015-06-03 04:00:00'])

        test_data = xr.Dataset({"testCol": ("time", xr.DataArray([-2, -1, 0, 1, 2])),
                                "predictCol1": ("time", xr.DataArray([2, -3, 3, 1, -2])),
                                "predictCol2": ("time", xr.DataArray([4, 4, 3, -2, 1])), "time": time})

        test_result = self.mape_calculator.transform(y=test_data['testCol'], gt=test_data['testCol'],
                                                     pred1=test_data['predictCol1'],
                                                     pred2=test_data['predictCol2'])

        expected_result = xr.DataArray(np.array([[0.0, 3 / 2, 23 / 8]]),
                                       coords={"time": result_time, "predictions": ["gt", "pred1", "pred2"]},
                                       dims=["time", "predictions"])

        xr.testing.assert_equal(test_result, expected_result)

    def test_transform_without_predictions(self):
        self.mape_calculator.set_params()

        time = pd.to_datetime(['2015-06-03 00:00:00', '2015-06-03 01:00:00',
                               '2015-06-03 02:00:00', '2015-06-03 03:00:00',
                               '2015-06-03 04:00:00'])

        test_data = xr.Dataset({"testCol": ("time", xr.DataArray([-2, -1, 0, 1, 2])),
                                "predictCol1": ("time", xr.DataArray([2, -3, 3, 1, -2])),
                                "predictCol2": ("time", xr.DataArray([4, 4, 3, -2, 1])), "time": time})

        with pytest.raises(InputNotAvailable) as e_info:
            self.mape_calculator.transform(y=test_data['testCol'])

        self.assertEqual(e_info.value.message,
                         "No predictions are provided as input for the MAPE Calculator. You should add the predictions "
                         "by a seperate key word arguments if you add the MAPE Calculator to the pipeline.")
