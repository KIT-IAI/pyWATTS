import unittest

import pytest
import xarray as xr
import pandas as pd
from pywatts.core.exceptions.input_not_available import InputNotAvailable

from pywatts.modules.root_mean_squared_error import RmseCalculator
import numpy as np


class TestRMSECalculator(unittest.TestCase):

    def setUp(self) -> None:
        self.rmse_calculator = RmseCalculator()

    def tearDown(self) -> None:
        self.rmse_calculator = None

    def test_get_params(self):
        self.assertEqual(self.rmse_calculator.get_params(),
                         {'offset': 0, 'rolling': False, 'window': 24})

    def test_set_params(self):
        self.rmse_calculator.set_params(offset=24, rolling=True, window=2)
        self.assertEqual(self.rmse_calculator.get_params(),
                         {'offset': 24, 'rolling': True, 'window': 2})

    def test_transform_rolling(self):
        self.rmse_calculator.set_params(rolling=True, window=2)
        time = pd.to_datetime(['2015-06-03 00:00:00', '2015-06-03 01:00:00',
                               '2015-06-03 02:00:00', '2015-06-03 03:00:00',
                               '2015-06-03 04:00:00'])

        test_data = xr.Dataset({"testCol": ("time", xr.DataArray([-2, -1, 0, 1, 2])),
                                "predictCol1": ("time", xr.DataArray([2, -3, 3, 1, -2])),
                                "predictCol2": ("time", xr.DataArray([4, 4, 3, -2, 1])), "time": time})

        test_result = self.rmse_calculator.transform(y=test_data['testCol'], gt=test_data['testCol'],
                                                     pred1=test_data['predictCol1'],
                                                     pred2=test_data['predictCol2'])
        expected_result = xr.DataArray(np.array([[np.nan, np.nan, np.nan],
                                                 [0.0, np.sqrt(10), np.sqrt(30.5)],
                                                 [0.0, np.sqrt(6.5), np.sqrt(17)],
                                                 [0.0, np.sqrt(4.5), 3],
                                                 [0.0, np.sqrt(8), np.sqrt(5)], ]),
                                       coords={"time": time, "predictions": ["gt", "pred1", "pred2"]},
                                       dims=["time", "predictions"])

        xr.testing.assert_allclose(test_result, expected_result)

    def test_transform(self):
        self.rmse_calculator.set_params()

        time = pd.to_datetime(['2015-06-03 00:00:00', '2015-06-03 01:00:00',
                               '2015-06-03 02:00:00', '2015-06-03 03:00:00',
                               '2015-06-03 04:00:00'])

        result_time = pd.to_datetime(['2015-06-03 04:00:00'])

        test_data = xr.Dataset({"testCol": ("time", xr.DataArray([-2, -1, 0, 1, 2])),
                                "predictCol1": ("time", xr.DataArray([2, -3, 3, 1, -2])),
                                "predictCol2": ("time", xr.DataArray([4, 4, 3, -2, 1])), "time": time})

        test_result = self.rmse_calculator.transform(y=test_data['testCol'], gt=test_data['testCol'],
                                                     pred1=test_data['predictCol1'],
                                                     pred2=test_data['predictCol2'])

        expected_result = xr.DataArray(np.array([[0.0, 3.0, 4.0]]),
                                       coords={"time": result_time, "predictions": ["gt", "pred1", "pred2"]},
                                       dims=["time", "predictions"])

        xr.testing.assert_equal(test_result, expected_result)

    def test_transform_without_predictions(self):
        self.rmse_calculator.set_params()

        time = pd.to_datetime(['2015-06-03 00:00:00', '2015-06-03 01:00:00',
                               '2015-06-03 02:00:00', '2015-06-03 03:00:00',
                               '2015-06-03 04:00:00'])

        test_data = xr.Dataset({"testCol": ("time", xr.DataArray([-2, -1, 0, 1, 2])),
                                "predictCol1": ("time", xr.DataArray([2, -3, 3, 1, -2])),
                                "predictCol2": ("time", xr.DataArray([4, 4, 3, -2, 1])), "time": time})

        with pytest.raises(InputNotAvailable) as e_info:
            self.rmse_calculator.transform(y=test_data['testCol'])

        self.assertEqual(e_info.value.message,
                         "No predictions are provided as input for the RMSE Calculator. You should add the predictions "
                         "by a seperate key word arguments if you add the RMSECalculator to the pipeline.")
