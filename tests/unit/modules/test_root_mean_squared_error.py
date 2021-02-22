import unittest
import xarray as xr
import pandas as pd

from pywatts.modules.root_mean_squared_error import RmseCalculator
import numpy as np


class TestRMSECalculator(unittest.TestCase):

    def setUp(self) -> None:
        self.rmse_calculator = RmseCalculator()

    def tearDown(self) -> None:
        self.rmse_calculator = None

    def test_get_params(self):
        self.assertEqual(self.rmse_calculator.get_params(),
                         {})

    def test_transform(self):
        self.rmse_calculator.set_params(target="testCol", prediction=["predictCol1", "predictCol2"])

        time = pd.to_datetime(['2015-06-03 00:00:00', '2015-06-03 01:00:00',
                               '2015-06-03 02:00:00', '2015-06-03 03:00:00',
                               '2015-06-03 04:00:00'])

        result_time = pd.to_datetime(['2015-06-03 04:00:00'])

        test_data = xr.Dataset({"testCol": ("time", xr.DataArray([-2, -1, 0, 1, 2])),
                                "predictCol1": ("time", xr.DataArray([2, -3, 3, 1, -2])),
                                "predictCol2": ("time", xr.DataArray([4, 4, 3, -2, 1])), "time": time})

        test_result = self.rmse_calculator.transform(y=test_data['testCol'], y_hat=test_data['testCol'])  # This fails

        expected_result = xr.DataArray(np.array([0.0]), coords={"time": result_time}, dims=["time"])

        xr.testing.assert_equal(test_result, expected_result)
