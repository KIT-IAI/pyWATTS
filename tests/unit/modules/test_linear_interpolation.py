import unittest
import xarray as xr
import numpy as np
import pandas as pd

from pywatts.modules import LinearInterpolater


class TestLinearInterpolater(unittest.TestCase):

    def setUp(self) -> None:
        self.linear_interpolater = LinearInterpolater()

    def tearDown(self) -> None:
        self.linear_interpolater = None

    def test_get_params(self):
        self.assertEqual(self.linear_interpolater.get_params(),
                         {
                             "method": "linear",
                             "dim": "time",
                             "fill_value": "extrapolate"
                         })

    def test_set_params(self):
        self.assertEqual(self.linear_interpolater.get_params(),
                         {
                             "method": "linear",
                             "dim": "time",
                             "fill_value": "extrapolate"
                         })
        self.linear_interpolater.set_params(method="index", dim="location", fill_value="inside")
        self.assertEqual(self.linear_interpolater.get_params(),
                         {
                             "method": "index",
                             "dim": "location",
                             "fill_value": "inside"
                         })
        self.linear_interpolater.set_params(method="linear", dim="time", fill_value="extrapolate")

    def test_transform(self):
        time = pd.to_datetime(['2015-06-03 00:00:00', '2015-06-03 01:00:00',
                               '2015-06-03 02:00:00', '2015-06-03 03:00:00',
                               '2015-06-03 04:00:00'])
        test_data = xr.Dataset({"test": ("time", xr.DataArray([1, 2, np.nan, 4, 5]).data),
                                "test2": ("time", xr.DataArray([np.nan, 2, 3, 4, 5]).data),
                                "test3": ("time", xr.DataArray([1, 2, 3, 4, np.nan]).data),
                                "test4": ("time", xr.DataArray([1, np.nan, np.nan, np.nan, 5]).data), "time": time})
        test_result = self.linear_interpolater.transform(test_data)
        expected_result = xr.Dataset({"test": ("time", xr.DataArray([1., 2., 3., 4., 5.]).data),
                                      "test2": ("time", xr.DataArray([1., 2., 3., 4., 5.]).data),
                                      "test3": ("time", xr.DataArray([1., 2., 3., 4., 5.]).data),
                                      "test4": ("time", xr.DataArray([1., 2., 3., 4., 5.]).data), "time": time})
        xr.testing.assert_equal(test_result, expected_result)
