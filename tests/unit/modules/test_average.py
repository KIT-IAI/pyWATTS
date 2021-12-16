import unittest

import pandas as pd
import xarray as xr

from pywatts.modules import Average


class TestAverage(unittest.TestCase):
    def setUp(self) -> None:
        self.averager = Average()

    def tearDown(self) -> None:
        self.averager = None

    def test_get_params(self):
        self.assertEqual(
            self.averager.get_params(),
            {
                "weights": None,
            }
        )

    def test_set_params(self):
        self.assertEqual(
            self.averager.get_params(),
            {
                "weights": None,
            }
        )
        self.averager.set_params(weights=[1, 2, 3])
        self.assertEqual(
            self.averager.get_params(),
            {
                "weights": [1, 2, 3],
            }
        )

    def test_transform_averaging(self):
        time = pd.date_range('2002-01-01', freq='24H', periods=7)

        da1 = xr.DataArray([2, 3, 4, 5, 6, 7, 8], dims=["time"], coords={'time': time})
        da2 = xr.DataArray([3, 4, 5, 6, 7, 8, 9], dims=["time"], coords={'time': time})

        result = self.averager.transform(y1=da1, y2=da2)

        expected_result = xr.DataArray([2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5], dims=["time"], coords={'time': time})

        xr.testing.assert_equal(result, expected_result)

    def test_transform_weighting(self):
        time = pd.date_range('2002-01-01', freq='24H', periods=7)

        da1 = xr.DataArray([2, 3, 4, 5, 6, 7, 8], dims=["time"], coords={'time': time})
        da2 = xr.DataArray([3, 4, 5, 6, 7, 8, 9], dims=["time"], coords={'time': time})

        self.averager.set_params(weights=[1, 0])
        result = self.averager.transform(y1=da1, y2=da2)
        expected_result = xr.DataArray([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dims=["time"], coords={'time': time})
        xr.testing.assert_equal(result, expected_result)

        self.averager.set_params(weights=[0, 1])
        result = self.averager.transform(y1=da1, y2=da2)
        expected_result = xr.DataArray([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dims=["time"], coords={'time': time})
        xr.testing.assert_equal(result, expected_result)

        self.averager.set_params(weights=[1, 1])
        result = self.averager.transform(y1=da1, y2=da2)
        expected_result = xr.DataArray([2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5], dims=["time"], coords={'time': time})
        xr.testing.assert_equal(result, expected_result)

        self.averager.set_params(weights=[0.75, 0.25])
        result = self.averager.transform(y1=da1, y2=da2)
        expected_result = xr.DataArray([2.25, 3.25, 4.25, 5.25, 6.25, 7.25, 8.25], dims=["time"], coords={'time': time})
        xr.testing.assert_equal(result, expected_result)

        self.averager.set_params(weights=[0.25, 0.75])
        result = self.averager.transform(y1=da1, y2=da2)
        expected_result = xr.DataArray([2.75, 3.75, 4.75, 5.75, 6.75, 7.75, 8.75], dims=["time"], coords={'time': time})
        xr.testing.assert_equal(result, expected_result)
