import unittest

import pandas as pd
import xarray as xr

from pywatts.modules import RollingGroupBy, RollingVariance


class TestRollingVariance(unittest.TestCase):
    def setUp(self) -> None:
        self.rolling_variance = RollingVariance(window_size=3)

    def tearDown(self) -> None:
        self.rolling_variance = None

    def test_get_params(self):
        self.assertEqual(
            self.rolling_variance.get_params(),
            {
                "window_size": 3,
                "window_size_unit": "d",
                "group_by": RollingGroupBy.No,
                "country": "Germany",
                "continent": "Europe"
            }
        )

    def test_set_params(self):
        self.assertEqual(
            self.rolling_variance.get_params(),
            {
                "window_size": 3,
                "window_size_unit": "d",
                "group_by": RollingGroupBy.No,
                "country": "Germany",
                "continent": "Europe"
            }
        )
        self.rolling_variance.set_params(window_size=5)
        self.assertEqual(
            self.rolling_variance.get_params(),
            {
                "window_size": 5,
                "window_size_unit": "d",
                "group_by": RollingGroupBy.No,
                "country": "Germany",
                "continent": "Europe"
            }
        )

    def test_transform_groupbyNo(self):
        time = pd.date_range('2002-01-01', freq='24H', periods=7)

        da = xr.DataArray([2, 3, 4, 5, 6, 7, 8], dims=["time"], coords={'time': time})

        result = self.rolling_variance.transform(da)

        expected_result = xr.DataArray([0., 0, 0.5, 1., 1., 1., 1.], dims=["time"], coords={'time': time})

        xr.testing.assert_equal(result, expected_result)

    def test_transform_groupbyWeekend(self):
        time = pd.date_range('2002-01-01', freq='24H', periods=7)
        self.rolling_variance.set_params(group_by=RollingGroupBy.WorkdayWeekend)
        da = xr.DataArray([2, 3, 4, 5, 6, 7, 8], dims=["time"], coords={'time': time})

        result = self.rolling_variance.transform(da)

        expected_result = xr.DataArray([0., 0, 0.5, 1., 0, 0, 0], dims=["time"], coords={'time': time})
        xr.testing.assert_equal(result, expected_result)

    def test_transform_groupbyWeekendHoliday(self):
        time = pd.date_range('2002-01-01', freq='24H', periods=7)
        self.rolling_variance.set_params(group_by=RollingGroupBy.WorkdayWeekendAndHoliday)
        da = xr.DataArray([2, 3, 4, 5, 6, 7, 8], dims=["time"], coords={'time': time})

        result = self.rolling_variance.transform(da)

        expected_result = xr.DataArray([0., 0, 0, .5, 0, 0, 0], dims=["time"], coords={'time': time})
        xr.testing.assert_equal(result, expected_result)
