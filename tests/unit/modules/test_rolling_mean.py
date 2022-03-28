import unittest

import pandas as pd
import xarray as xr

from pywatts.modules import RollingGroupBy, RollingMean


class TestRollingMean(unittest.TestCase):
    def setUp(self) -> None:
        self.rolling_mean = RollingMean(window_size=3)

    def tearDown(self) -> None:
        self.rolling_mean = None

    def test_get_params(self):
        self.assertEqual(
            self.rolling_mean.get_params(),
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
            self.rolling_mean.get_params(),
            {
                "window_size": 3,
                "window_size_unit": "d",
                "group_by": RollingGroupBy.No,
                "country": "Germany",
                "continent": "Europe"
            }
        )
        self.rolling_mean.set_params(window_size=5)
        self.assertEqual(
            self.rolling_mean.get_params(),
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

        ds = xr.DataArray([2, 3, 4, 5, 6, 7, 8], dims=["time"], coords={'time': time})

        result = self.rolling_mean.transform(ds)

        expected_result = xr.DataArray([0., 2., 2.5, 3., 4., 5., 6.], dims=["time"], coords={'time': time})

        xr.testing.assert_equal(result, expected_result)

    def test_transform_groupbyWeekendWeekday(self):
        self.rolling_mean.set_params(group_by=RollingGroupBy.WorkdayWeekend)
        time = pd.date_range('2002-01-01', freq='12H', periods=14)

        ds = xr.DataArray([2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 42, 42], dims=["time"], coords={'time': time})

        result = self.rolling_mean.transform(ds)

        expected_result = xr.DataArray([0., 0., 2., 2., 2.5, 2.5, 3., 3., 0., 0., 6., 6., 5., 5.], dims=["time"],
                                       coords={'time': time})

        xr.testing.assert_equal(result, expected_result)

    def test_transform_groupbyWeekendWeekdayAndHoliday(self):
        time = pd.date_range('2002-01-01', freq='12H', periods=14)
        self.rolling_mean.set_params(window_size=4, group_by=RollingGroupBy.WorkdayWeekendAndHoliday)

        ds = xr.DataArray([2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 42, 42], dims=["time"], coords={'time': time})

        result = self.rolling_mean.transform(ds)

        expected_result = xr.DataArray([0., 0., 0., 0., 3, 3, 3.5, 3.5, 2, 2., 6.,6,4.5, 4.5], dims=["time"], coords={'time': time})

        xr.testing.assert_equal(result, expected_result)

    def test_get_min_data(self):
        r_mean = RollingMean(window_size=12, window_size_unit="h")
        self.assertEqual(r_mean.get_min_data(), pd.Timedelta(hours=12))
