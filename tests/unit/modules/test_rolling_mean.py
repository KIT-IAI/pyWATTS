import unittest

import numpy as np
import pandas as pd
import xarray as xr

from pywatts.core.exceptions.wrong_parameter_exception import WrongParameterException
from pywatts.modules.rolling_mean import RollingMean, RollingMeanGroupBy


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
                "group_by": RollingMeanGroupBy.No,
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
                "group_by": RollingMeanGroupBy.No,
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
                "group_by": RollingMeanGroupBy.No,
                "country": "Germany",
                "continent": "Europe"
            }
        )

    def test_transform_groupbyNo(self):
        time = pd.date_range('2002-01-01', freq='24H', periods=7)

        ds = xr.DataArray([2, 3, 4, 5, 6, 7, 8], dims=["time"] , coords={'time': time})

        result = self.rolling_mean.transform(ds)

        expected_result = xr.DataArray([0. , 2. , 2.5, 3. , 4. , 5. , 6. ], dims=["time"] , coords={'time': time})

        xr.testing.assert_equal(result, expected_result)

    def test_transform_groupbyWeekendWeekday(self):
        self.rolling_mean.set_params(group_by=RollingMeanGroupBy.WorkdayWeekend)
        time = pd.date_range('2002-01-01', freq='24H', periods=7)

        ds = xr.DataArray([2, 3, 4, 5, 6, 7, 42], dims=["time"], coords={'time': time})

        result = self.rolling_mean.transform(ds)

        expected_result = xr.DataArray([0., 2., 2.5, 3., 0., 6., 5.], dims=["time"], coords={'time': time})

        xr.testing.assert_equal(result, expected_result)

    def test_transform_groupbyWeekendWeekdayAndHoliday(self):
        time = pd.date_range('2002-01-01', freq='24H', periods=7)
        self.rolling_mean.set_params(window_size=4, group_by=RollingMeanGroupBy.WorkdayWeekendAndHoliday)

        ds = xr.DataArray([2, 3, 4, 5, 6, 7, 42], dims=["time"], coords={'time': time})

        result = self.rolling_mean.transform(ds)

        expected_result = xr.DataArray([0., 0., 3, 3.5, 2., 6., 4.5], dims=["time"], coords={'time': time})

        xr.testing.assert_equal(result, expected_result)
