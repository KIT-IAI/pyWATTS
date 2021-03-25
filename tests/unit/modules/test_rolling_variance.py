import unittest

import pandas as pd
import pytest
import xarray as xr
import numpy as np

from pywatts.core.exceptions.wrong_parameter_exception import WrongParameterException
from pywatts.modules.rolling_base import RollingGroupBy
from pywatts.modules.rolling_variance import RollingVariance


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
