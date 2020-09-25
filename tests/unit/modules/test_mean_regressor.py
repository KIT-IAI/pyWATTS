import unittest

import pandas as pd
import pytest
import xarray as xr
import numpy as np

from pywatts.core.exceptions.wrong_parameter_exception import WrongParameterException
from pywatts.modules.rolling_mean import RollingMean


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
                "indexes": []
            }
        )

    def test_set_params(self):
        self.assertEqual(
            self.rolling_mean.get_params(),
            {
                "window_size": 3,
                "indexes": []
            }
        )
        self.rolling_mean.set_params(window_size=5, indexes=["test"])
        self.assertEqual(
            self.rolling_mean.get_params(),
            {
                "window_size": 5,
                "indexes": ["test"]
            }
        )

    def test_transform_explicit_index(self):
        time = pd.date_range('2000-01-01', freq='24H', periods=7)
        self.rolling_mean.set_params(indexes=["time"])

        ds = xr.Dataset({'foo': ('time', [2, 3, 4, 5, 6, 7, 8]), 'time': time})

        result = self.rolling_mean.transform(ds)

        expected_result = xr.Dataset({'foo': ('time', [np.nan, np.nan, 3., 4., 5., 6., 7.]), 'time': time})

        xr.testing.assert_equal(result, expected_result)

    def test_transform(self):
        time = pd.date_range('2000-01-01', freq='24H', periods=7)

        ds = xr.Dataset({'foo': ('time', [2, 3, 4, 5, 6, 7, 8]), 'time': time})

        result = self.rolling_mean.transform(ds)

        expected_result = xr.Dataset({'foo': ('time', [np.nan, np.nan, 3., 4., 5., 6., 7.]), 'time': time})

        xr.testing.assert_equal(result, expected_result)

    def test_transform_exception(self):
        time = pd.date_range('2000-01-01', freq='24H', periods=7)
        self.rolling_mean.set_params(indexes=["FOO"])

        ds = xr.Dataset({'foo': ('time', [2, 3, 4, 5, 6, 7, 8]), 'time': time})
        with self.assertRaises(WrongParameterException) as context:
            self.rolling_mean.transform(ds)
        self.assertEqual(context.exception.message,
                         "Not all indexes (['FOO']) are in the indexes of x (['time']). "
                         "Perhaps you set the wrong indexes with set_params or during the initialization of mean regressor.")
