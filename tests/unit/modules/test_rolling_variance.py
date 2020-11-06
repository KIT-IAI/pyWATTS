import unittest

import pandas as pd
import pytest
import xarray as xr
import numpy as np

from pywatts.core.exceptions.wrong_parameter_exception import WrongParameterException
from pywatts.modules.rolling_variance import RollingVariance


class TestRollingVariance(unittest.TestCase):
    def setUp(self) -> None:
        self.rolling_variance = RollingVariance(window_size=2)

    def tearDown(self) -> None:
        self.rolling_variance = None

    def test_get_params(self):
        self.assertEqual(
            self.rolling_variance.get_params(),
            {
                "window_size": 2,
                "indexes": [],
                "ddof": 1,
            }
        )

    def test_set_params(self):
        self.assertEqual(
            self.rolling_variance.get_params(),
            {
                "window_size": 2,
                "indexes": [],
                "ddof": 1,
            }
        )
        self.rolling_variance.set_params(window_size=5, indexes=["test"], ddof=0)
        self.assertEqual(
            self.rolling_variance.get_params(),
            {
                "window_size": 5,
                "indexes": ["test"],
                "ddof": 0,
            }
        )

    def test_transform_explicit_index(self):
        time = pd.date_range('2000-01-01', freq='24H', periods=6)
        self.rolling_variance.set_params(indexes=["time"])

        ds = xr.Dataset({'foo': ('time', [1., 1., 1., 2., 2., 2.]), 'time': time})

        result = self.rolling_variance.transform(ds)
        expected_result = xr.Dataset({'foo': ('time', [np.nan, 0.0, 0.0, 0.5, 0.0, 0.0]), 'time': time})
        xr.testing.assert_equal(result, expected_result)

    def test_transform(self):
        time = pd.date_range('2000-01-01', freq='24H', periods=6)

        ds = xr.Dataset({'foo': ('time', [1., 1., 1., 2., 2., 2.]), 'time': time})

        result = self.rolling_variance.transform(ds)
        expected_result = xr.Dataset({'foo': ('time', [np.nan, 0.0, 0.0, 0.5, 0.0, 0.0]), 'time': time})
        xr.testing.assert_equal(result, expected_result)

        self.rolling_variance.set_params(ddof=0)
        result = self.rolling_variance.transform(ds)
        expected_result = xr.Dataset({'foo': ('time', [np.nan, 0.0, 0.0, 0.25, 0.0, 0.0]), 'time': time})
        xr.testing.assert_equal(result, expected_result)

    def test_transform_exception(self):
        time = pd.date_range('2000-01-01', freq='24H', periods=7)
        self.rolling_variance.set_params(indexes=["FOO"])

        ds = xr.Dataset({'foo': ('time', [2, 3, 4, 5, 6, 7, 8]), 'time': time})
        with self.assertRaises(WrongParameterException) as context:
            self.rolling_variance.transform(ds)
        self.assertEqual(context.exception.message,
                         "Not all indexes (['FOO']) are in the indexes of x (['time']). "
                         "Perhaps you set the wrong indexes with set_params or during the initialization of "
                         "variance regressor.")
