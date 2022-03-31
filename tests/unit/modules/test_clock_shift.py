import unittest
import pandas as pd
import numpy as np
import xarray as xr

from pywatts.core.exceptions.wrong_parameter_exception import WrongParameterException
from pywatts.modules import ClockShift


class TestClockShift(unittest.TestCase):
    def setUp(self):
        self.clock_shift = ClockShift(lag=2)

    def tearDown(self):
        self.clock_shift = None

    def test_get_params(self):
        params = self.clock_shift.get_params()

        self.assertEqual({
            "lag": 2,
            "indexes": None
        }, params)

    def test_set_params(self):
        params = self.clock_shift.get_params()

        self.assertEqual({
            "lag": 2,
            "indexes": None
        }, params)

        self.clock_shift.set_params(indexes=["time"], lag=24)
        params = self.clock_shift.get_params()

        self.assertEqual({
            "lag": 24,
            "indexes": ["time"]
        }, params)

    def test_transform(self):
        time = pd.date_range('2000-01-01', freq='24H', periods=7)

        ds = xr.Dataset({'foo': ('time', [2, 3, 4, 5, 6, 7, 8]), 'time': time})

        result = self.clock_shift.transform(ds)

        time = pd.date_range('2000-01-01', freq='24H', periods=7)
        expected_result = xr.Dataset({'foo': ('time', [np.nan, np.nan, 2., 3., 4.5, 5., 6.]), 'time': time})

        self.assertEqual(result.values(), expected_result)

    def test_transform_exception(self):
        time = pd.date_range('2000-01-01', freq='24H', periods=7)
        self.clock_shift.set_params(indexes=["FOO"])

        ds = xr.Dataset({'foo': ('time', [2, 3, 4, 5, 6, 7, 8]), 'time': time})
        with self.assertRaises(WrongParameterException) as context:
            self.clock_shift.transform(ds)
        self.assertEqual(context.exception.message,
                         "Not all indexes (['FOO']) are in the indexes of x (['time']). "
                         "Perhaps you set the wrong indexes with set_params or during the initialization of the ClockShift.")

    def test_get_min_data(self):
        clock_shift = ClockShift(lag=24)
        self.assertEqual(clock_shift.get_min_data(), 24)
