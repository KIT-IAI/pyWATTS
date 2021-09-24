import unittest

from pywatts.modules import ChangeDirection
import pandas as pd
import xarray as xr
import numpy as np


class TestChangeDirection(unittest.TestCase):

    def setUp(self) -> None:
        self.change_direction = ChangeDirection()

    def tearDown(self) -> None:
        self.change_direction = None

    def test_transform(self):
        time = pd.date_range('2000-01-01', freq='24H', periods=7)

        ds = xr.Dataset({'foo': ('time', [2, 3, 4, 3, 3, 1, 2]), 'time': time})

        result = self.change_direction.transform(ds)

        time = pd.date_range('2000-01-01', freq='24H', periods=7)
        expected_result = xr.Dataset({'foo': ('time', [np.nan, 1, 1, -1, 0, -1, 1]), 'time': time})

        xr.testing.assert_equal(result, expected_result)

    def test_get_params(self):
        self.assertEqual({}, self.change_direction.get_params())
