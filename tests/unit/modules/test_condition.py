import unittest

import pandas as pd
import xarray as xr

from pywatts.modules import Condition


def x_geq_zero(x):
    return x >= 0


def x_leq_zero(x):
    return x <= 0


class TestCondition(unittest.TestCase):
    def setUp(self) -> None:
        self.condition = Condition(condition=x_geq_zero)

    def tearDown(self) -> None:
        self.condition = None

    def test_get_params(self):
        self.assertEqual(
            self.condition.get_params(),
            {
                "condition": x_geq_zero
            }
        )

    def test_set_params(self):
        self.assertEqual(
            self.condition.get_params(),
            {
                "condition": x_geq_zero
            }
        )
        self.condition.set_params(condition=x_leq_zero)
        self.assertEqual(
            self.condition.get_params(),
            {
                "condition": x_leq_zero
            }
        )

    def test_transform(self):
        time = pd.date_range('2002-01-01', freq='24H', periods=7)
        self.condition.set_params(condition=x_geq_zero)
        depend = xr.DataArray([3, 2, 1, 0, -1, -2, -3], dims=["time"], coords={'time': time})
        if_true = xr.DataArray([3, 2, 1, 0, -1, -2, -3], dims=["time"], coords={'time': time})
        if_false = xr.DataArray([0, 0, 0, 0, 0, 0, 0], dims=["time"], coords={'time': time})

        result = self.condition.transform(dependency=depend, if_true=if_true, if_false=if_false)

        expected_result = xr.DataArray([3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0], dims=["time"], coords={'time': time})

        xr.testing.assert_equal(result, expected_result)
