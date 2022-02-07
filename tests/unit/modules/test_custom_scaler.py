import unittest

import pandas as pd
import xarray as xr

from pywatts.modules import CustomScaler
from pywatts.core.exceptions import WrongParameterException


class TestCustomScaler(unittest.TestCase):
    def setUp(self) -> None:
        self.custom_scaler = CustomScaler()

    def tearDown(self) -> None:
        self.custom_scaler = None

    def test_get_params(self):
        self.assertEqual(
            self.custom_scaler.get_params(),
            {
                "multiplier": 1.0,
                "bias": 0.0,
            }
        )

    def test_set_params(self):
        self.assertEqual(
            self.custom_scaler.get_params(),
            {
                "multiplier": 1.0,
                "bias": 0.0,
            }
        )
        self.custom_scaler.set_params(multiplier=2.7, bias=1.2)
        self.assertEqual(
            self.custom_scaler.get_params(),
            {
                "multiplier": 2.7,
                "bias": 1.2,
            }
        )

    def test_transform_bias(self):
        time = pd.date_range('2002-01-01', freq='24H', periods=7)
        self.custom_scaler.set_params(multiplier=None, bias=1.0)
        da = xr.DataArray([2, 3, 4, 5, 6, 7, 8], dims=["time"], coords={'time': time})

        result = self.custom_scaler.transform(da)

        expected_result = xr.DataArray([3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dims=["time"], coords={'time': time})

        xr.testing.assert_equal(result, expected_result)

    def test_transform_multiplier(self):
        time = pd.date_range('2002-01-01', freq='24H', periods=7)
        self.custom_scaler.set_params(multiplier=2.0, bias=None)
        da = xr.DataArray([2, 3, 4, 5, 6, 7, 8], dims=["time"], coords={'time': time})

        result = self.custom_scaler.transform(da)

        expected_result = xr.DataArray([4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0], dims=["time"], coords={'time': time})

        xr.testing.assert_equal(result, expected_result)

    def test_transform_bias_and_multiplier(self):
        time = pd.date_range('2002-01-01', freq='24H', periods=7)
        self.custom_scaler.set_params(multiplier=2.0, bias=1.0)
        da = xr.DataArray([2, 3, 4, 5, 6, 7, 8], dims=["time"], coords={'time': time})

        result = self.custom_scaler.transform(da)

        expected_result = xr.DataArray([5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0], dims=["time"], coords={'time': time})

        xr.testing.assert_equal(result, expected_result)

    def test_inverse_transform_bias(self):
        time = pd.date_range('2002-01-01', freq='24H', periods=7)
        self.custom_scaler.set_params(multiplier=None, bias=1.0)
        da = xr.DataArray([2, 3, 4, 5, 6, 7, 8], dims=["time"], coords={'time': time})

        da = self.custom_scaler.transform(da)
        result = self.custom_scaler.inverse_transform(da)

        expected_result = xr.DataArray([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dims=["time"], coords={'time': time})

        xr.testing.assert_equal(result, expected_result)

    def test_inverse_transform_multiplier(self):
        time = pd.date_range('2002-01-01', freq='24H', periods=7)
        self.custom_scaler.set_params(multiplier=2.0, bias=None)
        da = xr.DataArray([2, 3, 4, 5, 6, 7, 8], dims=["time"], coords={'time': time})

        da = self.custom_scaler.transform(da)
        result = self.custom_scaler.inverse_transform(da)

        expected_result = xr.DataArray([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dims=["time"], coords={'time': time})

        xr.testing.assert_equal(result, expected_result)

    def test_inverse_transform_bias_and_multiplier(self):
        time = pd.date_range('2002-01-01', freq='24H', periods=7)
        self.custom_scaler.set_params(multiplier=2.0, bias=1.0)
        da = xr.DataArray([2, 3, 4, 5, 6, 7, 8], dims=["time"], coords={'time': time})

        da = self.custom_scaler.transform(da)
        result = self.custom_scaler.inverse_transform(da)

        expected_result = xr.DataArray([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dims=["time"], coords={'time': time})

        xr.testing.assert_equal(result, expected_result)

    def test_set_params_exception(self):
        with self.assertRaises(WrongParameterException) as context:
            self.custom_scaler.set_params(multiplier=0)
        self.assertEqual(context.exception.message,
                         "Multiplication by 0 is not possible. "
                         "During initialisation set a multiplier different to 0.")
