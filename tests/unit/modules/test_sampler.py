import unittest
import pandas as pd
import xarray as xr

from pywatts.core.exceptions.wrong_parameter_exception import WrongParameterException
from pywatts.modules import Sampler


class TestSampler(unittest.TestCase):
    def setUp(self) -> None:
        self.sampler = Sampler(sample_size=3)

    def tearDown(self) -> None:
        self.sampler = None

    def test_get_params(self):
        params = self.sampler.get_params()

        self.assertEqual(params,
                         {
                             "sample_size": 3,
                             "indexes": [],
                         })

    def test_set_params(self):
        self.assertEqual(self.sampler.get_params(),
                         {
                             "sample_size": 3,
                             "indexes": [],
                         })
        self.sampler.set_params(indexes=["Foo"], sample_size=12)
        self.assertEqual(self.sampler.get_params(),
                         {
                             "sample_size": 12,
                             "indexes": ["Foo"],
                         })

    def test_transform(self):
        time = pd.date_range('2000-01-01', freq='24H', periods=7)

        da = xr.DataArray([1, 2, 3, 4, 5, 6, 7], dims=['time'], coords={"time": time})

        self.sampler.set_params(sample_size=3)
        result = self.sampler.transform(da)

        time = pd.date_range('2000-01-01', freq='24H', periods=7)

        expected_result = xr.DataArray([[0, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]],
                                       dims=["time", "horizon"], coords={"time": time})

        xr.testing.assert_equal(result, expected_result)

    def test_transform_exception(self):
        time = pd.date_range('2000-01-01', freq='24H', periods=7)
        self.sampler.set_params(indexes=["FOO"])

        da = xr.DataArray([2, 3, 4, 5, 6, 7, 8], dims=['time'], coords={"time": time})
        with self.assertRaises(WrongParameterException) as context:
            self.sampler.transform(da)
        self.assertEqual(context.exception.message,
                         "Not all indexes (['FOO']) are in the indexes of x (['time']). "
                         "Perhaps you set the wrong indexes with set_params or during the initialization of the Sampler.")

    def test_set_params_exception(self):
        with self.assertRaises(WrongParameterException) as context:
            self.sampler.set_params(sample_size=-3)
        self.assertEqual(context.exception.message,
                         "Sample size cannot be less than or equal to zero. "
                         "Please define a sample size greater than zero.")

    def test_get_min_data(self):
        sampler = Sampler(sample_size=24)
        self.assertEqual(sampler.get_min_data(), 24)
