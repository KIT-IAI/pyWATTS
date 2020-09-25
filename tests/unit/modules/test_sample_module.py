import unittest
import pandas as pd
import xarray as xr

from pywatts.core.exceptions.wrong_parameter_exception import WrongParameterException
from pywatts.modules.sample_module import Sampler


class TestSampler(unittest.TestCase):
    def setUp(self) -> None:
        self.sampler = Sampler(sample_size=2)

    def tearDown(self) -> None:
        self.sampler = None

    def test_get_params(self):
        params = self.sampler.get_params()

        self.assertEqual(params,
                         {
                             "lag": 2,
                             "indeces": [],
                             "data_var_names": []
                         })

    def test_set_params(self):
        self.assertEqual(self.sampler.get_params(),
                         {
                             "lag": 2,
                             "indeces": [],
                             "data_var_names" : []
                         })
        self.sampler.set_params(indexes=["Foo"], sample_size=12, data_var_names=["BAR"])
        self.assertEqual(self.sampler.get_params(),
                         {
                             "lag": 12,
                             "indeces": ["Foo"],
                             "data_var_names": ["BAR"]
                         })

    def test_transform(self):
        time = pd.date_range('2000-01-01', freq='24H', periods=7)

        ds = xr.Dataset({'foo': ('time', [2, 3, 4, 5, 6, 7, 8]), 'time': time})

        result = self.sampler.transform(ds)

        time = pd.date_range('2000-01-01', freq='24H', periods=7)

        expected_result = xr.Dataset(
            {'foo': (['time', 'horizon'], [[2, 0], [3, 2], [4,3], [5,4], [6,5], [7,6], [8,7]]),
             'time': time})

        xr.testing.assert_equal(result, expected_result)

    def test_transform_bar(self):
        self.sampler.set_params(data_var_names=["BAR"])
        time = pd.date_range('2000-01-01', freq='24H', periods=7)

        ds = xr.Dataset({'foo': ('time', [2, 3, 4, 5, 6, 7, 8]), 'time': time})

        result = self.sampler.transform(ds)

        time = pd.date_range('2000-01-01', freq='24H', periods=7)

        expected_result = xr.Dataset(
            {'BAR': (['time', 'horizon'], [[2, 0], [3, 2], [4,3], [5,4], [6,5], [7,6], [8,7]]),
             'time': time})

        xr.testing.assert_equal(result, expected_result)

    def test_transform_exception(self):
        time = pd.date_range('2000-01-01', freq='24H', periods=7)
        self.sampler.set_params(indexes=["FOO"])

        ds = xr.Dataset({'foo': ('time', [2, 3, 4, 5, 6, 7, 8]), 'time': time})
        with self.assertRaises(WrongParameterException) as context:
            self.sampler.transform(ds)
        self.assertEqual(context.exception.message,
                         "Not all indexes (['FOO']) are in the indexes of x (['time']). "
                         "Perhaps you set the wrong indexes with set_params or during the initialization of the Sampler.")

