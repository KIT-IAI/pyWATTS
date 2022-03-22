import unittest

import numpy as np
import pandas as pd
import xarray as xr

from pywatts.modules import Resampler


class TestResampler(unittest.TestCase):
    def setUp(self):
        self.start_time = "2000-01-01"
        self.times = 31 * 24
        self.load_data = np.random.randint(40, 80, (self.times))
        self.linear_data = np.arange(0, self.times)
        self.dataset = xr.Dataset({
                # example load data with two dimensions
                "load": (["time", "giga_watts"], self.load_data.reshape(-1, 1)),
                # example load as usual one dimensional vector
                "load[GW]": ("time", self.load_data),
                "linear": ("time", self.linear_data)
            }, coords={
                "time": pd.date_range(self.start_time, freq="H", periods=self.times)
            }
        )

    def tearDown(self):
        self.load_data = None
        self.linear_data = None
        self.dataset = None

    def _test_base(self, data_index, target_time, method, expected_time, expected_data,
                   time_index="time", method_args=None):
        # resample data using resampler object
        resampler = Resampler(time_index=time_index, target_time=target_time,
                              method=method, method_args=method_args)
        resampled_dataset = resampler.transform(self.dataset)

        # check if resampled data matches expected ones
        self.assertTrue((resampled_dataset.time == expected_time).all())
        self.assertTrue(((resampled_dataset[data_index].values.flatten() - expected_data) ** 2 < 0.01).all())

    def test_get_set_params(self):
        # define parameters and check if they are set correctly
        params = {
            "time_index": "[TARGET_INDEX]",
            "target_time": "[TARGET_TIME]",
            "method": "[METHOD]",
            "method_args": {"arg1": 0},
        }
        resampler = Resampler(**params)
        self.assertTrue(params == resampler.get_params())

        # define new params and try to set them
        new_params = {
            "time_index": "[NEW_TARGET_INDEX]",
            "target_time": "[NEW_TARGET_TIME]",
            "method": "[NEW_METHOD]",
            "method_args": {"arg1": 0, "arg2": 1},
        }
        resampler.set_params(**new_params)
        params.update(new_params)
        self.assertTrue(params == resampler.get_params())

    def test_downscaling_mean(self):
        # test mean resampling for one and two dimensional dataset
        target_times = [("6H", 6), ("12H", 12), ("1D", 24)]
        for target_time, divider in target_times:
            # calculate expected time and load vector
            expected_time = pd.date_range(self.start_time, freq=target_time, periods=self.times / divider)
            expected_data = np.mean(self.load_data.reshape(-1, divider), axis=1)
            for index in ["load", "load[GW]"]:
                self._test_base(index, target_time, "mean", expected_time, expected_data)

    def test_downscaling_median(self):
        # test median resampling for one and two dimensional dataset
        target_times = [("6H", 6), ("12H", 12), ("1D", 24)]
        for target_time, divider in target_times:
            # calculate expected time and load vector
            expected_time = pd.date_range(self.start_time, freq=target_time, periods=self.times / divider)
            expected_data = np.median(self.load_data.reshape(-1, divider), axis=1)
            for index in ["load", "load[GW]"]:
                self._test_base(index, target_time, "median", expected_time, expected_data)

    def test_upscaling_interpolation(self):
        # test interpolation resampling
        # calculate expected time and load vector
        expected_data = np.arange(0, self.times - 0.5, step=0.5)
        expected_time = pd.date_range(self.start_time, freq="1800s", periods=len(expected_data))
        self._test_base("linear", "1800s", "interpolate", expected_time, expected_data, method_args={"kind": "linear"})

    def test_get_min_data(self):
        resampler = Resampler(target_time="12h")
        self.assertEqual(resampler.get_min_data(), pd.Timedelta(hours=12))
