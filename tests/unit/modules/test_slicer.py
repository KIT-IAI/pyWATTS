import unittest

import numpy as np
import pandas as pd
import xarray as xr

from pywatts.modules import Slicer


class TestSlicer(unittest.TestCase):
    def setUp(self):
        self.start_time = "2000-01-01"
        self.times = 365
        self.data = np.arange(0, self.times)
        self.dataset = xr.Dataset({
                "data": ("time", self.data)
            }, coords={
                "time": pd.date_range(self.start_time, freq="D", periods=self.times)
            }
        )

    def tearDown(self):
        self.data = None
        self.dataset = None

    def test_set_get_params(self):
        # check parameters a set in __init__ and returned in get_params correctly
        # also set new parameters via set_params and check get_params
        init_params = {
            "start": None,
            "end": 2
        }
        obj = Slicer(**init_params)
        self.assertEqual(obj.get_params(), init_params)

        test_params = [
            {
                "start": 1,
                "end": 2
            },
            {
                "start": 1,
                "end": None
            },
            {
                "start": None,
                "end": None
            },
        ]
        for new_params in test_params:
            obj.set_params(**new_params)
            self.assertEqual(obj.get_params(), new_params)

    def test_start_only(self):
        slicer = Slicer(start=0)
        data = slicer.transform(x=self.dataset["data"])
        np.testing.assert_array_equal(self.dataset["data"].values, data.values)
        slicer.set_params(start=1)
        data = slicer.transform(x=self.dataset["data"])
        np.testing.assert_array_equal(self.dataset["data"][1:].values, data.values)

    def test_end_only(self):
        slicer = Slicer(end=-1)
        data = slicer.transform(x=self.dataset["data"])
        np.testing.assert_array_equal(self.dataset["data"][:-1].values, data.values)
        slicer.set_params(end=-2)
        data = slicer.transform(x=self.dataset["data"])
        np.testing.assert_array_equal(self.dataset["data"][:-2].values, data.values)

    def test_both_none(self):
        slicer = Slicer()
        data = slicer.transform(x=self.dataset["data"])
        np.testing.assert_array_equal(self.dataset["data"].values, data.values)

    def test_start_end(self):
        slicer = Slicer(start=0, end=-1)
        data = slicer.transform(x=self.dataset["data"])
        np.testing.assert_array_equal(self.dataset["data"][:-1].values, data.values)
        slicer.set_params(start=1, end=-1)
        data = slicer.transform(x=self.dataset["data"])
        np.testing.assert_array_equal(self.dataset["data"][1:-1].values, data.values)
