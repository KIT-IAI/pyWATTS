import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import xarray as xr

from pywatts_pipeline.core.exceptions.wrong_parameter_exception import WrongParameterException
from pywatts.modules.postprocessing.merger import Merger


class TestMerger(unittest.TestCase):
    def setUp(self) -> None:
        self.merger = Merger()

    def tearDown(self) -> None:
        self.merger = None

    def test_get_set_params(self):
        self.assertEqual(
            self.merger.get_params(),
            {"method": "mean",
            'name': 'merger'}
        )
        self.merger.set_params(method=-5)
        self.assertEqual(
            self.merger.get_params(),
            {"method": -5,
            'name': 'merger'}
        )

    def test_set_params_invalid_params(self):
        self.assertRaises(WrongParameterException, self.merger.set_params, method="Foo")

    def test_init_with_invalid_params(self):
        self.assertRaises(WrongParameterException, Merger, method="Foo")

    def test_transform_mean(self):
        time = pd.date_range('2000-01-01', freq='24H', periods=7)
        da = xr.DataArray([[0, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]],
                                       dims=["time", "horizon"], coords={"time": time})
        self.merger.set_params(method="mean")
        result = self.merger.transform(da)

        expected_result = xr.DataArray([np.nan, np.nan, 1, 2, 3, 4, 5],
                                       dims=["time"], coords={"time": time})
        xr.testing.assert_equal(result, expected_result)

    def test_transform_median(self):
        time = pd.date_range('2000-01-01', freq='24H', periods=7)
        da = xr.DataArray([[0, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]],
                                       dims=["time", "horizon"], coords={"time": time})
        self.merger.set_params(method="median")
        result = self.merger.transform(da)

        expected_result = xr.DataArray([np.nan, np.nan, 1, 2, 3, 4, 5],
                                       dims=["time"], coords={"time": time})
        xr.testing.assert_equal(result, expected_result)

    def test_transform_integer(self):
        time = pd.date_range('2000-01-01', freq='24H', periods=7)
        da = xr.DataArray([[0, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]],
                                       dims=["time", "horizon"], coords={"time": time})
        self.merger.set_params(method=1)
        result = self.merger.transform(da)

        expected_result = xr.DataArray([0, 1, 2, 3, 4, 5, 6],
                                       dims=["time"], coords={"time": time})
        xr.testing.assert_equal(result, expected_result)

    def test_transform_negative_integer(self):
        time = pd.date_range('2000-01-01', freq='24H', periods=7)
        da = xr.DataArray([[0, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]],
                          dims=["time", "horizon"], coords={"time": time})
        self.merger.set_params(method=-1)
        result = self.merger.transform(da)

        expected_result = xr.DataArray([1, 2, 3, 4, 5, 6, 7],
                                       dims=["time"], coords={"time": time})
        xr.testing.assert_equal(result, expected_result)

    def test_transform_integer_too_big(self):
        time = pd.date_range('2000-01-01', freq='24H', periods=7)
        da = xr.DataArray([[0, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]],
                          dims=["time", "horizon"], coords={"time": time})
        self.merger.set_params(method=3)
        result = self.merger.transform(da)

        expected_result = xr.DataArray([1, 2, 3, 4, 5, 6, 7],
                                       dims=["time"], coords={"time": time})
        xr.testing.assert_equal(result, expected_result)

    def test_transform_integer_too_small(self):
        time = pd.date_range('2000-01-01', freq='24H', periods=7)
        da = xr.DataArray([[0, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]],
                                       dims=["time", "horizon"], coords={"time": time})
        self.merger.set_params(method=-10)
        result = self.merger.transform(da)

        expected_result = xr.DataArray([0, 0, 1, 2, 3, 4, 5],
                                       dims=["time"], coords={"time": time})
        xr.testing.assert_equal(result, expected_result)

    def test_save_load(self):
        merger = Merger(method=10)
        json = merger.save(fm=MagicMock())

        merger_reloaded = Merger.load(json)

        self.assertEqual(
            merger.get_params(),
            merger_reloaded.get_params()
        )
