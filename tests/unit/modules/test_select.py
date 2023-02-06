import unittest
import pandas as pd
import pytest
import xarray as xr
from pywatts_pipeline.core.exceptions import WrongParameterException

from pywatts.modules.preprocessing.select import Select


class TestSelect(unittest.TestCase):
    def setUp(self) -> None:
        self.select = Select(start=1, stop=4, step=1)

    def tearDown(self) -> None:
        self.select = None

    def test_get_params(self):
        params = self.select.get_params()

        self.assertEqual(params,
                         {
                             "start": 1,
                             "stop": 4,
                             "step": 1,
                         })

    def test_set_params(self):
        self.assertEqual(self.select.get_params(),
                         {
                             "start": 1,
                             "stop": 4,
                             "step": 1,
                         })
        self.select.set_params(start=3, stop=20, step=12)
        self.assertEqual(self.select.get_params(),
                         {
                             "start": 3,
                             "stop": 20,
                             "step": 12,
                         })

    def test_transform(self):
        time = pd.date_range('2000-01-01', freq='24H', periods=7)
        da = xr.DataArray([1, 2, 3, 4, 5, 6, 7], dims=['time'], coords={"time": time})

        for i, (select_params, expected) in enumerate([
            ({"start": 1, "stop": 4, "step": 1},
             xr.DataArray([[2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]],
                          dims=["time", "horizon"],
                          coords={"time": pd.date_range('2000-01-01', freq='24H', periods=4)})),
            ({"start": 1, "stop": 4, "step": 10},

             # Only one value is in a slicing object with step 10 between start 1 and stop 4
             xr.DataArray([2, 3, 4, 5, 6, 7],
                          dims=["time"],
                          coords={"time": pd.date_range('2000-01-01', freq='24H', periods=6)})),
            ({"start": -2, "stop": 1, "step": 1},
             xr.DataArray([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]],
                          dims=["time", "horizon"],
                          coords={"time": pd.date_range('2000-01-03', freq='24H', periods=5)})),
            ({"start": -2, "stop": 1, "step": None},
             xr.DataArray([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]],
                          dims=["time", "horizon"],
                          coords={"time": pd.date_range('2000-01-03', freq='24H', periods=5)})),
            ({"start": 2, "stop": None, "step": None},
             xr.DataArray([3, 4, 5, 6, 7],
                          dims=["time"],
                          coords={"time": pd.date_range('2000-01-01', freq='24H', periods=5)})),
            ({"start": [2, 3], "stop": None, "step": None},
             xr.DataArray([[3, 4], [4, 5], [5, 6], [6, 7]],
                          dims=["time", "horizon"],
                          coords={"time": pd.date_range('2000-01-01', freq='24H', periods=4)}))]):
            with self.subTest(i=i):
                select = Select(**select_params)
                result = select.transform(da)
                xr.testing.assert_equal(result, expected)

    def test_exceptions(self):
        for i, (select_params, expected) in enumerate([
            ({"start": 1.0, "stop": 2, "step": 1},
             "The start parameter is a float. But it needs to be an integer or a list."
             ),
            ({"start": 2, "stop":2, "step": 1},
             "The start parameter is greater than or equals stop. But it needs to be smaller than stop."
             )
        ]):
            with self.subTest(i=i):
                with self.assertRaises(WrongParameterException) as context:
                    self.select.set_params(**select_params)
                self.assertEqual(context.exception.message, expected)


    def test_get_min_data(self):
        select = Select(start=-24)
        self.assertEqual(select.get_min_data(), 24)
