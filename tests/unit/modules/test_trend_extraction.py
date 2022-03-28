import unittest

import pandas as pd
import xarray as xr

from pywatts.modules import TrendExtraction


class TestTrendExtraction(unittest.TestCase):
    def setUp(self) -> None:
        self.trend_extractor = TrendExtraction(period=2, length=3)

    def tearDown(self) -> None:
        self.trend_extractor = None

    def test_get_params(self):
        params = self.trend_extractor.get_params()

        self.assertEqual({
            "period": 2,
            "length": 3,
            "indexes": []
        }, params)

    def test_set_params(self):
        self.assertEqual(self.trend_extractor.get_params(),
                         {
                             "period": 2,
                             "length": 3,
                             "indexes": []
                         })
        self.trend_extractor.set_params(indexes=["Foo"], period=12, length=5)
        self.assertEqual(self.trend_extractor.get_params(),
                         {
                             "period": 12,
                             "length": 5,
                             "indexes": ["Foo"]
                         })

    def test_transform(self):
        time = pd.date_range('2000-01-01', freq='24H', periods=10)

        da = xr.DataArray([2, 3, 4, 5, 6, 7, 8, 9, 10, 42], dims=["time"], coords={'time': time})

        result = self.trend_extractor.transform(da)

        expected_result = xr.DataArray(
            [[0, 0, 0], [0, 0, 0], [2, 0, 0], [3, 0, 0], [4, 2, 0], [5, 3, 0], [6, 4, 2], [7, 5, 3], [8, 6, 4],
             [9, 7, 5]], dims=["time", "length"], coords={'time': time})

        xr.testing.assert_equal(result, expected_result)

    def test_get_min_data(self):
        trend = TrendExtraction(period=12, length=2)
        self.assertEqual(trend.get_min_data(), 36)
