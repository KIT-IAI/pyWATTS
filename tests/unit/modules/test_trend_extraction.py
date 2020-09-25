import unittest

import pandas as pd
import xarray as xr

from pywatts.modules.trend_extraction import TrendExtraction


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

        ds = xr.Dataset({'foo': ('time', [2, 3, 4, 5, 6, 7, 8, 9, 10, 42]), 'time': time})

        result = self.trend_extractor.transform(ds)

        expected_result = xr.Dataset(
            {'foo_trend': (['length', 'time'], [[0, 0, 2, 3, 4, 5, 6, 7, 8, 9],
                                                [0, 0, 0, 0, 2, 3, 4, 5, 6, 7],
                                                [0, 0, 0, 0, 0, 0, 2, 3, 4, 5]]),
        'time': time})

        xr.testing.assert_equal(result, expected_result)
