import unittest

import numpy as np
import pandas as pd
import xarray as xr
from pywatts.modules import StatisticExtraction, StatisticFeature

from unittest.mock import MagicMock
class TestStatisticExtraction(unittest.TestCase):

    def test_get_set_params(self):
        statistic_extraction = StatisticExtraction()
        self.assertEqual(
            {"dim": "horizon",
             "name":"statistics",
             "features": [StatisticFeature.min, StatisticFeature.max, StatisticFeature.std, StatisticFeature.mean]},
            statistic_extraction.get_params()
        )
        statistic_extraction.set_params(dim=2, features=[StatisticFeature.min])
        self.assertEqual(
            {"dim": 2,
             "name": "statistics",
             "features": [StatisticFeature.min]},
            statistic_extraction.get_params()
        )

    def test_transform(self):
        time = pd.date_range('2000-01-01', freq='24H', periods=3)
        da = xr.DataArray([[0, 0], [1, 3], [1, 5]],
                                       dims=["time", "horizon"], coords={"time": time})
        statistic_extraction = StatisticExtraction(dim="horizon")
        expected_result = xr.DataArray([[0, 0, 0,0], [1, 3, 1,2], [1, 5, 2,3]],
                                       dims=["time", "stat_features"], coords={"time": time, "stat_features": [1, 2, 3, 4]})

        result = statistic_extraction.transform(da)
        xr.testing.assert_equal(result, expected_result)

    def test_save_load(self):
        statistic_extraction = StatisticExtraction(dim="horizon", features=[StatisticFeature.max])
        json = statistic_extraction.save(fm=MagicMock())

        statistic_extraction_reloaded = StatisticExtraction.load(json)

        self.assertEqual(
            statistic_extraction.get_params(),
            statistic_extraction_reloaded.get_params()
        )
