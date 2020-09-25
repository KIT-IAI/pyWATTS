import unittest
import pandas as pd
import xarray as xr
import numpy as np

from pywatts.modules.missing_value_detection import MissingValueDetector


class TestMissingValueDetector(unittest.TestCase):

    def setUp(self) -> None:
        self.missing_value_detector = MissingValueDetector()

    def tearDown(self) -> None:
        self.missing_value_detector = None

    def test_get_params(self):
        params = self.missing_value_detector.get_params()
        self.assertEqual(bool(params),False)

    def test_set_params(self):
        self.missing_value_detector.set_params()


    def test_transform(self):
        time = pd.to_datetime(['2015-06-03 00:00:00', '2015-06-03 01:00:00',
                               '2015-06-03 02:00:00', '2015-06-03 03:00:00',
                               '2015-06-03 04:00:00'])
        test_data = xr.Dataset({"test": ("time", xr.DataArray([1, 2, np.nan, 4, 5])),
                                "test2": ("time", xr.DataArray([np.nan, 2, 3, 4, 5])),
                                "test3": ("time", xr.DataArray([1, 2, 3, 4, np.nan])),
                                "test4": ("time", xr.DataArray([1, np.nan, np.nan, np.nan, 5])), "time": time})
        test_result = self.missing_value_detector.transform(test_data)

        expected_result = xr.Dataset({"test": ("time", xr.DataArray([False, False, True, False, False])),
                                      "test2": ("time", xr.DataArray([True, False, False, False, False])),
                                      "test3": ("time", xr.DataArray([False, False, False, False, True])),
                                      "test4": ("time", xr.DataArray([False, True, True, True, False])), "time": time})
        xr.testing.assert_equal(test_result,expected_result)

