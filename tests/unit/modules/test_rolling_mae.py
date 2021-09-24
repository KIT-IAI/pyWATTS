import unittest

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pywatts.core.exceptions.input_not_available import InputNotAvailable
from pywatts.modules import RollingMAE


class TestRollingMAE(unittest.TestCase):

    def setUp(self) -> None:
        self.rolling_mae = RollingMAE()

    def tearDown(self) -> None:
        self.rolling_mae = None

    def test_get_params(self):
        self.assertEqual(self.rolling_mae.get_params(), {'window_size': 24, 'window_size_unit': "d"})

    def test_set_params(self):
        self.rolling_mae.set_params(window_size=2, window_size_unit="m")
        self.assertEqual(self.rolling_mae.get_params(),
                         {"window_size": 2, "window_size_unit": "m"})

    def test_transform(self):
        self.rolling_mae.set_params(window_size=2, window_size_unit="h")
        time = pd.to_datetime(['2015-06-03 00:00:00', '2015-06-03 01:00:00',
                               '2015-06-03 02:00:00', '2015-06-03 03:00:00',
                               '2015-06-03 04:00:00'])

        test_data = xr.Dataset({"testCol": ("time", xr.DataArray([-2, -1, 0, 1, 2]).data),
                                "predictCol1": ("time", xr.DataArray([2, -3, 3, 1, -2]).data),
                                "predictCol2": ("time", xr.DataArray([4, 4, 3, -2, 1]).data), "time": time})

        test_result = self.rolling_mae.transform(y=test_data['testCol'], gt=test_data['testCol'],
                                                 pred1=test_data['predictCol1'],
                                                 pred2=test_data['predictCol2'])
        expected_result = xr.DataArray(np.array([[0.0, 4, 6],
                                                 [0.0, 3, 5.5],
                                                 [0.0, 2.5, 4],
                                                 [0.0, 1.5, 3],
                                                 [0.0, 2, 2]]),
                                       coords={"time": time, "predictions": ["gt", "pred1", "pred2"]},
                                       dims=["time", "predictions"])


        xr.testing.assert_allclose(test_result, expected_result)

    def test_transform_without_predictions(self):
        time = pd.to_datetime(['2015-06-03 00:00:00', '2015-06-03 01:00:00',
                               '2015-06-03 02:00:00', '2015-06-03 03:00:00',
                               '2015-06-03 04:00:00'])

        test_data = xr.Dataset({"testCol": ("time", xr.DataArray([-2, -1, 0, 1, 2]).data),
                                "predictCol1": ("time", xr.DataArray([2, -3, 3, 1, -2]).data),
                                "predictCol2": ("time", xr.DataArray([4, 4, 3, -2, 1]).data), "time": time})

        with pytest.raises(InputNotAvailable) as e_info:
            self.rolling_mae.transform(y=test_data['testCol'])

        self.assertEqual(e_info.value.message,
                         "No predictions are provided as input for the RollingMAE. You should add the predictions by a"
                         " seperate key word arguments if you add the RollingMAE to the pipeline.")
