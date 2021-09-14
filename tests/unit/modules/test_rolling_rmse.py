import unittest

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pywatts.core.exceptions.input_not_available import InputNotAvailable
from pywatts.modules import RollingRMSE


class TestRollingRMSE(unittest.TestCase):

    def setUp(self) -> None:
        self.rolling_rmse = RollingRMSE()

    def tearDown(self) -> None:
        self.rolling_rmse = None

    def test_get_params(self):
        self.assertEqual(self.rolling_rmse.get_params(), {'window_size': 24, 'window_size_unit': "d"})

    def test_set_params(self):
        self.rolling_rmse.set_params(window_size=2, window_size_unit="m")
        self.assertEqual(self.rolling_rmse.get_params(),
                         {"window_size": 2, "window_size_unit": "m"})

    def test_transform(self):
        self.rolling_rmse.set_params(window_size=2, window_size_unit="h")
        time = pd.to_datetime(['2015-06-03 00:00:00', '2015-06-03 01:00:00',
                               '2015-06-03 02:00:00', '2015-06-03 03:00:00',
                               '2015-06-03 04:00:00'])

        test_data = xr.Dataset({"testCol": ("time", xr.DataArray([-2, -1, 0, 1, 2]).data),
                                "predictCol1": ("time", xr.DataArray([2, -3, 3, 1, -2]).data),
                                "predictCol2": ("time", xr.DataArray([4, 4, 3, -2, 1]).data), "time": time,
                                "timeTest": time})

        y = xr.DataArray(np.array([-2, -1, 0, 1, 2]),
                         coords={"timeTest": time},
                         dims=["timeTest"])

        test_result = self.rolling_rmse.transform(y=y, gt=test_data['testCol'],
                                                  pred1=test_data['predictCol1'],
                                                  pred2=test_data['predictCol2'])
        expected_result = xr.DataArray(np.array([[0.0, 4, 6],
                                                 [0.0, np.sqrt(10), np.sqrt(30.5)],
                                                 [0.0, np.sqrt(6.5), np.sqrt(17)],
                                                 [0.0, np.sqrt(4.5), 3],
                                                 [0.0, np.sqrt(8), np.sqrt(5)], ]),
                                       coords={"timeTest": time, "predictions": ["gt", "pred1", "pred2"]},
                                       dims=["timeTest", "predictions"])

        xr.testing.assert_allclose(test_result, expected_result)

    def test_transform_without_predictions(self):
        time = pd.to_datetime(['2015-06-03 00:00:00', '2015-06-03 01:00:00',
                               '2015-06-03 02:00:00', '2015-06-03 03:00:00',
                               '2015-06-03 04:00:00'])

        test_data = xr.Dataset({"testCol": ("time", xr.DataArray([-2, -1, 0, 1, 2]).data),
                                "predictCol1": ("time", xr.DataArray([2, -3, 3, 1, -2]).data),
                                "predictCol2": ("time", xr.DataArray([4, 4, 3, -2, 1]).data), "time": time})

        with pytest.raises(InputNotAvailable) as e_info:
            self.rolling_rmse.transform(y=test_data['testCol'])

        self.assertEqual(e_info.value.message,
                         "No predictions are provided as input for the RollingRMSE. You should add the predictions by a"
                         " seperate key word arguments if you add the RollingRMSE to the pipeline.")
