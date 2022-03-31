import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pywatts.core.exceptions.input_not_available import InputNotAvailable
from pywatts.summaries import MAPE
from tests.unit.summaries.metrics.test_base_metric_base import BaseTestMetricBase


class TestMASE(BaseTestMetricBase, unittest.TestCase):
    load_information = {'params': {'offset': 24, "cuts": []}, 'name': 'NAME', 'class': 'RMSE',
                        'module': 'pywatts.summaries.mae_summary', 'filter': 'filter_path'}

    def test_transform_with_filter(self):
        filter_mock = MagicMock()
        mae = MAPE(filter_method=filter_mock)

        time = pd.to_datetime(['2015-06-03 00:00:00', '2015-06-03 01:00:00',
                               '2015-06-03 02:00:00', '2015-06-03 03:00:00',
                               '2015-06-03 04:00:00'])

        test_data = xr.Dataset({"testCol": ("time", xr.DataArray([-2, -1, 0, 1, 2]).data),
                                "predictCol1": ("time", xr.DataArray([2, -3, 3, 1, -2]).data),  "time": time})

        filter_mock.return_value = (test_data["predictCol1"].values, test_data["testCol"].values)

        test_result = mae.transform(file_manager=MagicMock(), y=test_data['testCol'],
                                          pred1=test_data['predictCol1'])

        filter_mock.assert_called_once()

        np.testing.assert_equal(filter_mock.call_args[0][0], test_data["predictCol1"])
        np.testing.assert_equal(filter_mock.call_args[0][1], test_data["testCol"])

        self.assertEqual(test_result.k_v, {"pred1": 3/2 * 100})

    def test_transform(self):
        time = pd.to_datetime(['2015-06-03 00:00:00', '2015-06-03 01:00:00',
                               '2015-06-03 02:00:00', '2015-06-03 03:00:00',
                               '2015-06-03 04:00:00'])

        test_data = xr.Dataset({"testCol": ("time", xr.DataArray([-2, -1, 0, 1, 2]).data),
                                "predictCol1": ("time", xr.DataArray([2, -3, 3, 1, -2]).data),
                                "predictCol2": ("time", xr.DataArray([4, 4, 3, -2, 1]).data), "time": time})

        test_result = self.metric.transform(file_manager=MagicMock(), y=test_data['testCol'], gt=test_data['testCol'],
                                          pred1=test_data['predictCol1'],
                                          pred2=test_data['predictCol2'])
        expected_result = {"gt": 0.0, "pred1": 3/2 * 100, "pred2": 23/8 * 100}

        self.assertEqual(test_result.k_v, expected_result)

    def test_transform_cutouts(self):
        time = pd.to_datetime(['2015-06-03 00:00:00', '2015-06-03 01:00:00',
                               '2015-06-03 02:00:00', '2015-06-03 03:00:00',
                               '2015-06-03 04:00:00'])

        test_data = xr.Dataset({"testCol": ("time", xr.DataArray([-2, -1, 0, 1, 2]).data),
                                "predictCol1": ("time", xr.DataArray([2, -3, 3, 1, -2]).data),
                                "predictCol2": ("time", xr.DataArray([4, 4, 3, -2, 1]).data), "time": time})

        self.metric.set_params(cuts=[(pd.Timestamp('2015-06-03 01:00:00'), pd.Timestamp('2015-06-03 03:00:00'))])
        test_result = self.metric.transform(file_manager=MagicMock(), y=test_data['testCol'],
                                          pred1=test_data['predictCol1'])
        expected_result = {"pred1": 3/2 * 100,
                           "pred1: Cut from 2015-06-03 01:00:00 to 2015-06-03 03:00:00" : (2/1 + 0/1)/2 * 100}

        self.assertEqual(test_result.k_v, expected_result)


    def test_transform_without_predictions(self):
        time = pd.to_datetime(['2015-06-03 00:00:00', '2015-06-03 01:00:00',
                               '2015-06-03 02:00:00', '2015-06-03 03:00:00',
                               '2015-06-03 04:00:00'])

        test_data = xr.Dataset({"testCol": ("time", xr.DataArray([-2, -1, 0, 1, 2]).data),
                                "predictCol1": ("time", xr.DataArray([2, -3, 3, 1, -2]).data),
                                "predictCol2": ("time", xr.DataArray([4, 4, 3, -2, 1]).data), "time": time})

        with pytest.raises(InputNotAvailable) as e_info:
            self.metric.transform(file_manager=MagicMock(), y=test_data['testCol'])

        self.assertEqual(e_info.value.message,
                         "No predictions are provided as input for the MAPE.  You should add the predictions by a "
                         "seperate key word arguments if you add the MAPE to the pipeline.")

    def get_metric(self):
        return MAPE

