import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pywatts.core.exceptions.input_not_available import InputNotAvailable
from pywatts.summaries import RMSE


class TestRMSE(unittest.TestCase):

    def setUp(self) -> None:
        self.rmse = RMSE(name="NAME")

    def tearDown(self) -> None:
        self.rmse = None

    def test_get_params(self):
        self.assertEqual(self.rmse.get_params(),
                         {'offset': 0})

    def test_set_params(self):
        self.rmse.set_params(offset=24)
        self.assertEqual(self.rmse.get_params(),
                         {'offset': 24})

    def test_transform_with_filter(self):
        filter_mock = MagicMock()
        rmse = RMSE(filter_method=filter_mock)

        time = pd.to_datetime(['2015-06-03 00:00:00', '2015-06-03 01:00:00',
                               '2015-06-03 02:00:00', '2015-06-03 03:00:00',
                               '2015-06-03 04:00:00'])

        test_data = xr.Dataset({"testCol": ("time", xr.DataArray([-2, -1, 0, 1, 2]).data),
                                "predictCol1": ("time", xr.DataArray([2, -3, 3, 1, -2]).data),  "time": time})

        filter_mock.return_value = (test_data["predictCol1"].values, test_data["testCol"].values)

        test_result = rmse.transform(file_manager=MagicMock(), y=test_data['testCol'],
                                          pred1=test_data['predictCol1'])

        filter_mock.assert_called_once()

        np.testing.assert_equal(filter_mock.call_args[0][0], test_data["predictCol1"])
        np.testing.assert_equal(filter_mock.call_args[0][1], test_data["testCol"])

        expected_result = '  * pred1: 3.0\n'

        self.assertEqual(test_result, expected_result)
    def test_transform(self):
        time = pd.to_datetime(['2015-06-03 00:00:00', '2015-06-03 01:00:00',
                               '2015-06-03 02:00:00', '2015-06-03 03:00:00',
                               '2015-06-03 04:00:00'])

        test_data = xr.Dataset({"testCol": ("time", xr.DataArray([-2, -1, 0, 1, 2]).data),
                                "predictCol1": ("time", xr.DataArray([2, -3, 3, 1, -2]).data),
                                "predictCol2": ("time", xr.DataArray([4, 4, 3, -2, 1]).data), "time": time})

        test_result = self.rmse.transform(file_manager=MagicMock(), y=test_data['testCol'], gt=test_data['testCol'],
                                          pred1=test_data['predictCol1'],
                                          pred2=test_data['predictCol2'])

        expected_result = '  * gt: 0.0\n  * pred1: 3.0\n  * pred2: 4.0\n'

        self.assertEqual(test_result, expected_result)

    def test_transform_without_predictions(self):
        time = pd.to_datetime(['2015-06-03 00:00:00', '2015-06-03 01:00:00',
                               '2015-06-03 02:00:00', '2015-06-03 03:00:00',
                               '2015-06-03 04:00:00'])

        test_data = xr.Dataset({"testCol": ("time", xr.DataArray([-2, -1, 0, 1, 2]).data),
                                "predictCol1": ("time", xr.DataArray([2, -3, 3, 1, -2]).data),
                                "predictCol2": ("time", xr.DataArray([4, 4, 3, -2, 1]).data), "time": time})

        with pytest.raises(InputNotAvailable) as e_info:
            self.rmse.transform(file_manager=MagicMock(), y=test_data['testCol'])

        self.assertEqual(e_info.value.message,
                         "No predictions are provided as input for the RMSE.  You should add the predictions by a "
                         "seperate key word arguments if you add the RMSE to the pipeline.")

    @patch("builtins.open")
    @patch("pywatts.summaries.metric_base.cloudpickle")
    def test_save(self, cloudpickle_mock, open_mock):
        fm_mock = MagicMock()
        fm_mock.get_path.return_value = "filter_path"
        filter_mock = MagicMock()

        rmse = RMSE(name="NAME", filter_method=filter_mock)

        json = rmse.save(fm_mock)

        fm_mock.get_path.assert_called_once_with("NAME_filter.pickle")
        open_mock.assert_called_once_with("filter_path", "wb")

        cloudpickle_mock.dump.assert_called_once_with(filter_mock, open_mock().__enter__.return_value)
        self.assertEqual(json["filter"], "filter_path")
        self.assertEqual(json["params"], {"offset": 0})


    @patch("builtins.open")
    @patch("pywatts.summaries.metric_base.cloudpickle")
    def test_load(self, cloudpickle_mock, open_mock):
        load_information = {'params': {'offset': 24}, 'name': 'NAME', 'class': 'RMSE',
                            'module': 'pywatts.summaries.rmse_summary', 'filter': 'filter_path'}
        filter_mock = MagicMock()
        cloudpickle_mock.load.return_value = filter_mock

        rmse = RMSE.load(load_information)

        open_mock.assert_called_once_with("filter_path", "rb")
        cloudpickle_mock.load.assert_called_once_with(open_mock().__enter__.return_value)

        self.assertEqual(rmse.name, "NAME")
        self.assertEqual(rmse.filter_method, filter_mock)
        self.assertEqual(rmse.get_params(), {"offset": 24})
