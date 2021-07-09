import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pywatts.core.exceptions.input_not_available import InputNotAvailable
from pywatts.summaries.mae_summary import MAE


class TestMAE(unittest.TestCase):

    def setUp(self) -> None:
        self.mae = MAE(name="NAME")

    def tearDown(self) -> None:
        self.mae = None

    def test_get_params(self):
        self.assertEqual(self.mae.get_params(),
                         {'offset': 0})

    def test_set_params(self):
        self.mae.set_params(offset=24)
        self.assertEqual(self.mae.get_params(),
                         {'offset': 24})

    def test_transform_with_filter(self):
        filter_mock = MagicMock()
        mae = MAE(filter_method=filter_mock)

        time = pd.to_datetime(['2015-06-03 00:00:00', '2015-06-03 01:00:00',
                               '2015-06-03 02:00:00', '2015-06-03 03:00:00',
                               '2015-06-03 04:00:00'])

        test_data = xr.Dataset({"testCol": ("time", xr.DataArray([-2, -1, 0, 1, 2])),
                                "predictCol1": ("time", xr.DataArray([2, -3, 3, 1, -2])),  "time": time})

        filter_mock.return_value = (test_data["predictCol1"].values, test_data["testCol"].values)

        test_result = mae.transform(file_manager=MagicMock(), y=test_data['testCol'],
                                          pred1=test_data['predictCol1'])

        filter_mock.assert_called_once()

        np.testing.assert_equal(filter_mock.call_args[0][0], test_data["predictCol1"])
        np.testing.assert_equal(filter_mock.call_args[0][1], test_data["testCol"])

        expected_result = '  * pred1: 2.6\n'

        self.assertEqual(test_result, expected_result)
    def test_transform(self):
        time = pd.to_datetime(['2015-06-03 00:00:00', '2015-06-03 01:00:00',
                               '2015-06-03 02:00:00', '2015-06-03 03:00:00',
                               '2015-06-03 04:00:00'])

        test_data = xr.Dataset({"testCol": ("time", xr.DataArray([-2, -1, 0, 1, 2])),
                                "predictCol1": ("time", xr.DataArray([2, -3, 3, 1, -2])),
                                "predictCol2": ("time", xr.DataArray([4, 4, 3, -2, 1])), "time": time})

        test_result = self.mae.transform(file_manager=MagicMock(), y=test_data['testCol'], gt=test_data['testCol'],
                                         pred1=test_data['predictCol1'],
                                         pred2=test_data['predictCol2'])

        expected_result = '  * gt: 0.0\n  * pred1: 2.6\n  * pred2: 3.6\n'

        self.assertEqual(test_result, expected_result)

    def test_transform_without_predictions(self):
        time = pd.to_datetime(['2015-06-03 00:00:00', '2015-06-03 01:00:00',
                               '2015-06-03 02:00:00', '2015-06-03 03:00:00',
                               '2015-06-03 04:00:00'])

        test_data = xr.Dataset({"testCol": ("time", xr.DataArray([-2, -1, 0, 1, 2])),
                                "predictCol1": ("time", xr.DataArray([2, -3, 3, 1, -2])),
                                "predictCol2": ("time", xr.DataArray([4, 4, 3, -2, 1])), "time": time})

        with pytest.raises(InputNotAvailable) as e_info:
            self.mae.transform(file_manager=MagicMock(), y=test_data['testCol'])

        self.assertEqual(e_info.value.message,
                         "No predictions are provided as input for the MAE.  You should add the predictions by a "
                         "seperate key word arguments if you add the MAE to the pipeline.")

    @patch("builtins.open")
    @patch("pywatts.summaries.mae_summary.cloudpickle")
    def test_save(self, cloudpickle_mock, open_mock):
        fm_mock = MagicMock()
        fm_mock.get_path.return_value = "filter_path"
        filter_mock = MagicMock()

        mae = MAE(name="NAME", filter_method=filter_mock)

        json = mae.save(fm_mock)

        fm_mock.get_path.assert_called_once_with("NAME_filter.pickle")
        open_mock.assert_called_once_with("filter_path", "wb")

        cloudpickle_mock.dump.assert_called_once_with(filter_mock, open_mock().__enter__.return_value)
        self.assertEqual(json["filter"], "filter_path")
        self.assertEqual(json["params"], {"offset": 0})


    @patch("builtins.open")
    @patch("pywatts.summaries.mae_summary.cloudpickle")
    def test_load(self, cloudpickle_mock, open_mock):
        load_information = {'params': {'offset': 24}, 'name': 'NAME', 'class': 'RMSE',
                            'module': 'pywatts.summaries.mae_summary', 'filter': 'filter_path'}
        filter_mock = MagicMock()
        cloudpickle_mock.load.return_value = filter_mock

        mae = MAE.load(load_information)

        open_mock.assert_called_once_with("filter_path", "rb")
        cloudpickle_mock.load.assert_called_once_with(open_mock().__enter__.return_value)

        self.assertEqual(mae.name, "NAME")
        self.assertEqual(mae.filter_method, filter_mock)
        self.assertEqual(mae.get_params(), {"offset": 24})
