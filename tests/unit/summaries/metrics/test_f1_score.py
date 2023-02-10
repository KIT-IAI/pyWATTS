import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pywatts_pipeline.core.exceptions.input_not_available import InputNotAvailable
from pywatts_pipeline.core.exceptions.invalid_input_exception import InvalidInputException
from pywatts.summaries.f1_summary import F1Score
from tests.unit.summaries.metrics.test_base_metric_base import BaseTestMetricBase


class TestF1Score(BaseTestMetricBase, unittest.TestCase):
    load_information = {'params': {'offset': 24}, 'name': 'NAME', 'class': 'F1',
                        'module': 'pywatts.summaries.f1_score_summary', 'filter': 'filter_path'}

    def get_metric(self):
        return F1Score

    @patch("pywatts.summaries.f1_summary.f1_score")
    def test_transform_with_filter(self, f1_score_mock):
        filter_mock = MagicMock()
        f1_score = F1Score(filter_method=filter_mock, average="blub")

        time = pd.to_datetime(['2015-06-03 00:00:00', '2015-06-03 01:00:00',
                               '2015-06-03 02:00:00', '2015-06-03 03:00:00',
                               '2015-06-03 04:00:00'])
        test_data = xr.Dataset({"testCol": ("time", xr.DataArray([-2, -1, 0, 1, 2]).data),
                                "predictCol1": ("time", xr.DataArray([2, -3, 3, 1, -2]).data),  "time": time})
        filter_mock.return_value = (test_data["predictCol1"].values, test_data["testCol"].values)
        f1_score.transform(file_manager=MagicMock(), y=test_data['testCol'],
                                          pred1=test_data['predictCol1'])

        filter_mock.assert_called_once()
        f1_score_mock.assert_called_once()
        np.testing.assert_equal(
            f1_score_mock.call_args[0][0],
            np.array([2, -3, 3, 1, -2])
        )
        np.testing.assert_equal(
            f1_score_mock.call_args[0][1],
            np.array([-2, -1, 0, 1, 2])
        )
        self.assertEqual(
            f1_score_mock.call_args[1],
            {"average": "blub"}
        )

    @patch("pywatts.summaries.f1_summary.f1_score")
    def test_transform_with_wrong_shape_solvable_by_reshaping(self, f1_score_mock):
        time = pd.to_datetime(['2015-06-03 00:00:00', '2015-06-03 01:00:00',
                               '2015-06-03 02:00:00', '2015-06-03 03:00:00',
                               '2015-06-03 04:00:00'])

        test_data = xr.Dataset({"testCol": ("time", xr.DataArray([-2, -1, 0, 1, 2]).data),
                                "predictCol1": (("time", "dim1"), xr.DataArray([[2], [-3], [3], [1], [-2]]).data),
                                "predictCol2": (("time", "dim1"), xr.DataArray([[4], [4], [3], [-2], [1]]).data), "time": time})

        self.metric.transform(file_manager=MagicMock(), y=test_data['testCol'], gt=test_data['testCol'],
                                         pred1=test_data['predictCol1'],
                                         pred2=test_data['predictCol2'])

        self.assertEqual(f1_score_mock.call_count, 3)

    def test_transform_with_invalid_shape(self):
        time = pd.to_datetime(['2015-06-03 00:00:00', '2015-06-03 01:00:00',
                               '2015-06-03 02:00:00', '2015-06-03 03:00:00',
                               '2015-06-03 04:00:00'])

        test_data = xr.Dataset({"testCol": ("time", xr.DataArray([-2, -1, 0, 1, 2]).data),
                                "predictCol1": (("time", "dim12"), xr.DataArray([[2, 2], [-3, 2], [3, 2], [1, 2], [-2,2 ]]).data),
                                "predictCol2": (("time", "dim1"), xr.DataArray([[4], [4], [3], [-2], [1]]).data), "time": time})

        with self.assertRaises(InvalidInputException) as cm:
            self.metric.transform(file_manager=MagicMock(), y=test_data['testCol'], gt=test_data['testCol'],
                                         pred1=test_data['predictCol1'],
                                         pred2=test_data['predictCol2'])

        self.assertEqual(cm.exception.message,
                         "The prediction pred1 does not match to the shape of the ground truth y in the instance NAME of class F1Score.")


    @patch("pywatts.summaries.f1_summary.f1_score")
    def test_transform_cutouts(self, f1_score_mock):
        time = pd.to_datetime(['2015-06-03 00:00:00', '2015-06-03 01:00:00',
                               '2015-06-03 02:00:00', '2015-06-03 03:00:00',
                               '2015-06-03 04:00:00'])

        test_data = xr.Dataset({"testCol": ("time", xr.DataArray([-2, -1, 0, 1, 2]).data),
                                "predictCol1": ("time", xr.DataArray([2, -3, 3, 1, -2]).data),
                                "predictCol2": ("time", xr.DataArray([4, 4, 3, -2, 1]).data), "time": time})

        self.metric.set_params(cuts=[(pd.Timestamp('2015-06-03 01:00:00'), pd.Timestamp('2015-06-03 03:00:00'))])
        test_result = self.metric.transform(file_manager=MagicMock(), y=test_data['testCol'],
                                             pred1=test_data['predictCol1'])
        self.assertEqual(f1_score_mock.call_count, 2)


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
                         "No predictions are provided as input for the F1Score.  You should add the predictions by a "
                         "seperate key word arguments if you add the F1Score to the pipeline.")

    def test_set_params(self):
        self.metric.set_params(offset=24, cuts=[("Test", "test")], average="BLUB")
        self.assertEqual(self.metric.get_params(),
                         {'offset': 24,
                          "filter_method":None,
                          "average" : "BLUB",
                          "cuts": [("Test", "test")]})

    def get_default_params(self):
        return {'offset': 0, "cuts":[], "average":"micro", "filter_method": None}

    def get_load_params(self):
        return {'offset': 24, "cuts":[], "average":"micro", "filter_method": None}
