from unittest.mock import MagicMock, patch, call
from unittest import TestCase

import numpy as np
import pandas as pd
import xarray as xr
import pywatts.summaries
from pywatts.summaries import TSNESummary
from pywatts.summaries.confusion_matrix import ConfusionMatrix


class TestTSNESummary(TestCase):
    def setUp(self) -> None:
        self.confusion_matrix = ConfusionMatrix()

    def tearDown(self) -> None:
        self.confusion_matrix = None

    def test_get_params(self):
        self.assertEqual(
            self.confusion_matrix.get_params(),
            {}
        )

    def test_transform_all(self):
        time = pd.to_datetime(['2015-06-03 00:00:00', '2015-06-03 01:00:00',
                               '2015-06-03 02:00:00', '2015-06-03 03:00:00',
                               '2015-06-03 04:00:00'])

        test_data = xr.Dataset({"testCol": (["time", "horizon"], xr.DataArray([[0], [1], [0], [1], [1]]).data),
                                "predictCol1": (["time", "horizon"], xr.DataArray([[1], [1], [1], [1], [1]]).data),
                                "predictCol2": (["time", "horizon"], xr.DataArray([[0], [0], [0], [0], [0]]).data),
                                "time": time})

        test_result = self.confusion_matrix.transform(file_manager=MagicMock(), gt=test_data['testCol'],
                                                      pred1=test_data['predictCol1'], pred2=test_data['predictCol2'])

        np.testing.assert_equal(
            test_result.k_v["pred1"],
            np.array([[0, 2], [0, 3]])
        )

        np.testing.assert_equal(
            test_result.k_v["pred2"],
            np.array([[2, 0], [3, 0]])
        )
