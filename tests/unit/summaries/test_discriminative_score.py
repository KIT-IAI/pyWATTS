from unittest.mock import MagicMock, patch, call
from unittest import TestCase

import numpy as np
import pandas as pd
import xarray as xr
import pywatts.summaries
from pywatts.summaries import DiscriminativeScore


class TestDiscriminativeScore(TestCase):
    def setUp(self) -> None:
        self.get_model_mock = MagicMock()
        self.discriminative_score = DiscriminativeScore(repetitions=42, get_model=self.get_model_mock)

    def tearDown(self) -> None:
        self.discriminative_score = None

    def test_get_set_params(self):
        new_model = MagicMock()
        self.assertEqual(
            self.discriminative_score.get_params(),
            {"repetitions": 42, "get_model": self.get_model_mock, "test_size": 0.3,
             "name":"DiscriminativeScore",
             "fit_kwargs": {"epochs": 10, "validation_split": 0.2}}
        )
        self.discriminative_score.set_params(get_model=new_model, test_size=0.8, fit_kwargs={}, repetitions=1)
        self.assertEqual(
            self.discriminative_score.get_params(),
            {"repetitions": 1,"name":"DiscriminativeScore", "get_model": new_model, "test_size": 0.8, "fit_kwargs": {}}
        )

    def test_transform(self):
        get_model_mock = MagicMock()
        model_mock = MagicMock()
        model_mock.predict.return_value = np.array([1, 0,1])
        get_model_mock.return_value = model_mock
        self.discriminative_score.set_params(get_model=get_model_mock)

        time = pd.to_datetime(['2015-06-03 00:00:00', '2015-06-03 01:00:00',
                               '2015-06-03 02:00:00', '2015-06-03 03:00:00',
                               '2015-06-03 04:00:00'])

        test_data = xr.Dataset({"testCol": (["time", "horizon"], xr.DataArray([[1], [1], [0], [1], [0]]).data),
                                "predictCol1": (["time", "horizon"], xr.DataArray([[2], [-3], [3], [1], [-2]]).data),
                                "predictCol2": (["time", "horizon"], xr.DataArray([[-2], [-1], [0], [1], [2]]).data),
                                "time": time})

        fm_mock = MagicMock()

        self.discriminative_score.transform(file_manager=fm_mock, gt=test_data['testCol'],
                                     pred1=test_data['predictCol1'], pred2=test_data['predictCol2'])

        self.assertEqual(2 * 42, get_model_mock.call_count)
        model_mock.fit.assert_called()
        model_mock.predict.assert_called()

    @patch('pywatts.summaries.discriminative_score.cloudpickle')
    @patch("builtins.open")
    def test_save_load(self, open_mock, cloudpickle_mock):
        json = self.discriminative_score.save(fm=MagicMock())

        reloaded = DiscriminativeScore.load(json)
        params = self.discriminative_score.get_params()

        params["get_model"] = cloudpickle_mock.load()
        params["fit_kwargs"] = cloudpickle_mock.load()

        self.assertEqual(
            params,
            reloaded.get_params()
        )

        self.assertEqual(cloudpickle_mock.dump.call_count, 2)
