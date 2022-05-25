from unittest.mock import MagicMock, patch, call
from unittest import TestCase

import numpy as np
import pandas as pd
import xarray as xr
import pywatts.summaries
from pywatts.summaries.train_synthetic_test_real import TrainSyntheticTestReal, TSTRTask


class TestTSTRSummary(TestCase):
    def setUp(self) -> None:
        self.get_model_mock = MagicMock()
        self.tstr = TrainSyntheticTestReal(repetitions=42, get_model=self.get_model_mock)

    def tearDown(self) -> None:
        self.tstr = None

    def test_get_set_params(self):
        new_model = MagicMock()
        self.assertEqual(
            self.tstr.get_params(),
            {"repetitions": 42, "get_model": self.get_model_mock, "train_test_split": 0.66, "task": TSTRTask.Regression,
             "metrics": ["rmse", "mae"], "fit_kwargs": {"epochs": 100, "validation_split": 0.2}, "n_targets": 1}
        )
        self.tstr.set_params(n_targets=42, get_model=new_model, train_test_split=0.8, fit_kwargs={}, repetitions=1,
                             task=TSTRTask.Classification, metrics=["accuracy"])
        self.assertEqual(
            self.tstr.get_params(),
            {"repetitions": 1, "get_model": new_model, "train_test_split": 0.8,
             "fit_kwargs": {}, "n_targets": 42, "metrics": ["accuracy"], "task": TSTRTask.Classification}
        )

    def test_transform(self):
        regressor_mock = MagicMock()
        model_mock = MagicMock()
        model_mock.predict.return_value = np.array([2, 0])
        regressor_mock.return_value = model_mock
        self.tstr.set_params(get_model=regressor_mock)

        time = pd.to_datetime(['2015-06-03 00:00:00', '2015-06-03 01:00:00',
                               '2015-06-03 02:00:00', '2015-06-03 03:00:00',
                               '2015-06-03 04:00:00'])

        test_data = xr.Dataset({"testCol": (["time", "horizon"], xr.DataArray([[-2], [-1], [0], [1], [2]]).data),
                                "predictCol1": (["time", "horizon"], xr.DataArray([[2], [-3], [3], [1], [-2]]).data),
                                "predictCol2": (["time", "horizon"], xr.DataArray([[-2], [-1], [0], [1], [2]]).data),
                                "time": time})

        fm_mock = MagicMock()

        self.tstr.transform(file_manager=fm_mock, real=test_data['testCol'],
                            pred1=test_data['predictCol1'], pred2=test_data['predictCol2'])

        self.assertEqual(regressor_mock.call_count, 3 * 42)
        model_mock.fit.assert_called()
        model_mock.predict.assert_called()

    @patch('pywatts.summaries.train_synthetic_test_real.cloudpickle')
    @patch("builtins.open")
    def test_save_load(self, open_mock, cloudpickle_mock):
        json = self.tstr.save(fm=MagicMock())

        reloaded = TrainSyntheticTestReal.load(json)
        params = self.tstr.get_params()

        params["get_model"] = cloudpickle_mock.load()
        params["fit_kwargs"] = cloudpickle_mock.load()

        self.assertEqual(
            params,
            reloaded.get_params()
        )

        self.assertEqual(cloudpickle_mock.dump.call_count, 2)
