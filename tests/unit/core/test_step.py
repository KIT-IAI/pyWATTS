import os
import unittest
from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd
import xarray as xr

from pywatts.core.computation_mode import ComputationMode
from pywatts.core.step import Step


class TestStep(unittest.TestCase):
    def setUp(self) -> None:
        self.module_mock = MagicMock()
        self.module_mock.name = "test"
        self.step_mock = MagicMock()
        self.step_mock.stop = False
        self.step_mock.id = 2

    def tearDown(self) -> None:
        self.module_mock = None
        self.step_mock = None

    def test_fit(self):
        step = Step(self.module_mock, self.step_mock, file_manager=MagicMock())
        input_mock = MagicMock()

        step._fit(input_mock, None)

        self.module_mock.fit.assert_called_once_with(input_mock, None)

    @patch("builtins.open")
    @patch("pywatts.core.step.cloudpickle")
    def test_store_load_of_step_with_condition(self, cloudpickle_mock, open_mock):
        condition_mock = MagicMock()
        step = Step(self.module_mock, self.step_mock, None, condition=condition_mock)
        fm_mock = MagicMock()
        fm_mock.get_path.return_value = os.path.join("folder", "test_condition.pickle")
        json = step.get_json(fm_mock)
        reloaded_step = Step.load(json, [self.step_mock], targets=None, module=self.module_mock,
                                  file_manager=MagicMock())

        # One call in load and one in save
        open_mock.assert_has_calls(
            [call(os.path.join("folder", "test_condition.pickle"), "wb"),
             call(os.path.join("folder", "test_condition.pickle"), "rb")],
            any_order=True)
        self.assertEqual(json, {
            "target_ids": [],
            "input_ids": [2],
            "id": -1,
            'computation_mode': 4,
            "train_if": None,
            "module": "pywatts.core.step",
            "class": "Step",
            "name": "test",
            "last": True,
            'plot': False,
            'to_csv': False,
            'condition': os.path.join("folder", "test_condition.pickle")}, json)

        self.assertEqual(reloaded_step.module, self.module_mock)
        self.assertEqual(reloaded_step.inputs, [self.step_mock])
        cloudpickle_mock.load.assert_called_once_with(open_mock().__enter__.return_value)
        cloudpickle_mock.dump.assert_called_once_with(condition_mock, open_mock().__enter__.return_value)

    @patch("builtins.open")
    @patch("pywatts.core.step.cloudpickle")
    def test_store_load_of_step_with_train_if(self, cloudpickle_mock, open_mock):
        train_if_mock = MagicMock()
        step = Step(self.module_mock, self.step_mock, None, train_if=train_if_mock)
        fm_mock = MagicMock()
        fm_mock.get_path.return_value = os.path.join("folder", "test_train_if.pickle")
        json = step.get_json(fm_mock)
        reloaded_step = Step.load(json, [self.step_mock], targets=None, module=self.module_mock,
                                  file_manager=MagicMock())

        # One call in load and one in save
        open_mock.assert_has_calls(
            [call(os.path.join("folder", "test_train_if.pickle"), "wb"),
             call(os.path.join("folder", "test_train_if.pickle"), "rb")],
            any_order=True)
        self.assertEqual(json, {
            "target_ids": [],
            "input_ids": [2],
            "id": -1,
            'computation_mode': 4,
            "train_if": os.path.join("folder", "test_train_if.pickle"),
            "module": "pywatts.core.step",
            "class": "Step",
            "name": "test",
            "last": True,
            'plot': False,
            'to_csv': False,
            'condition': None}, json),

        self.assertEqual(reloaded_step.module, self.module_mock)
        self.assertEqual(reloaded_step.inputs, [self.step_mock])
        cloudpickle_mock.load.assert_called_once_with(open_mock().__enter__.return_value)
        cloudpickle_mock.dump.assert_called_once_with(train_if_mock, open_mock().__enter__.return_value)

    @patch("pywatts.core.base_step._get_time_indeces", return_value=["time"])
    @patch("pywatts.core.base_step.xr")
    def test_transform_batch_with_existing_buffer(self, xr_mock, *args):
        input_step = MagicMock()
        input_step.stop = False
        transform_result = MagicMock()
        self.module_mock.transform.return_value = transform_result
        already_existing_buffer = MagicMock()

        step = Step(self.module_mock, input_step, file_manager=MagicMock())
        step.buffer = already_existing_buffer

        step.get_result(None, None)

        # Two calls, once in should_stop and once in _transform
        xr_mock.concat.assert_called_once_with([already_existing_buffer, transform_result], dim="time")
        input_step.get_result.assert_has_calls([call(None, None), call(None, None)])

    def test_get_result(self):
        input_step = MagicMock()
        input_step.stop = False
        input_step_result = MagicMock()
        input_step.get_result.return_value = input_step_result

        step = Step(self.module_mock, input_step, file_manager=MagicMock())
        step.get_result(None, None)

        # Two calls, once in should_stop and once in _transform
        input_step.get_result.assert_has_calls([call(None, None), call(None, None)])

    def test_further_elements_input_false(self):
        input_step = MagicMock()
        input_step.further_elements.return_value = False
        step = Step(self.module_mock, input_step, file_manager=MagicMock())
        result = step.further_elements("2000.12.12")

        input_step.further_elements.assert_called_once_with("2000.12.12")
        self.assertFalse(result)

    def test_further_elements_target_false(self):
        target_step = MagicMock()
        target_step.further_elements.return_value = False
        step = Step(self.module_mock, self.step_mock, target=target_step, file_manager=MagicMock())
        result = step.further_elements("2000.12.12")
        target_step.further_elements.assert_called_once_with("2000.12.12")
        self.assertFalse(result)

    def test_further_elements_already_buffered(self):
        time = pd.date_range('2000-01-01', freq='24H', periods=7)

        ds = xr.Dataset({'foo': ('time', [2, 3, 4, 5, 6, 7, 8]), 'time': time})
        step = Step(self.module_mock, self.step_mock, file_manager=MagicMock())
        step.buffer = ds
        result = step.further_elements(pd.Timestamp("2000-01-05"))
        self.step_mock.further_elements.assert_not_called()
        self.assertEqual(result, True)

    def test_input_finished(self):
        step = Step(self.module_mock, self.step_mock, file_manager=MagicMock())
        input_mock = MagicMock()
        input_mock.has_further_elements.return_value = [False]
        step._fit(input_mock, None)

        self.module_mock.fit.assert_called_once_with(input_mock, None)

    def test_fit_with_targets(self):
        target_mock = MagicMock()
        input_mock = MagicMock()

        step = Step(self.module_mock, self.step_mock, MagicMock(), target=MagicMock())
        step._fit(input_mock, target_mock)

        self.module_mock.fit.assert_called_once_with(input_mock, target_mock)

    def test_transform(self):
        step = Step(self.module_mock, self.step_mock, None)

        input_mock = MagicMock()
        step._fit(input_mock, None)
        step._transform(input_mock)
        self.module_mock.transform.assert_called_once_with(input_mock)

    def test_load(self):
        step_config = {
            "target_ids": [],
            "input_ids": [2],
            'computation_mode': 3,
            "id": -1,
            "module": "pywatts.core.step",
            "class": "Step",
            "condition": None,
            "train_if": None,
            "name": "test",
            "last": False,
            "to_csv": False,
            "plot": False
        }
        step = Step.load(step_config, [self.step_mock], None, self.module_mock, None)

        self.assertEqual(step.name, "test")
        self.assertEqual(step.id, -1)
        self.assertEqual(step.get_json("file"), step_config)
        self.assertEqual(step.module, self.module_mock)

    def test_get_json(self):
        step = Step(self.module_mock, self.step_mock, None)
        json = step.get_json("file")
        self.assertEqual({
            "target_ids": [],
            "input_ids": [2],
            'condition': None,
            'train_if': None,
            "id": -1,
            'computation_mode': 4,
            "module": "pywatts.core.step",
            "class": "Step",
            "name": "test",
            "last": True,
            'plot': False,
            'to_csv': False}, json)

    @patch("pywatts.utils.lineplot.plt.close")
    @patch("pywatts.utils.lineplot.plt.clf")
    @patch("pywatts.utils.lineplot.plt.savefig")
    @patch("pywatts.utils.lineplot.plt.Figure")
    def test_plot(self, mock_figure, mock_save, mock_clf, mock_close):
        fm_mock = MagicMock()
        step = Step(self.module_mock, self.step_mock, fm_mock)
        step._plot(xr.Dataset({"test": xr.DataArray(np.asarray([1, 2, 3, 4, 5]))}))
        mock_figure.assert_called_once()
        fm_mock.get_path.assert_called_with("test_test.png")
        mock_figure().suptitle.assert_called_with("test")
        mock_save.assert_called_once()
        mock_clf.assert_called_once()
        mock_close.assert_called_once()

    def test_to_csv(self):
        fm_mock = MagicMock()
        fm_mock.get_path.return_value = "path/test.csv"
        dataset_mock = MagicMock()
        df_mock = MagicMock()
        dataset_mock.to_dataframe.return_value = df_mock

        step = Step(self.module_mock, self.step_mock, fm_mock, to_csv=True)

        # perform to csv and check results
        step._to_csv(dataset_mock)

        fm_mock.get_path.assert_called_with("test.csv")
        dataset_mock.to_dataframe.assert_called_once()

        df_mock.to_csv.assert_called_once_with("path/test.csv", sep=";")

    def test_set_computation_mode(self):
        step = Step(MagicMock(), MagicMock(), MagicMock())
        step.set_computation_mode(ComputationMode.FitTransform)

        assert step.computation_mode == ComputationMode.FitTransform

        step.set_computation_mode(ComputationMode.Transform)
        assert step.computation_mode == ComputationMode.Transform

    def test_set_computation_mode_specified(self):
        step = Step(MagicMock(), MagicMock(), MagicMock(), computation_mode=ComputationMode.FitTransform)
        assert step.computation_mode == ComputationMode.FitTransform

        step.set_computation_mode(ComputationMode.Transform)
        assert step.computation_mode == ComputationMode.FitTransform

    def test_reset(self):
        step = Step(MagicMock(), MagicMock(), MagicMock())
        buffer_mock = MagicMock()
        step.buffer = MagicMock()
        step.computation_mode = ComputationMode.Transform
        step.stop = True
        step.finished = True
        step.reset()

        xr.testing.assert_equal(step.buffer, xr.Dataset())
        assert step.computation_mode == ComputationMode.Default
        assert step.stop == False
        assert step.finished == False
