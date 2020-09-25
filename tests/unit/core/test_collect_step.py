import unittest
import warnings
from unittest.mock import MagicMock, patch, call

from pywatts.core.collect_step import CollectStep
import pandas as pd
import xarray as xr


class TestCollectStep(unittest.TestCase):
    def setUp(self) -> None:
        self.step_one = MagicMock()
        self.step_one.stop = False
        self.step_one.id = 2
        self.step_one_result = MagicMock()
        self.merged_result = MagicMock()
        self.step_one_result.merge.return_value = self.merged_result

        self.merged_result.indexes = {"time": xr.DataArray([pd.Timestamp("2000.12.24 12:00:00").to_numpy()])}

        self.step_one.get_result.return_value = self.step_one_result

        self.step_two = MagicMock()
        self.step_two_result = MagicMock()
        self.step_two.get_result.return_value = self.step_two_result
        self.step_two.stop = False
        self.step_two.id = 1

    def tearDown(self) -> None:
        self.step_two = None
        self.step_one = None

    def test_get_result(self):
        step = CollectStep([self.step_one, self.step_two])

        step.get_result(None, None)
        self.step_one_result.merge.assert_called_once_with(self.step_two_result)

        # Two calls, since get_result is called by further_elements and _transform
        self.step_two.get_result.assert_has_calls([call(None, None), call(None, None)])
        assert step.finished == True

    @patch("pywatts.core.base_step._get_time_indeces", return_value=["time"])
    def test_get_result_batches(self, get_time_indeces_mock):
        step = CollectStep([self.step_one, self.step_two])

        step.get_result(pd.Timestamp("2000-12-24 12:00:00"), pd.Timestamp("2000-12-24 14:00:00"))

        get_time_indeces_mock.assert_called_with(step.buffer)
        self.step_one_result.merge.assert_called_once_with(self.step_two_result)
        # Two calls, since get_result is called by further_elements and _transform
        self.step_two.get_result.assert_has_calls(
            [call(pd.Timestamp("2000-12-24 12:00:00"), pd.Timestamp("2000-12-24 14:00:00")),
            call(pd.Timestamp("2000-12-24 12:00:00"), pd.Timestamp("2000-12-24 14:00:00"))])
        assert step.finished == False

    def test_get_result_naming_conflict(self):
        time = pd.date_range('2000-01-01', freq='24H', periods=7)

        ds_one = xr.Dataset({'foo': ('time', [2, 3, 4, 5, 6, 7, 8]), 'time': time})
        ds_two = xr.Dataset({'foo': ('time', [2, 2, 2, 2, 2, 2, 2]), 'time': time})
        ds_result = xr.Dataset({'foo': ('time', [2, 3, 4, 5, 6, 7, 8]),
                                'foo_0': ('time', [2, 2, 2, 2, 2, 2, 2]),
                                'time': time})
        self.step_one.get_result.return_value = ds_one
        self.step_two.get_result.return_value = ds_two

        step = CollectStep([self.step_one, self.step_two])
        with warnings.catch_warnings(record=True) as w:
            step.get_result(None, None)

        self.assertEqual(len(w), 1)
        self.assertEqual(str(w[-1].message), "There was a naming conflict. Therefore, we renamed:{'foo': 'foo_0'}")
        result = step.buffer
        self.assertEqual(list(result.data_vars.keys()), ["foo", "foo_0"])

        # Two calls, since get_result is called by further_elements and _transform
        self.step_one.get_result.assert_has_calls([call(None, None), call(None, None)])
        self.step_two.get_result.assert_has_calls([call(None, None), call(None, None)])

        xr.testing.assert_equal(step.buffer, ds_result)

    def test_load(self):
        params = {
            "target_ids": [],
            "input_ids": [2, 1],
            'computation_mode': 4,
            "id": -1,
            "module": "pywatts.core.collect_step",
            "class": "CollectStep",
            "name": "CollectStep",
            "last": False
        }
        step = CollectStep.load(params, [self.step_one, self.step_two], None, None, None)
        json = step.get_json("file_path")
        self.assertEqual(params, json)
