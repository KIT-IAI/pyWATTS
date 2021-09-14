import unittest
from unittest.mock import MagicMock

from pywatts.core.either_or_step import EitherOrStep
import xarray as xr
import pandas as pd


class TestEitherOrStep(unittest.TestCase):
    def setUp(self) -> None:
        self.step_one = MagicMock()
        self.step_one.id = 1
        self.step_two = MagicMock()
        self.result_mock_step_2 = MagicMock()
        self.result_mock_step_1 = MagicMock()
        time = pd.date_range('2000-01-01', freq='24H', periods=7)

        self.da2 = xr.DataArray([[2, 0], [3, 2], [4, 3], [5, 4], [6, 5], [7, 6], [8, 7]],
                                dims=["time", "horizon"], coords={"time": time, "horizon": [0, 1]})
        self.da1 = xr.DataArray([[5, 5], [5, 5], [4, 5], [5, 4], [6, 5], [7, 6], [8, 7]],
                                dims=["time", "horizon"], coords={"time": time, "horizon": [0, 1]})
        self.step_two.get_result.return_value = self.da2
        self.step_one.get_result.return_value = self.da1
        self.step_two.id = 2

    def tearDown(self) -> None:
        self.step_two = None
        self.step_one = None

    def test_transform(self):
        self.step_one.get_result.return_value = None
        step = EitherOrStep({"option1": self.step_one, "option2": self.step_two})
        step.get_result(None, None)
        xr.testing.assert_equal(step.buffer["EitherOr"], self.da2)

    def test_load(self):
        params = {
            "target_ids": {},
            "input_ids": {2: "stepTwo", 1: "stepOne"},
            'default_run_setting': {'computation_mode': 4},
            "id": -1,
            "module": "pywatts.core.either_or_step",
            "class": "EitherOrStep",
            "name": "EitherOrStep",
            "last": False
        }
        step = EitherOrStep.load(params, {"stepOne": self.step_one, "stepTwo": self.step_two}, None, None, None)
        json = step.get_json("file")
        self.assertEqual(params, json)
