import unittest
import pandas as pd
import xarray as xr

from unittest.mock import MagicMock

from pywatts.core.start_step import StartStep


class TestStartStep(unittest.TestCase):

    def test_load(self):
        params = {
            "index": "SomeIndex",
            "target_ids": {},
            "input_ids": {},
            "id": -1,
            'default_run_setting': {'computation_mode': 4},
            "module": "pywatts.core.start_step",
            "class": "StartStep",
            "name": "StartStep",
            "last": False
        }
        step = StartStep(None).load(params, None, None, None, None)
        json = step.get_json("file")
        self.assertEqual(params, json)

    def test_get_result_copying(self):
        # Tests if the get_result method calls correctly the previous step and the module

        input_step = MagicMock()
        input_step_result_mock = MagicMock()
        input_step.get_result.return_value = input_step_result_mock
        input_step._should_stop.return_value = False

        time = pd.date_range('2000-01-01', freq='1H', periods=7)
        step = StartStep("x")
        step.buffer = {"x": xr.DataArray([2, 3, 4, 3, 3, 1, 2], dims=["time"],
                                         coords={'time': time})}
        result1 = step.get_result(None, None)
        result2 = step.get_result(None, None)

        # result1 and result2 has to be different objects
        result1[1] = 20

        self.assertNotEqual(result1[1], result2[1])
