import unittest
from unittest.mock import MagicMock

from pywatts.core.either_or_step import EitherOrStep


class TestEitherOrStep(unittest.TestCase):
    def setUp(self) -> None:
        self.step_one = MagicMock()
        self.step_one.stop = True
        self.step_one.id = 2
        self.step_two = MagicMock()
        self.result_mock_step_2 = MagicMock()
        self.result_mock_step_1 = MagicMock()
        self.step_two.stop = False
        self.step_two.get_result.return_value = self.result_mock_step_2
        self.step_one.get_result.return_value = self.result_mock_step_1
        self.step_two.id = 1

    def tearDown(self) -> None:
        self.step_two = None
        self.step_one = None

    def test_transform(self):
        self.step_one.get_result.return_value = None
        step = EitherOrStep([self.step_one, self.step_two])
        step.get_result(None, None)
        self.assertEqual(step.buffer, self.result_mock_step_2)

    def test_load(self):
        params = {
            "target_ids": [],
            "input_ids": [2, 1],
            'computation_mode': 4,
            "id": -1,
            "module": "pywatts.core.either_or_step",
            "class": "EitherOrStep",
            "name": "EitherOrStep",
            "last":False
        }
        step = EitherOrStep.load(params, [self.step_one, self.step_two], None, None, None)
        json = step.get_json("file")
        self.assertEqual(params, json)