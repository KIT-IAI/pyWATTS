import unittest
from unittest.mock import MagicMock

from pywatts.core.exceptions.kind_of_transform_does_not_exist_exception import \
    KindOfTransformDoesNotExistException, KindOfTransform
from pywatts.core.probabilistic_step import ProbablisticStep
import pandas as pd


class TestProbabilisticStep(unittest.TestCase):

    def setUp(self) -> None:
        self.probabilistic_module = MagicMock()
        self.input_step = MagicMock()
        self.input_step.get_result.return_value = MagicMock(), False
        self.input_step._should_stop.return_value = False
        self.probabilistic_step = ProbablisticStep(self.probabilistic_module, {"x": self.input_step},
                                                   file_manager=MagicMock())

    def tearDown(self) -> None:
        self.probabilistic_module = None
        self.input_step = None
        self.probabilistic_step = None

    def test_transform(self):
        input_mock = MagicMock()
        self.probabilistic_step._transform(input_mock)

        self.probabilistic_module.predict_proba.assert_called_once_with(input_mock)

    def test_get_result_stop(self):
        self.input_step.get_result.return_value = None

        self.probabilistic_step.get_result(pd.Timestamp("2000.01.01"), pd.Timestamp("2000.01.02"))

        self.probabilistic_module.predict_proba.assert_not_called()
        self.assertTrue(self.probabilistic_step._should_stop(pd.Timestamp("2000.01.01"), pd.Timestamp("2000.01.02")))

    def test_transform_no_prob_method(self):
        self.probabilistic_module.has_predict_proba = False
        self.probabilistic_module.name = "Magic"

        with self.assertRaises(KindOfTransformDoesNotExistException) as context:
            self.probabilistic_step.get_result(None, None)
        self.assertEqual(f"The module Magic has no probablisitic transform", context.exception.message)
        self.assertEqual(KindOfTransform.PROBABILISTIC_TRANSFORM, context.exception.method)

        self.probabilistic_module.predict_proba.assert_not_called()
