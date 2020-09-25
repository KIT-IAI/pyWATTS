import unittest
from unittest.mock import MagicMock

from pywatts.core.exceptions.kind_of_transform_does_not_exist_exception import \
    KindOfTransformDoesNotExistException, KindOfTransform
from pywatts.core.probabilistic_step import ProbablisticStep


class TestProbabilisticStep(unittest.TestCase):

    def setUp(self) -> None:
        self.probabilistic_module = MagicMock()
        self.input_step = MagicMock()
        self.input_step.get_result.return_value = MagicMock(), False
        self.input_step.stop = False
        self.probabilistic_step = ProbablisticStep(self.probabilistic_module, self.input_step, file_manager=MagicMock())

    def tearDown(self) -> None:
        self.probabilistic_module = None
        self.input_step = None
        self.probabilistic_step = None

    def test_transform(self):
        input_mock = MagicMock()
        self.probabilistic_step._transform(input_mock)

        self.probabilistic_module.predict_proba.assert_called_once_with(input_mock)

    def test_get_result_stop(self):
        self.input_step.stop = True
        self.probabilistic_step.get_result(None, None)

        self.probabilistic_module.predict_proba.assert_not_called()
        self.assertTrue(self.probabilistic_step.stop)

    def test_transform_no_prob_method(self):
        self.probabilistic_module.has_predict_proba = False
        self.probabilistic_module.name = "Magic"

        with self.assertRaises(KindOfTransformDoesNotExistException) as context:
            self.probabilistic_step.get_result(None, None)
        self.assertEqual(f"The module Magic has no probablisitic transform", context.exception.message)
        self.assertEqual(KindOfTransform.PROBABILISTIC_TRANSFORM, context.exception.method)

        self.probabilistic_module.predict_proba.assert_not_called()

    def test_to_csv(self):
        fm_mock = MagicMock()
        fm_mock.get_path.return_value = "path/test.csv"
        dataset_mock = MagicMock()
        df_mock = MagicMock()
        dataset_mock.to_dataframe.return_value = df_mock
        self.probabilistic_module.name = "test"

        step = ProbablisticStep(self.probabilistic_module, self.input_step, fm_mock, to_csv=True)

        # perform to csv and check results
        step._to_csv(dataset_mock)

        fm_mock.get_path.assert_called_with("test.csv")
        dataset_mock.to_dataframe.assert_called_once()

        df_mock.to_csv.assert_called_once_with("path/test.csv", sep=";")
