import unittest
from unittest.mock import MagicMock

from pywatts.core.exceptions.kind_of_transform_does_not_exist_exception import \
    KindOfTransformDoesNotExistException, \
    KindOfTransform
from pywatts.core.inverse_step import InverseStep


class TestInverseTransform(unittest.TestCase):

    def setUp(self) -> None:
        self.inverse_module = MagicMock()
        self.input_step = MagicMock()
        self.input_step.stop = False
        self.inverse_step = InverseStep(self.inverse_module, self.input_step, file_manager=MagicMock())
        self.inverse_step_result = MagicMock()
        self.input_step.get_result.return_value = self.inverse_step_result

    def tearDown(self) -> None:
        self.inverse_module = None
        self.input_step = None
        self.inverse_step = None

    def test_get_result(self):
        self.inverse_step.get_result(None, None)

        self.inverse_module.inverse_transform.assert_called_once_with(self.input_step.get_result())

    def test_get_result_stop(self):
        self.input_step.stop = True
        self.inverse_step.get_result(None,None)

        self.inverse_module.inverse_transform.assert_not_called()
        self.assertTrue(self.inverse_step.stop)

    def test_transform_no_inverse_method(self):
        self.inverse_module.has_inverse_transform = False
        self.inverse_module.name = "Magic"

        with self.assertRaises(KindOfTransformDoesNotExistException) as context:
            self.inverse_step.get_result(None, None)
        self.assertEqual(f"The module Magic has no inverse transform", context.exception.message)
        self.assertEqual(KindOfTransform.INVERSE_TRANSFORM, context.exception.method)

        self.inverse_module.inverse_transform.assert_not_called()

    def test_to_csv(self):
        fm_mock = MagicMock()
        fm_mock.get_path.return_value = "path/test.csv"
        dataset_mock = MagicMock()
        df_mock = MagicMock()
        dataset_mock.to_dataframe.return_value = df_mock
        self.inverse_module.name = "test"

        step = InverseStep(self.inverse_module, self.input_step, fm_mock, to_csv=True)

        # perform to csv and check results
        step._to_csv(dataset_mock)

        fm_mock.get_path.assert_called_with("test.csv")
        dataset_mock.to_dataframe.assert_called_once()

        df_mock.to_csv.assert_called_once_with("path/test.csv", sep=";")
