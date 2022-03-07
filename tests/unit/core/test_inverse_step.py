import unittest
from unittest.mock import MagicMock

from pywatts.core.exceptions.kind_of_transform_does_not_exist_exception import \
    KindOfTransformDoesNotExistException, \
    KindOfTransform
from pywatts.core.inverse_step import InverseStep
import pandas as pd
import xarray as xr


class TestInverseTransform(unittest.TestCase):

    def setUp(self) -> None:
        self.inverse_module = MagicMock()
        self.input_step = MagicMock()
        time = pd.date_range('2000-01-01', freq='24H', periods=7)

        self.inverse_module.inverse_transform.return_value = xr.DataArray(
            [[5, 5], [5, 5], [4, 5], [5, 4], [6, 5], [7, 6], [8, 7]],
            dims=["time", "horizon"], coords={"time": time, "horizon": [0, 1]})
        self.input_step._should_stop.return_value = False
        self.inverse_step = InverseStep(self.inverse_module, {"input": self.input_step}, file_manager=MagicMock())
        self.inverse_step_result = MagicMock()
        self.input_step.get_result.return_value = self.inverse_step_result

    def tearDown(self) -> None:
        self.inverse_module = None
        self.input_step = None
        self.inverse_step = None

    def test_get_result(self):
        # This test checks if the get_result methods works corerctly, i.e. if it returns the correct result of the step and
        # calculate it if necessary.
        self.inverse_step.get_result(pd.Timestamp("2000.01.01"), None)
        self.inverse_module.inverse_transform.assert_called_once_with(input=self.input_step.get_result())

    def test_get_result_stop(self):
        self.input_step.get_result.return_value = None
        self.inverse_step.get_result(pd.Timestamp("2000.01.01"), None)

        self.inverse_module.inverse_transform.assert_not_called()
        self.assertTrue(self.inverse_step._should_stop(None, None))

    def test_transform_no_inverse_method(self):
        self.inverse_module.has_inverse_transform = False
        self.inverse_module.name = "Magic"

        with self.assertRaises(KindOfTransformDoesNotExistException) as context:
            self.inverse_step.get_result(None, None)
        self.assertEqual(f"The module Magic has no inverse transform", context.exception.message)
        self.assertEqual(KindOfTransform.INVERSE_TRANSFORM, context.exception.method)

        self.inverse_module.inverse_transform.assert_not_called()
