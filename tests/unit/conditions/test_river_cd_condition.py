import unittest
from unittest.mock import MagicMock, call

import pandas as pd
import pytest
import xarray as xr

from pywatts.conditions.cd_condition import RiverDriftDetectionCondition
from pywatts_pipeline.core.exceptions.step_creation_exception import StepCreationException


class TestRiverCDCondition(unittest.TestCase):
    def test_evaluate(self):
        time = pd.date_range('2000-01-01', freq='24H', periods=4)

        y = xr.DataArray([1, 2, 3, 4], dims=['time'], coords={"time": time})
        y_hat = xr.DataArray([2, 3, 4, 5], dims=['time'], coords={"time": time})

        drift_detector_mock = MagicMock()
        drift_detector_mock.change_detected = False
        detection_condition = RiverDriftDetectionCondition(drift_detection=drift_detector_mock)

        self.assertFalse(detection_condition.evaluate(y=y, y_hat=y_hat))
        drift_detector_mock.change_detected = True
        y_hat = xr.DataArray([3, 4, 5, 6], dims=['time'], coords={"time": time})

        self.assertTrue(detection_condition.evaluate(y=y, y_hat=y_hat))
        drift_detector_mock.change_detected = False

        y_hat = xr.DataArray([2, 3, 4, 5], dims=['time'], coords={"time": time})
        self.assertFalse(detection_condition.evaluate(y=y, y_hat=y_hat))

        update_calls = [call(1.0), call(2.0), call(1.0)]

        drift_detector_mock.update.assert_has_calls(update_calls)
        drift_detector_mock.reset.assert_called_once()

    def test_add_to_little_steps(self):
        with pytest.raises(StepCreationException):
            RiverDriftDetectionCondition()(y=MagicMock())

    def test_add_to_much_steps(self):
        with pytest.raises(StepCreationException):
            RiverDriftDetectionCondition()(y=MagicMock(), y_hat=MagicMock(), x=MagicMock())