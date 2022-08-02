import numpy as np
import xarray as xr
from river.drift import ADWIN

from pywatts_pipeline.core.exceptions.invalid_input_exception import InvalidInputException
from pywatts_pipeline.core.condition.base_condition import BaseCondition


class RiverDriftDetectionCondition(BaseCondition):
    """
    A Drift Detection Condition that wraps the detection algorithms from the River library, based on the RMSE metric.

    :param drift_detection: The Drift Detection Algorithm from the River library. The default algorithm is ADWIN.
    """

    def __init__(self, name="CDCondition", drift_detection=ADWIN()):
        super().__init__(name=name)
        self.drift_occured = False
        self.drift_detection = drift_detection
        self.counter = 0

    def evaluate(self, y: xr.DataArray, y_hat: xr.DataArray):
        """
        Returns True if the specified drift detection algorithm detects a drift.
        :param y: GT Time Series
        :type y: xr.DataArray
        :param y_hat: Forecast Time Series
        :type y_hat: xr.DataArray
        """
        rmse = np.sqrt(np.mean((y_hat.values - y.values) ** 2))
        if not np.isnan(rmse):
            self.drift_detection.update(rmse)
            self.counter += 1

        if self.drift_detection.change_detected:
            self.reset()
            return True
        else:
            return False
