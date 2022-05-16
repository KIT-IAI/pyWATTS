import numpy as np
import xarray as xr
from river.drift import ADWIN

from pywatts.core.base_condition import BaseCondition


class RiverDriftDetectionCondition(BaseCondition):
    """
    A Drift Detection Condition that wraps the detection algorithms from the River library, based on the RMSE metric.

    :param drift_detection: The Drift Detection Algorithm from the River library. The default algorithm is ADWIN.
    """

    def __init__(self, name="CDCondition", drift_detection=ADWIN()):
        super().__init__(name=name)
        self.drift_detection = drift_detection

    def evaluate(self, start, end):
        """
        Returns True if the specified drift detection algorithm detects a drift.
        :param start: start of the batch
        :type start: pd.Timestamp
        :param end: end of the batch
        :type end: pd.Timestamp
        """
        y, y_hat = self._get_inputs(start, end)

        if not self._is_evaluated:
            if self.drift_detection.change_detected:
                self.drift_detection.reset()
            rmse = np.sqrt(np.mean((y_hat.values - y.values) ** 2))
            if not np.isnan(rmse):
                self.drift_detection.update(rmse)

        if self.drift_detection.change_detected:
            return True
        else:
            return False
