import numpy as np
import pandas as pd
from river.drift import ADWIN

from pywatts.core.base_condition import BaseCondition
from pywatts.core.exceptions.invalid_input_exception import InvalidInputException


class RiverDriftDetectionCondition(BaseCondition):
    """
    A Drift Detection Condition that wraps the detection algorithms from the River library, based on the RMSE metric.

    :param drift_detection: The Drift Detection Algorithm from the River library. The default algorithm is ADWIN.
    """

    def __init__(self, name="CDCondition", refit_batch: pd.Timedelta = pd.Timedelta(hours=10), refit_params: dict = None,
                 drift_detection=ADWIN()):
        super().__init__(name=name, refit_batch=refit_batch, refit_params=refit_params)
        self.drift_detection = drift_detection

    def evaluate(self, start, end):
        """
        Returns True if the specified drift detection algorithm detects a drift.
        :param start: start of the batch
        :type start: pd.Timestamp
        :param end: end of the batch
        :type end: pd.Timestamp
        """
        inputs = self._get_inputs(start, end)

        if not self._is_evaluated(end):
            if self.drift_detection.change_detected:
                self.drift_detection.reset()
            if len(inputs) == 1:
                self.drift_detection.update(inputs)
            elif len(inputs) == 2:
                y, y_hat = inputs.values()
                rmse = np.sqrt(np.mean((y_hat.values - y.values) ** 2))
                if not np.isnan(rmse):
                    self.drift_detection.update(rmse)
            else:
                raise InvalidInputException(
                    f"More than two inputs are given for the instance {self.name} of class {self.__class__.__name__}.")

        if self.drift_detection.change_detected:
            return True
        else:
            return False
