import numpy as np
import pandas as pd

from pywatts.modules.metrics.rolling_metric_base import RollingMetricBase


class RollingMAE(RollingMetricBase):
    """
    Module to calculate the Rolling Mean Absolute Error (MAE)
    :param window_size: Determine the window size for the rolling mae. Default 24
    :type window_size: int
    :param window_size_unit: Determine the unit of the window size. Default Day (d)"
    :type window_size_unit: str

    """

    def _apply_rolling_metric(self, p, t, index):
        result = pd.DataFrame(np.mean((p - t), axis=-1), index=index).rolling(
            f"{self.window_size}{self.window_size_unit}").apply(
            lambda x: np.mean(np.abs(x)))
        result.loc[result.index[0]: result.index[0] + pd.Timedelta(f"{self.window_size}{self.window_size_unit}")] = np.nan

        return result.values
