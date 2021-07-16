import numpy as np
import pandas as pd

from pywatts.modules.rolling_metric_base import RollingMetricBase
from pywatts.utils._xarray_time_series_utils import _get_time_indices


class RollingRMSE(RollingMetricBase):
    """
    Module to calculate the Rolling Root Mean Squared Error (RMSE)
    :param window_size: Determine the window size of the rolling rmse. Default 24
    :type window_size: int
    :param window_size_unit: Determine the unit of the window size. Default Day (d)"
    :type window_size_unit: str

    """

    def _apply_rolling_metric(self, p_, t_, index):
        return pd.DataFrame(np.mean((p_ - t_) ** 2, axis=-1),
                            index=index).rolling(
            f"{self.window_size}{self.window_size_unit}").apply(
            lambda x: np.sqrt(np.mean(x))).values
