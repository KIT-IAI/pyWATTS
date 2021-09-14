from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import xarray as xr

from pywatts.core.base import BaseTransformer
from pywatts.core.exceptions import InputNotAvailable
from pywatts.utils._xarray_time_series_utils import _get_time_indexes


class RollingMetricBase(BaseTransformer, ABC):
    """
    Module to calculate a Rolling Metric
    :param window_size: Determine the window size for the rolling metric. Default 24
    :type window_size: int
    :param window_size_unit: Determine the unit of the window size. Default Day (d)"
    :type window_size_unit: str

    """

    def __init__(self, name: str = None, window_size=24, window_size_unit="d"):
        super().__init__(name if name is not None else self.__class__.__name__)
        self.window_size_unit = window_size_unit
        self.window_size = window_size

    def get_params(self) -> Dict[str, object]:
        """
        Returns a dict of parameters used in the RollingMetric.

        :return: Parameters set for the RollingMetric
        :rtype: Dict[str, object]
        """
        return {
            "window_size_unit": self.window_size_unit,
            "window_size": self.window_size
        }

    def set_params(self, window_size=None, window_size_unit=None):
        """
        Set the parameter for the RollingMetric.

        :param window_size: Determine the window size if a rolling metric should be calculated.
                            Ignored if rolling is set to False. Default 24
        :type window_size: int
        :param window_size_unit: Determine the unit of the window size. Default Day (d)"
        :type window_size_unit: str

        """
        if window_size is not None:
            self.window_size = window_size
        if window_size_unit is not None:
            self.window_size_unit = window_size_unit

    def transform(self, y: xr.DataArray, **kwargs: xr.DataArray) -> xr.DataArray:
        """
        Calculates the MAE based on the predefined target and predictions variables.

        :param x: the input dataset
        :type x: Optional[xr.DataArray]

        :return: The calculated MAE
        :rtype: xr.DataArray
        """

        if kwargs == {}:
            error_message = f"No predictions are provided as input for the {self.__class__.__name__}. You should add the predictions" \
                            f" by a seperate key word arguments if you add the {self.__class__.__name__} to the pipeline."
            self.logger.error(error_message)
            raise InputNotAvailable(error_message)

        t = y.values
        results = {}
        for key, y_hat in kwargs.items():
            p = y_hat.values
            p_, t_ = p.reshape((len(p), -1)), t.reshape((len(t), -1))
            index = y.indexes[_get_time_indexes(y)[0]]
            results[key] = self._apply_rolling_metric(p_, t_, index)
        time = y.indexes[_get_time_indexes(y)[0]]

        return xr.DataArray(np.concatenate(list(results.values()), axis=1),
                            coords={_get_time_indexes(y)[0]: time, "predictions": list(results.keys())},
                            dims=[_get_time_indexes(y)[0], "predictions"])

    @abstractmethod
    def _apply_rolling_metric(self, p_, t_, index):
        pass
