from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import pandas as pd
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

        self._t_buffer = None
        self._p_buffer = {}

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

        index = y.indexes[_get_time_indexes(y)[0]]
        if self._t_buffer is None:
            # the buffer is required for the batch mode
            self._t_buffer = pd.DataFrame(y.values.reshape((len(y), -1)), index=index)
        else:
            # append/overwrite with new data
            self._t_buffer = pd.DataFrame(y.values.reshape((len(y), -1)), index=index).combine_first(self._t_buffer)

        results = {}
        for key, y_hat in kwargs.items():
            if key not in self._p_buffer.keys():
                # the buffer is required for the batch mode
                self._p_buffer[key] = pd.DataFrame(y_hat.values.reshape((len(y_hat), -1)), index=index)
            else:
                # append/overwrite with new data
                self._p_buffer[key] = pd.DataFrame(y_hat.values.reshape((len(y_hat), -1)), index=index)\
                    .combine_first(self._p_buffer[key])

            result = self._apply_rolling_metric(p=self._p_buffer[key].values,
                                                t=self._t_buffer.values,
                                                index=self._t_buffer.index).loc[index]  # crop result
            result.loc[self._t_buffer.index[0]:  # set value
                       self._t_buffer.index[0] + pd.Timedelta(f"{self.window_size}{self.window_size_unit}")] = np.nan
            results[key] = result
        time = y.indexes[_get_time_indexes(y)[0]]

        return xr.DataArray(np.concatenate(list(results.values()), axis=1),
                            coords={_get_time_indexes(y)[0]: time, "predictions": list(results.keys())},
                            dims=[_get_time_indexes(y)[0], "predictions"])

    @abstractmethod
    def _apply_rolling_metric(self, p, t, index):
        pass
