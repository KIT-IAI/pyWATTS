from typing import Dict, List, Union

import xarray as xr
from pywatts_pipeline.core.exceptions import WrongParameterException
from pywatts_pipeline.core.transformer.base import BaseTransformer
from pywatts_pipeline.utils._xarray_time_series_utils import _get_time_indexes


class Select(BaseTransformer):
    """
    This module selects for each time stamp one or multiple values from the original time series.
    E.g. it can be used for building input samples, target samples, or for shifting the time series.

    :param start: The offset for shifting the time series. If this is a list, stop and step are ignored and
                  the list specifies the values that should be selected.
    :type start: Union[int, List[int]
    :param stop: The offset for shifting the time series
    :type stop: int
    :param step: The offset for shifting the time series
    :type step: int
    """

    def __init__(
        self,
        start: Union[int, List[int]],
        stop: int = None,
        step: int = None,
        name: str = "SampleModule",
    ):
        super().__init__(name)
        if self.__check_params(start, stop, step):
            self.start = start
            self.stop = stop
            self.step = step

    def __check_params(self, start, stop, step):
        if not isinstance(start, (int, list)):
            raise WrongParameterException(f"The start parameter is a {type(start).__name__}.",
                                          "But it needs to be an integer or a list.",
                                          self)
        elif not (stop is None) and start >= stop:
            raise WrongParameterException(f"The start parameter is greater than or equals stop.",
                                          "But it needs to be smaller than stop.",
                                          self)
        return True

    def get_min_data(self):
        if isinstance(self.start, list):
            return abs(min(self.start))
        return abs(self.start) if self.start < 0 else 0

    def set_params(
        self, start: Union[int, List[int]] = None, stop: int = None, step: int = None, **kwargs
    ):
        """
        Set params.


        :param start: The offset for shifting the time series. If this is a list, stop and step are ignored and
                      the list specifies the values that should be selected.
        :type start: Union[int, List[int]
        :param stop: The offset for shifting the time series
        :type stop: int
        :param step: The offset for shifting the time series
        :type step: int
        """
        if start:
            self.start = start
        if stop:
            self.stop = stop
        if step:
            self.step = step
        self.__check_params(self.start, self.stop, self.step)

    def transform(self, x: xr.DataArray) -> xr.DataArray:
        """
        Select the desired subset from the given time series x

        :param x: the input
        :type x: xr.DataArray
        :return: A shifted time series.
        :rtype: xr.DataArray
        """
        indexes = _get_time_indexes(x)
        if isinstance(self.start, list):
            to_select = self.start
        elif not (self.stop is None):
            if not (self.step is None):
                to_select = list(range(self.start, self.stop, self.step))
            else:
                to_select = list(range(self.start, self.stop, 1))
        else:
            to_select = [self.start]
        if len(to_select) > 1:
            r = [x.shift({index: -1 * i for index in indexes}) for i in to_select]
            result = xr.concat(r, dim="horizon").dropna(indexes[0])
            return result.transpose(indexes[0], "horizon", ...)
        return x.shift({index: -1 * to_select[0] for index in indexes}).dropna(indexes[0])
