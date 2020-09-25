from typing import Optional, Dict

import xarray as xr

from pywatts.core.base import BaseTransformer
from pywatts.core.exceptions.wrong_parameter_exception import WrongParameterException
from pywatts.utils._xarray_time_series_utils import _get_time_indeces


class RollingMean(BaseTransformer):
    """
     Module which calculates a rolling mean over a specific window size

     :param name: Name of the new variable
     :type name: str
     :param window_size: Window size for which to calculate the mean
     :type window_size: int
     :param indexes: Specifies over which dimension in the xarray the mean is calculated (default: time)
     :type indexes:

    """

    def __init__(self, name: str = "RollingMean", window_size=24 * 7, indexes=[]):

        super().__init__(name)
        self.window_size = window_size
        self.indexes = indexes

    def get_params(self) -> Dict[str, object]:
        return {
            "window_size": self.window_size,
            "indexes": self.indexes
        }

    def set_params(self, window_size=None, indexes=None):
        if window_size:
            self.window_size = window_size
        if indexes:
            self.indexes = indexes

    def transform(self, x: Optional[xr.Dataset]) -> xr.Dataset:
        indexes = self.indexes
        if not indexes:
            indexes = _get_time_indeces(x)
        try:
            return x.rolling(**{index: self.window_size for index in indexes}).mean()
        except KeyError:
            raise WrongParameterException(
                f"Not all indexes ({indexes}) are in the indexes of x ({list(x.indexes.keys())}).",
                f"Perhaps you set the wrong indexes with set_params or during the initialization of mean regressor.",
                module=self.name)
