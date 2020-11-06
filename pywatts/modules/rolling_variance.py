from typing import Optional, Dict

import xarray as xr

from pywatts.core.base import BaseTransformer
from pywatts.core.exceptions.wrong_parameter_exception import WrongParameterException
from pywatts.utils._xarray_time_series_utils import _get_time_indeces


class RollingVariance(BaseTransformer):
    """
     Module which calculates a rolling variance over a specific window size

     :param name: Name of the new variable
     :type name: str
     :param window_size: Window size for which to calculate the variance
     :type window_size: int
     :param indexes: Specifies over which dimension in the xarray the variance is calculated (default: time)
     :type indexes: list
     :param ddof: Delta Degrees of Freedom. The divisor used in N - ddof, where N represents the number of elements.
     :type ddof: int
     """

    def __init__(self, name: str = "RollingVariance", window_size=24 * 7, indexes: list = None, ddof: int = 1):

        super().__init__(name)
        if indexes is None:
            indexes = []
        self.window_size = window_size
        self.indexes = indexes
        self.kwargs = {'ddof': ddof}

    def get_params(self) -> Dict[str, object]:
        return {
            "window_size": self.window_size,
            "indexes": self.indexes,
            "ddof": self.kwargs['ddof'],
        }

    def set_params(self, window_size=None, indexes=None, ddof=None):
        if window_size is not None:
            self.window_size = window_size
        if indexes is not None:
            self.indexes = indexes
        if ddof is not None:
            self.kwargs.update({'ddof': ddof})

    def transform(self, x: Optional[xr.Dataset]) -> xr.Dataset:
        indexes = self.indexes
        if not indexes:
            indexes = _get_time_indeces(x)
        try:
            return x.rolling(**{index: self.window_size for index in indexes}).var(**self.kwargs)
        except KeyError:
            raise WrongParameterException(
                f"Not all indexes ({indexes}) are in the indexes of x ({list(x.indexes.keys())}).",
                f"Perhaps you set the wrong indexes with set_params or during the initialization of "
                f"variance regressor.",
                module=self.name)
