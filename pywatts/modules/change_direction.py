from typing import Dict

import xarray as xr

from pywatts.core.base import BaseTransformer
from pywatts.core.exceptions.wrong_parameter_exception import WrongParameterException
from pywatts.utils._xarray_time_series_utils import _get_time_indexes


class ChangeDirection(BaseTransformer):
    """
    This module calculates a time series that indicates whether the next value is higher, lower, or the same.

    :param name: The name of the ChangeDirection module
    :type name: str
    """

    def __init__(self, name="change_direction"):
        super().__init__(name)

    def get_params(self) -> Dict[str, object]:
        """
        Returns a list of all parameters (note that this module has no parameters)

        :return: List of parameters
        :rtype: Dict
        """
        return {}

    def set_params(self):
        """
        Sets the parameters (note that this module has no parameters)
        """

    def transform(self, x: xr.DataArray) -> xr.DataArray:
        """
        Transforms the time series in a time series that indicates whether the next value is higher, lower, or the same

        :param x: The time series that should be transformed
        :type x: xr.DataArray, optional
        :return: A time series, where 1 indicates that the next value is higher, -1 that the next value
        is lower, and 0 that the next value is the same
        :rtype: xr.DataArray
        :raises WrongParameterException: If not all indexes are part of x
        """
        indexes = _get_time_indexes(x)
        try:
            return xr.ufuncs.sign(x - x.shift({index: 1 for index in indexes}))
        except ValueError as exc:
            raise WrongParameterException(
                f"Not all indexes ({indexes}) are in the indexes of x ({list(x.indexes.keys())}).",
                "Either correct the indexes which you passed to that module or assert that this index occurs in the "
                "data which are passed by the previous modules to the current one.", module=self.name) from exc
