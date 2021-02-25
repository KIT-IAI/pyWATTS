from typing import Dict

import xarray as xr

from pywatts.core.exceptions.wrong_parameter_exception import WrongParameterException
from pywatts.core.base import BaseTransformer


class LinearInterpolater(BaseTransformer):
    """
    This module creates a linear interpolator.

    :param name: Name of the linear interpolator
    :type name: str
    :param method: The method used for interpolation (e.g. linear)
    :type method: str
    :param dim: The dimension used
    :type dim: str
    :param fill_value: Handling of missing values (see
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html)
    :type fill_value: str
    """

    def __init__(self, name: str = "LinearInterpolater", method: str = "linear",
                 dim: str = "time", fill_value: str = "extrapolate"):
        super().__init__(name)
        self.method = method
        self.dim = dim
        self.fill_value = fill_value

    def get_params(self) -> Dict[str, object]:
        """
        Returns a list of parameters used for the linear interpolation

        :return: Parameters set for the linear interpolation
        :rtype: Dict
        """
        return {"method": self.method,
                "dim": self.dim,
                "fill_value": self.fill_value}

    def set_params(self, method: str = None, dim: str = None, fill_value: str = None):
        """
        Sets the parameters for the linear interpolation

        :param method: The method used for interpolation (e.g. linear)
        :type method: str
        :param dim: The dimension used
        :type dim: str
        :param fill_value: Handling of missing values (see
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html)
        :type fill_value: str
        """
        if method is not None:
            self.method = method
        if dim is not None:
            self.dim = dim
        if fill_value is not None:
            self.fill_value = fill_value

    def transform(self, x=xr.DataArray) -> xr.DataArray:
        """
        Transforms the input

        :param x: Input xarray dataset
        :type x: xr.DataArray
        :return: Interpolated dataset
        :rtype: xr.DataArray
        """
        if self.dim not in x.dims:
            raise WrongParameterException(
                f"The dimension {self.dim} is not part of the input dataset.",
                "Assert that the previous modules provide the corresponding dimension.",
                module=self.name
            )
        return x.interpolate_na(method=self.method, dim=self.dim, fill_value=self.fill_value)
