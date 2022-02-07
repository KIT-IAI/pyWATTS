from typing import Dict

import xarray as xr

from pywatts.core.base import BaseTransformer
from pywatts.core.exceptions import WrongParameterException


class CustomScaler(BaseTransformer):
    """
    Scaling step to scale a time series individually by a multiplier and a bias.
    By default the scaling does not affect the time series, i.e., the multiplier is 1.0 and the bias 0.0.
    """

    def __init__(self, multiplier: float = 1.0, bias: float = 0.0, name: str = "CustomScaler"):
        """ Initialize the scaling step.
        :param multiplier: Value that is multiplied to every value in the time series.
        :type multiplier: float, optional
        :param bias: Value that is added to every value in the time series
        :type bias: float, optional
        """
        super().__init__(name)
        self.has_inverse_transform = True

        self.validate_multiplier(multiplier)

        self.multiplier = multiplier
        self.bias = bias

    def get_params(self) -> Dict[str, object]:
        """ Get parameters for the CustomScaler object.
        :return: Parameters as dict object.
        :rtype: Dict[str, object]
        """
        return {
            "multiplier": self.multiplier,
            "bias": self.bias
        }

    def set_params(self, multiplier: float = None, bias: float = None):
        """ Set or change CustomScaler object parameters.
        :param multiplier: Value that is multiplied to every value in the time series.
        :type multiplier: float, optional
        :param bias: Value that is added to every value in the time series
        :type bias: float, optional
        """
        if multiplier is not None:
            self.validate_multiplier(multiplier)
            self.multiplier = multiplier
        if bias is not None:
            self.bias = bias

    def transform(self, x: xr.DataArray) -> xr.DataArray:
        """ Apply the scaling to xarray dataset.
        :param x: xarray DataArray to apply scaling on.
        :type x: xr.DataArray
        :return: Xarray dataset scaled according to the specified multiplier and bias.
        :rtype: xr.DataArray
        """

        x = x * self.multiplier
        x = x + self.bias

        return x

    def inverse_transform(self, x: xr.DataArray) -> xr.DataArray:
        """ Apply the inverse scaling to xarray dataset.
        :param x: xarray DataArray to apply differentiation on.
        :type x: xr.DataArray
        :return: Xarray dataset containing the n-th order differentiations.
        :rtype: xr.DataArray
        """

        x = x - self.bias
        x = x / self.multiplier

        return x

    def validate_multiplier(self, multiplier):
        if multiplier == 0:
            raise WrongParameterException(
                "Multiplication by 0 is not possible.",
                "During initialisation set a multiplier different to 0.",
                module=self.__class__)
