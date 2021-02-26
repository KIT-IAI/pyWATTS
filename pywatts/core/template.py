from typing import Optional, Dict

import xarray as xr

from pywatts.core.base import BaseEstimator


class Template(BaseEstimator):
    """
    Template for a estimator
    For a transformer replace the base class BaseEstimator with BaseTransformer and delete the fit method

    :param name: The name of the module
    :type name: str
    Add further parameters of the module
    """

    def __init__(self, name: str):
        super().__init__(name)
        # example for a parameter initialization
        # self.windows_size = 10

        # If this module implements predict_proba, set the has_predict_proba to True.
        # self.has_predict_proba = True

        # If this module implements inverse_transform, set the has_inverse_transform to True.
        # self.has_inverse_transform = True

    def get_params(self) -> Dict[str, object]:
        """
        Returns all parameters in a dict

        :return: Dict with params
        :rtype: Dict
        """
        params = {}

        # example for the window size parameter
        # params["window_size"] = self.window_size

        return params

    def set_params(self, **kwargs):
        """
        This method sets the parameters. If there is no parameter, write pass in the method's body.

        :param Parameter1: Parameter1 is responsible for ...
        :type Parameter1: The type of Parameter 1
        ...
        """
        # Example for setting a parameter
        # if parameter1 is not None:
        #     self.parameter1 = parameter1

        pass

    def fit(self, x: xr.DataArray, y: xr.DataArray):
        """
        Fit the model, e.g. optimize parameters such that model(x) = y

        :param x: input
        :type x: xarray.Dataset
        :param y: target
        :type y: xarray.Dataset
        """
        # Write the code for fitting the module/model

    def transform(self, x: Optional[xr.DataArray]) -> xr.DataArray:
        """
        Transforms the input

        :param x: the input
        :type x: xarray.Dataset
        :return: The transformed input
        :rtype: xarray.Dataset
        """
        # Write here code for transforming the input

    def predict_proba(self, x: xr.DataArray) -> xr.DataArray:
        """
        Probabilistic transform, necessary, for example, for methods for probabilist forecasts.

        Note if you implement this method, the flag "self.has_predict_proba" must be set to true in the constructor

        :param x: the input
        :type x: xarray.Dataset
        :return: The transformed input
        :rtype: xarray.Dataset
        """

        # add here code for the probabilist transform
        return xr.DataArray()

    def inverse_transform(self, x: xr.DataArray) -> xr.DataArray:
        """
        Inverse transform, necessary, for example, for methods for that can perform the inverse of the transform,
        e.g., Scaler.

        Note if you implement this method, the flag "self.has_inverse_transform" must be set to true in the constructor

        :param x: the input
        :type x: xarray.Dataset
        :return: The transformed input
        :rtype: xarray.Dataset
        """

        # Add here code for the inverse transform
        return xr.DataArray()
