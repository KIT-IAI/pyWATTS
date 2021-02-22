from typing import Dict, List

import xarray as xr

from pywatts.core.base import BaseTransformer
from pywatts.core.exceptions.wrong_parameter_exception import WrongParameterException
from pywatts.utils._xarray_time_series_utils import _get_time_indeces


class Sampler(BaseTransformer):
    """
    This module creates samples with a size specified by sample_size. I.e., if sample_size is 24h. It creates for each
    timestamp a vector containing all values of the past 24 hours.
    E.g., this module is useful if it forecasting algorithms needs the values of the past 24 hours as input.

    :param sample_size: The offset for shifting the time series
    :type sample_size: int
    :param indices: The indices which should be shifted through time
    :type indices: List[str]
    :param data_var_names: Name for the shifted time series
    :type data_var_names: List[str]

     """

    def __init__(self, sample_size: int, name: str = "SampleModule", indeces: List[str] = None, data_var_names=None):
        super().__init__(name)
        if indeces is None:
            indeces = []
        if data_var_names is None:
            data_var_names = []
        self.sample_size = sample_size
        self.indexes = indeces
        self.data_var_names = data_var_names

    def get_params(self) -> Dict[str, object]:
        """
        Returns a list of all hyperparameters/ user defined parameters

        :return: Dict with params
        :rtype: dict
        """
        return {
            "lag": self.sample_size,
            "indeces": self.indexes,
            "data_var_names": self.data_var_names
        }

    def set_params(self, sample_size: int = None, indexes: List[str] = None, data_var_names=[]):
        """
        Set params.

        :param sample_size: The offset for shifting the time series
        :type sample_size: int
        :param indices: The indices which should be shifted through time
        :type indices: List[str]
        :param data_var_names: Name for the shifted time series
        :type data_var_names: List[str]

        """
        if sample_size:
            self.sample_size = sample_size
        if indexes is not None:
            # Do not use if indexes here, since this would be false if indexes is empty.
            self.indexes = indexes
        if data_var_names:
            self.data_var_names = data_var_names

    def transform(self, x: xr.Dataset) -> xr.Dataset:
        """
        Sample the given time series x by the lag.

        :param x: the input
        :type x: xr.Dataset
        :return: A shifted time series.
        :rtype: xr.Dataset
        """
        indexes = self.indexes
        if not indexes:
            indexes = _get_time_indeces(x)
        try:
            result = x
            for i in range(1, self.sample_size):
                result = xr.concat([result, x.shift({index: i for index in indexes}, fill_value=0)], dim="horizon")
        except ValueError:
            raise WrongParameterException(
                f"Not all indexes ({indexes}) are in the indexes of x ({list(x.indexes.keys())}).",
                "Perhaps you set the wrong indexes with set_params or during the initialization of the Sampler.",
                module=self.name)
        result = result.transpose(_get_time_indeces(x)[0], "horizon", ...)
        if self.data_var_names:
            return result.rename(dict(zip(x.data_vars.keys(), self.data_var_names)))
        return result
