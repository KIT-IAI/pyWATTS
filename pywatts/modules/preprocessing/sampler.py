from typing import Dict, List

import pandas as pd
import xarray as xr

from pywatts.core.base import BaseTransformer
from pywatts.core.exceptions.wrong_parameter_exception import WrongParameterException
from pywatts.utils._xarray_time_series_utils import _get_time_indexes
import numpy as np


class Sampler(BaseTransformer):
    """
    This module creates samples with a size specified by sample_size. I.e., if sample_size is 24h. It creates for each
    timestamp a vector containing all values of the past 24 hours.
    E.g., this module is useful if it forecasting algorithms needs the values of the past 24 hours as input.

    :param sample_size: The offset for shifting the time series
    :type sample_size: int
    :param indexes: The indexes which should be shifted through time
    :type indexes: List[str]

     """

    def __init__(self, sample_size: int, name: str = "SampleModule", indexes: List[str] = None):
        super().__init__(name)
        if indexes is None:
            indexes = []
        if sample_size <= 0:
            raise WrongParameterException(
                "Sample size cannot be less than or equal to zero.",
                "Please define a sample size greater than zero.",
                module=self.name)
        self.sample_size = sample_size
        self.indexes = indexes

    def get_min_data(self):
        return self.sample_size

    def get_params(self) -> Dict[str, object]:
        """
        Returns a list of all hyperparameters/ user defined parameters

        :return: Dict with params
        :rtype: dict
        """
        return {
            "sample_size": self.sample_size,
            "indexes": self.indexes,
        }

    def set_params(self, sample_size: int = None, indexes: List[str] = None):
        """
        Set params.

        :param sample_size: The offset for shifting the time series
        :type sample_size: int
        :param indexes: The indexes which should be shifted through time
        :type indexes: List[str]

        """
        if sample_size:
            self.sample_size = sample_size
            if sample_size <= 0:
                raise WrongParameterException(
                    "Sample size cannot be less than or equal to zero.",
                    "Please define a sample size greater than zero.",
                    module=self.name)
        if indexes is not None:
            # Do not use if indexes here, since this would be false if indexes is empty.
            self.indexes = indexes

    def transform(self, x: xr.DataArray) -> xr.DataArray:
        """
        Sample the given time series x by the lag.

        :param x: the input
        :type x: xr.DataArray
        :return: A shifted time series.
        :rtype: xr.DataArray
        """
        indexes = self.indexes
        if not indexes:
            indexes = _get_time_indexes(x)
        try:
            r = [x.shift({index: i for index in indexes}, fill_value=0) for i in range(self.sample_size - 1, -1, -1)]
        except ValueError as exc:
            raise WrongParameterException(
                f"Not all indexes ({indexes}) are in the indexes of x ({list(x.indexes.keys())}).",
                "Perhaps you set the wrong indexes with set_params or during the initialization of the Sampler.",
                module=self.name) from exc
        result = xr.DataArray(np.stack(r, axis=-1), dims=(*x.dims, "horizon"), coords=x.coords)
        return result.transpose(_get_time_indexes(x)[0], "horizon", ...)
