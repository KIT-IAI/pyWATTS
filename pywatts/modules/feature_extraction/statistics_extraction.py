from enum import IntEnum
from typing import Dict, Optional, List

import xarray as xr
import numpy as np
from pywatts_pipeline.core.transformer.base import BaseTransformer
from pywatts_pipeline.utils._xarray_time_series_utils import _get_time_indexes


class StatisticFeature(IntEnum):
    """
    The available statistic features, that are extractable by the statistic extraction module:

    min: Extracting the min of the time series sample.
    max: Extracting the max of the time series sample.
    std: Extracting the std of the time series sample.
    mean: Extracting the mean of the time series sample.
    """
    min = 1
    max = 2
    std = 3
    mean = 4


class StatisticExtraction(BaseTransformer):
    """
    This module extracts statistical features based on samples of time series defined by a DataArray input.
    It can calculate the min, max, std, and mean of the samples.

    :param name: Name of this module.
    :type name: str
    :param features: The features that should be extracted. The following features exist: min, max, std, and mean.
                     (Default: List[min, max, std, mean)
    :type features: Optional[List[StatisticFeature]]
    :param dim: The dimension on which the statistics should be extracted
    :type dim: str
    """

    def __init__(self, name: str = "statistics", features: List[StatisticFeature] = None, dim="horizon"):
        super().__init__(name)
        self.dim = dim
        if features is None:
            self.features = [StatisticFeature.min, StatisticFeature.max, StatisticFeature.std, StatisticFeature.mean]
        else:
            self.features = features

    def _extract(self, feature, x):
        if feature == StatisticFeature.min:
            return x.min(dim=self.dim)
        if feature == StatisticFeature.max:
            return x.max(dim=self.dim)
        if feature == StatisticFeature.std:
            return x.std(dim=self.dim)
        if feature == StatisticFeature.mean:
            return x.mean(dim=self.dim)

    def transform(self, x: xr.DataArray) -> xr.DataArray:
        """ Add statistic features to xarray dataarray as configured.

        :param x: Xarray dataarray containing a time series
        :return: The xarray dataarray with statistic features.
        """
        time = _get_time_indexes(x)[0]
        data = []
        for feature in self.features:
            data.append(self._extract(feature, x))
        return xr.DataArray(np.array(data).swapaxes(0, 1).reshape(len(x), len(self.features)), coords=[getattr(x, time), self.features],
                            dims=[time, "stat_features"])

    def get_params(self) -> Dict[str, object]:
        """ Get parameters of this calendar extraction processing step.

        :return: Json dict containing the parameters.
        """
        return {
            "dim": self.dim,
            "features": self.features
        }

    def set_params(self, dim=None, features=None):
        """ Set parameters of the statistic extraction module.

        :param features: The features that should be extracted. The following features exist: min, max, std, and mean.
                         (Default: List[min, max, std, mean)
        :type features: Optional[List[StatisticFeature]]
        :param dim: The dimension on which the statistics should be extracted
        :type dim: str
        """

        if dim:
            self.dim = dim
        if features:
            self.features = features
