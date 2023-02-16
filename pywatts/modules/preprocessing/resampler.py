from typing import Dict, Optional, Any

import pandas as pd
import xarray as xr

from pywatts_pipeline.core.transformer.base import BaseTransformer


class Resampler(BaseTransformer):
    """
    Module to resample time series based data to a given target time (both up and down sampling).
    All methods given by pandas' resample method are provided because of xarray's
    data set structure. See http://xarray.pydata.org/en/stable/generated/xarray.Dataset.resample.html for more details.

    This class resamples time series data based on xarray's resample method
    which is in turn based on pandas' resampling implementation.

    :param name: Name of this processing step (default: "Resampler").
    :type name: str
    :param time_index: Index of the dataset specifying the time series to be resampled (default: "time").
    :type time_index: str
    :param target_time: Target time after the resampling given in string datetime format (default: "1H").
                        For example, "6H"/"6h" for 6 hours, "s"/"S" for seconds, "m"/"M" for months.
    :type target_time: str
    :param method: Method to use for down- or upsampling the data (default: "mean").
                   For example, "mean", "min", "sum", "median", "reduce", "map".
                   http://xarray.pydata.org/en/stable/generated/xarray.core.resample.DatasetResample.html
    :type method: str
    :param method_args: Optional parameters for the selected method as a dict (default: "None").
                        Note: Some methods like reduce or map require parameters!
    :type method_args: Optional[Dict[str, Any]]

    Example:

        # downsample dataset to 30 Minutes (1800s) by using mean method
        Resampler(target_time="1800s", method="mean")

        # downsample dataset to 1 day by summing up all data for one day
        Resampler(target_time="1d", method="sum")

        # upsample "time_series" index of the dataset to 1 Minute by using interpolate method
        Resampler(time_index="time_series", target_time="60s", method="interpolate")

        # resample dataset index to 1 hour by using some costum (in this case first element)
        Resampler(target_time="1h", method="map", method_args={"func": lambda x: x.mean()})

    """

    def __init__(self, name: str = "Resampler", time_index: str = "time", target_time: str = "1H",
                 method: str = "mean", method_args: Optional[Dict[str, Any]] = None):

        super().__init__(name)
        self.time_index = time_index
        self.target_time = target_time
        self.method = method
        self.method_args = method_args

    def transform(self, x: xr.DataArray) -> xr.DataArray:
        """ Resamples the dataset x as specified in the constructor.

        :param x: dataset which should be resampled.
        :type x: xr.DataArray
        :return: Resampled xarray dataset as xarray dataset.
        :rtype: xr.DataArray
        """
        if self.method_args is not None:
            args = self.method_args
        else:
            args = {}

        return getattr(
            x.resample(**{self.time_index: self.target_time}), self.method
        )(**args)

    def get_min_data(self):
        return pd.Timedelta(self.target_time)
