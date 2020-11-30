from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import workalendar.africa
import workalendar.america
import workalendar.asia
import workalendar.europe
import workalendar.oceania
import workalendar.usa
import xarray as xr

from pywatts.core.base import BaseTransformer
from pywatts.core.exceptions.wrong_parameter_exception import WrongParameterException
from pywatts.utils._xarray_time_series_utils import _get_time_indeces


class CalendarExtraction(BaseTransformer):
    """
    This pipeline stepp will extract date features based on a timeseries defined by 'time_index'.
    It will calculate the year, month, weekday, day, weekend, workday, and holiday based
    on the timeseries given a variable encoding ('numerical' or 'sine' currently).
    For the holidays it is importent to set the correct continent and country/region.
    E.g. 'Europe' and 'BadenWurttemberg' or 'Germany'
    """

    def __init__(self, name: str = "CalendarExtraction", time_index: str = None,
                 encoding: str = "numerical", prefix: str = "", feature=None,
                 continent: str = "Europe", country: str = "Germany", one_data_var: bool = False):
        """ Initialize the calendar extration step.
            For correct holidays please set valid continent and country.

        :param name: Name of this processing step.
        :type name: str
        :param time_index: Index of the timeseries of the xarray dataset.
        :type time_index: str
        :param encoding: Selected encoding for the features (e.g. numerical [0, 1, 2] or sine)
        :type encoding: str
        :param prefix: Prefix for the features that will be added to the xarray dataset (default "").
        :type prefix: str
        :param continent: Continent where the country or region is located
                          (important for importing calendar module).
        :type continent: str
        :param country: Country or region to use for holiday calendar (default 'Germany')
        :type country: str
        :param one_data_var: Flag indicate if there should be one datavar for each feature or if all features should be
                             only in one data variable
        :type one_data_var: bool
        :raises WrongParameterException: If 'continent' and/or 'country' is invalid.
        """
        super().__init__(name)
        self.time_index = time_index
        self.encoding = encoding
        self.prefix = prefix
        self.continent = continent
        self.country = country
        self.calendar = self._init_calendar(continent, country)

        # check if correct encoding is selected
        # each encoding is implemented in a different method
        # named after the encoding strategy
        if not hasattr(self, f"_encode_{self.encoding}"):
            raise WrongParameterException("Please set a valid encoding strategy.",
                                          f"Change to the allowed strategies numerical or sine",
                                          module=self.name)

    def _init_calendar(self, continent: str, country: str):
        """ Check if continent and country are correct and return calendar object.

        :param continent: Continent where the country or region is located.
        :type continent: str
        :param country: Country or region to use for the calendar object.
        :type country: str
        :return: Returns workalendar object to use for holiday lookup.
        :rtype: workalendar object
        :raises WrongParameterException: If the wrong country or continent are set
        """
        if hasattr(workalendar, continent.lower()):
            module = getattr(workalendar, continent.lower())
            if hasattr(module, country):
                return getattr(module, country)()
            else:
                raise WrongParameterException("Please set a valid country for the CalendarExtraction step.",
                                              f"See the documentation of workkalendar for valid countries",
                                              module=self.name)
        else:
            raise WrongParameterException("Please set a valid continent for the CalendarExtraction step.",
                                          "See the documentation of workkalendar for valid continents",
                                          module=self.name)

    def _encode_numerical(self, feature: str, timeseries: pd.Series):
        """ Encode a specific feature numerical given a pandas series timeseries.

        :param feature: Feature to calculate (e.g. year, month, weekday, ...)
        :type feature: str
        :param timeseries: Datetime[ns] timeseries as pandas Series (fast and easy map method)
        :type timeseries: pd.Series
        """
        if feature == "year":
            return timeseries.map(lambda element: element.year)
        elif feature == "month":
            return timeseries.map(lambda element: element.month - 1)
        elif feature == "day":
            return timeseries.map(lambda element: element.day - 1)
        elif feature == "weekday":
            return timeseries.map(lambda element: element.weekday())
        elif feature == "hour":
            return timeseries.map(lambda element: element.hour)
        elif feature == "weekend":
            return timeseries.map(lambda element: (element.weekday() >= 5) * 1)
        elif feature == "workday":
            return timeseries.map(
                lambda element: (self.calendar.is_working_day(element)) * 1
            )
        elif feature == "holiday":
            return timeseries.map(
                lambda element: (self.calendar.is_holiday(element)) * 1
            )

    def _encode_sine(self, feature: str, timeseries):
        """ Encode a specific feature as sine given a pandas series timeseries.
            For some features sine encoding makes no sense and numerical encoding is selected.

        :param feature: Feature to calculate (e.g. year, month, weekday, ...)
        :type feature: str
        :param timeseries: Datetime[ns] timeseries as pandas Series (fast and easy map method)
        :type timeseries: pd.Series
        """
        if feature == "year":
            # because year isn't a cyclic feature it makes no sense to encode it with sine
            return timeseries.map(lambda element: element.year)
        elif feature == "month":
            return timeseries.map(lambda element: np.sin(2 * np.pi * (element.month - 1) / 11))
        elif feature == "day":
            return timeseries.map(lambda element: np.sin(2 * np.pi * (element.day - 1) / 30))
        elif feature == "weekday":
            return timeseries.map(lambda element: np.sin(2 * np.pi * element.weekday() / 6))
        elif feature == "hour":
            return timeseries.map(lambda element: np.sin(2 * np.pi * element.hour / 23))
        elif feature == "weekend":
            # because this features is only 0 or 1 it makes no sense to encode it with sine
            return timeseries.map(lambda element: (element.weekday() >= 5) * 1)
        elif feature == "workday":
            # because this features is only 0 or 1 it makes no sense to encode it with sine
            return timeseries.map(
                lambda element: (self.calendar.is_working_day(element)) * 1
            )
        elif feature == "holiday":
            # because this features is only 0 or 1 it makes no sense to encode it with sine
            return timeseries.map(
                lambda element: (self.calendar.is_holiday(element)) * 1
            )

    def get_params(self) -> Dict[str, object]:
        """ Get parameters of this calendar extraction processing step.

        :return: Json dict containing the parameters.
        """
        return {
            "time_index": self.time_index,
            "encoding": self.encoding,
            "prefix": self.prefix,
            "continent": self.continent,
            "country": self.country
        }

    def set_params(self, time_index: Optional[str] = None,
                   encoding: Optional[str] = None, prefix: Optional[str] = None,
                   continent: Optional[str] = None, country: Optional[str] = None):
        """ Set parameters of the calendar extraction processing step.

        :param time_index: Index of the timeseries of the xarray dataset.
        :type time_index: str
        :param encoding: Selected encoding for the features (e.g. numerical [0, 1, 2] or sine)
        :type encoding: str
        :param prefix: Prefix for the features that will be added to the xarray dataset (default "").
        :type prefix: str
        :param continent: Continent where the country or region is located
                          (important for importing calendar module).
        :type continent: str
        :param country: Country or region to use for holiday calendar (default 'Germany')
        :type country: str
        :raises WrongParameterException: If 'continent' and/or 'country' is invalid.
        """
        if time_index is not None:
            self.time_index = time_index
        if encoding is not None:
            self.encoding = encoding
        if prefix is not None:
            self.prefix = prefix
        if continent is not None:
            self.continent = continent
        if country is not None:
            self.country = country
        self.calendar = self._init_calendar(self.continent, self.country)
        if not hasattr(self, f"_encode_{self.encoding}"):
            raise WrongParameterException("Please set a valid encoding strategy.",
                                          f"Change to the allowed strategies numerical or sine",
                                          module=self.name)

    def transform(self, x: xr.Dataset) -> xr.Dataset:
        """ Add date features to xarray dataset as configured.

        :param x: Xarray dataset containing a timeseries specified by the object's 'time_index'
        :return: The xarray dataset with date features added.
        """
        features = [
            "year", "month", "day", "weekday", "hour", "weekend", "workday", "holiday"
        ]
        time_index = self.time_index
        if time_index is None:
            time_index = _get_time_indeces(x)[0]
        data = dict()
        for feature in features:
            data[f"{self.prefix}{feature}"] = getattr(self, f"_encode_{self.encoding}")(
                feature, getattr(x, time_index).to_series()
            )
        return xr.DataArray(np.stack(data.values(), axis=-1), coords=(x[time_index], features))