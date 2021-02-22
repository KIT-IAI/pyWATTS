from typing import Optional, Dict

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


class CalendarExtraction(BaseTransformer):
    """
    This pipeline stepp will extract date features based on a timeseries defined by a DataArray input.
    It can calculate the year, month, weekday, day, weekend, workday, and holiday based
    on the timeseries given a variable encoding ('numerical' or 'sine' currently).
    For the holidays it is importent to set the correct continent and country/region.
    E.g. 'Europe' and 'BadenWurttemberg' or 'Germany'
    """

    def __init__(self, name: str = "CalendarExtraction", calendar_feature: str = "month",
                 encoding: str = "numerical", continent: str = "Europe", country: str = "Germany"):
        """ Initialize the calendar extration step.
            For correct holidays please set valid continent and country.

        :param name: Name of this processing step.
        :type name: str
        :param calendar_feature: Feature to extract from the input time series.
        :type calendar_feature: str
        :param encoding: Selected encoding for the features (e.g. numerical [0, 1, 2] or sine)
        :type encoding: str
        :param continent: Continent where the country or region is located
                          (important for importing calendar module).
        :type continent: str
        :param country: Country or region to use for holiday calendar (default 'Germany')
        :type country: str
        :raises WrongParameterException: If 'calendar_features' is invalid.
        :raises WrongParameterException: If 'encoding' is not in ['numerical', 'sine'].
        :raises WrongParameterException: If 'continent' and/or 'country' is invalid.
        """
        super().__init__(name)
        self.calendar_feature = calendar_feature
        self.encoding = encoding
        self.continent = continent
        self.country = country
        self.calendar = self._init_calendar(continent, country)

        calendar_features = ["year", "month", "day", "weekday", "hour", "weekend", "workday", "holiday"]
        if self.calendar_feature not in calendar_features:
            raise WrongParameterException(
                "Invalid calendar feature selected.",
                "Please select a valid calendar feature within [" + ", ".join(calendar_features),
                module=self.name
            )

        # check if correct encoding is selected
        # each encoding is implemented in a different method
        # named after the encoding strategy
        if not hasattr(self, f"_encode_{self.encoding}"):
            raise WrongParameterException(
                "Please set a valid encoding strategy.",
                "Change to the allowed strategies numerical or sine",
                module=self.name
            )

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
        if not hasattr(workalendar, continent.lower()):
            raise WrongParameterException(
                "Please set a valid continent for the CalendarExtraction step.",
                "See the documentation of workkalendar for valid continents",
                module=self.name
            )
        module = getattr(workalendar, continent.lower())
        if not hasattr(module, country):
            raise WrongParameterException(
                "Please set a valid country for the CalendarExtraction step.",
                "See the documentation of workkalendar for valid countries",
                module=self.name
            )
        return getattr(module, country)()

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
            "calendar_feature": self.calendar_feature,
            "encoding": self.encoding,
            "continent": self.continent,
            "country": self.country
        }

    def set_params(self, calendar_feature: Optional[str] = None, encoding: Optional[str] = None,
                   continent: Optional[str] = None, country: Optional[str] = None):
        """ Set parameters of the calendar extraction processing step.

        :param calendar_feature: Feature to extract from the input time series.
        :type calendar_feature: str
        :param encoding: Selected encoding for the features (e.g. numerical [0, 1, 2] or sine)
        :type encoding: str
        :param continent: Continent where the country or region is located
                          (important for importing calendar module).
        :type continent: str
        :param country: Country or region to use for holiday calendar (default 'Germany')
        :type country: str
        :raises WrongParameterException: If 'calendar_features' is invalid.
        :raises WrongParameterException: If 'encoding' is not in ['numerical', 'sine'].
        :raises WrongParameterException: If 'continent' and/or 'country' is invalid.
        """
        if calendar_feature is not None:
            calendar_features = ["year", "month", "day", "weekday", "hour", "weekend", "workday", "holiday"]
            if calendar_feature in calendar_features:
                self.calendar_feature = calendar_feature
            else:
                raise WrongParameterException(
                    "Invalid calendar feature selected.",
                    "Please select a valid calendar feature within [" + ", ".join(calendar_features),
                    module=self.name
                )
        if encoding is not None:
            self.encoding = encoding
        if continent is not None:
            self.continent = continent
        if country is not None:
            self.country = country

        self.calendar = self._init_calendar(self.continent, self.country)
        if not hasattr(self, f"_encode_{self.encoding}"):
            raise WrongParameterException(
                "Please set a valid encoding strategy.",
                "Change to the allowed strategies numerical or sine",
                module=self.name
            )

    def transform(self, x: xr.DataArray) -> xr.DataArray:
        """ Add date features to xarray dataset as configured.

        :param x: xarray DataArray containing a timeseries.
        :return: The xarray dataset with date features added.
        """
        series = getattr(self, f"_encode_{self.encoding}")(self.calendar_feature, x.to_series().index)
        return xr.DataArray(series, dims=x.dims, coords=x.coords)
