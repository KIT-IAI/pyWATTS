from typing import Optional, Dict, List
from enum import IntEnum

import numpy as np
import pandas as pd
import xarray as xr

from pywatts.core.base import BaseTransformer
from pywatts.core.exceptions.util_exception import UtilException
from pywatts.core.exceptions.wrong_parameter_exception import WrongParameterException
from pywatts.utils._workalendar_utils import _init_calendar
from pywatts.utils._xarray_time_series_utils import _get_time_indexes


class CalendarFeature(IntEnum):
    """
    The available calendar features, that are extractable by the calendar extraction module:

    year: Extracting the year of the time series element.
    month: Extracting the month of the time series element.
    month_sine: Extracting the month of the time series element and encodes it with sine.
    month_cos: Extracting the month of the time series element and encodes it with cos.
    day: Extracting the day of the time series element.
    day_sine: Extracting the day of the time series element and encodes it with sine.
    day_cos: Extracting the day of the time series element and encodes it with cos.
    hour: Extracting the hour of the time series element.
    hour_sine: Extracting the hour of the time series element and encodes it with sine.
    hour_cos: Extracting the hour of the time series element and encodes it with cos.
    weekday: Extracting the weekday of the time series element.
    weekday_sine: Extracting the weekday of the time series element and encodes it with sine.
    weekday_cos: Extracting the weekday of the time series element and encodes it with cos.
    monday: Extracting a flag indicating if the element of the time series element is a monday or not.
    tuesday: Extracting a flag indicating if the element of the time series element is a tuesday or not.
    wednesday: Extracting a flag indicating if the element of the time series element is a wednesday or not.
    thursday: Extracting a flag indicating if the element of the time series element is a thursday or not.
    friday: Extracting a flag indicating if the element of the time series element is a friday or not.
    saturday: Extracting a flag indicating if the element of the time series element is a saturday or not.
    sunday: Extracting a flag indicating if the element of the time series element is a sunday or not.
    weekend: Extracting a flag indicating if the element of the time series element is a weekend or not.
    workday: Extracting a flag indicating if the element of the time series element is a workday or not.
    holiday: Extracting a flag indicating if the element of the time series element is a holiday or not.
    """
    year = 1
    month = 2
    month_sine = 3
    month_cos = 4
    day = 5
    day_sine = 6
    day_cos = 7
    hour = 8
    hour_sine = 9
    hour_cos = 10
    weekday = 11
    weekday_sine = 12
    weekday_cos = 13
    monday = 14
    tuesday = 15
    wednesday = 16
    thursday = 17
    friday = 18
    saturday = 19
    sunday = 20
    weekend = 21
    workday = 22
    holiday = 23


class CalendarExtraction(BaseTransformer):
    """
    This pipeline step will extract date features based on a timeseries defined by a DataArray input.
    It can calculate the year, month, month_sine, month_cos, day, day_sine, day_cos, hour, hour_sine, hour_cos, weekday,
    weekday_sine, weekday_cos, monday, tuesday, wednesday, thursday, friday, saturday, sunday, weekend, workday, and
    holiday. based on the timeseries.
    For the holidays it is importent to set the correct continent and country/region.
    E.g. 'Europe' and 'BadenWurttemberg' or 'Germany'

    :param name: Name of this processing step.
    :type name: str
    :param continent: Continent where the country or region is located
                      (important for importing calendar module).
    :type continent: str
    :param country: Country or region to use for holiday calendar (default 'Germany')
    :type country: str
    :param features: The features that should be extracted. The following features exist: year, month, month_sine,
                     month_cos, day, day_sine, day_cos, hour, hour_sine, hour_cos, weekday, weekday_sine, weekday_cos,
                     monday, tuesday, wednesday, thursday, friday, saturday, sunday, weekend, workday, holiday.
                     (Default: month, day, weekday, hour)
    :type features: Optional[List[CalendarFeature]]
    :raises WrongParameterException: If 'continent' and/or 'country' is invalid.
    """

    def __init__(self, name: str = "CalendarExtraction",
                 continent: str = "Europe", country: str = "Germany", features: Optional[List[CalendarFeature]] = None):

        super().__init__(name)
        self.continent = continent
        self.country = country
        try:
            self.calendar = _init_calendar(self.continent, self.country)
        except UtilException as exc:
            raise WrongParameterException(exc.message,
                                          "Please set a valid country or continent for.", CalendarExtraction) from exc

        if features is None:
            self.features: List[CalendarFeature] = [CalendarFeature.month, CalendarFeature.day, CalendarFeature.hour,
                                                    CalendarFeature.weekday]
        else:
            self.features: List[CalendarFeature] = features

    def _encode(self, feature: CalendarFeature, timeseries: pd.Series):
        """ Encode a specific feature numerical given a pandas series timeseries.

        :param feature: Feature to calculate (e.g. year, month, weekday, ...)
        :type feature: str
        :param timeseries: Datetime[ns] timeseries as pandas Series (fast and easy map method)
        :type timeseries: pd.Series
        """
        if feature == CalendarFeature.year:
            return timeseries.map(lambda element: element.year)
        elif feature == CalendarFeature.month:
            return timeseries.map(lambda element: element.month - 1)
        elif feature == CalendarFeature.day:
            return timeseries.map(lambda element: element.day - 1)
        elif feature == CalendarFeature.weekday:
            return timeseries.map(lambda element: element.weekday())
        elif feature == CalendarFeature.hour:
            return timeseries.map(lambda element: element.hour)
        elif feature == CalendarFeature.weekend:
            return timeseries.map(lambda element: (element.weekday() >= 5) * 1)
        elif feature == CalendarFeature.workday:
            return timeseries.map(lambda element: (self.calendar.is_working_day(element)) * 1)
        elif feature == CalendarFeature.holiday:
            return timeseries.map(lambda element: (self.calendar.is_holiday(element)) * 1)
        elif feature == CalendarFeature.monday:
            return timeseries.map(lambda element: element.weekday() == 0)
        elif feature == CalendarFeature.tuesday:
            return timeseries.map(lambda element: element.weekday() == 1)
        elif feature == CalendarFeature.wednesday:
            return timeseries.map(lambda element: element.weekday() == 2)
        elif feature == CalendarFeature.thursday:
            return timeseries.map(lambda element: element.weekday() == 3)
        elif feature == CalendarFeature.friday:
            return timeseries.map(lambda element: element.weekday() == 4)
        elif feature == CalendarFeature.saturday:
            return timeseries.map(lambda element: element.weekday() == 5)
        elif feature == CalendarFeature.sunday:
            return timeseries.map(lambda element: element.weekday() == 6)
        elif feature == CalendarFeature.month_sine:
            return timeseries.map(lambda element: np.sin(np.pi * 2 * (element.month - 1) / 11))
        elif feature == CalendarFeature.day_sine:
            return timeseries.map(lambda element: np.sin(np.pi * 2 * (element.day - 1) / element.days_in_month))
        elif feature == CalendarFeature.weekday_sine:
            return timeseries.map(lambda element: np.sin(np.pi * 2 * element.weekday() / 6))
        elif feature == CalendarFeature.hour_sine:
            return timeseries.map(lambda element: np.sin(np.pi * 2 * element.hour / 23))
        elif feature == CalendarFeature.month_cos:
            return timeseries.map(lambda element: np.cos(np.pi * 2 * (element.month - 1) / 11))
        elif feature == CalendarFeature.day_cos:
            return timeseries.map(lambda element: np.cos(np.pi * 2 * (element.day - 1) / element.days_in_month))
        elif feature == CalendarFeature.weekday_cos:
            return timeseries.map(lambda element: np.cos(np.pi * 2 * (element.weekday()) / 6))
        elif feature == CalendarFeature.hour_cos:
            return timeseries.map(lambda element: np.cos(np.pi * 2 * (element.hour) / 23))

    def get_params(self) -> Dict[str, object]:
        """ Get parameters of this calendar extraction processing step.

        :return: Json dict containing the parameters.
        """
        return {
            "continent": self.continent,
            "country": self.country,
            "features": self.features,
        }

    def set_params(self, continent: Optional[str] = None, country: Optional[str] = None,
                   features: Optional[List[CalendarFeature]] = None):
        """ Set parameters of the calendar extraction processing step.

        :param continent: Continent where the country or region is located
                          (important for importing calendar module).
        :type continent: str
        :param country: Country or region to use for holiday calendar (default 'Germany')
        :type country: str
        :param features: A list, which contains all features that should be calculated. Default all are calculated.
        :type features: List[str]
        :raises AttributeError: If 'continent' and/or 'country' is invalid.
        """

        if continent is not None:
            self.continent = continent
        if country is not None:
            self.country = country
        if features is not None:
            self.features: List[CalendarFeature] = features
        try:
            self.calendar = _init_calendar(self.continent, self.country)
        except UtilException as exc:
            raise WrongParameterException(exc.message,
                                          "Please set a valid country or continent for.", CalendarExtraction) from exc

    def transform(self, x: xr.DataArray) -> xr.DataArray:
        """ Add date features to xarray dataset as configured.

        :param x: Xarray dataset containing a timeseries specified by the object's 'time_index'
        :return: The xarray dataset with date features added.
        """

        time_index = _get_time_indexes(x)[0]
        data = [self._encode(feature, x[time_index].to_series()) for feature in self.features]
        return xr.DataArray(np.array(data).swapaxes(0, 1), coords=[getattr(x, time_index), self.features.copy()],
                            dims=[time_index, "features"])
