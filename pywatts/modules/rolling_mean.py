from enum import IntEnum
from typing import Dict, Optional

import xarray as xr

from pywatts.core.base import BaseTransformer
from pywatts.core.exceptions.wrong_parameter_exception import WrongParameterException
from pywatts.utils._workalendar_utils import _init_calendar
from pywatts.utils._xarray_time_series_utils import numpy_to_xarray


class RollingMeanGroupBy(IntEnum):
    No = 1
    WorkdayWeekend = 2
    WorkdayWeekendAndHoliday = 3


class RollingMean(BaseTransformer):
    """
     Module which calculates a rolling mean over a specific window size. Note, currently the smallest resolution of the
     generated profile is one minute.

     :param name: Name of the new variable
     :type name: str
     :param window_size: Window size for which to calculate the mean
     :type window_size: int
     :param window_size_unit: Unit of the window size (default: "d" [day])
     :type window_size_unit: str
     :param groupy_by: how the entries of the time series should be grouped
     :type group_by. RollingMeanGroupBy
     :param continent: If group_by is WorkdayAndHoliday: Continent where the country or region is located
                          (important for importing calendar module).
     :type continent: str
     :param country: If group_by is WorkdayAndHoliday: Country or region to use for holiday calendar (default 'Germany')
     :type country: str
     :param closed: If there array is closed left or right
     :type closed: str
    """

    def __init__(self, name: str = "RollingMean", window_size=24 * 7, window_size_unit="d",
                 group_by: RollingMeanGroupBy = RollingMeanGroupBy.No, continent: str = "Europe",
                 country: str = "Germany", closed="left"):

        super().__init__(name)
        self.window_size = window_size
        self.window_size_unit = "d"
        self.group_by = group_by
        self.window_size_unit = window_size_unit
        self.country = country
        self.continent = continent
        self.closed = closed
        self.cal = _init_calendar(self.continent, self.country)

    def get_params(self) -> Dict[str, object]:
        """
        Get the parameters of the rolling mean module as dict
        """
        return {
            "window_size": self.window_size,
            "window_size_unit": self.window_size_unit,
            "group_by": self.group_by,
            "country": self.country,
            "continent": self.continent,
        }

    def set_params(self, window_size: Optional[int] = None, window_size_unit: Optional[str] = None,
                   group_by: Optional[RollingMeanGroupBy] = None, continent: Optional[str] = None,
                   country: Optional[str] = None, closed: Optional[str] = None):
        """
        Set parameters of the rolling mean
        :param window_size: Window size for which to calculate the mean
        :type window_size: int
        :param window_size_unit: Unit of the window size (default: "d" [day])
        :type window_size_unit: str
        :param groupy_by: how the entries of the time series should be grouped
        :type group_by. RollingMeanGroupBy
        :param continent: If group_by is WorkdayAndHoliday: Continent where the country or region is located
                          (important for importing calendar module).
        :type continent: str
        :param country: If group_by is WorkdayAndHoliday: Country or region to use for holiday calendar (default 'Germany')
        :type country: str
        :param closed: If there array is closed left or right
        :type closed: str
        """
        if window_size:
            self.window_size = window_size
        if window_size_unit:
            self.window_size_unit = window_size_unit
        if group_by:
            self.group_by = group_by
        if closed:
            self.closed = closed
        if continent:
            self.continent = continent
        if country:
            self.country = country
        self.cal = _init_calendar(self.continent, self.country)

    def transform(self, x: xr.DataArray) -> xr.DataArray:
        """ Calculates a rolling mean

        :param x: Xarray dataset containing a timeseries specified by the object's 'time_index'
        :return: The xarray dataset with date features added.
        """

        df = x.to_dataframe("name")

        if self.group_by == RollingMeanGroupBy.No:
            rolling_mean = df.rolling(f"{self.window_size}{self.window_size_unit}", closed=self.closed).mean()
        elif self.group_by == RollingMeanGroupBy.WorkdayWeekend:
            mask = df.index.map(
                lambda element: element.minute + 1440 if element.weekday() >= 5 else element.hour).values
            rolling_mean = df.groupby(mask).rolling(f"{self.window_size}{self.window_size_unit}", closed=self.closed,
                                                    on=df.index).mean().reset_index(0).drop("level_0",
                                                                                            axis=1).sort_index()
        elif self.group_by == RollingMeanGroupBy.WorkdayWeekendAndHoliday:
            mask = df.index.map(lambda element:
                                element.minute + 1440 if self.cal.is_holiday(element) or element.weekday() >= 5
                                else element.hour).values
            rolling_mean = df.groupby(mask).rolling(f"{self.window_size}{self.window_size_unit}", closed=self.closed,
                                                    on=df.index).mean().reset_index(0).drop("level_0",
                                                                                            axis=1).sort_index()
        else:
            raise WrongParameterException(
                "GroupBy has to be RollingMeanGroupBy.No, RollingMeanGroupBy.WorkdayWeekend, "
                "RollingMeanGroupBy.WorkdayWeekendAndHoliday.",
                "During initialisation set either  RollingMeanGroupBy.No, RollingMeanGroupBy.WorkdayWeekend, or "
                "RollingMeanGroupBy.WorkdayWeekendAndHoliday.",
                module=RollingMean)

        rolling_mean.fillna(inplace=True, value=0)
        return numpy_to_xarray(rolling_mean.values.reshape((len(rolling_mean),)), x, self.name)
