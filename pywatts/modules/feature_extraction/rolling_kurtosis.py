from pywatts.modules.feature_extraction.rolling_base import RollingBase


class RollingKurtosis(RollingBase):
    """
     Module which calculates a rolling kurtosis over a specific window size. Note, currently the smallest resolution of
     the generated profile is one minute.

     For the documentation of the methods see :class:`pywatts.modules.rolling_base.RollingBase`.

     :param name: Name of the new variable
     :type name: str
     :param window_size: Window size for which to calculate the kurtosis
     :type window_size: int
     :param window_size_unit: Unit of the window size (default: "d" [day])
     :type window_size_unit: str
     :param groupy_by: how the entries of the time series should be grouped
     :type group_by. RollingGroupBy
     :param continent: If group_by is WorkdayAndHoliday: Continent where the country or region is located
                          (important for importing calendar module).
     :type continent: str
     :param country: If group_by is WorkdayAndHoliday: Country or region to use for holiday calendar (default 'Germany')
     :type country: str
     :param closed: If there array is closed left or right
     :type closed: str
    """

    def _get_rolling(self, df):
        return df.rolling(f"{self.window_size}{self.window_size_unit}", closed=self.closed).kurt()
