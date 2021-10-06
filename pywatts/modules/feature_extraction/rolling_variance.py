from pywatts.modules.feature_extraction.rolling_base import RollingBase, RollingGroupBy
import pandas as pd

class RollingVariance(RollingBase):
    """
     Module which calculates a rolling variance over a specific window size. Note, currently the smallest resolution of
     the generated profile is one minute.

     For the documentation of the methods see :class:`pywatts.modules.rolling_base.RollingBase`.

     :param name: Name of the new variable
     :type name: str
     :param window_size: Window size for which to calculate the variance
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

    def __init__(self, name: str = "RollingVar", window_size=24 * 7, window_size_unit="d",
                 group_by: RollingGroupBy = RollingGroupBy.No, continent: str = "Europe",
                 country: str = "Germany", closed="left", alpha=None):
        super().__init__(name=name, window_size=window_size, window_size_unit=window_size_unit, group_by=group_by,
                         continent=continent, country=country, closed=closed)
        self.alpha = alpha

    def _get_rolling(self, df):
        if self.alpha:
            ewm = lambda x: pd.Series(x).ewm(alpha=self.alpha).var().iloc[-1]
            return df.rolling(f"{self.window_size}{self.window_size_unit}", closed=self.closed).apply(ewm)
        else:
            return df.rolling(f"{self.window_size}{self.window_size_unit}", closed=self.closed).var()
