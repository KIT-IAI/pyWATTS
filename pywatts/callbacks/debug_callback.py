import xarray as xr

from pywatts.callbacks.base_callback import BaseCallback


class PrintCallback(BaseCallback):
    """
    Print callback class to print out result data into terminal for debugging.

    :param BaseCallback: Base callback class.
    :type BaseCallback: BaseCallback
    """

    def __call__(self, x: xr.DataArray):
        """
        Implementation of abstract base method to print out
        pipeline result data into terminal.

        :param x: Data that should be printed into terminal.
        :type x: xr.DataArray
        """
        # NOTE: print out pandas arrays is a little bit more understandable IMO.
        print(x.to_pandas())


class StatisticCallback(BaseCallback):
    """
    Statistic callback class to print out statistical information about the results
    into terminal for better understanding and debugging.

    :param BaseCallback: Base callback class.
    :type BaseCallback: BaseCallback
    """

    def __call__(self, x: xr.DataArray):
        """
        Implementation of abstract base method to print out
        pipeline step results into terminal.

        :param x: Data that should be printed into terminal.
        :type x: xr.DataArray
        """
        print(x.to_pandas().describe())
