import xarray as xr
from typing import Dict

from pywatts.callbacks.base_callback import BaseCallback


class PrintCallback(BaseCallback):
    """
    Print callback class to print out result data into terminal for debugging.

    :param BaseCallback: Base callback class.
    :type BaseCallback: BaseCallback
    """

    def __call__(self, data_dict: xr.DataArray):
        """
        Implementation of abstract base method to print out
        pipeline result data into terminal.

        :param data_dict: Dict of DataArrays that should be printed out into terminal.
        :type data_dict: Dict[str, xr.DataArray]
        """
        # NOTE: print out pandas arrays is a little bit more understandable IMO.
        print("\n# Print Callback")
        for key in data_dict:
            print(f"## {key}")
            print(data_dict[key].to_pandas())


class StatisticCallback(BaseCallback):
    """
    Statistic callback class to print out statistical information about the results
    into terminal for better understanding and debugging.

    :param BaseCallback: Base callback class.
    :type BaseCallback: BaseCallback
    """

    def __call__(self, data_dict: Dict[str, xr.DataArray]):
        """
        Implementation of abstract base method to print out
        pipeline statistical information of step results into terminal.

        :param data_dict: Dict of DataArrays that statistical information should be printed out.
        :type data_dict: Dict[str, xr.DataArray]
        """
        print("\n# Statistical Callback")
        for key in data_dict:
            print(f"## {key}")
            print(data_dict[key].to_pandas().describe())
