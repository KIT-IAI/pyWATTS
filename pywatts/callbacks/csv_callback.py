from typing import Optional, Dict

import xarray as xr

from pywatts.callbacks.base_callback import BaseCallback


class CSVCallback(BaseCallback):
    """
    Callback class to save csv files.

    :param BaseCallback: Base callback as parent class.
    :type BaseCallback: BaseCallback
    """

    def __init__(self, prefix: str, use_filemanager: Optional[bool] = None):
        """
        Initialise csv callback class given a prefix and optional use_filemanager flag.

        :param prefix: Prefix of the CSV file that should be written.
        :type prefix: str
        :param use_filemanager: Optional flag to set if the filemanager of the pipeline should be used.
        :type use_filemanager: Optional[bool]
        """
        if use_filemanager is None:
            # use base class default if use_filemanager is not set
            super().__init__()
        else:
            super().__init__(use_filemanager)
        self.prefix = prefix

    def __call__(self, data_dict: Dict[str, xr.DataArray]):
        """
        Implementation of abstract base __call__ method
        to write the csv file to a given location based on the filename.

        :param data_dict: Dict of DataArrays that should be written to CSV files.
        :type data_dict: Dict[str, xr.DataArray]
        """
        for key in data_dict:
            data_dict[key].to_pandas().to_csv(self.get_path(f"{self.prefix}_{key}.csv"))
