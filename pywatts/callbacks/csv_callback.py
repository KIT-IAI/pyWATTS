from typing import Optional

import xarray as xr

from pywatts.callbacks.base_callback import BaseCallback


class CSVCallback(BaseCallback):
    """
    Callback class to save csv files.

    :param BaseCallback: Base callback as parent class.
    :type BaseCallback: BaseCallback
    """

    def __init__(self, filename: str, use_filemanager: Optional[bool] = None):
        """
        Initialise csv callback class given a filename and optional use_filemanager flag.

        :param filename: Name of the CSV file that should be written.
        :type filename: str
        :param use_filemanager: Optional flag to set if the filemanager of the pipeline should be used.
        :type use_filemanager: Optional[bool]
        """
        if use_filemanager is None:
            # use base class default if use_filemanager is not set
            super().__init__()
        else:
            super().__init__(use_filemanager)
        self.filename = filename

    def __call__(self, x: xr.DataArray):
        """
        Implementation of abstract base __call__ method
        to write the csv file to a given location based on the filename.

        :param x: Data that should be saved as a CSV file.
        :type x: xr.DataArray
        """
        x.to_pandas().to_csv(self.get_path(self.filename))
