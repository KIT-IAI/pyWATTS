from typing import Optional

import numpy as np
import xarray as xr
from pywatts_pipeline.core.callbacks.base_callback import BaseCallback


class NPYCallback(BaseCallback):
    """
    Callback to save the data as npy file.
    """

    def __init__(self, prefix: str, use_filemanager: Optional[bool] = None):
        """
        Initialise NPYCallback given a filename and
        optional use_filemanager flag.

        :param prefix: Prefix to use for the stored file.
        :type prefix: str
        :param use_filemanager: Flag to denote if the filemanager of the pipeline should be used.
        :type use_filemanager: Optional[bool]
        """
        if use_filemanager is None:
            # use base class default if use_filemanager is not set
            super().__init__()
        else:
            super().__init__(use_filemanager)
        self.prefix = prefix

    def __call__(self, data_dict: xr.DataArray):
        """
        Implementation of abstract __call__ base method to store the npy file.

        :param data_dict: Dict of DataArrays that should be stored as npy file.
        :type data_dict: Dict[str, xr.DataArray]
        """
        for key in data_dict:
            with open(self.get_path(f"{self.prefix}_{key}.npy"), "wb") as file:
                np.save(file, data_dict[key].values)
