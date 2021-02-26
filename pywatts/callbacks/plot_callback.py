from typing import Optional

import xarray as xr

import matplotlib.pyplot as plt

from pywatts.callbacks.base_callback import BaseCallback


class LinePlotCallback(BaseCallback):
    """
    Callback to save a line plot.

    :param BaseCallback: Base callback class.
    :type BaseCallback: BaseCallback
    """

    def __init__(self, prefix: str, use_filemanager: Optional[bool] = None):
        """
        Initialise line plot callback object given a filename and
        optional use_filemanager flag.

        :param prefix: Prefix to use for the line plot output file.
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
        Implementation of abstract __call__ base method to save a line plot.

        :param data_dict: Dict of DataArrays that should be plotted.
        :type data_dict: Dict[str, xr.DataArray]
        """
        for key in data_dict:
            data_dict[key].to_pandas().plot.line()
            plt.tight_layout()
            plt.savefig(self.get_path(f"{self.prefix}_{key}.png"))
            plt.close()


class ImagePlotCallback(BaseCallback):
    """
    Callback to save an image.

    :param BaseCallback: Base callback class.
    :type BaseCallback: BaseCallback
    """

    def __init__(self, prefix: str, use_filemanager: Optional[bool] = None):
        """
        Initialise image plot callback object given a filename and
        optional use_filemanager flag.

        :param prefix: Prefix to use for the line plot output file.
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
        Implementation of abstract __call__ base method to save an image plot.

        :param data_dict: Dict of DataArrays that should be plotted.
        :type data_dict: Dict[str, xr.DataArray]
        """
        for key in data_dict:
            img = data_dict[key].to_pandas().to_numpy()
            if len(img.shape) > 1:
                img = img.T
                plt.imshow(img)
                plt.tight_layout()
                plt.savefig(self.get_path(f"{self.prefix}_{key}.png"))
                plt.close()
