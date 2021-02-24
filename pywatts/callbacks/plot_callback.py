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

    def __init__(self, filename: str, use_filemanager: Optional[bool] = None):
        """
        Initialise line plot callback object given a filename and
        optional use_filemanager flag.

        :param filename: Filename to use for the line plot output file.
        :type filename: str
        :param use_filemanager: Flag to denote if the filemanager of the pipeline should be used.
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
        Implementation of abstract __call__ base method to save a line plot.

        :param x: Data that should be plotted and saved.
        :type x: xr.DataArray
        """
        x.to_pandas().plot.line()
        plt.savefig(self.get_path(self.filename))
        plt.close()


class ImagePlotCallback(BaseCallback):
    """
    Callback to save an image.

    :param BaseCallback: Base callback class.
    :type BaseCallback: BaseCallback
    """

    def __init__(self, filename: str, use_filemanager: Optional[bool] = None):
        """
        Initialise image plot callback object given a filename and
        optional use_filemanager flag.

        :param filename: Filename to use for the line plot output file.
        :type filename: str
        :param use_filemanager: Flag to denote if the filemanager of the pipeline should be used.
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
        Implementation of abstract __call__ base method to save an image plot.

        :param x: Data that should be plotted and saved.
        :type x: xr.DataArray
        """
        img = x.to_pandas().to_numpy()
        img = img.T
        plt.imshow(img)
        plt.savefig(self.get_path(self.filename))
        plt.close()
