from typing import Dict

import xarray as xr

from pywatts.core.base import BaseTransformer


class MissingValueDetector(BaseTransformer):
    """
    Module to detect missing values (NaN, NA)
    """

    def __init__(self, name: str = "missingValueDetector"):
        super().__init__(name)

    def get_params(self) -> Dict[str, object]:
        """
        Get params
        """
        return {}

    def set_params(self):
        """
        Set params
        """
        pass

    def transform(self, dataset: xr.DataArray) -> xr.DataArray:
        """
        Detects the indexes that correspond to the input having missing values

        :param dataset: Dataset in which missing values should be detected
        :type dataset: xr.DataArray
        :return: Returns a dataset with binary values, true if value is missing and false otherwise
        :rtype: xr.DataArray
        """

        return dataset.isnull()
