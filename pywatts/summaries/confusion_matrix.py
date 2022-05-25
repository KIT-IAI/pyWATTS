from typing import Dict

from pywatts.core.base_summary import BaseSummary
from pywatts.core.filemanager import FileManager

import xarray as xr
from pywatts.core.summary_object import SummaryObjectTable
from sklearn.metrics import confusion_matrix


class ConfusionMatrix(BaseSummary):
    """
    Summary to calculate the confusion matrix

    :param name: Name of the confusion matrix
    :type name: str
    """
    def __init__(self, name: str = "Confusion Matrix"):
        super().__init__(name)

    def transform(self, file_manager: FileManager, gt: xr.DataArray, **kwargs: xr.DataArray) -> SummaryObjectTable:
        """
        Calculates the confusion matrix for all predictions
        :param file_manager: The filemanager
        :type file_manager: FileManager
        :param gt: The ground truth data
        :type gt: xr.DataArray
        :param kwargs: The predictions
        :type kwargs: xr.DataArray
        :return: A summary containing all confusion matrices
        :rtype: SummaryObjectTable
        """
        summary = SummaryObjectTable("Confusion Matrix")
        for key, value in kwargs.items():
            cm = confusion_matrix(gt, value.astype(int))
            summary.set_kv(key, cm)
        return summary

    def get_params(self) -> Dict[str, object]:
        """
        Get all parameters of the Confusion Matrix Summary
        """
        return {}

    def set_params(self):
        """
        Set all parameters of the Confusion Matrix Summary
        """
        pass