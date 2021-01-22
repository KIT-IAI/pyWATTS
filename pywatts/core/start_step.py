from typing import Dict

from pywatts.core.base_step import BaseStep
from pywatts.core.filemanager import FileManager

from pywatts.utils._xarray_time_series_utils import _get_time_indeces


class StartStep(BaseStep):
    """
    Start Step of the pipeline.
    """

    def __init__(self, index: str):
        super().__init__()
        self.name = "StartStep"
        self.index = index

    @classmethod
    def load(cls, stored_step: dict, inputs, targets, module, file_manager):
        """
        A classmethod which reloads a previously stored step.

        :param stored_step:
        :param inputs:
        :param targets:
        :param module:
        :return:
        """
        step = cls(index=stored_step["index"])
        step.id = stored_step["id"]
        step.name = stored_step["name"]
        step.last = stored_step["last"]
        return step

    def further_elements(self, counter):
        """
        Checks if there exist at least one data for the time after counter.

        :param counter: The timestamp for which it should be tested if there exist further data after it.
        :type counter: pd.Timestamp
        :return: True if there exist further data
        :rtype: bool
        """
        indeces = _get_time_indeces(self.buffer)
        if len(indeces) == 0 or counter > self.buffer.indexes[indeces[0]][-1]:
            return False
        else:
            return True

    def get_json(self, fm: FileManager) -> Dict:
        json = super().get_json(fm)
        json["index"] = self.index
        return json
