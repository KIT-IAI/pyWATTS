from typing import List, Dict

from pywatts.core.base_step import BaseStep
import pandas as pd

from pywatts.core.filemanager import FileManager
from pywatts.utils._xarray_time_series_utils import _get_time_indeces


class EitherOrStep(BaseStep):
    """
    This step merges the result of multiple input steps, by choosing the first step in the input list which
    contains data for the current data.

    :param input_step: The input_steps for the either_or_step
    :type input_step: List[BaseStep]
    """

    def __init__(self, input_step: List[BaseStep]):
        super().__init__(list(input_step))
        self.name = "EitherOr"

    def _compute(self, start, end):
        input_data = self._get_input(start, end)
        return self._transform(input_data)

    def _get_input(self, start, batch):
        inputs = []
        # TODO why is input_steps here a list and not a dict?
        for step in self.input_steps:
            inp = step.get_result(start, batch)
            inputs.append(inp)
        return inputs

    def _transform(self, input_step):
        # Chooses the first input_step which calculation is not stopped.
        for in_step in input_step:
            if not in_step is None:
                # This buffer is never changed in this step. Consequently, no copy is necessary..
                self._post_transform(in_step)
                return

    @classmethod
    def load(cls, stored_step: dict, inputs, targets, module, file_manager):
        """
        Load the Either or step from a stored step.

        :param stored_step: Information about the stored either or step
        :param inputs: the input steps
        :param targets: Does not exist for eitherOr
        :param module: Does not exist for either or step
        :param file_manager: The filemanager used for saving informations.
        :return: The restored eitherOrStep
        """
        step = cls(inputs)
        step.id = stored_step["id"]
        step.name = stored_step["name"]
        step.last = stored_step["last"]
        return step

    def _should_stop(self, start, end):
        input_data = self._get_input(start, end)
        return input_data and (all(map(lambda x: x is None, input_data)))

    def get_json(self, fm: FileManager) -> Dict:
        """
        Returns a dictionary containing all information needed for restoring the step.

        :param fm: The filemanager which can be used by the step for storing the state of the step.
        :type fm: FileManager
        :return: A dictionary containing all information needed for restoring the step.
        :rtype: Dict
        """
        return {
            "target_ids": list(map(lambda x: x.id, self.targets.values())),
            "input_ids": [step.id for step in self.input_steps],
            "id": self.id,
            "module": self.__module__,
            "class": self.__class__.__name__,
            "name": self.name,
            "last": self.last,
            "computation_mode": int(self.computation_mode)
        }
