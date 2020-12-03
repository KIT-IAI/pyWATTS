from typing import List

from pywatts.core.base_step import BaseStep
import pandas as pd

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
        for step in self.input_steps:
            inp = step.get_result(start, batch)
            inputs.append(inp)
        return inputs

    def further_elements(self, counter: pd.Timestamp) -> bool:
        """
        Checks if there exist at least one data for the time after counter.

        :param counter: The timestampe for which it should be tested if there exist further data after it.
        :type counter: pd.Timestamp
        :return: True if there exist further data
        :rtype: bool
        """
        if self.buffer is None or counter < self.buffer.indexes[_get_time_indeces(self.buffer)[0]][-1]:
            return True
        for input_step in self.input_steps:
            if not input_step.further_elements(counter):
                return False
        for target_step in self.targets:
            if not target_step.further_elements(counter):
                return False
        return True

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
