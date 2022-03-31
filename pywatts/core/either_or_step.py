import pandas as pd
from pywatts.core.base_step import BaseStep


class EitherOrStep(BaseStep):
    """
    This step merges the result of multiple input steps, by choosing the first step in the input list which
    contains data for the current data.

    :param input_step: The input_steps for the either_or_step
    :type input_step: List[BaseStep]
    """

    def __init__(self, input_steps):
        super().__init__(input_steps)
        self.name = "EitherOr"

    def _compute(self, start, end, minimum_data):
        input_data = self._get_input(start, end, minimum_data)
        return self._transform(input_data)

    def _get_input(self, start, batch, minimum_data=(0, pd.Timedelta(0))):
        inputs = []
        for step in self.input_steps.values():
            inp = step.get_result(start, batch, minimum_data=minimum_data)
            inputs.append(inp)
        return inputs

    def _transform(self, input_step):
        # Chooses the first input_step which calculation is not stopped.
        for in_step in input_step:
            if in_step is not None:
                # This buffer is never changed in this step. Consequently, no copy is necessary..
                return self._post_transform(in_step)

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
