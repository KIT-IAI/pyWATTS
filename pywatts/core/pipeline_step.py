from typing import Optional, List, Dict

from pywatts.core.base import Base
from pywatts.core.base_step import BaseStep
from pywatts.core.computation_mode import ComputationMode
from pywatts.core.filemanager import FileManager
from pywatts.core.pipeline import Pipeline
from pywatts.core.step import Step
import pandas as pd


class ResultStep(BaseStep):

    def __init__(self, input_steps, buffer_element: str):
        super().__init__(input_steps=input_steps)
        self.buffer_element = buffer_element

    def get_result(self, start: pd.Timestamp, end: Optional[pd.Timestamp]):
        return list(self.input_steps.values())[0].get_result(start, end, self.buffer_element)

    def get_json(self, fm: FileManager) -> Dict:
        json_dict = super(ResultStep, self).get_json(fm)
        json_dict["buffer_element"] = self.buffer_element
        return json_dict

    @classmethod
    def load(cls, stored_step: dict, inputs, targets, module, file_manager):
        """
        Load a stored ResultStep.

        :param stored_step: Informations about the stored step
        :param inputs: The input step of the stored step
        :param targets: The target step of the stored step
        :param module: The module wrapped by this step
        :return: Step
        """
        step = cls(inputs, stored_step["buffer_element"])
        step.id = stored_step["id"]
        step.name = stored_step["name"]
        step.last = stored_step["last"]
        return step


class PipelineStep(Step):
    """
    This step is necessary for subpipelining. Since it contains functionality for adding a pipeline as a
    subpipeline to a other pipeline.

    :param module: The module which is wrapped by the step-
    :type module: Pipeline
    :param input_step: The input_step of the module.
    :type input_step: Step
    :param file_manager: The file_manager which is used for storing data.
    :type file_manager: FileManager
    :param target: The step against which's output the module of the current step should be fitted. (Default: None)
    :type target: Optional[Step]
    :param computation_mode: The computation mode which should be for this step. (Default: ComputationMode.Default)
    :type computation_mode: ComputationMode
    :param plot: Flag if the result of this step should be plotted.
    :type plot: bool
    :param to_csv: Flag if the result of this step should be written in a csv file.
    :type to_csv: bool
    :param summary: Flag if a summary of the result should be printed.
    :type summary: bool
    :param condition: A callable which checks if the step should be executed with the current data.
    :type condition: Callable[xr.Dataset, xr.Dataset, bool]
    """
    module: Pipeline

    def __init__(self, module: Base, input_steps: Dict[str, BaseStep], file_manager, targets, plot,
                 summary,
                 computation_mode,
                 to_csv, condition, batch_size, train_if):

        super().__init__(module, input_steps, file_manager, targets=targets, plot=plot,
                         summary=summary,
                         computation_mode=computation_mode,
                         to_csv=to_csv, condition=condition, batch_size=batch_size, train_if=train_if)
        self.result_steps: Dict[str, ResultStep] = {}

    def set_computation_mode(self, computation_mode: ComputationMode):
        """
        Sets the computation mode of the step for the current run. Note that after reset the all mode is restored.
        Moreover, setting the computation_mode is only possible if the computation_mode is not set explicitly while
        adding the corresponding module to the pipeline.
        Moreover, it sets also the computation_mode of all steps in the subpipeline.

        :param computation_mode: The computation mode which should be set.
        :type computation_mode: ComputationMode
        """
        if self._original_compuation_mode == computation_mode.Default:
            self.computation_mode = computation_mode
            for step in self.module.id_to_step.values():
                step.set_computation_mode(computation_mode)

    def reset(self):
        """
        Resets all information of the step concerning a specific run. Furthermore, it resets also all steps
        of the subpipeline.
        """
        super().reset()
        for step in self.module.id_to_step.values():
            step.reset()

    def get_result(self, start: pd.Timestamp, end: Optional[pd.Timestamp], buffer_element: str = None):
        result = super().get_result(start, end)
        if buffer_element is None:
            return result
        return result[buffer_element]

    def get_result_step(self, item: str):

        if item not in self.result_steps:
            self.result_steps[item] = ResultStep(input_steps={"result": self}, buffer_element=item)
        return self.result_steps[item]
