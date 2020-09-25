from pywatts.core.computation_mode import ComputationMode
from pywatts.core.pipeline import Pipeline
from pywatts.core.step import Step


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
