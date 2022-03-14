from pywatts.core.run_setting import RunSetting
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
    :param callbacks: Callbacks to use after results are processed.
    :type callbacks: List[Union[BaseCallback, Callable[[Dict[str, xr.DataArray]], None]]]
    :param condition: A callable which checks if the step should be executed with the current data.
    :type condition: Callable[xr.DataArray, xr.DataArray, bool]
    """
    module: Pipeline

    def set_run_setting(self, run_setting: RunSetting):
        """
        Sets the run settings of the step for the current run. Note that after reset old setting is restored.
        Moreover, setting the computation_mode is only possible if the computation_mode is not set explicitly while
        adding the corresponding module to the pipeline.
        Moreover, it sets also the computation_mode of all steps in the subpipeline.

        :param computation_mode: The computation mode which should be set.
        :type computation_mode: ComputationMode
        """
        self.current_run_setting = self.default_run_setting.update(run_setting=run_setting)
        for step in self.module.id_to_step.values():
            step.set_run_setting(run_setting)
        self.module.current_run_setting = self.current_run_setting

    def reset(self, keep_buffer=False):
        """
        Resets all information of the step concerning a specific run. Furthermore, it resets also all steps
        of the subpipeline.
        """
        super().reset(keep_buffer=keep_buffer)
        for step in self.module.id_to_step.values():
            step.reset(keep_buffer=keep_buffer)
