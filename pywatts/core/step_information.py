from pywatts.core.base_step import BaseStep


class StepInformation:
    """
    This steps contains information necesary for creating a pipeline and steps by the step factory

    :param step: The step
    :param pipeline: The pipeline
    """

    def __init__(self, step: BaseStep, pipeline):
        self.step = step
        self.pipeline = pipeline

    def __getitem__(self, item: str):
        from pywatts.core.pipeline_step import PipelineStep
        if isinstance(self.step, PipelineStep):
            self.step.last = False # TODO this should be a part of the step_factory
            return StepInformation(self.step.get_result_step(item), self.pipeline)
        else:
            return self
