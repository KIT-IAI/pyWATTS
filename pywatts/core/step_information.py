from typing import TYPE_CHECKING

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
        from pywatts.core.step import Step
        if isinstance(self.step, Step):
            self.step.last = False  # TODO this should be a part of the step_factory
            result_step = self.step.get_result_step(item)
            id = self.pipeline.add(module=result_step, input_ids=[self.step.id])
            result_step.id = id
            return StepInformation(result_step, self.pipeline)
        else:
            return self


class SummaryInformation:
    def __init__(self, step, pipeline):
        self.step = step
        self.pipeline = pipeline
