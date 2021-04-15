import inspect
from typing import Tuple, Dict, Union, List, Callable

import xarray as xr

from pywatts.core.base import Base
from pywatts.core.base_step import BaseStep
from pywatts.core.base_summary import BaseSummary
from pywatts.core.either_or_step import EitherOrStep
from pywatts.core.exceptions.step_creation_exception import StepCreationException
from pywatts.core.inverse_step import InverseStep
from pywatts.core.pipeline import Pipeline
from pywatts.core.pipeline_step import PipelineStep
from pywatts.core.probabilistic_step import ProbablisticStep
from pywatts.core.step import Step
from pywatts.core.step_information import StepInformation, SummaryInformation
from pywatts.callbacks import BaseCallback
from pywatts.core.summary_step import SummaryStep


class StepFactory:
    """
    A factory for creating the appropriate step for the current sitation.
    """

    def create_step(self,
                    module: Base,
                    kwargs: Dict[str, Union[StepInformation, Tuple[StepInformation, ...]]],
                    use_inverse_transform: bool, use_predict_proba: bool,
                    callbacks: List[Union[BaseCallback, Callable[[Dict[str, xr.DataArray]], None]]],
                    condition,
                    batch_size,
                    computation_mode,
                    train_if):
        """
        Creates a appropriate step for the current situation.

        :param module: The module which should be added to the pipeline
        :param kwargs: The input steps for the current step
        :param targets: The target steps for the currrent step
        :param use_inverse_transform: Should inverse_transform be called instead of transform
        :param use_predict_proba: Should probabilistic_transform be called instead of transform
        :param callbacks: Callbacks to use after results are processed.
        :param condition: A function returning True or False which indicates if the step should be performed
        :param batch_size: The size of the past time range which should be used for relearning the module
        :param computation_mode: The computation mode of the step
        :param train_if: A method for determining if the step should be fitted at a specific timestamp.
        :return: StepInformation
        """

        arguments = inspect.signature(module.transform).parameters.keys()

        if "kwargs" not in arguments and not isinstance(module, Pipeline):
            for argument in arguments:
                if argument not in kwargs.keys():
                    raise StepCreationException(
                        f"The module {module.name} miss {argument} as input. The module needs {arguments} as input. "
                        f"{kwargs} are given as input."
                        f"Add {argument}=<desired_input> when adding {module.name} to the pipeline.",
                        module
                    )

        # TODO needs to check that inputs are unambigious -> I.e. check that each input has only one output
        pipeline = self._check_ins(kwargs)

        input_steps, target_steps = self._split_input_target_steps(kwargs, pipeline)

        if isinstance(module, Pipeline):
            step = PipelineStep(module, input_steps, pipeline.file_manager, targets=target_steps,
                                callbacks=callbacks, computation_mode=computation_mode, condition=condition,
                                batch_size=batch_size, train_if=train_if)
        elif use_inverse_transform:
            step = InverseStep(module, input_steps, pipeline.file_manager, targets=target_steps,
                               callbacks=callbacks, computation_mode=computation_mode, condition=condition)
        elif use_predict_proba:
            step = ProbablisticStep(module, input_steps, pipeline.file_manager, targets=target_steps,
                                    callbacks=callbacks, computation_mode=computation_mode, condition=condition)
        else:
            step = Step(module, input_steps, pipeline.file_manager, targets=target_steps,
                        callbacks=callbacks, computation_mode=computation_mode, condition=condition,
                        batch_size=batch_size, train_if=train_if)

        step_id = pipeline.add(module=step,
                               input_ids=[step.id for step in input_steps.values()],
                               target_ids=[step.id for step in target_steps.values()])
        step.id = step_id

        if len(target_steps) > 1:
            step.last = False
            for target in target_steps:
                r_step = step.get_result_step(target)
                r_id = pipeline.add(module=step, input_ids=[step_id])
                r_step.id = r_id

        return StepInformation(step, pipeline)

    def _split_input_target_steps(self, kwargs, pipeline):
        input_steps: Dict[str, BaseStep] = dict()
        target_steps: Dict[str, BaseStep] = dict()
        for key, element in kwargs.items():
            if isinstance(element, StepInformation):
                element.step.last = False
                if key.startswith("target"):
                    target_steps[key] = element.step
                else:
                    input_steps[key] = element.step
                if isinstance(element.step, PipelineStep):
                    raise Exception(
                        f"Please specify which result of {element.step.name} should be used, since this steps"
                        f"may provide multiple results.")
            elif isinstance(element, tuple):
                if key.startswith("target"):
                    target_steps[key] = self._createEitherOrStep(element, pipeline).step
                else:
                    input_steps[key] = self._createEitherOrStep(element, pipeline).step
        return input_steps, target_steps

    def _createEitherOrStep(self, inputs: Tuple[StepInformation], pipeline):
        for input_step in inputs:
            input_step.step.last = False
        step = EitherOrStep({x.step.name + f"{i}": x.step for i, x in enumerate(inputs)})
        step_id = pipeline.add(module=step,
                               input_ids=list(map(lambda x: x.step.id, inputs)))
        step.id = step_id
        return StepInformation(step, pipeline)

    def _check_ins(self, kwargs):
        pipeline = None
        for input_step in kwargs.values():
            if isinstance(input_step, StepInformation):
                pipeline_temp = input_step.pipeline
                if len(input_step.step.targets) > 1:  # TODO define multi-output steps?
                    raise StepCreationException(
                        f"The step {input_step.step.name} has multiple outputs. "
                        "Adding such a step to the pipeline is ambigious. "
                        "Specifiy the desired column of your dataset by using step[<column_name>]",
                    )
            elif isinstance(input_step, Pipeline):
                raise StepCreationException(
                    "Adding a pipeline as input might be ambigious. "
                    "Specifiy the desired column of your dataset by using pipeline[<column_name>]",
                )
            elif isinstance(input_step, tuple):
                # We assume that a tuple consists only of step informations and do not contain a pipeline.
                pipeline_temp = input_step[0].pipeline

                if len(input_step[0].step.targets) > 1:
                    raise StepCreationException(
                        f"The step {input_step.step.name} has multiple outputs. Adding such a step to the pipeline is "
                        "ambigious. "
                        "Specifiy the desired column of your dataset by using step[<column_name>]",
                    )

                for step_information in input_step[1:]:

                    if len(step_information.step.targets) > 1:
                        raise StepCreationException(
                            f"The step {input_step.step.name} has multiple outputs. Adding such a step to the pipeline is "
                            "ambigious. "
                            "Specifiy the desired column of your dataset by using step[<column_name>]",
                        )
                    if not pipeline_temp == step_information.pipeline:
                        raise Exception()

            if pipeline_temp is None:
                raise Exception()  # TODO

            if pipeline is None:
                pipeline = pipeline_temp

            if not pipeline_temp == pipeline:
                raise StepCreationException(f"A step can only be part of one pipeline. Assure that all inputs {kwargs}"
                                            f"are part of the same pipeline.")
        return pipeline


    def create_summary(self,
                    module: BaseSummary,
                    kwargs: Dict[str, Union[StepInformation, Tuple[StepInformation, ...]]],
                    ) -> SummaryInformation:
        arguments = inspect.signature(module.transform).parameters.keys()

        if "kwargs" not in arguments and not isinstance(module, Pipeline):
            for argument in arguments:
                if argument not in kwargs.keys():
                    raise StepCreationException(
                        f"The module {module.name} miss {argument} as input. The module needs {arguments} as input. "
                        f"{kwargs} are given as input."
                        f"Add {argument}=<desired_input> when adding {module.name} to the pipeline.",
                        module
                    )

        # TODO needs to check that inputs are unambigious -> I.e. check that each input has only one output
        pipeline = self._check_ins(kwargs)
        input_steps, target_steps = self._split_input_target_steps(kwargs, pipeline)

        step = SummaryStep(module, input_steps, pipeline.file_manager,)

        step_id = pipeline.add(module=step,
                               input_ids=[step.id for step in input_steps.values()],
                               target_ids=[step.id for step in target_steps.values()])

        return SummaryInformation(step, pipeline)
