import inspect
from typing import Tuple, Union, Dict

from pywatts.core.base import Base
from pywatts.core.base_step import BaseStep
from pywatts.core.either_or_step import EitherOrStep
from pywatts.core.inverse_step import InverseStep
from pywatts.core.pipeline import Pipeline
from pywatts.core.pipeline_step import PipelineStep
from pywatts.core.probabilistic_step import ProbablisticStep
from pywatts.core.step import Step
from pywatts.core.step_information import StepInformation


class StepFactory:
    """
    A factory for creating the appropriate step for the current sitation.
    """

    def create_step(self,
                    module: Base,
                    kwargs: Dict[str, Union[StepInformation, Tuple[StepInformation, ...]]],
                    targets: Union[StepInformation, Tuple[StepInformation, ...], Pipeline],
                    use_inverse_transform: bool, use_predict_proba: bool, plot: bool, to_csv: bool, summary: bool,
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
        :param plot: Should the result be plotted
        :param to_csv:  Should the result be written in a csv file
        :param condition: A function returning True or False which indicates if the step should be performed
        :param batch_size: The size of the past time range which should be used for relearning the module
        :param computation_mode: The computation mode of the step
        :param train_if: A method for determining if the step should be fitted at a specific timestamp.
        :return: StepInformation
        """

        arguments = inspect.signature(module.transform).parameters.keys()
        # TODO Raise an exception if no kwargs are provided
        # TODO Check if kwargs is Dict and not a pipeline (e.g. when passing SKLearnWrapper(..)(x=self.pipeline))
        if not "kwargs" in arguments and not isinstance(module, Pipeline):

            for kwarg in arguments:
                assert kwarg in kwargs.keys()
        # TODO CHeck that arguemtns are in inputs

        pipeline = self._check_ins(kwargs, targets)

        input_steps: Dict[str, BaseStep] = dict()

        for key, element in kwargs.items():
            if isinstance(element, StepInformation):
                input_steps[key] = element.step
                if isinstance(element.step, PipelineStep):
                    raise Exception(
                        f"Please specify which result of {element.step.name} should be used, since this steps"
                        f"may provide multiple results.")
            elif isinstance(element, tuple):
                input_steps[key] = self._createEitherOrStep(element, pipeline).step

        if targets and isinstance(targets, tuple):
            target = self._createEitherOrStep(targets).step
        elif targets:
            target = targets.step
        else:
            target = None

        for input_step in input_steps.values():
            input_step.last = False

        if target:
            target.last = False

        if isinstance(module, Pipeline):
            step = PipelineStep(module, input_steps, pipeline.file_manager, target=target, plot=plot, summary=summary,
                                computation_mode=computation_mode,
                                to_csv=to_csv, condition=condition, batch_size=batch_size, train_if=train_if)
        elif use_inverse_transform:
            step = InverseStep(module, input_steps, pipeline.file_manager, target, computation_mode=computation_mode,
                               plot=plot, summary=summary,
                               to_csv=to_csv, condition=condition)
        elif use_predict_proba:
            step = ProbablisticStep(module, input_steps, pipeline.file_manager, target,
                                    computation_mode=computation_mode, plot=plot, summary=summary,
                                    to_csv=to_csv, condition=condition)
        else:
            step = Step(module, input_steps, pipeline.file_manager, target=target, plot=plot, summary=summary,
                        computation_mode=computation_mode,
                        to_csv=to_csv, condition=condition, batch_size=batch_size, train_if=train_if)

        if target:
            step_id = pipeline.add(module=step,
                                   input_ids=[],
                                   target_ids=[])
        else:
            step_id = pipeline.add(module=step,
                                   input_ids=[])
        step.id = step_id

        return StepInformation(step, pipeline)

    def _createEitherOrStep(self, inputs: Tuple[StepInformation], pipeline):
        for input_step in inputs:
            input_step.step.last = False
        step = EitherOrStep(list(map(lambda x: x.step, inputs)))
        step_id = pipeline.add(module=step,
                               input_ids=list(map(lambda x: x.step.id, inputs)))
        step.id = step_id
        return StepInformation(step, pipeline)

    # def _createCollectStep(self, inputs: List[Union[StepInformation, Tuple[StepInformation]]]):
    #     pipeline = self._check_ins(list(inputs), [])
    #     final_inputs = []
    #     for input_step in inputs:
    #         if isinstance(input_step, Tuple):
    #             input_step = self._createEitherOrStep(input_step)
    #         input_step.step.last = False
    #         final_inputs.append(input_step.step)
    #
    #     step = CollectStep(final_inputs)
    #     step_id = pipeline.add(module=step,
    #                            input_ids=list(map(lambda x: x.id, final_inputs)))
    #     step.id = step_id
    #     return StepInformation(step, pipeline)

    def _check_ins(self, kwargs, target):
        pipeline = None
        for key, input_step in kwargs.items():
            if isinstance(input_step, StepInformation):
                pipeline_temp = input_step.pipeline
            elif isinstance(input_step, Pipeline):
                # TODO custom exception needded here...
                raise Exception("This might be ambigious if you input data. Specifiy the desired column of your dataset by using pipeinling[<column_name>]")
            elif isinstance(input_step, tuple):
                # We assume that a tuple consists only of step informations and do not contain a pipeline.
                pipeline_temp = input_step[0].pipeline
                for step_information in input_step[1:]:
                    if not pipeline_temp == step_information.pipeline:
                        raise Exception()

            if pipeline_temp is None:
                raise Exception()

            if pipeline is None:
                pipeline = pipeline_temp

            if not pipeline_temp == pipeline:
                # TODO
                raise Exception()

        if isinstance(target, StepInformation):
            pipeline_temp = target.pipeline
        elif isinstance(target, Tuple):
            # We assume that a tuple consists only of step informations and do not contain a pipeline.
            pipeline_temp = target[0].pipeline
            for step_information in target[1:]:
                if not pipeline_temp == step_information.pipeline:
                    raise Exception()
        elif isinstance(target, Pipeline):
            pipeline_temp = target

        if target and not pipeline_temp == pipeline:
            raise Exception()

            # raise StepCreationException(f"A step information can only be part of one pipeline. "
            #                             f"Assert that you added {input_step.step.name} to the correct pipeline. "
            #                             f"However, if you want to use the module {input_step.step.name} in"
            #                             f"distinct pipeine. Assert that you add the module multiple times and not "
            #                             f"the step_information.",
            #                             )
            #
            # raise StepCreationException(f"A step information can only be part of one pipeline"
            #                             f"Assert that you added {target.step.name} to the correct pipeline. "
            #                             f"However,"
            #                             f"if you want to use the module {target.step.name} in distinct pipeine. "
            #                             f"Assert that you add the module multiple times and not the "
            #                             f"step_information.",
            #                             )
        return pipeline
