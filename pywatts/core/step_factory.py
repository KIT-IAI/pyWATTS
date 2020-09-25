from typing import Tuple, List, Generator, Union

from pywatts.core.collect_step import CollectStep
from pywatts.core.either_or_step import EitherOrStep
from pywatts.core.exceptions.step_creation_exception import StepCreationException
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
                    module,
                    inputs: Union[
                        List[Union[StepInformation, Tuple[StepInformation]]], Pipeline, List[Pipeline], StepInformation,
                        Tuple[StepInformation]],
                    targets: Union[
                        List[Union[StepInformation, Tuple[StepInformation]]], StepInformation, Tuple[StepInformation]],
                    use_inverse_transform: bool, use_predict_proba: bool, plot: bool, to_csv: bool, summary:bool,
                    condition,
                    batch_size,
                    computation_mode,
                    train_if):
        """
        Creates a appropriate step for the current situation.

        :param module: The module which should be added to the pipeline
        :param inputs: The input steps for the current step
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
        if targets is None:
            targets = []

        pipeline = self._check_ins(inputs, targets)
        if isinstance(inputs, Pipeline):
            input_step = pipeline.start_step
        elif isinstance(inputs, StepInformation):
            input_step = inputs.step
        elif isinstance(inputs, list):
            if len(inputs) == 0:
                raise StepCreationException(f"There has to be at least one input for a module"
                                            f"Check if the input list while adding the module {module.name} is empty",
                                            module.name)
            elif len(inputs) > 1:
                input_step = self._createCollectStep(inputs).step
            elif isinstance(inputs[0], tuple):
                input_step = self._createEitherOrStep(inputs[0]).step
            elif isinstance(inputs[0], Pipeline):
                input_step = pipeline.start_step
            else:
                input_step = inputs[0].step
        else:
            raise StepCreationException(
                f"The inputs of {module.name} is not a pipeline, list of StepInformation or a list of tuples of "
                f"stepinformation. Try to provide the stepinformations of the previous modules",
                module.name)

        if len(targets) > 1:
            target = self._createCollectStep(targets).step
        elif targets and isinstance(targets[0], tuple):
            target = self._createEitherOrStep(targets[0]).step
        elif targets:
            target = targets[0].step
        else:
            target = None

        if input_step is not None:
            input_step.last = False
        if target is not None:
            target.last = False
        if isinstance(module, Pipeline):
            step = PipelineStep(module, input_step, pipeline.file_manager, target=target, plot=plot, summary=summary,
                                computation_mode=computation_mode,
                                to_csv=to_csv, condition=condition, batch_size=batch_size, train_if=train_if)
        elif use_inverse_transform:
            step = InverseStep(module, input_step, pipeline.file_manager, target, computation_mode=computation_mode,
                               plot=plot,  summary=summary,
                               to_csv=to_csv, condition=condition)
        elif use_predict_proba:
            step = ProbablisticStep(module, input_step, pipeline.file_manager, target,
                                    computation_mode=computation_mode, plot=plot, summary=summary,
                                    to_csv=to_csv, condition=condition)
        else:
            step = Step(module, input_step, pipeline.file_manager, target=target, plot=plot,  summary=summary,
                        computation_mode=computation_mode,
                        to_csv=to_csv, condition=condition, batch_size=batch_size, train_if=train_if)

        if target:
            step_id = pipeline.add(module=step,
                                   input_ids=[step.inputs[0].id],
                                   target_ids=[step.targets[0].id])
        else:
            step_id = pipeline.add(module=step,
                                   input_ids=[step.inputs[0].id])
        step.id = step_id

        return StepInformation(step, pipeline)

    def _createCollectStep(self, inputs: List[Union[StepInformation, Tuple[StepInformation]]]):
        pipeline = self._check_ins(list(inputs), [])
        final_inputs = []
        for input_step in inputs:
            if isinstance(input_step, Tuple):
                input_step = self._createEitherOrStep(input_step)
            input_step.step.last = False
            final_inputs.append(input_step.step)

        step = CollectStep(final_inputs)
        step_id = pipeline.add(module=step,
                               input_ids=list(map(lambda x: x.id, final_inputs)))
        step.id = step_id
        return StepInformation(step, pipeline)

    def _createEitherOrStep(self, inputs: Tuple[StepInformation]):
        pipeline = self._check_ins(list(inputs), [])
        for input_step in inputs:
            input_step.step.last = False
        step = EitherOrStep(list(map(lambda x: x.step, inputs)))
        step_id = pipeline.add(module=step,
                               input_ids=list(map(lambda x: x.step.id, inputs)))
        step.id = step_id
        return StepInformation(step, pipeline)

    def _check_ins(self, inputs, targets):
        if isinstance(inputs, StepInformation):
            return inputs.pipeline
        if isinstance(inputs, Pipeline):
            return inputs
        elif isinstance(inputs, StepInformation):
            inputs = [inputs]
        elif isinstance(inputs, Tuple):
            inputs = [inputs]
        elif len(inputs) == 1 and isinstance(inputs[0], Pipeline):
            return inputs[0]

        if isinstance(inputs, StepInformation):
            targets = [targets]
        elif isinstance(inputs, Tuple):
            targets = [targets]

        flatten_input = list(self._flatten_input(inputs))
        flatten_target = list(self._flatten_input(targets))
        pipeline = flatten_input[0].pipeline
        for input_step in flatten_input:
            if not pipeline is input_step.pipeline:
                raise StepCreationException(f"A step information can only be part of one pipeline. "
                                            f"Assert that you added {input_step.step.name} to the correct pipeline. "
                                            f"However, if you want to use the module {input_step.step.name} in"
                                            f"distinct pipeine. Assert that you add the module multiple times and not "
                                            f"the step_information.",
                                            )
        for target in flatten_target:
            if not pipeline is target.pipeline:
                raise StepCreationException(f"A step information can only be part of one pipeline"
                                            f"Assert that you added {target.step.name} to the correct pipeline. "
                                            f"However,"
                                            f"if you want to use the module {target.step.name} in distinct pipeine. "
                                            f"Assert that you add the module multiple times and not the "
                                            f"step_information.",
                                            )
        return pipeline

    def _flatten_input(self, container) -> Generator[StepInformation, None, None]:
        # https://stackoverflow.com/questions/10823877/what-is-the-fastest-way-to-flatten-arbitrarily-nested-lists-in-python
        for i in container:
            if isinstance(i, (list, tuple)):
                for j in self._flatten_input(i):
                    yield j
            else:
                yield i
