from typing import Optional, Dict

from pywatts.core.computation_mode import ComputationMode
from pywatts.core.exceptions.kind_of_transform_does_not_exist_exception import \
    KindOfTransformDoesNotExistException, KindOfTransform
from pywatts.core.base import Base
from pywatts.core.step import BaseStep, Step


class ProbablisticStep(Step):
    """
    This step calls the inverse_transform method of the corresponding module.

    :param module: The module which is wrapped by the step-
    :type module: Base
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
    :param summary: Flag if the summary of the result of this step should be printed.
    :type summary: bool
    :param condition: A callable which checks if the step should be executed with the current data.
    :type condition: Callable[xr.Dataset, xr.Dataset, bool]
    """

    def __init__(self, module: Base, input_steps: Dict[str, BaseStep], file_manager, target=None,
                 computation_mode=ComputationMode.Default, plot: bool = False,
                 to_csv: bool = False, summary: bool = False, condition=None):
        super().__init__(module=module, input_steps=input_steps, target=target, file_manager=file_manager,
                         computation_mode=computation_mode,
                         plot=plot, to_csv=to_csv, summary=summary, condition=condition)

    def _transform(self, input_step):
        # Call probabilistic transform of the encapsulated module

        if not self.module.has_predict_proba:
            raise KindOfTransformDoesNotExistException(f"The module {self.module.name} has no probablisitic transform",
                                                       KindOfTransform.PROBABILISTIC_TRANSFORM)

        self._post_transform(self.module.predict_proba(input_step))
