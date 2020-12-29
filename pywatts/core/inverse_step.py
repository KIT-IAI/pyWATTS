from typing import Callable, Optional, Dict

import xarray as xr

from pywatts.core.base_step import BaseStep
from pywatts.core.filemanager import FileManager
from pywatts.core.computation_mode import ComputationMode
from pywatts.core.exceptions.kind_of_transform_does_not_exist_exception import \
    KindOfTransformDoesNotExistException, KindOfTransform
from pywatts.core.step import Step
from pywatts.core.base import Base


class InverseStep(Step):
    """
    This step calls the inverse_transform method of the corresponding module.

    :param module: The module which is wrapped by the step-
    :type module: Base
    :param input_steps: The input_step of the module.
    :type input_steps: Step
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
    :param condition: A callable which checks if the step should be executed with the current data.
    :type condition: Callable[xr.Dataset, xr.Dataset, bool]
    """

    def __init__(self, module: Base, input_steps: Dict[str, BaseStep], file_manager: FileManager,
                 target: Optional[Base] = None,
                 computation_mode=ComputationMode.Default, plot: bool = False, to_csv: bool = False,
                 summary: bool = False,
                 condition: Callable[[xr.Dataset, xr.Dataset], bool] = None):
        super().__init__(module=module, input_steps=input_steps, file_manager=file_manager,
                         computation_mode=computation_mode,
                         target=target, plot=plot, to_csv=to_csv, summary=summary, condition=condition)

    def _transform(self, input_step):
        # Calls the inverse_transform of the encapsulated module, if the input is not stopped.

        if not self.module.has_inverse_transform:
            raise KindOfTransformDoesNotExistException(f"The module {self.module.name} has no inverse transform",
                                                       KindOfTransform.INVERSE_TRANSFORM)

        self._post_transform(self.module.inverse_transform(x=input_step))
