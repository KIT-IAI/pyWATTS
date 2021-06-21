from pywatts.core.exceptions.kind_of_transform_does_not_exist_exception import \
    KindOfTransformDoesNotExistException, KindOfTransform
from pywatts.core.step import Step


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
    :param callbacks: Callbacks to use after results are processed.
    :type callbacks: List[Union[BaseCallback, Callable[[Dict[str, xr.DataArray]], None]]]
    :param condition: A callable which checks if the step should be executed with the current data.
    :type condition: Callable[xr.DataArray, xr.DataArray, bool]
    """

    def _transform(self, input_step):
        # Calls the inverse_transform of the encapsulated module, if the input is not stopped.

        if not self.module.has_inverse_transform:
            raise KindOfTransformDoesNotExistException(f"The module {self.module.name} has no inverse transform",
                                                       KindOfTransform.INVERSE_TRANSFORM)

        return self._post_transform(self.module.inverse_transform(**input_step))
