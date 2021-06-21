import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict

import pandas as pd
import xarray as xr

from pywatts.core.computation_mode import ComputationMode
from pywatts.core.filemanager import FileManager
from pywatts.utils._xarray_time_series_utils import _get_time_indeces

logger = logging.getLogger(__name__)


class BaseStep(ABC):
    """
    The base class of all steps.
    :param input_steps: The input steps
    :type input_steps: Optional[Dict[str, BaseStep]]
    :param targets: The target steps
    :type targets: Optional[Dict[str, BaseStep]]
    :param condition: A function which evaluates to False or True for detecting if the module should be executed.
    :type condition: Callable
    :param computation_mode: The computation mode for this module
    :type computation_mode: ComputationMode
    """

    def __init__(self, input_steps: Optional[Dict[str, "BaseStep"]] = None,
                 targets: Optional[Dict[str, "BaseStep"]] = None, condition=None,
                 computation_mode=ComputationMode.Default):
        self._original_compuation_mode = computation_mode
        self.computation_mode = computation_mode
        self.input_steps: Dict[str, "BaseStep"] = dict() if input_steps is None else input_steps
        self.targets: Dict[str, "BaseStep"] = dict() if targets is None else targets
        self.condition = condition
        self.cached_result = None, None, None

        self.name = "BaseStep"

        self.id = -1
        self.finished = False
        self.last = True
        self._current_end = None
        self.buffer: Dict[str, xr.DataArray] = {}
        self.training_time = None

    def get_result(self, start: pd.Timestamp, end: Optional[pd.Timestamp], buffer_element: str = None,
                   return_all=False):
        """
        This method is responsible for providing the result of this step.
        Therefore,
        this method triggers the get_input and get_target data methods.
        Additionally, it triggers the computations and checks if all data are processed.

        :param start: The start date of the requested results of the step
        :type start: pd.Timedstamp
        :param end: The end date of the requested results of the step (exclusive)
        :type end: Optional[pd.Timestamp]
        :param buffer_element: if the buffer of the step contains multiple results, this determines the result which is
                               returned.
        :type buffer_element: str
        :param return_all: Flag that indicates if all results in the buffer should be returned.
        :type return_all: bool
        :return: The resulting data or None if no data are calculated
        """
        # Check if step should be executed.
        if self._should_stop(start, end):
            return None

        # Trigger fit and transform if necessary
        if not self.finished and not (end is not None and self._current_end is not None and end <= self._current_end):
            if not self.buffer or not self._current_end or end > self._current_end:
                self.cached_result = self._compute(start, end), start, end
                self._current_end = end
            if not end:
                self.finished = True
            else:
                self.finished = not self.further_elements(end)
            # Only call callbacks if the step is finished
            if self.finished:
                self._callbacks()

        if self.cached_result[0] is not None and self.cached_result[1] == start and self.cached_result[2] == end:
            return self.cached_result[0] if return_all else self.cached_result[0][
                buffer_element] if buffer_element is not None else list(self.cached_result[0].values())[0]
        return self._pack_data(start, end, buffer_element, return_all=return_all)

    def _compute(self, start, end) -> Dict[str, xr.DataArray]:
        pass

    def further_elements(self, counter: pd.Timestamp) -> bool:
        """
        Checks if there exist at least one data for the time after counter.

        :param counter: The timestampe for which it should be tested if there exist further data after it.
        :type counter: pd.Timestamp
        :return: True if there exist further data
        :rtype: bool
        """
        if not self.buffer or all(
                [counter < b.indexes[_get_time_indeces(self.buffer)[0]][-1] for b in self.buffer.values()]):
            return True
        for input_step in self.input_steps.values():
            if not input_step.further_elements(counter):
                return False
        for target_step in self.targets.values():
            if not target_step.further_elements(counter):
                return False
        return True

    def _pack_data(self, start, end, buffer_element=None, return_all=False):
        # Provide requested data
        time_index = _get_time_indeces(self.buffer)
        if end and start and end > start:
            index = list(self.buffer.values())[0].indexes[time_index[0]]
            start = max(index[0], start.to_numpy())
            # After sel copy is not needed, since it returns a new array.
            if buffer_element is not None:
                return self.buffer[buffer_element].sel(
                    **{time_index[0]: index[(index >= start) & (index < end.to_numpy())]})
            elif return_all:
                return {key: b.sel(**{time_index[0]: index[(index >= start) & (index < end.to_numpy())]}) for
                        key, b in self.buffer.items()}
            else:
                return list(self.buffer.values())[0].sel(
                    **{time_index[0]: index[(index >= start) & (index < end.to_numpy())]})
        else:
            self.finished = True
            if buffer_element is not None:
                return self.buffer[buffer_element].copy()
            elif return_all:
                return self.buffer.copy()
            else:
                return list(self.buffer.copy().values())[0]

    def _transform(self, input_step):
        pass

    def _fit(self, input_step, target_step):
        pass

    def _callbacks(self):
        pass

    def _post_transform(self, result):
        if isinstance(result, dict) and len(result) <= 1:
            result = {self.name: list(result.values())[0]}
        elif not isinstance(result, dict):
            result = {self.name: result}

        if not self.buffer:
            self.buffer = result
        else:
            # Time dimension is mandatory, consequently there dim has to exist
            dim = _get_time_indeces(result)[0]
            for key in self.buffer.keys():
                self.buffer[key] = xr.concat([self.buffer[key], result[key]], dim=dim)
        return result

    def get_json(self, fm: FileManager) -> Dict:
        """
        Returns a dictionary containing all information needed for restoring the step.

        :param fm: The filemanager which can be used by the step for storing the state of the step.
        :type fm: FileManager
        :return: A dictionary containing all information needed for restoring the step.
        :rtype: Dict
        """
        return {
            "target_ids": {step.id: key for key, step in self.targets.items()},
            "input_ids": {step.id: key for key, step in self.input_steps.items()},
            "id": self.id,
            "module": self.__module__,
            "class": self.__class__.__name__,
            "name": self.name,
            "last": self.last,
            "computation_mode": int(self._original_compuation_mode)
        }

    @classmethod
    @abstractmethod
    def load(cls, stored_step: dict, inputs, targets, module, file_manager):
        """
        Restores the step.

        :param stored_step: Information about the stored step
        :param inputs: The input steps of the step which should be restored
        :param targets: The target steps of the step which should be restored
        :param module: The module which is contained by this step
        :param file_manager: The filemanager of the step
        :return: The restored step.
        """

    def _get_input(self, start, batch):
        return None

    def _get_target(self, start, batch):
        return None

    def _should_stop(self, start, end) -> bool:
        # Fetch input and target data
        input_result = self._get_input(start, end)
        target_result = self._get_target(start, end)

        return (self.condition is not None and not self.condition(input_result, target_result)) or \
               (input_result is not None and len(input_result) > 0 and
                any(map(lambda x: x is None, input_result.values()))) or \
               (target_result is not None and len(target_result) > 0 and any(map(lambda x: x is None, target_result.values())))

    def reset(self):
        """
        Resets all information of the step concerning a specific run.
        """
        self.buffer = {}
        self.finished = False
        self.computation_mode = self._original_compuation_mode

    def set_computation_mode(self, computation_mode: ComputationMode):
        """
        Sets the computation mode of the step for the current run. Note that after reset the all mode is restored.
        Moreover, setting the computation_mode is only possible if the computation_mode is not set explicitly while
        adding the corresponding module to the pipeline.

        :param computation_mode: The computation mode which should be set.
        :type computation_mode: ComputationMode
        """
        if self._original_compuation_mode == computation_mode.Default:
            self.computation_mode = computation_mode
