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
    Attributes:
        stop -- Flag which indicates if the step is stopped. I.e. the condition mechanism stops the execution of this
                step for the current data
        finished -- Flag which indicates that all data are processed.
        buffer -- contains the results of this step.

    :param input_steps: The input steps
    :type input_steps: Step
    :param targets: The target steps
    :type targets: step
    :param condition: A function which evaluates to False or True for detecting if the module should be executed.
    :param computation_mode: The computation mode for this module
    """
    stop: bool = False
    finished = False
    buffer: xr.Dataset = xr.Dataset()

    def __init__(self, input_steps: Optional[Dict[str, "BaseStep"]] = None, targets=None, condition=None,
                 computation_mode=ComputationMode.Default):
        self._original_compuation_mode = computation_mode
        self.computation_mode = computation_mode
        self.input_steps: Dict[str, "BaseStep"] = dict() if input_steps is None else input_steps
        self.targets = [] if targets is None else targets
        self.condition = condition

        self.name = "BaseStep"

        self.id = -1
        self.last = True
        self._current_end = None

    def get_result(self, start: pd.Timestamp, end: Optional[pd.Timestamp]):
        """
        This method is responsible for providing the result of this step.
        Therefore,
        this method triggers the get_input and get_target data methods.
        Additionally, it triggers the computations and checks if all data are processed.

        :param start: The start date of the requested results of the step
        :type start: pd.Timedstamp
        :param end: The end date of the requested results of the step (exclusive)
        :type end: Optional[pd.Timestamp]
        :return: The resulting data or None if no data are calculated
        """
        self.stop = False
        # Check if step should be executed.
        if self._should_stop(start, end):
            self.stop = True
            return None

        # Trigger fit and transform if necessary
        if not self.finished:
            if self.buffer is None or not self._current_end or end > self._current_end:
                self._compute(start, end)
                self._current_end = end
            if not end:
                self.finished = True
            else:
                self.finished = not self.further_elements(end)
            self._outputs()

        return self._pack_data(start, end)

    def _compute(self, start, end):
        pass

    def further_elements(self, counter: pd.Timestamp) -> bool:
        """
        Checks if there exist at least one data for the time after counter.

        :param counter: The timestampe for which it should be tested if there exist further data after it.
        :type counter: pd.Timestamp
        :return: True if there exist further data
        :rtype: bool
        """
        if self.buffer is None or counter < self.buffer.indexes[_get_time_indeces(self.buffer)[0]][-1]:
            return True
        for input_step in self.input_steps.values():
            if not input_step.further_elements(counter):
                return False
        for target_step in self.targets:
            if not target_step.further_elements(counter):
                return False
        return True

    def _pack_data(self, start, end):
        # Provide requested data
        time_index = _get_time_indeces(self.buffer)
        if end and start and end > start:
            index = self.buffer.indexes[time_index[0]]
            start = max(index[0], start.to_numpy())
            result = self.buffer.copy().sel(**{time_index[0]: index[(index >= start) & (index < end.to_numpy())]})
        else:
            self.finished = True
            result = self.buffer.copy()
        return result

    def _transform(self, input_step):
        pass

    def _fit(self, input_step, target_step):
        pass

    def _outputs(self):
        pass

    def _post_transform(self, result):
        if self.buffer is not None:
            self.buffer = result
        else:
            # Time dimension is mandatory, consequently there dim has to exist
            dim = _get_time_indeces(result)[0]
            if self.buffer is None:
                self.buffer = result
            else:
                for key in result.keys():
                    self.buffer[key] = xr.concat([self.buffer[key], result[key]], dim=dim)

    def get_json(self, fm: FileManager) -> Dict:
        """
        Returns a dictionary containing all information needed for restoring the step.

        :param fm: The filemanager which can be used by the step for storing the state of the step.
        :type fm: FileManager
        :return: A dictionary containing all information needed for restoring the step.
        :rtype: Dict
        """
        return {
            "target_ids": list(map(lambda x: x.id, self.targets)),
            "input_ids": {step.id: key for key, step in self.input_steps.items()},
            "id": self.id,
            "module": self.__module__,
            "class": self.__class__.__name__,
            "name": self.name,
            "last": self.last,
            "computation_mode": int(self.computation_mode)
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

    def _should_stop(self, start, end):
        # Fetch input and target data
        input_step = self._get_input(start, end)
        target_step = self._get_target(start, end)

        return (self.condition is not None and not self.condition(input_step, target_step)) or \
               (self.input_steps and any(map(lambda x: x.stop, self.input_steps.values()))) \
               or (self.targets and any(map(lambda x: x.stop, self.targets)))

    def reset(self):
        """
        Resets all information of the step concerning a specific run.
        """
        self.buffer = None
        self.finished = False
        self.stop = False
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

    # TODO how to handle if there a multiple return value of a module...
