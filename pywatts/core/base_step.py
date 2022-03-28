import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict
import copy

import pandas as pd
import xarray as xr

from pywatts.core.computation_mode import ComputationMode
from pywatts.core.filemanager import FileManager
from pywatts.core.run_setting import RunSetting
from pywatts.utils._xarray_time_series_utils import _get_time_indexes
from pywatts.core.summary_object import SummaryObjectList, SummaryCategory

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
                 computation_mode=ComputationMode.Default, name="BaseStep"):
        self.default_run_setting = RunSetting(computation_mode=computation_mode)
        self.current_run_setting = self.default_run_setting.clone()
        self.input_steps: Dict[str, "BaseStep"] = dict() if input_steps is None else input_steps
        self.targets: Dict[str, "BaseStep"] = dict() if targets is None else targets
        self.condition = condition

        self.name = name

        self.id = -1
        self.finished = False
        self.last = True
        self._current_end = None
        self.buffer: Dict[str, xr.DataArray] = {}
        self.training_time = SummaryObjectList(self.name + " Training Time", category=SummaryCategory.FitTime)
        self.transform_time = SummaryObjectList(self.name + " Transform Time", category=SummaryCategory.TransformTime)

    def get_result(self, start: pd.Timestamp, end: Optional[pd.Timestamp], buffer_element: str = None,
                   return_all=False, minimum_data=(0, pd.Timedelta(0))):
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

        # Only execute the module if the step is not finished and the results are not yet calculated
        if not self.finished and not (end is not None and self._current_end is not None and end <= self._current_end):
            if not self.buffer or not self._current_end or end > self._current_end:
                self._compute(start, end, minimum_data)
                self._current_end = end
            if not end:
                self.finished = True
            else:
                self.finished = not self.further_elements(end)

            # Only call callbacks if the step is finished
            if self.finished:
                self._callbacks()

        return self._pack_data(start, end, buffer_element, return_all=return_all, minimum_data=minimum_data)

    def _compute(self, start, end, minimum_data) -> Dict[str, xr.DataArray]:
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
                [counter < b.indexes[_get_time_indexes(self.buffer)[0]][-1] for b in self.buffer.values()]):
            return True
        for input_step in self.input_steps.values():
            if not input_step.further_elements(counter):
                return False
        for target_step in self.targets.values():
            if not target_step.further_elements(counter):
                return False
        return True

    def _pack_data(self, start, end, buffer_element=None, return_all=False, minimum_data=(0, pd.Timedelta(0))):
        # Provide requested data
        time_index = _get_time_indexes(self.buffer)
        if start:
            index = list(self.buffer.values())[0].indexes[time_index[0]]
            if len(index) > 1:
                freq = index[1] - index[0]
            else:
                freq = 0
            start = start.to_numpy() - pd.Timedelta(minimum_data[0] * freq) - minimum_data[1]
            # If end is not set, all values should be considered. Thus we add a small timedelta to the last index entry.
            end = end.to_numpy() if end is not None else (index[-1] + pd.Timedelta(nanoseconds=1)).to_numpy()
            # After sel copy is not needed, since it returns a new array.
            if buffer_element is not None:
                return self.buffer[buffer_element].sel(
                    **{time_index[0]: index[(index >= start) & (index < end)]})
            elif return_all:
                return {key: b.sel(**{time_index[0]: index[(index >= start) & (index < end)]}) for
                        key, b in self.buffer.items()}
            else:
                return list(self.buffer.values())[0].sel(
                    **{time_index[0]: index[(index >= start) & (index < end)]})
        else:
            self.finished = True
            if buffer_element is not None:
                return self.buffer[buffer_element].copy()
            elif return_all:
                return copy.deepcopy(self.buffer)
            else:
                return list(self.buffer.values())[0].copy()

    def _transform(self, input_step):
        pass

    def _fit(self, input_step, target_step):
        pass

    def _callbacks(self):
        pass

    def _post_transform(self, result):
        if not isinstance(result, dict):
            result = {self.name: result}

        if not self.buffer:
            self.buffer = result
        else:
            # Time dimension is mandatory, consequently there dim has to exist
            dim = _get_time_indexes(result)[0]
            for key in self.buffer.keys():
                last = self.buffer[key][dim].values[-1]
                self.buffer[key] = xr.concat([self.buffer[key], result[key][result[key][dim] > last]], dim=dim)
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
            "default_run_setting": self.default_run_setting.save()
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

    def _get_input(self, start, batch, minimum_data=(0, pd.Timedelta(0))):
        return None

    def _get_target(self, start, batch, minimum_data=(0, pd.Timedelta(0))):
        return None

    def _should_stop(self, start, end) -> bool:
        # Fetch input and target data
        input_result = self._get_input(start, end)
        target_result = self._get_target(start, end)

        # Check if either the condition is True or some of the previous steps stopped (return_value is None)
        return (self.condition is not None and not self.condition(input_result, target_result)) or \
               self._input_stopped(input_result) or self._input_stopped(target_result)

    @staticmethod
    def _input_stopped(input_data):
        return (input_data is not None and len(input_data) > 0 and any(map(lambda x: x is None, input_data.values())))

    def reset(self, keep_buffer=False):
        """
        Resets all information of the step concerning a specific run.
        :param keep_buffer: Flag indicating if the buffer should be resetted too.
        """
        if not keep_buffer:
            self.buffer = {}
        self.finished = False
        self.current_run_setting = self.default_run_setting.clone()

    def set_run_setting(self, run_setting: RunSetting):
        """
        Sets the computation mode of the step for the current run. Note that after reset the all mode is restored.
        Moreover, setting the computation_mode is only possible if the computation_mode is not set explicitly while
        adding the corresponding module to the pipeline.

        :param computation_mode: The computation mode which should be set.
        :type computation_mode: ComputationMode
        """
        self.current_run_setting = self.default_run_setting.update(run_setting)
