import logging
import time
import warnings
from typing import Optional, Dict, Union, Callable, List

import cloudpickle
import numpy as np
import pandas as pd
import xarray as xr

from pywatts.callbacks import BaseCallback
from pywatts.core.base import Base, BaseEstimator
from pywatts.core.base_condition import BaseCondition
from pywatts.core.base_step import BaseStep
from pywatts.core.run_setting import RunSetting
from pywatts.core.computation_mode import ComputationMode
from pywatts.core.exceptions.not_fitted_exception import NotFittedException
from pywatts.core.filemanager import FileManager
from pywatts.core.result_step import ResultStep
from pywatts.utils._xarray_time_series_utils import _get_time_indexes

logger = logging.getLogger(__name__)


class Step(BaseStep):
    """
    This step encapsulates modules and manages all information for executing a pipeline step.
    Including fetching the input from the input and target step.

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
    :param callbacks: Callbacks to use after results are processed.
    :type callbacks: List[Union[BaseCallback, Callable[[Dict[str, xr.DataArray]], None]]]
    :param condition: A callable which checks if the step should be executed with the current data.
    :type condition: Callable[xr.DataArray, xr.DataArray, bool]
    :param refit_conditions: A List of Callables of BaseConditions, which contains a condition that indicates if
                                 the module should be trained or not
    :type refit_conditions: List[Union[BaseCondition, Callable]]
    :param lag: Needed for online learning. Determines what data can be used for retraining.
            E.g., when 24 hour forecasts are performed, a lag of 24 hours is needed, else the retraining would
            use future values as target values.
    :type lag: pd.Timedelta
    :param retrain_batch: Needed for online learning. Determines how much data should be used for retraining.
    :type retrain_batch: pd.Timedelta
    """

    def __init__(self, module: Base, input_steps: Dict[str, BaseStep], file_manager, *,
                 targets: Optional[Dict[str, "BaseStep"]] = None,
                 computation_mode=ComputationMode.Default,
                 callbacks: List[Union[BaseCallback, Callable[[Dict[str, xr.DataArray]], None]]] = [],
                 condition=None,
                 batch_size: Optional[None] = None,
                 refit_conditions=[],
                 retrain_batch=pd.Timedelta(hours=24),
                 lag=pd.Timedelta(hours=24)):
        super().__init__(input_steps, targets, condition=condition,
                         computation_mode=computation_mode, name=module.name)
        self.file_manager = file_manager
        self.module = module
        self.retrain_batch = retrain_batch
        self.callbacks = callbacks
        self.batch_size = batch_size
        if self.current_run_setting.computation_mode is not ComputationMode.Refit and len(refit_conditions) > 0:
            message = "You added a refit_condition without setting the computation_mode to refit." \
                      " The condition will be ignored."
            warnings.warn(message)
            logger.warning(message)
        self.lag = lag
        self.refit_conditions = refit_conditions
        self.result_steps: Dict[str, ResultStep] = {}

    def _fit(self, inputs: Dict[str, BaseStep], target_step):
        # Fit the encapsulate module, if the input and the target is not stopped.
        self.module.fit(**inputs, **target_step)

    def _callbacks(self):
        # plots and writs the data if the step is finished.
        for callback in self.callbacks:
            dim = _get_time_indexes(self.buffer)[0]

            if self.current_run_setting.online_start is not None:
                to_plot = {k: self.buffer[k][self.buffer[k][dim] >= self.current_run_setting.online_start] for k in
                           self.buffer.keys()}
            else:
                to_plot = self.buffer
            if isinstance(callback, BaseCallback):
                callback.set_filemanager(self.file_manager)
            if isinstance(self.buffer, xr.DataArray) or isinstance(self.buffer, xr.Dataset):
                # DEPRECATED: direct DataArray or Dataset passing is depricated
                callback({"deprecated": self.buffer})
            else:
                callback(to_plot)

    def _transform(self, input_step):
        if isinstance(self.module, BaseEstimator) and not self.module.is_fitted:
            message = f"Try to call transform in {self.name} on not fitted module {self.module.name}"
            logger.error(message)
            raise NotFittedException(message, self.name, self.module.name)
        result = self.module.transform(**input_step)
        return self._post_transform(result)

    @classmethod
    def load(cls, stored_step: Dict, inputs, targets, module, file_manager):
        """
        Load a stored step.

        :param stored_step: Informations about the stored step
        :param inputs: The input step of the stored step
        :param targets: The target step of the stored step
        :param module: The module wrapped by this step
        :return: Step
        """
        if stored_step["condition"]:
            with open(stored_step["condition"], 'rb') as pickle_file:
                condition = cloudpickle.load(pickle_file)
        else:
            condition = None
        refit_conditions = []
        for refit_condition in stored_step["refit_conditions"]:
            with open(refit_condition, 'rb') as pickle_file:
                refit_conditions.append(cloudpickle.load(pickle_file))

        callbacks = []
        for callback_path in stored_step["callbacks"]:
            with open(callback_path, 'rb') as pickle_file:
                callback = cloudpickle.load(pickle_file)
            callback.set_filemanager(file_manager)
            callbacks.append(callback)

        step = cls(module, inputs, targets=targets, file_manager=file_manager, condition=condition,
                   refit_conditions=refit_conditions, callbacks=callbacks, batch_size=stored_step["batch_size"])
        step.default_run_setting = RunSetting.load(stored_step["default_run_setting"])
        step.current_run_setting = step.default_run_setting.clone()
        step.id = stored_step["id"]
        step.name = stored_step["name"]
        step.last = stored_step["last"]

        return step

    def _compute(self, start, end, minimum_data):
        input_data = self._get_input(start, end, minimum_data)
        target = self._get_target(start, end, minimum_data)
        if self.current_run_setting.computation_mode in [ComputationMode.Default, ComputationMode.FitTransform,
                                                         ComputationMode.Train]:
            # Fetch input_data and target data
            if self.batch_size:
                input_batch = self._get_input(end - self.batch_size, end, minimum_data)
                target_batch = self._get_target(end - self.batch_size, end, minimum_data)
                start_time = time.time()
                self._fit(input_batch, target_batch)
                self.training_time.set_kv("", time.time() - start_time)
            else:
                start_time = time.time()
                self._fit(input_data, target)
                self.training_time.set_kv("", time.time() - start_time)
        elif self.module is BaseEstimator:
            logger.info("%s not fitted in Step %s", self.module.name, self.name)

        start_time = time.time()
        result = self._transform(input_data)
        self.transform_time.set_kv("", time.time() - start_time)
        result_dict = {}
        for key, res in result.items():
            index = res.indexes[_get_time_indexes(result)[0]]
            start = max(index[0], start.to_numpy())
            result_dict[key] = res.sel(**{_get_time_indexes(res)[0]: index[(index >= start)]})
        return result_dict

    def _get_target(self, start, batch, minimum_data=(0, pd.Timedelta(0))):
        min_data_module = self.module.get_min_data()
        if isinstance(min_data_module, (int, np.integer)):
            minimum_data = minimum_data[0] + min_data_module, minimum_data[1]
        else:
            minimum_data = minimum_data[0], minimum_data[1] + min_data_module
        return {
            key: target.get_result(start, batch, minimum_data=minimum_data)
            for key, target in self.targets.items()
        }

    def _get_input(self, start, batch, minimum_data=(0, pd.Timedelta(0))):
        min_data_module = self.module.get_min_data()
        if isinstance(min_data_module, (int, np.integer)):
            minimum_data = minimum_data[0] + min_data_module, minimum_data[1]
        else:
            minimum_data = minimum_data[0], minimum_data[1] + min_data_module
        return {
            key: input_step.get_result(start, batch, minimum_data=minimum_data) for
            key, input_step in self.input_steps.items()
        }

    def get_json(self, fm: FileManager):
        json = super().get_json(fm)
        condition_path = None
        refit_conditions_paths = []
        callbacks_paths = []
        if self.condition:
            condition_path = fm.get_path(f"{self.name}_condition.pickle")
            with open(condition_path, 'wb') as outfile:
                cloudpickle.dump(self.condition, outfile)
        for refit_condition in self.refit_conditions:
            refit_conditions_path = fm.get_path(f"{self.name}_refit_conditions.pickle")
            with open(refit_conditions_path, 'wb') as outfile:
                cloudpickle.dump(refit_condition, outfile)
            refit_conditions_paths.append(refit_conditions_path)
        for callback in self.callbacks:
            callback_path = fm.get_path(f"{self.name}_callback.pickle")
            with open(callback_path, 'wb') as outfile:
                cloudpickle.dump(callback, outfile)
            callbacks_paths.append(callback_path)
        json.update({"callbacks": callbacks_paths,
                     "condition": condition_path,
                     "refit_conditions": refit_conditions_paths,
                     "batch_size": self.batch_size})
        return json

    def refit(self, start: pd.Timestamp, end: pd.Timestamp):
        """
        Refits the module of the step.
        :param start: The date of the first data used for retraining.
        :param end: The date of the last data used for retraining.
        """
        if self.current_run_setting.computation_mode in [ComputationMode.Refit] and isinstance(self.module,
                                                                                               BaseEstimator):
            for refit_condition in self.refit_conditions:
                if isinstance(refit_condition, BaseCondition):
                    condition_input = {key: value.step.get_result(start, end) for key, value in
                                       refit_condition.kwargs.items()}
                    if refit_condition.evaluate(**condition_input):
                        self._refit(end)
                        break
                elif isinstance(refit_condition, Callable):
                    input_data = self._get_input(start, end)
                    target = self._get_target(start, end)
                    if refit_condition(input_data, target):
                        self._refit(end)
                        break

    def _refit(self, end):
        refit_input = self._get_input(end - self.retrain_batch, end)
        refit_target = self._get_target(end - self.retrain_batch, end)
        self.module.refit(**refit_input, **refit_target)

    def get_result_step(self, item: str):
        if item not in self.result_steps:
            self.result_steps[item] = ResultStep(input_steps={"result": self}, buffer_element=item)
        return self.result_steps[item]
