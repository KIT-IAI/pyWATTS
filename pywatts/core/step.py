import logging
import time
from typing import Optional, Dict, Union, Callable, List

import cloudpickle
import xarray as xr

from pywatts.callbacks import BaseCallback
from pywatts.core.base import Base, BaseEstimator
from pywatts.core.base_step import BaseStep
from pywatts.core.computation_mode import ComputationMode
from pywatts.core.exceptions.not_fitted_exception import NotFittedException
from pywatts.core.filemanager import FileManager
from pywatts.core.result_step import ResultStep

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
    :param train_if: A callable which checks if the train_if step should be executed or not.
    :type train_if: Callable[xr.DataArray, xr.DataArray, bool]
    """

    def __init__(self, module: Base, input_steps: Dict[str, BaseStep], file_manager, *,
                 targets: Optional[Dict[str, "BaseStep"]] = None,
                 computation_mode=ComputationMode.Default,
                 callbacks: List[Union[BaseCallback, Callable[[Dict[str, xr.DataArray]], None]]] = [],
                 condition=None,
                 batch_size: Optional[None] = None,
                 train_if=None):

        super().__init__(input_steps, targets, condition=condition,
                         computation_mode=computation_mode)
        self.name = module.name
        self.file_manager = file_manager
        self.module = module
        self.callbacks = callbacks
        self.batch_size = batch_size
        self.train_if = train_if
        self.result_steps: Dict[str, ResultStep] = {}

    def _fit(self, inputs: Dict[str, BaseStep], target_step):
        # Fit the encapsulate module, if the input and the target is not stopped.
        self.module.fit(**inputs, **target_step)

    def _callbacks(self):
        # plots and writs the data if the step is finished.
        for callback in self.callbacks:
            if isinstance(callback, BaseCallback):
                callback.set_filemanager(self.file_manager)
            if isinstance(self.buffer, xr.DataArray) or isinstance(self.buffer, xr.Dataset):
                # DEPRECATED: direct DataArray or Dataset passing is depricated
                callback({"deprecated": self.buffer})
            else:
                callback(self.buffer)

    def _transform(self, input_step):
        if isinstance(self.module, BaseEstimator) and not self.module.is_fitted:
            message = f"Try to call transform in {self.name} on not fitted module {self.module.name}"
            logger.error(message)
            raise NotFittedException(message, self.name, self.module.name)
        result = self.module.transform(**input_step)
        self._post_transform(result)
        return result

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
        if stored_step["train_if"]:
            with open(stored_step["train_if"], 'rb') as pickle_file:
                train_if = cloudpickle.load(pickle_file)
        else:
            train_if = None
        callbacks = []
        for callback_path in  stored_step["callbacks"]:
            with open(callback_path, 'rb') as pickle_file:
                callback = cloudpickle.load(pickle_file)
            callback.set_filemanager(file_manager)
            callbacks.append(callback)

        step = cls(module, inputs, targets=targets, file_manager=file_manager,
                   computation_mode=ComputationMode(stored_step["computation_mode"]), condition=condition,
                   train_if=train_if, callbacks=callbacks, batch_size=stored_step["batch_size"])
        step.id = stored_step["id"]
        step.name = stored_step["name"]
        step.last = stored_step["last"]

        return step

    def _compute(self, start, end):
        input_data = self._get_input(start, end)
        target = self._get_target(start, end)
        if self.computation_mode in [ComputationMode.Default, ComputationMode.FitTransform, ComputationMode.Train] and (
                not self.train_if or self.train_if(input_data, target)):
            # Fetch input_data and target data
            start_time = time.time()
            if self.batch_size:
                input_batch = self._get_input(end - self.batch_size, end)
                target_batch = self._get_target(end - self.batch_size, end)
                self._fit(input_batch, target_batch)
            else:
                self._fit(input_data, target)
            self.training_time = time.time() - start_time
        elif self.module is BaseEstimator:
            logger.info("%s not fitted in Step %s", self.module.name, self.name)

        self._transform(input_data)

    def _get_target(self, start, batch):
        return {
            key: target.get_result(start, batch) for key, target in self.targets.items()
        }

    def _get_input(self, start, batch):
        return {
            key: input_step.get_result(start, batch) for key, input_step in self.input_steps.items()
        }

    def get_json(self, fm: FileManager):
        json = super().get_json(fm)
        condition_path = None
        train_if_path = None
        callbacks_paths = []
        if self.condition:
            condition_path = fm.get_path(f"{self.name}_condition.pickle")
            with open(condition_path, 'wb') as outfile:
                cloudpickle.dump(self.condition, outfile)
        if self.train_if:
            train_if_path = fm.get_path(f"{self.name}_train_if.pickle")
            with open(train_if_path, 'wb') as outfile:
                cloudpickle.dump(self.train_if, outfile)
        for callback in self.callbacks:
            callback_path = fm.get_path(f"{self.name}_callback.pickle")
            with open(callback_path, 'wb') as outfile:
                cloudpickle.dump(callback, outfile)
            callbacks_paths.append(callback_path)
        json.update({"callbacks": callbacks_paths,
                     "condition": condition_path,
                     "train_if": train_if_path,
                     "batch_size": self.batch_size})
        return json

    def get_result_step(self, item: str):
        if item not in self.result_steps:
            self.result_steps[item] = ResultStep(input_steps={"result": self}, buffer_element=item)
        return self.result_steps[item]
