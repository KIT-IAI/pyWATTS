import logging
from typing import Optional, Dict

import cloudpickle

from pywatts.core.base import Base, BaseEstimator
from pywatts.core.base_step import BaseStep
from pywatts.core.computation_mode import ComputationMode
from pywatts.core.exceptions.not_fitted_exception import NotFittedException
from pywatts.core.filemanager import FileManager
from pywatts.utils.lineplot import _recursive_plot
from pywatts.utils.summary import _xarray_summary

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
    :param plot: Flag if the result of this step should be plotted.
    :type plot: bool
    :param to_csv: Flag if the result of this step should be written in a csv file.
    :type to_csv: bool
    :param condition: A callable which checks if the step should be executed with the current data.
    :type condition: Callable[xr.Dataset, xr.Dataset, bool]
    :param train_if: A callable which checks if the train_if step should be executed or not.
    :type train_if: Callable[xr.Dataset, xr.Dataset, bool]
    """

    def __init__(self, module: Base, input_steps: Dict[str, BaseStep], file_manager, *, target=None,
                 computation_mode=ComputationMode.Default,
                 plot=False, to_csv=False, summary: bool = True, condition=None,
                 batch_size: Optional[None] = None,
                 train_if=None):

        super().__init__(input_steps, [target] if target is not None else None, condition=condition,
                         computation_mode=computation_mode)
        self.name = module.name
        self.file_manager = file_manager
        self.module = module
        self.batch_size = batch_size
        self.plot = plot
        self.to_csv = to_csv
        self.summary = summary
        self.train_if = train_if

    def _fit(self, inputs: Dict[str, BaseStep], target_step):
        # Fit the encapsulate module, if the input and the target is not stopped.
        self.module.fit(**inputs, target=target_step)

    def _outputs(self):
        # plots and writs the data if the step is finished.
        if self.plot and self.finished:
            self._plot(self.buffer)
        if self.to_csv and self.finished:
            self._to_csv(self.buffer)
        if self.summary and self.finished:
            self._summary(self.buffer)

    def _transform(self, input_step):
        if isinstance(self.module, BaseEstimator) and not self.module.is_fitted:
            message = f"Try to call transform in {self.name} on not fitted module {self.module.name}"
            logger.error(message)
            raise NotFittedException(message, self.name, self.module.name)
        result = self.module.transform(**input_step)
        self._post_transform(result)
        return result

    def _plot(self, result):
        name = f"{self.module.name}"
        title = self.module.name
        _recursive_plot(result, filemanager=self.file_manager, name=name, title=title)

    def _to_csv(self, dataset):
        dataset.to_dataframe(self.name).to_csv(
            self.file_manager.get_path(f"{self.name}.csv"), sep=";"
        )

    def _summary(self, dataset):
        """
        Print out some basic information of the dataset
        like pandas DataFrame.describe method.

        :param dataset: xarray dataset to print out summary for.
        """
        _xarray_summary(dataset)

    # # simple pre-condition check if the buffer is available
    # if self.inputs[0] is None:
    #   raise RuntimeError('Input for Module ' + self.name + ' not available')

    @classmethod
    def load(cls, stored_step: dict, inputs, targets, module, file_manager):
        """
        Load a stored step.

        :param stored_step: Informations about the stored step
        :param inputs: The input step of the stored step
        :param targets: The target step of the stored step
        :param module: The module wrapped by this step
        :return: Step
        """
        step = cls(module, inputs, targets)
        step.input_steps = inputs
        step.targets = targets if targets else []
        step.id = stored_step["id"]
        step.name = stored_step["name"]
        step.to_csv = stored_step["to_csv"]
        step.plot = stored_step["plot"]
        step.last = stored_step["last"]
        step.file_manager = file_manager
        step.computation_mode = ComputationMode(stored_step["computation_mode"])
        if stored_step["condition"]:
            with open(stored_step["condition"], 'rb') as pickle_file:
                step.condition = cloudpickle.load(pickle_file)
        if stored_step["train_if"]:
            with open(stored_step["train_if"], 'rb') as pickle_file:
                step.train_if = cloudpickle.load(pickle_file)
        return step

    def _get_input(self, start, batch):
        return {
            key: input_step.get_result(start, batch) for key, input_step in self.input_steps.items()
        }

    def _compute(self, start, end):
        input_data = self._get_input(start, end)
        target = self._get_target(start, end)
        if self.computation_mode in [ComputationMode.Default, ComputationMode.FitTransform, ComputationMode.Train] and (
                not self.train_if or self.train_if(input_data, target)):
            # Fetch input_data and target data
            if self.batch_size:
                input_batch = self._get_input(end - self.batch_size, end)
                target_batch = self._get_target(end - self.batch_size, end)
                self._fit(input_batch, target_batch)
            else:
                self._fit(input_data, target)
        elif self.module is BaseEstimator:
            logger.info("%s not fitted in Step %s", self.module.name, self.name)

        self._transform(input_data)

    def _get_target(self, start, batch):
        if not self.targets:
            return None
        return self.targets[0].get_result(start, batch)

    def get_json(self, fm: FileManager):
        json = super().get_json(fm)
        condition_path = None
        train_if_path = None
        if self.condition:
            condition_path = fm.get_path(f"{self.name}_condition.pickle")
            with open(condition_path, 'wb') as outfile:
                cloudpickle.dump(self.condition, outfile)
        if self.train_if:
            train_if_path = fm.get_path(f"{self.name}_train_if.pickle")
            with open(train_if_path, 'wb') as outfile:
                cloudpickle.dump(self.train_if, outfile)
        json.update({"plot": self.plot,
                     "to_csv": self.to_csv,
                     "condition": condition_path,
                     "train_if": train_if_path})
        return json
