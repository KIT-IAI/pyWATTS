"""
Module containing a pipeline
"""
import glob
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Union, List, Dict, Optional

import pandas as pd
import xarray as xr

from pywatts.core.base import BaseTransformer
from pywatts.core.base_step import BaseStep
from pywatts.core.run_setting import RunSetting
from pywatts.core.computation_mode import ComputationMode
from pywatts.core.exceptions.io_exceptions import IOException
from pywatts.core.filemanager import FileManager
from pywatts.core.start_step import StartStep
from pywatts.core.step import Step
from pywatts.core.step_information import StepInformation
from pywatts.core.exceptions.wrong_parameter_exception import WrongParameterException
from pywatts.core.summary_step import SummaryStep
from pywatts.utils._xarray_time_series_utils import _get_time_indexes
from pywatts.utils._pywatts_json_encoder import PyWATTSJsonEncoder
from pywatts.core.summary_formatter import SummaryMarkdown, SummaryJSON, SummaryFormatter

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename='pywatts.log',
                    level=logging.ERROR)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

logging.getLogger('matplotlib').setLevel(logging.WARN)


class Pipeline(BaseTransformer):
    """
    The pipeline class is the central class of pyWATTS. It is responsible for
    * The interaction with the user
    * starting the execution of the pipeline
    * loading and saving the pipeline

    :param path: The path where the results of the pipeline should be stored (Default: ".")
    :type path: str
    :param batch: If specified then the pipeline does not process the whole data per step.
                  Instead it processes only the data in the given interval. (Default: None)
    :type batch: Optional[pd.Timedelta]
    """

    def __init__(self, path: Optional[str] = ".", batch: Optional[pd.Timedelta] = None, name="Pipeline"):
        super().__init__(name)
        self.batch = batch
        self.counter = None
        self.start_steps = dict()
        self.id_to_step: Dict[int, BaseStep] = {}
        if path is None:
            self.file_manager = None
        else:
            self.file_manager = FileManager(path)

    def transform(self, **x: xr.DataArray) -> xr.DataArray:
        """
        Transform the input into output, by performing all the step in this pipeline.
        Moreover, this method collects the results of the last steps in this pipeline.

        Note, this method is necessary for enabling subpipelining.

        :param x: The input data
        :type x: xr.DataArray
        :return:The transformed data
        :rtype: xr.DataArray
        """
        if self.current_run_setting.online_start is not None:
            time_index_name = _get_time_indexes(x)[0]
            time_index = list(x.values())[0][time_index_name]
            if time_index[-1].values < self.current_run_setting.online_start:
                # Complete data should not be executed online and no summaries should be calculated. Thus
                # _transform is called directly.
                return self._transform(x, None)
            else:
                return self._comp(x, False, self.current_run_setting.summary_formatter, self.batch)
        else:
            return self._comp(x, False, self.current_run_setting.summary_formatter, self.batch)

    def _transform(self, x, batch=None):
        for step in self.id_to_step.values():
            step.finished = False
        for key, (start_step, _) in self.start_steps.items():
            if not start_step.buffer:
                start_step.buffer = {key: x[key].copy()}
            else:
                dim = _get_time_indexes(start_step.buffer[key])[0]
                last = start_step.buffer[key][dim].values[-1]
                start_step.buffer[key] = xr.concat([start_step.buffer[key], x[key][x[key][dim] > last]], dim=dim)
            start_step.finished = True
        time_index = _get_time_indexes(x)
        self.counter = list(x.values())[0].indexes[time_index[0]][0]  # The start date of the input time series.
        last_steps = list(filter(lambda x: x.last, self.id_to_step.values()))
        if not batch:
            return self._collect_results(last_steps)
        return self._collect_batches(last_steps)

    def _collect_batches(self, last_steps):
        result = dict()
        while all(map(lambda step: step.further_elements(self.counter), last_steps)):
            print(self.counter)
            if not result:
                result = self._collect_results(last_steps, use_batch=not self.batch is None)
            else:
                input_results = self._collect_results(last_steps, use_batch=not self.batch is None)
                if input_results is not None:
                    dim = _get_time_indexes(input_results)[0]
                    for key in input_results.keys():
                        result[key] = xr.concat([result[key], input_results[key]], dim=dim)
                else:
                    message = f"From {self.counter} until {self.counter + self.batch} no data are calculated"
                    warnings.warn(message)
                    logger.info(message)
            self.refit(self.counter, self.counter + self.batch if self.batch is not None else self.counter)
            self.counter += self.batch
        return result

    def _collect_results(self, inputs, use_batch=False):
        # Note the return value is None if none of the inputs provide a result for this step...
        end = None if not use_batch else self.counter + self.batch
        result = dict()
        for i, step in enumerate(inputs):
            if not isinstance(step, SummaryStep):
                res = step.get_result(self.counter, end, return_all=True)
                for key, value in res.items():
                    result = self._add_to_result(i, key, value, result)
        return result

    def _add_to_result(self, i, key, res, result):
        if key in result.keys():
            message = f"Naming Conflict: {key} is renamed to. {key}_{i}"
            warnings.warn(message)
            logger.info(message)
            result[f"{key}_{i}"] = res
        else:
            result[key] = res
        return result

    def get_params(self) -> Dict[str, object]:
        """
        Returns the parameter of a pipeline module
        :return: Dictionary containing information about this module
        :rtype: Dict
        """
        return {"batch": self.batch}

    def set_params(self, batch=None):
        """
        Set params of pipeline module.

        :param batch: The time period length for which in each online learning step the pipeline should be executed.
        :type batch: Optional[pd.Timedelta]
        """
        if batch:
            self.batch = batch

    def draw(self):
        """
        Should draw a graph of the pipeline
        Draws the graph with the names of th modules
        :return:
        """

        # TODO built the graph which should be drawn by starting with the last steps...

    def test(self, data: Union[pd.DataFrame, xr.Dataset], summary: bool = True,
             summary_formatter: SummaryFormatter = SummaryMarkdown(), online_start=None):
        """
        Executes all modules in the pipeline in the correct order. This method call only transform on every module
        if the ComputationMode is Default. I.e. if no computationMode is specified during the addition of the module to
        the pipeline.

        :param data: dataset which should be processed by the data
        :type path: Union[pd.DataFrame, xr.Dataset]
        :param summary: A flag indicating if an additional summary should be returned or not.
        :type summary: bool
        :param summary_formatter: Determines the format of the summary.
        :type summary_formatter: SummaryFormatter
        :return: The result of all end points of the pipeline
        :rtype: Dict[xr.DataArray]
        """
        return self._run(data, ComputationMode.Transform, summary, summary_formatter, online_start)

    def train(self, data: Union[pd.DataFrame, xr.Dataset], summary: bool = True,
              summary_formatter: SummaryFormatter = SummaryMarkdown()):
        """
        Executes all modules in the pipeline in the correct order. This method calls fit and transform on each module
        if the ComputationMode is Default. I.e. if no computationMode is specified during the addition of the module to
        the pipeline.

        :param data: dataset which should be processed by the data
        :type path: Union[pd.DataFrame, xr.Dataset]
        :param summary: A flag indicating if an additional summary should be returned or not.
        :type summary: bool
        :param summary_formatter: Determines the format of the summary.
        :type summary_formatter: SummaryFormatter
        :return: The result of all end points of the pipeline
        :rtype: Dict[xr.DataArray]
        """

        return self._run(data, ComputationMode.FitTransform, summary, summary_formatter)

    def _run(self, data: Union[pd.DataFrame, xr.Dataset], mode: ComputationMode, summary: bool,
             summary_formatter: SummaryFormatter, online_start=None):

        self.current_run_setting = RunSetting(computation_mode=mode,
                                              summary_formatter=summary_formatter,
                                              online_start=online_start,
                                              return_summary=summary)

        for step in self.id_to_step.values():
            step.reset()
            step.set_run_setting(self.current_run_setting)

        if isinstance(data, pd.DataFrame):
            data = data.to_xarray()
            data = {key: data[key] for key in data.data_vars}
        elif isinstance(data, xr.Dataset):
            data = {key: data[key] for key in data.data_vars}
        elif isinstance(data, dict):
            for key in data:
                if not isinstance(data[key], xr.DataArray):
                    raise WrongParameterException(
                        "Input Dict does not contain xr.DataArray objects.",
                        "Make sure to pass Dict[str, xr.DataArray].",
                        self.name)
        else:
            raise WrongParameterException(
                "Unkown data type to pass to pipeline steps.",
                "Make sure to use pandas DataFrames, xarray Datasets, or Dict[str, xr.DataArray].",
                self.name)

        if self.current_run_setting.online_start is not None:
            # First only _transform should be called (no summary, no online) on the data before online_start.
            # Afterwards, comp is called (_transform and summaries using online simulation)
            index_name = _get_time_indexes(data)[0]
            self._transform({key: data[key].sel(
                **{index_name: data[key][index_name] < self.current_run_setting.online_start }) for key in data}, False)
            for step in self.id_to_step.values():
                step.reset(keep_buffer=True)
                step.set_run_setting(self.current_run_setting.clone())
            return self._comp({key: data[key].sel(**{index_name: data[key][index_name] >= self.current_run_setting.online_start}) for key in data},
                              self.current_run_setting.return_summary, summary_formatter, self.batch, start=self.current_run_setting.online_start)
        else:
            return self._comp(data, self.current_run_setting.return_summary, summary_formatter, self.batch)


    def _comp(self, data, return_summary, summary_formatter, batch, start=None):
        result = self._transform(data, batch)
        summary = self._create_summary(summary_formatter, start)
        return (result, summary) if return_summary else result

    def add(self, *,
            module: Union[BaseStep],
            input_ids: List[int] = None,
            target_ids: List[int] = None):
        """
        Add a new module with all of it's inputs to the pipeline.

        :param target_ids: The target determines the module which provides the target value.
        :param module: The module which should be added
        :param input_ids: A list of modules, whose input is needed for this steps
        :return: None
        """
        if input_ids is None:
            input_ids = []

        if target_ids is None:
            target_ids = []

        # register modules in the pipeline and get ids
        step_id = self._register_step(module)

        logger.info("Add %s to the pipeline. Inputs are %s%s",
                    self.id_to_step[step_id],
                    [self.id_to_step[input_id] for input_id in input_ids],
                    "." if not input_ids else f" and the target is "
                                              f"{[self.id_to_step[target_id] for target_id in target_ids]}.")

        return step_id

    def _register_step(self, step) -> int:
        """
        Registers the module in the pipeline and inits the wrappers as well as the id.

        :param step: the step to be registered
        :return:
        """
        # if the module is not there, find new id
        if self.id_to_step:
            step_id = max(self.id_to_step) + 1
        else:
            step_id = 1

        self.id_to_step[step_id] = step
        return step_id

    def save(self, fm: FileManager):
        """
        Saves the pipeline. Note You should not call this method from outside of pyWATTS. If you want to store your
        pipeline then you should use to_folder.
        """
        json_module = super().save(fm)
        path = os.path.join(str(fm.basic_path), self.name)
        if os.path.isdir(path):
            number = len(glob.glob(f'{path}*'))
            path = f"{path}_{number + 1}"
        self.to_folder(path)
        json_module["pipeline_path"] = path
        json_module["params"] = {
            "batch": str(self.batch) if self.batch else None
        }
        return json_module

    @classmethod
    def load(cls, load_information):
        """
        Loads the pipeline.  Note You should not call this method from outside of pyWATTS. If you want to store your
        pipeline then you should use from_folder.
        """
        pipeline = cls.from_folder(load_information["pipeline_path"])
        return pipeline

    def to_folder(self, path: Union[str, Path]):
        """
        Saves the pipeline in pipeline.json in the specified folder.

        :param path: path of the folder
        :return: None
        """
        if not isinstance(path, Path):
            path = Path(path)
        save_file_manager = FileManager(path, time_mode=False)

        modules = []
        # 1. Iterate over steps and collect all modules -> With step_id to module_id
        #    Create for each step dict with information for restorage
        steps_for_storing = []
        for step in self.id_to_step.values():
            step_json = step.get_json(save_file_manager)
            if isinstance(step, Step):
                if step.module in modules:
                    step_json["module_id"] = modules.index(step.module)
                else:
                    modules.append(step.module)
                    step_json["module_id"] = len(modules) - 1
            steps_for_storing.append(step_json)

        # 2. Iterate over all modules and create Json and save as pickle or h5 ... if necessary...
        modules_for_storing = []
        for module in modules:
            stored_module = module.save(save_file_manager)
            modules_for_storing.append(stored_module)

        # 3. Put everything together and dump it.
        stored_pipeline = {
            "name": "Pipeline",
            "id": 1,
            "version": 1,
            "modules": modules_for_storing,
            "steps": steps_for_storing,
            "path": self.file_manager.basic_path if self.file_manager else None,
            "batch": str(self.batch) if self.batch else None,
        }
        file_path = save_file_manager.get_path('pipeline.json')
        with open(file_path, 'w') as outfile:
            json.dump(obj=stored_pipeline, fp=outfile, sort_keys=False, indent=4, cls=PyWATTSJsonEncoder)

    @staticmethod
    def from_folder(load_path, file_manager_path=None):
        """
        Loads the pipeline from the pipeline.json in the specified folder
        .. warning::
            Sometimes from_folder use unpickle for loading modules. Note that this is not safe.
            Consequently, load only pipelines you trust with `from_folder`.
            For more details about pickling see https://docs.python.org/3/library/pickle.html


        :param load_path: path to the pipeline.json
        :type load_path: str
        :param file_manager_path: path for the results and outputs
        :type file_manager_path: str
        """
        if not os.path.isdir(load_path):
            logger.error("Path %s for loading pipeline does not exist", load_path)
            raise IOException(f"Path {load_path} does not exist"
                              f"Check the path which you passed to the from_folder method.")

        # load json file
        file_path = os.path.join(load_path, 'pipeline.json')
        with open(file_path, 'r') as outfile:
            json_dict = json.load(outfile)

        # load general pipeline config
        if file_manager_path is None:
            file_manager_path = json_dict.get('path', ".")

        batch = pd.Timedelta(json_dict.get("batch")) if json_dict.get("batch") else None

        pipeline = Pipeline(file_manager_path, batch)
        # 1. load all modules
        modules = {}  # create a dict of all modules with their id from the json
        for i, json_module in enumerate(json_dict["modules"]):
            modules[i] = pipeline._load_modules(json_module)

        # 2. Load all steps
        for step in json_dict["steps"]:
            step = pipeline._load_step(modules, step)
            pipeline.id_to_step[step.id] = step

        pipeline.start_steps = {element.index: (element, StepInformation(step=element, pipeline=pipeline))
                                for element in filter(lambda x: isinstance(x, StartStep), pipeline.id_to_step.values())}

        return pipeline

    def _load_modules(self, json_module):
        mod = __import__(json_module["module"], fromlist=json_module["class"])
        klass = getattr(mod, json_module["class"])
        return klass.load(json_module)

    def _load_step(self, modules, step):
        mod = __import__(step["module"], fromlist=step["class"])
        klass = getattr(mod, step["class"])
        module = None
        if isinstance(klass, Step) or issubclass(klass, Step):
            module = modules[step["module_id"]]
        loaded_step = klass.load(step,
                                 inputs={key: self.id_to_step[int(step_id)] for step_id, key in
                                         step["input_ids"].items()},
                                 targets={key: self.id_to_step[int(step_id)] for step_id, key in
                                          step["target_ids"].items()},
                                 module=module,
                                 file_manager=self.file_manager)
        return loaded_step

    def __getitem__(self, item: str):
        """
        Returns the step_information for the start step corresponding to the item
        """
        if item not in self.start_steps.keys():
            start_step = StartStep(item)
            self.start_steps[item] = start_step, StepInformation(step=start_step, pipeline=self)
            start_step.id = self.add(module=start_step, input_ids=[], target_ids=[])
        return self.start_steps[item][-1]

    def _create_summary(self, summary_formatter, start=None, end=None):
        summaries = []
        for step in self.id_to_step.values():
            if isinstance(step, SummaryStep):
                summaries.append(step.get_summary(start, end))
            summaries.extend([step.transform_time, step.training_time])
        return summary_formatter.create_summary(summaries, self.file_manager)

    def refit(self, start, end):
        """
        Refits all steps inside of the pipeline.
        :param start: The date of the first data used for retraining.
        :param end: The date of the last data used for retraining.
        """
        for step in self.id_to_step.values():
            # A lag is needed, since if we have a 24 hour forecast we can evaluate the forecast not until 24 hours
            # are gone, since before not all target variables are available
            if isinstance(step, Step):
                step.refit(start - step.lag, end - step.lag)
