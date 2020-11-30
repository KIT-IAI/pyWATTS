"""
Module containing a pipeline
"""

import json
import logging
import os
import warnings
from pathlib import Path
from typing import Union, List, Dict, Optional

import networkx as nx
import pandas as pd
import xarray as xr
from xarray import MergeError

from pywatts.core.computation_mode import ComputationMode
from pywatts.core.base import BaseTransformer
from pywatts.core.base_step import BaseStep
from pywatts.core.filemanager import FileManager
from pywatts.core.start_step import StartStep
from pywatts.core.step import Step
from pywatts.core.step_information import StepInformation
from pywatts.utils._xarray_time_series_utils import _get_time_indeces

from pywatts.core.exceptions.io_exceptions import IOException

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

    def __init__(self, path: str = ".", batch: Optional[pd.Timedelta] = None):
        super().__init__("Pipeline")
        self.batch = batch
        self.counter = None
        self.start_steps = dict()
        self.id_to_step: Dict[int, BaseStep] = {}
        self.computational_graph = nx.DiGraph()
        self.target_graph = nx.DiGraph()
        self.file_manager = FileManager(path)

    def transform(self, x: Optional[xr.Dataset]) -> xr.Dataset:
        """
        Transform the input into output, by performing all the step in this pipeline.
        Moreover, this method collects the results of the last steps in this pipeline.

        Note, this method is necessary for enabling subpipelining.

        :param x: The input data
        :type x: xr.Dataset
        :return:The transformed data
        :rtype: xr.Dataset
        """
        queue = nx.dag.topological_sort(nx.compose(self.computational_graph, self.target_graph))
        for key, (start_step, _) in self.start_steps.items():
            start_step.buffer = x[key].copy()
            start_step.finished = True

        time_index = _get_time_indeces(x)
        self.counter = x.indexes[time_index[0]][0]  # The start date of the input time series.

        last_steps = list(map(lambda x: self.id_to_step[x], filter(lambda x: self.id_to_step[x].last, queue)))

        if not self.batch:
            return self._collect_results(last_steps)
        return self._collect_batches(last_steps, time_index)

    def _collect_batches(self, last_steps, time_index):
        result = xr.Dataset()
        while all(map(lambda step: step.further_elements(self.counter), last_steps)):
            print(self.counter)
            if not result:
                result = self._collect_results(last_steps)
            else:
                input_results = self._collect_results(last_steps)
                if input_results is not None:
                    result = xr.concat([result, input_results], dim=time_index[0])
                else:
                    message = f"From {self.counter} until {self.counter + self.batch} no data are calculated"
                    warnings.warn(message)
                    logger.info(message)
            self.counter += self.batch
        return result

    def _collect_results(self, inputs):
        # Note the return value is None if none of the inputs provide a result for this step...
        end = None if not self.batch else self.counter + self.batch
        results = [step.get_result(self.counter, end) for step in inputs]

        result = results[0]
        for i, step_result in enumerate(results[1:]):
            if step_result is not None and result is None:
                # Possible if the first ds is None
                result = step_result
            elif step_result is not None:
                try:
                    result = result.merge(step_result)
                except MergeError:
                    step_result = step_result.rename({key: key + f"_{i}" for key in step_result.data_vars})
                    message = f'There was a naming conflict. Therefore, we renamed:' + str(
                        {key: f"{key}_{i}" for key in step_result.data_vars})
                    logger.info(message)
                    warnings.warn(message)
                    result = result.merge(step_result)
        return result

    def get_params(self) -> Dict[str, object]:
        """
        Returns the parameter of a pipeline module
        :return: Dictionary containing information about this module
        :rtype: dict
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
        complete_graph: nx.DiGraph = nx.compose(self.computational_graph, self.target_graph)
        queue = nx.topological_sort(complete_graph)
        graph = nx.DiGraph()
        for module_id in queue:
            graph.add_node(self.id_to_step[module_id].name + str(module_id))
            for successor_id in complete_graph.successors(module_id):
                graph.add_node(self.id_to_step[successor_id].name + str(successor_id))
                graph.add_edge(self.id_to_step[module_id].name + str(module_id),
                               self.id_to_step[successor_id].name + str(successor_id))

        nx.draw_planar(graph, with_labels=True)

    def test(self, data: Union[pd.DataFrame, xr.Dataset]):
        """
        Executes all modules in the pipeline in the correct order. This method call only transform on every module
        if the ComputationMode is Default. I.e. if no computationMode is specified during the addition of the module to
        the pipeline.

        :param data: dataset which should be processed by the data
        :type path: Union[pd.DataFrame, xr.Dataset]
        :return: The result of all end points of the pipeline
        :rytpe: Dict[xr.Dataset]
        """
        return self._run(data, ComputationMode.Transform)

    def train(self, data: Union[pd.DataFrame, xr.Dataset]):
        """
        Executes all modules in the pipeline in the correct order. This method calls fit and transform on each module
        if the ComputationMode is Default. I.e. if no computationMode is specified during the addition of the module to
        the pipeline.

        :param data: dataset which should be processed by the data
        :type path: Union[pd.DataFrame, xr.Dataset]
        :return: The result of all end points of the pipeline
        :rytpe: Dict[xr.Dataset]
        """

        return self._run(data, ComputationMode.FitTransform)

    def _run(self, data: Union[pd.DataFrame, xr.Dataset], mode: ComputationMode):

        for step in self.id_to_step.values():
            step.reset()
            step.set_computation_mode(mode)

        if isinstance(data, pd.DataFrame):
            data = data.to_xarray()

        return self.transform(data)

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

        # register modules not in the pipeline and get ids
        module_id = self._register_step(module)

        self._add_to_graph(input_ids, module_id, target_ids)

        logger.info("Add %s to the pipeline. Inputs are %s%s",
                    self.id_to_step[module_id],
                    [self.id_to_step[input_id] for input_id in input_ids],
                    "." if not input_ids else f" and the target is "
                                              f"{[self.id_to_step[target_id] for target_id in target_ids]}.")

        return module_id

    def _add_to_graph(self, input_ids, module_id, target_ids):
        # create graph based on the pipeline ids
        self.computational_graph.add_node(module_id)
        self.target_graph.add_node(module_id)
        for input_id in input_ids:
            self.computational_graph.add_edge(input_id, module_id)
        for target_id in target_ids:
            self.target_graph.add_edge(target_id, module_id)

    def _register_step(self, module) -> int:
        """
        Registers the module in the pipeline and inits the wrapper as well as the id.

        :param module: the module to be registered
        :return:
        """
        # if the module is not there, find new id
        if self.id_to_step:
            module_id = max(self.id_to_step) + 1
        else:
            module_id = 1

        self.id_to_step[module_id] = module
        return module_id

    def save(self, fm: FileManager):
        json_module = super().save(fm)
        path = os.path.join(str(fm.basic_path), self.name)
        self.to_folder(path)
        json_module["pipeline_path"] = path
        json_module["params"] = {
            "batch": str(self.batch) if self.batch else None
        }
        return json_module

    @classmethod
    def load(cls, load_information):
        pipeline = Pipeline()
        pipeline = pipeline.from_folder(load_information["pipeline_path"])
        return pipeline

    def to_folder(self, path: Union[str, Path]):
        """
        TODO Update this for handling multiple start steps...
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
            "path": self.file_manager.basic_path,
            "batch": str(self.batch) if self.batch else None,
        }
        file_path = save_file_manager.get_path('pipeline.json')
        with open(file_path, 'w') as outfile:
            json.dump(obj=stored_pipeline, fp=outfile, sort_keys=True, indent=4)

    def from_folder(self, load_path, file_manager_path=None):
        """
        TODO Update this for handling multiple start steps...
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

        self.batch = pd.Timedelta(json_dict.get("batch")) if json_dict.get("batch") else None

        # pipeline = cls(file_manager_path, batch=batch)

        # 1. load all modules
        modules = {}  # create a dict of all modules with their id from the json
        for i, json_module in enumerate(json_dict["modules"]):
            mod = __import__(json_module["module"], fromlist=json_module["class"])
            klass = getattr(mod, json_module["class"])
            module = klass.load(json_module)
            modules[i] = module

        # 2. Load all steps
        for step in json_dict["steps"]:
            mod = __import__(step["module"], fromlist=step["class"])
            klass = getattr(mod, step["class"])
            module = None
            if isinstance(klass, Step) or issubclass(klass, Step):
                module = modules[step["module_id"]]
            step = klass.load(step,
                              inputs=list(map(lambda x: self.id_to_step[x], step["input_ids"])),
                              targets=list(map(lambda x: self.id_to_step[x], step["target_ids"])),
                              module=module, file_manager=self.file_manager)
            self.id_to_step[step.id] = step
            self._add_to_graph(input_ids=list(map(lambda x: x.id, step.input_steps), ),
                               module_id=step.id,
                               target_ids=list(map(lambda x: x.id, step.targets)))
        self.start_step = self.id_to_step[1]
        return self

    def __getitem__(self, item: str):
        """
        Returns the step_information for the start step corresponding to the item
        """
        if not item in self.start_steps.keys():
            start_step = StartStep(item)
            self.start_steps[item] = start_step, StepInformation(step=start_step, pipeline=self)
            start_step.id = self.add(module=start_step, input_ids=[], target_ids=[])
        return self.start_steps[item][-1]
