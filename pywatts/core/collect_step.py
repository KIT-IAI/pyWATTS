import logging
import warnings
from typing import List, Union, Tuple

import xarray as xr
from xarray import MergeError

from pywatts.core.base_step import BaseStep

logger = logging.getLogger(__name__)

class CollectStep(BaseStep):
    """
    This step collects the inputs of all previous steps.

    :param inputs: The inputs of this step.
    :type inputs: List[Union[BaseStep, Tuple[BaseStep]]]
    """

    def __init__(self, inputs: List[Union[BaseStep, Tuple[BaseStep]]]):
        super().__init__(inputs, [])
        self.inputs = inputs
        self.name = "Collect"

    def _get_input(self, start, batch):
        inputs = []
        for step in self.inputs:
            inp = step.get_result(start, batch)
            inputs.append(inp)
        return inputs

    def _compute(self, start, end):
        # Fetch input and target data
        input_data = self._get_input(start, end)
        self._transform(input_data)

    def _transform(self, input_step: List[xr.Dataset]):
        #Collects all results of the previous steps
        result = input_step[0]
        # Appending the id of the module to each data_var for avoiding naming conflict during the merge of the datasets.
        # result = result.rename({key: key + f"_{predecessors[0]}" for i, key in enumerate(result.data_vars)})
        for i, ds in enumerate(input_step[1:]):
            try:
                result = result.merge(ds)
            except MergeError:
                message = f'There was a naming conflict. Therefore, we renamed:' + str(
                    {key: f"{key}_{i}" for key in ds.data_vars})
                logger.info(message)
                warnings.warn(message)
                ds = ds.rename({key: key + f"_{i}" for key in ds.data_vars})
                result = result.merge(ds)
        self._post_transform(result)

    @classmethod
    def load(cls, stored_step: dict, inputs, targets, module, file_manager):
        """
        Load a stored collect step.

        :param stored_step:
        :param inputs:
        :param targets:
        :param module:
        :return: The restored step
        """
        step = cls(inputs)
        step.id = stored_step["id"]
        step.name = stored_step["name"]
        step.last = stored_step["last"]
        return step
