import logging
import pandas as pd
from typing import Dict, Optional

from pywatts.core.base import Base
from pywatts.core.base_step import BaseStep
from pywatts.core.base_summary import BaseSummary
from pywatts.core.filemanager import FileManager
from pywatts.core.step import Step

logger = logging.getLogger(__name__)


class SummaryStep(Step):
    """
    This step encapsulates modules and manages all information for executing a pipeline step.
    Including fetching the input from the input and target step.

    :param module: The module which is wrapped by the step-
    :type module: Base
    :param input_step: The input_step of the module.
    :type input_step: Step
    :param file_manager: The file_manager which is used for storing data.
    :type file_manager: FileManager
    """

    def __init__(self, module: BaseSummary, input_steps: Dict[str, BaseStep], file_manager):
        super().__init__(module, input_steps, file_manager)
        self.name = module.name
        self.file_manager = file_manager
        self.module = module

    def _transform(self, input_step):
        return self.module.transform(file_manager=self.file_manager, **input_step)

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
        step.inputs = inputs
        step.id = stored_step["id"]
        step.name = stored_step["name"]
        step.last = stored_step["last"]
        step.file_manager = file_manager
        return step

    def get_summary(self):
        # TODO Docs
        input_data = self._get_input(None, None)
        return self._transform(input_data)
