import inspect
from abc import ABC, abstractmethod
from typing import List

import pandas as pd
import xarray as xr

from pywatts.core.exceptions.step_creation_exception import StepCreationException
from pywatts.core.step_information import StepInformation


class BaseCondition(ABC):
    """
    This module contains a function which returns either True or False. The input of this function is the output of one
    or more modules.
    :param name: The name of the condition
    :type name: str
    """

    def __init__(self, name, refit_batch: pd.Timedelta = pd.Timedelta(hours=24), refit_params: dict = None):
        self.name = name
        self.kwargs = {}
        self.refit_batch = refit_batch
        self.refit_params = refit_params
        self._end = None

    @abstractmethod
    def evaluate(self, start, end) -> bool:
        """
        This method evaluates the Condition
        :param start: start of the batch
        :type start: pd.Timestamp
        :param end: end of the batch
        :type end: pd.Timestamp
        """

    def __call__(self, **kwargs: StepInformation):
        """
        This method adds a Condition to the pipeline.
        """
        arguments = inspect.signature(self.evaluate).parameters.keys()

        if arguments != kwargs.keys():
            raise StepCreationException(
                f"The given kwargs does not fit to the inputs of the Condition{self.__class__.__name__} {self.name}."
                f"The module only needs and accepts {inspect.signature(self.evaluate).parameters.keys()} as input. "
                f"However, {kwargs.keys()} are given as input. ",
                self
            )
        self.kwargs = kwargs

    def _is_evaluated(self, end):
        if end == self._end:
            return True
        self._end = end
        return False

    def _get_inputs(self, start, end):
        return {key: value.step.get_result(start, end) for key, value in self.kwargs.items()}
