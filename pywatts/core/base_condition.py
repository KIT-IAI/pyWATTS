import inspect
from abc import ABC, abstractmethod
from typing import List

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

    def __init__(self, name):
        self.name = name
        self.kwargs = {}

    @abstractmethod
    def evaluate(self, **kwargs: List[xr.DataArray]) -> bool:
        """
        This method evaluates the Condition
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
