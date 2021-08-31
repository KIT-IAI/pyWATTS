from abc import ABC, abstractmethod

import pandas as pd

from pywatts.core.step_information import StepInformation


class ConditionObject(ABC):
    """
    This module contains a function which returns either True or False. The input of this function is the output of one
    or more modules.

    A condition object can be passed to the train_if function of steps
    """

    def __init__(self, name):
        # self.function = function
        self.name = name

    @abstractmethod
    def save(self):
        """
        TODO
        """

    @abstractmethod
    def load(self):
        """
        TODO
        """

    @abstractmethod
    def evaluate(self, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        """
        TODO
        """

    def __call__(self, **kwargs: StepInformation):
        """
        TODO
        """
        # TODO?
