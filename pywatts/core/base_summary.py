# pylint: disable=W0233
from __future__ import annotations

import warnings
import xarray as xr
from abc import ABC, abstractmethod
from typing import Dict, Tuple, TYPE_CHECKING

from pywatts.core.base import Base
from pywatts.core.filemanager import FileManager
from pywatts.core.step_information import SummaryInformation

if TYPE_CHECKING:
    from pywatts.core.step_factory import StepInformation


class BaseSummary(Base, ABC):
    """
    This is the base class of the modules. It manages the basic functionality of modules. BaseTransformer and
    BaseEstimator inherit from this class.

    :param name: Name of the module
    :type name: str
    """

    def fit(self, **kwargs):
        """
        Dummy method of fit, which does nothing
        :return:
        """

    @abstractmethod
    def transform(self, file_manager: FileManager, *args, **kwargs: xr.DataArray) -> str:
        """
        Transform method. Here the summary should be calculated.
        :param file_manager: The filemanager, it can be used to store data that corresponds to the summary as a file.
        :type: file_manager: FileManager
        :param kwargs: The input data for which a summary should be calculated.
        :type kwargs: xr.DataArray
        :return: A markdown formatted string that contains the summary.
        :rtype: str
        """

    def save(self, fm: FileManager) -> Dict:
        """
        Saves the modules and the state of the module and returns a dictionary containing the relevant information.

        :param fm: the filemanager which can be used by the module for saving information about the module.
        :type fm: FileManager
        :return: A dictionary containing the information needed for restoring the module
        :rtype: Dict
        """
        return {"params": self.get_params(),
                "name": self.name,
                "class": self.__class__.__name__,
                "module": self.__module__}

    @classmethod
    def load(cls, load_information: Dict):
        """
        Uses the data in load_information for restoring the state of the module.

        :param load_information: The data needed for restoring the state of the module
        :type load_information: Dict
        :return: The restored module
        :rtype: Base
        """
        params = load_information["params"]
        name = load_information["name"]
        return cls(name=name, **params)

    def __call__(self, **kwargs) -> SummaryInformation:
        """
        Adds this module to pipeline by creating step and step information

        :param inputs: The input for the current step. If the input is a pipeline, then the corresponding module and
                       step is a starting step in the pipeline. If inputs is a list then the elements of the list have
                       to be a StepInformation or a tuple of Stepinformations. If it is a StepInformation then the input
                       has to be provided for calculating the next step. If it is a tuple, at least the result of one of
                       the steps in the tuple of step information must be provided for calculating the next step. The
                       tuples can be used for merging to path after an if statement.
        :type inputs: Union[Pipeline, List[Union[StepInformation, Tuple[StepInformation]]]

        :rtype: SummaryInformation
        """

        non_supported_kwargs = ["use_inverse_transform", "train_if", "callbacks", "condition", "computation_mode",
                                "batch_size"]

        for kwa in non_supported_kwargs:
            if kwa in kwargs:
                warnings.warn(f"{kwa} is set for {self.name}. However, {self.name} is a SummaryModule and the"
                              f" corresponding step do not support {kwa}.")

        from pywatts.core.step_factory import StepFactory

        return StepFactory().create_summary(self, kwargs)
