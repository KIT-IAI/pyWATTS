# pylint: disable=W0233
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Union, Tuple, Callable, TYPE_CHECKING

import pandas as pd
import xarray as xr

from pywatts.core.filemanager import FileManager
from pywatts.core.computation_mode import ComputationMode
from pywatts.core.exceptions.kind_of_transform_does_not_exist_exception import KindOfTransformDoesNotExistException, \
    KindOfTransform

if TYPE_CHECKING:
    from pywatts.core.pipeline import Pipeline
    from pywatts.core.step_factory import StepInformation


class Base(ABC):
    """
    This is the base class of the modules. It manages the basic functionality of modules. BaseTransformer and
    BaseEstimator inherit from this class.

    :param name: Name of the module
    :type name: str
    """

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.is_wrapper = False

        self.has_inverse_transform = False
        self.has_predict_proba = False

    @abstractmethod
    def get_params(self) -> Dict[str, object]:
        """
        Get params
        :return: Dict with params
        """

    @abstractmethod
    def set_params(self, *args, **kwargs):
        """
        Set params
        :return:
        """

    @abstractmethod
    def fit(self, **kwargs):
        """
        Fit the model, e.g. optimize parameters such that model(x) = y

        :param x: input
        :param y: target
        :return:
        """

    @abstractmethod
    def transform(self, x: xr.Dataset) -> xr.Dataset:
        """
        Transforms the input.

        :param x: the input
        :return: The transformed input
        """

    def inverse_transform(self, x: xr.Dataset) -> xr.Dataset:
        """
        Performs the inverse transformation if available.
        Note for developers of modules: if this method is implemented the flag "self.has_inverse_transform" must be set
        to True in the constructor.
         I.e. self.has_inverse_transform = True (must be called after "super().__init__(name)")

        :param x: the input
        :return: The transformed input
        """
        # if this method is not overwritten and hence not implemented, raise an exception
        raise KindOfTransformDoesNotExistException(f"The module {self.name} does not have a inverse transformation",
                                                   KindOfTransform.INVERSE_TRANSFORM)

    def predict_proba(self, x: xr.Dataset) -> xr.Dataset:
        """
        Performs the probabilistic transformation if available.
        Note for developers of modules: if this method is implemented, the flag "self.has_predict_proba" must be set to
        True in the constructor.
        I.e. self.has_inverse_transform = True (must be called after "super().__init__(name)")

        :param x: the input
        :return: The transformed input
        """
        # if this method is not overwritten and hence not implemented, raise an exception
        raise KindOfTransformDoesNotExistException(
            f"The module {self.name} does not have a probablistic transformation",
            KindOfTransform.PROBABILISTIC_TRANSFORM
        )

    def save(self, fm: FileManager) -> Dict:
        """
        Saves the modules and the state of the module and returns a dictionary containing the relevant information.

        :param fm: the filemanager which can be used by the module for saving information about the module.
        :type fm: FileManager
        :return: A dictionary containing the information needed for restoring the module
        :rtype:Dict
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

    def __call__(self,
                 use_inverse_transform: bool = False,
                 use_prob_transform: bool = False,
                 plot: bool = False,
                 to_csv: bool = False,
                 summary: bool = False,
                 condition: Optional[Callable] = None,
                 computation_mode: ComputationMode = ComputationMode.Default,
                 batch_size: Optional[pd.Timedelta] = None,
                 train_if: Optional[Union[Callable, bool]] = None,
                 **kwargs: Union[StepInformation, Tuple[StepInformation, ...]]
                 ) -> StepInformation:
        """
        Adds this module to pipeline by creating step and step information

        :param inputs: The input for the current step. If the input is a pipeline, then the corresponding module and
                       step is a starting step in the pipeline. If inputs is a list then the elements of the list have
                       to be a StepInformation or a tuple of Stepinformations. If it is a StepInformation then the input
                       has to be provided for calculating the next step. If it is a tuple, at least the result of one of
                       the steps in the tuple of step information must be provided for calculating the next step. The
                       tuples can be used for merging to path after an if statement.
        :type inputs: Union[Pipeline, List[Union[StepInformation, Tuple[StepInformation]]]
        :param targets: The steps which provide the target value for the current step. For the meaning of the tuples see
            inputs
        :type targets: List[Union[StepInformation, Tuple[StepInformation]]
        :param use_inverse_transform: Indicate if inverse transform should be called instead of transform.
                                      (default false)
        :type use_inverse_transform: bool
        :param use_prob_transform: Indicate if prob predict should be called instead of transform. (default false)
        :type use_prob_transform: bool
        :param plot: Indicate if the result of the current step should be plotted. (default false)
        :type plot: bool
        :param to_csv: Indicate if the result of the current step should be saved as csv. (default false)
        :type to_csv: bool
        :param train_if: A callable, which contains a condition that indicates if the module should be trained or not
        :type train_if: Optional[Callable]
        :param batch_size: Determines how much data from the past should be used for training
        :type batch_size: pd.Timedelta
        :param computation_mode: Determines the computation mode of the step. Could be ComputationMode.Train,
                                 ComputationMode.Transform, and Computation.FitTransform
        :type computation_mode: ComputationMode
        :return: Tuple of two step informations. The first one is the original step_information. The second one is
              the stepinformation with the inversed condition
        :rtype: Tuple[StepInformation]
        """

        from pywatts.core.step_factory import StepFactory

        return StepFactory().create_step(self, kwargs=kwargs,
                                         use_inverse_transform=use_inverse_transform,
                                         use_predict_proba=use_prob_transform, plot=plot, to_csv=to_csv,
                                         summary=summary,
                                         condition=condition,
                                         computation_mode=computation_mode, batch_size=batch_size,
                                         train_if=train_if
                                         )


class BaseTransformer(Base, ABC):
    """
    The base class for all transformer modules. It provides a dummy fit method.
    """

    def fit(self, **kwargs):
        """
        Dummy method of fit, which does nothing
        :return:
        """


class BaseEstimator(Base, ABC):
    """
    The base class for all estimator modules.

    :param name: The name of the module
    :type name: str
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.is_fitted = False

    def save(self, fm: FileManager) -> Dict:
        """
        Saves the modules and the state of the module and returns a dictionary containing the relevant information.

        :param fm: the filemanager which can be used by the module for saving information about the module.
        :type fm: FileManager
        :return: A dictionary containing the information needed for restoring the module
        :rtype:Dict
        """
        json_module = super().save(fm)
        json_module["is_fitted"] = self.is_fitted
        return json_module

    @classmethod
    def load(cls, load_information) -> BaseEstimator:
        """
        Uses the data in load_information for restoring the state of the module.

        :param load_information: The data needed for restoring the state of the module
        :type load_information: Dict
        :return: The restored module
        :rtype: BaseEstimator
        """
        module = super().__class__.load(load_information)
        module.is_fitted = load_information["is_fitted"]
        return module
