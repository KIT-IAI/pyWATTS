# pylint: disable=W0233
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Union, Tuple, Callable, TYPE_CHECKING
import logging
import pandas as pd
import xarray as xr

from pywatts.core.computation_mode import ComputationMode
from pywatts.core.exceptions.kind_of_transform_does_not_exist_exception import KindOfTransformDoesNotExistException, \
    KindOfTransform
from pywatts.core.filemanager import FileManager
from pywatts.callbacks import BaseCallback

if TYPE_CHECKING:
    from pywatts.core.step_factory import StepInformation


class Base(ABC):
    """
    This is the base class of the modules. It manages the basic functionality of modules. BaseTransformer and
    BaseEstimator inherit from this class.

    :param name: Name of the module
    :type name: str
    """

    def __init__(self, name: str):
        self.name = name
        self.is_wrapper = False
        self.logger = logging.getLogger(name)

        self.has_inverse_transform = False
        self.has_predict_proba = False

    @abstractmethod
    def get_params(self) -> Dict[str, object]:
        """
        Get params
        :return: Dict with params
        """

    @abstractmethod
    def set_params(self, **kwargs):
        """
        Set params
        :return:
        """

    @abstractmethod
    def fit(self, **kwargs):
        """
        Fit the model, e.g. optimize parameters such that model(x) = y

        :param kwargs: key word arguments as input. If the key word starts with target, then it is a target variable.
        :return:
        """

    @abstractmethod
    def transform(self, **kwargs: Dict[str: xr.DataArray]) -> xr.DataArray:
        """
        Transforms the input.

        :param x: key word arguments as input. Note that it is not allowed to use key words that starts with target
                  here, since this are target variables. And this function shall not fit any models against a target
                  variable.
        :return: The transformed input
        """

    def inverse_transform(self, **kwargs: xr.DataArray) -> xr.DataArray:
        """
        Performs the inverse transformation if available.
        Note for developers of modules: if this method is implemented the flag "self.has_inverse_transform" must be set
        to True in the constructor.
        I.e. self.has_inverse_transform = True (must be called after "super().__init__(name)")

        :param x: key word arguments as input. Note that it is not allowed to use key words that starts with target
                  here, since this are target variables. And this function shall not fit any models against a target
                  variable.
        :return: The transformed input
        """
        # if this method is not overwritten and hence not implemented, raise an exception
        raise KindOfTransformDoesNotExistException(f"The module {self.name} does not have a inverse transformation",
                                                   KindOfTransform.INVERSE_TRANSFORM)

    def predict_proba(self, *kwargs: Dict[str: xr.DataArray]) -> xr.DataArray:
        """
        Performs the probabilistic transformation if available.
        Note for developers of modules: if this method is implemented, the flag "self.has_predict_proba" must be set to
        True in the constructor.
        I.e. self.has_inverse_transform = True (must be called after "super().__init__(name)")

        :param x: key word arguments as input. Note that it is not allowed to use key words that starts with target
                  here, since this are target variables. And this function shall not fit any models against a target
                  variable.
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

    def refit(self, **kwargs):
        """
        This method refits the module. If not overwritten it is the same as fit.
        :param kwargs: key word arguments as input. If the key word starts with target, then it is a target variable.
        """
        return self.fit(**kwargs)

    def get_min_data(self):
        return 0

    def __call__(self,
                 use_inverse_transform: bool = False,
                 use_prob_transform: bool = False,
                 callbacks: List[BaseCallback, Callable[[Dict[str, xr.DataArray]], None]] = [],
                 condition: Optional[Callable] = None,
                 computation_mode: ComputationMode = ComputationMode.Default,
                 batch_size: Optional[pd.Timedelta] = None,
                 refit_conditions: List[Union[Callable, bool]] = [],
                 lag: Optional[int] = pd.Timedelta(hours=0),
                 retrain_batch: Optional[int] = pd.Timedelta(hours=24),
                 **kwargs: Union[StepInformation, Tuple[StepInformation, ...]]
                 ) -> StepInformation:
        """
        Adds this module to pipeline by creating step and step information

        :param kwargs: The inputs for the current step. The user has to choose as key words the key words of the fit and
                       transform method of the module which is added. Note, that all keywords that starts with "target"
                       are target variables and only passed to the fit method, for fitting the module.
                       Moreover, note that if the input is a pipeline or a module with multiple outputs, it
                       is important that the desired data column is specified by input[<desired_column>].

                       The input can also be a tuple, in that case at least the result of one of
                       the steps in the tuple must be provided for calculating the next step. The
                       tuples can be used for merging paths after a condition.
        :type kwargs: Union[StepInformation, Tuple[StepInformation]]
        :param use_inverse_transform: Indicate if inverse transform should be called instead of transform.
                                      (default false)
        :type use_inverse_transform: bool
        :param use_prob_transform: Indicate if prob predict should be called instead of transform. (default false)
        :type use_prob_transform: bool
        :param callbacks: Callbacks to use after results are processed.
        :type callbacks: List[BaseCallback, Callable[[Dict[str, xr.DataArray]]]]
        :param refit_conditions: A List of Callables of BaseConditions, which contains a condition that indicates if
                                 the module should be trained or not
        :type refit_conditions: List[Union[BaseCondition, Callable]]
        :param batch_size: Determines how much data from the past should be used for training
        :type batch_size: pd.Timedelta
        :param computation_mode: Determines the computation mode of the step. Could be ComputationMode.Train,
                                 ComputationMode.Transform, and Computation.FitTransform
        :type computation_mode: ComputationMode
        :param lag: Needed for online learning. Determines what data can be used for retraining.
                    E.g., when 24 hour forecasts are performed, a lag of 24 hours is needed, else the retraining would
                    use future values as target values.
        :type lag: pd.Timedelta
        :param retrain_batch: Needed for online learning. Determines how much data should be used for retraining.
        :type retrain_batch: pd.Timedelta
        :return: a step information.
        :rtype: StepInformation
        """

        from pywatts.core.step_factory import StepFactory

        return StepFactory().create_step(self, kwargs=kwargs,
                                         use_inverse_transform=use_inverse_transform,
                                         use_predict_proba=use_prob_transform,
                                         condition=condition,
                                         callbacks=callbacks,
                                         computation_mode=computation_mode, batch_size=batch_size,
                                         refit_conditions=refit_conditions if isinstance(refit_conditions, list) else [
                                             refit_conditions
                                         ],
                                         retrain_batch=retrain_batch,
                                         lag=lag
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
        module = super(BaseEstimator, cls).load(load_information)
        module.is_fitted = load_information["is_fitted"]
        return module
