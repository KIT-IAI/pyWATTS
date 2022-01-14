from typing import Dict, Union

import logging
import xarray as xr
import numpy as np

from pywatts.core.base import BaseTransformer
from pywatts.core.exceptions import WrongParameterException
from pywatts.core.step_information import StepInformation
from pywatts.utils._xarray_time_series_utils import numpy_to_xarray

logger = logging.getLogger(__name__)


class Ensemble(BaseTransformer):
    """
    Aggregation step to ensemble the given time series, ether by simple or weighted averaging.
    By default simple averaging is applied.
    """

    def __init__(self, weights: Union[str, list] = None, k_best: Union[str, int] = None,
                 loss: list = None, name: str = "Ensemble"):
        """ Initialize the ensemble step.
        :param weights: List of individual weights of the given forecasts for weighted averaging. Passing "auto"
        estimates the weights depending on the given loss values.
        :type weights: list, optional
        :param loss: List of the loss of the given forecasts for automated optimal weight estimation.
        :type loss: list, optional
        :param k_best: Drop poor forecasts in the automated weight estimation. Passing "auto" drops poor forecasts based
        on the given loss values by applying the 1.5*IQR rule.
        :type k_best: str or int, optional

        example for two given forecasts
        weights = None, k_best = None           -> averaging
        weights = None, k_best = 'auto'         -> averaging k-best with k based on loss values
        weights = None, k_best = k              -> averaging k-best with given k
        weights = [0.3,0.7], k_best = None      -> weighting based on given weights
        weights = [0.3,0.7], k_best = 'auto'    -> weighting based on given weights and k based on loss values
        weights = [0.3,0.7], k_best = k         -> weighting based on given weights and k
        weights = 'auto', k_best = None         -> weighting with weights based on loss values
        weights = 'auto', k_best = 'auto'       -> weighting k-best with weights and k based on loss values
        weights = 'auto', k_best = k            -> weighting k-best with weights based on loss values and given k
        """
        super().__init__(name)

        self.weights = weights
        self.loss = loss
        self.k_best = k_best

    def get_params(self) -> Dict[str, object]:
        """ Get parameters for the Ensemble object.
        :return: Parameters as dict object.
        :rtype: Dict[str, object]
        """
        return {
            "weights": self.weights,
            "loss": self.loss,
            "k_best": self.k_best
        }

    def set_params(self, weights: Union[str, list] = None, loss: list = None, k_best: Union[str, int] = None):
        """ Set or change Ensemble object parameters.
        :param weights: List of individual weights of the given forecasts for weighted averaging. Passing "auto"
        estimates the weights depending on the given loss values.
        :type weights: list, optional
        :param loss: List of the loss of the given forecasts for automated optimal weight estimation.
        :type loss: list, optional
        :param k_best: Drop poor forecasts in the automated weight estimation. Passing "auto" drops poor forecasts based
        on the given loss values by applying the 1.5*IQR rule.
        :type k_best: str or int, optional
        """
        if weights:
            self.weights = weights
        if loss:
            self.loss = loss
        if k_best:
            self.k_best = k_best

    def fit(self, **kwargs) -> xr.DataArray:
        if self.loss:
            # determine weights depending on in-sample loss
            if len(self.loss) is not len(kwargs):
                raise WrongParameterException(
                        "The number of the given loss values does not match the number of given forecasts.",
                        f"Make sure to pass {len(kwargs)} loss terms.",
                        self.name
                    )

            loss_values = []
            for item in self.loss:
                # temporary solution! RmseCalculator will be removed.
                if isinstance(item, StepInformation):
                    loss_values += [float(value) for value in item.step.buffer.values()]
                else:
                    loss_values += item

            loss_values_dropped = []
            if self.k_best is not None:
                # Do not sort the loss_values! Otherwise the weights do not match the given forecasts.
                if self.k_best == "auto":
                    q75, q25 = np.percentile(loss_values, [75, 25])
                    iqr = q75 - q25
                    upper_bound = q75 + 1.5 * iqr  # only check for outliers with high loss
                    loss_values_dropped = [value for value in loss_values if not (value <= upper_bound)]
                elif self.k_best > len(loss_values):
                    raise WrongParameterException(
                        "The given k is greater than the number of the given loss values.",
                        f"Make sure to define k <= {len(loss_values)}.",
                        self.name
                    )
                else:
                    loss_values_dropped = sorted(loss_values)[self.k_best:]

            # overwrite weights based on given loss values and zero weights of dropped forecasts
            if self.weights == "auto":  # weighted averaging depending on estimated weights
                self.weights = [0 if value in loss_values_dropped else 1/value for value in loss_values]
            elif self.weights is None:  # averaging
                self.weights = [0 if value in loss_values_dropped else 1 for value in loss_values]
            else:  # weighted averaging depending on specified weights
                self.weights = [0 if value in loss_values_dropped else weight for (value, weight) in zip(loss_values, self.weights)]

            self.is_fitted = True
        else:
            if self.weights == 'auto':
                raise WrongParameterException(
                        "Automated weight estimation requires the input of loss values.",
                        f"Make sure to pass a list of {len(kwargs)} loss values according to "
                        "the sequence of the given forecasts.",
                        self.name
                    )
            if self.k_best is not None:
                raise WrongParameterException(
                        "Averaging or weighted averaging of the k-best forecasts requires the input of loss values.",
                        f"Make sure to pass a list of {len(kwargs)} loss values according to "
                        "the sequence of the given forecasts.",
                        self.name
                    )

    def transform(self, **kwargs) -> xr.DataArray:
        """ Ensemble the given time series by simple or weighted averaging.
        :return: Xarray dataset aggregated by simple or weighted averaging.
        :rtype: xr.DataArray
        """

        if self.weights:
            if len(self.weights) is not len(kwargs):
                raise WrongParameterException(
                    "The number of the given weights does not match the number of given forecasts.",
                    f"Make sure to pass {len(kwargs)} weights.",
                    self.name
                )

        # normalize weights
        self.weights = [weight / sum(self.weights) for weight in self.weights]

        list_of_series = []
        list_of_indexes = []
        for series in kwargs.values():
            list_of_indexes.append(series.indexes)
            list_of_series.append(series.data)

        if not all(all(index) == all(list_of_indexes[0]) for index in list_of_indexes):
            raise ValueError("The indexes of the given time series for averaging do not match")

        result = np.average(list_of_series, axis=0, weights=self.weights)

        return numpy_to_xarray(result, series, self.name)
