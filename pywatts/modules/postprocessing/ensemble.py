from typing import Dict, Union

import logging
import xarray as xr
import numpy as np

from pywatts.core.base import BaseEstimator
from pywatts.core.exceptions import WrongParameterException
from pywatts.utils._split_kwargs import split_kwargs
from pywatts.utils._xarray_time_series_utils import numpy_to_xarray

logger = logging.getLogger(__name__)


class Ensemble(BaseEstimator):
    """
    Aggregation step to ensemble the given time series, ether by simple or weighted averaging.
    By default simple averaging is applied.
    """

    def __init__(self, weights: Union[str, list] = None, k_best: Union[str, int] = None,
                 loss_metric: str = "rmse", name: str = "Ensemble"):
        """ Initialize the ensemble step.
        :param weights: List of individual weights of the given forecasts for weighted averaging. Passing "auto"
        estimates the weights depending on the given loss values.
        :type weights: list, optional
        :param loss_metric: Specifies the loss metric for automated optimal weight estimation.
        :type loss_metric: str, optional
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
        self._weights = None
        self.loss_metric = loss_metric
        self.k_best = k_best
        self.is_fitted = False

    def get_params(self) -> Dict[str, object]:
        """ Get parameters for the Ensemble object.
        :return: Parameters as dict object.
        :rtype: Dict[str, object]
        """
        return {
            "weights": self.weights,
            "k_best": self.k_best,
            "loss_metric": self.loss_metric,
        }

    def set_params(self, weights: Union[str, list] = None, loss_metric: str = None, k_best: Union[str, int] = None):
        """ Set or change Ensemble object parameters.
        :param weights: List of individual weights of the given forecasts for weighted averaging. Passing "auto"
        estimates the weights depending on the given loss values.
        :type weights: list, optional
        :param loss_metric: List of the loss of the given forecasts for automated optimal weight estimation.
        :type loss_metric: str, optional
        :param k_best: Drop poor forecasts in the automated weight estimation. Passing "auto" drops poor forecasts based
        on the given loss values by applying the 1.5*IQR rule.
        :type k_best: str or int, optional
        """
        if weights is not None:
            self.weights = weights
        if loss_metric is not None:
            self.loss_metric = loss_metric
        if k_best is not None:
            self.k_best = k_best

    def fit(self, **kwargs):

        forecasts, targets = split_kwargs(kwargs)

        if self.weights == 'auto' or self.k_best is not None:
            # determine weights depending on in-sample loss
            loss_values = self._calculate_loss(ps=forecasts, ts=targets)
            # drop forecasts depending on in-sample loss
            loss_values_dropped = self._drop_forecasts(loss=loss_values)

            # overwrite weights based on given loss values and set weights of dropped forecasts to zero
            if self.weights == "auto":  # weighted averaging depending on estimated weights
                self._weights = [0 if value in loss_values_dropped else 1 / value for value in loss_values]
            elif self.weights is None:  # averaging
                self._weights = [0 if value in loss_values_dropped else 1 for value in loss_values]
            else:  # weighted averaging depending on given weights
                self._weights = [
                    0 if value in loss_values_dropped else weight for (value, weight) in zip(loss_values, self.weights)
                ]
        else:
            # use given weights
            if isinstance(self.weights, list):
                if len(self.weights) is not len(forecasts):
                    raise WrongParameterException(
                        "The number of the given weights does not match the number of given forecasts.",
                        f"Make sure to pass {len(forecasts)} weights.",
                        self.name
                    )
            self._weights = self.weights

        # normalize weights
        if self._weights:
            self._weights = [weight / sum(self._weights) for weight in self._weights]

        self.is_fitted = True

    def transform(self, **kwargs) -> xr.DataArray:
        """ Ensemble the given time series by simple or weighted averaging.
        :return: Xarray dataset aggregated by simple or weighted averaging.
        :rtype: xr.DataArray
        """

        forecasts, _ = split_kwargs(kwargs)

        list_of_series = []
        list_of_indexes = []
        for series in forecasts.values():
            list_of_indexes.append(series.indexes)
            list_of_series.append(series.data)

        if not all(all(index) == all(list_of_indexes[0]) for index in list_of_indexes):
            raise ValueError("The indexes of the given time series for averaging do not match")

        result = np.average(list_of_series, axis=0, weights=self._weights)

        return numpy_to_xarray(result, series, self.name)

    def _calculate_loss(self, ps, ts):

        t_ = np.array([t.values for t in ts.values()])
        loss_values = []
        for p in ps.values():
            p_ = p.values
            if self.loss_metric == "rmse":
                loss_values.append(np.sqrt(np.mean((p_ - t_) ** 2)))
            elif self.loss_metric == "mae":
                loss_values.append(np.mean(np.abs((p_ - t_))))
            else:
                WrongParameterException(
                    "The specified loss metric is not implemented.",
                    "Make sure to pass the loss metric 'rmse' or 'mae'.",
                    self.name
                )

        return loss_values

    def _drop_forecasts(self, loss: list):
        loss_values_dropped = []
        if self.k_best is not None:
            # Do not sort the loss_values! Otherwise, the weights do not match the given forecasts.
            if self.k_best == "auto":
                q75, q25 = np.percentile(loss, [75, 25])
                iqr = q75 - q25
                upper_bound = q75 + 1.5 * iqr  # only check for outliers with high loss
                loss_values_dropped = [value for value in loss if not (value <= upper_bound)]
            elif self.k_best > len(loss):
                raise WrongParameterException(
                    "The given k is greater than the number of the given loss values.",
                    f"Make sure to define k <= {len(loss)}.",
                    self.name
                )
            else:
                loss_values_dropped = sorted(loss)[self.k_best:]

        return loss_values_dropped
