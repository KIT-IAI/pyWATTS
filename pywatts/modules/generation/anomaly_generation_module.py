import numbers
from typing import Optional, Union, Dict, List

import numpy as np
import xarray as xr

from pywatts.core.base import BaseTransformer
from pywatts.core.exceptions.wrong_parameter_exception import WrongParameterException


class AnomalyGeneration(BaseTransformer):
    """
    Module for generating anomalies. Anomalies depend on their type and their length.
    Examples are not-a-number anomalies or constant values over a certain period of time.
    """

    def __init__(self, name: str = "AnomalyGeneration", count: Union[int, float] = 1,
                 anomaly: str = "gap", anomaly_params: Dict = {},
                 length_params: Dict = {}, label: Optional[int] = None, seed: int = 0):
        """
        Initialization routine for the anomaly creation module.

        :param name: Name of the module (default 'AnomalyGeneration').
        :type name: str
        :param count: How many anomalies should be inserted.
                      Could be a percentage value (float) or a specific number (int) (default 1).
        :type count: Union[int, float]
        :param anomaly: Type of anomaly to be created. E.g. 'gap', 'outlier', 'negate' or 'constant' (default 'gap').
        :type anomaly: str
        :param anomaly_params: Json dict containing anomaly method parameters (default {}).
        :type anomaly_params: Dict
        :param length_params: Json dict containing length distribution parameters (default {}).
        :type length_params: Dict
        :param label: Label to use for the anomaly labels (default None).
        :type label: Optional[int]
        :param seed: Seed to be used by the random generator (default 0).
        :type seed: int
        """
        super().__init__(name)
        self.count = count
        self._check_anomaly_type(anomaly)
        self.anomaly = anomaly
        self.anomaly_params = anomaly_params
        self.length_params = length_params
        self.label = label
        self.seed = seed

    def get_params(self) -> Dict[str, object]:
        """
        Get parameters of the anomaly generation module as Dict.

        :return: Dict containing all parameters.
        :rtype: Dict[str, object]
        """
        return {
            "count": self.count,
            "anomaly": self.anomaly,
            "anomaly_params": self.anomaly_params,
            "length_params": self.length_params,
            "seed": self.seed,
        }

    def set_params(self, count: Optional[Union[int, float]] = None,
                   anomaly: Optional[str] = None, anomaly_params: Optional[Dict] = None,
                   length_params: Optional[Dict] = None, label: Optional[str] = None,
                   seed: Optional[int] = None):
        """
        Set parameters of the anomaly generation module.

        :param count: Number of anomalies to be inserted.
                      It can be a percentage value (float) or a whole number (int).
        :type count: Optional[Union[int, float]]
        :param anomaly: Type of anomaly to be inserted, e.g. 'gap', 'outlier', 'negate' or 'constant.
        :type anomaly: Optional[str]
        :param anomaly_params: JSON Dict containing anomaly method parameters.
        :type anomaly_params: Optional[Dict]
        :param length_params: JSON Dict containing length distribution parameters.
        :type length_params: Optional[Dict]
        :param label: Label to use for the anomaly labels (default None).
        :type label: Optional[int]
        :param seed: Seed to be used by the random generator.
        :type seed: Optional[int]
        """
        if count is not None:
            self.count = count
        if anomaly is not None:
            self._check_anomaly_type(anomaly)
            self.anomaly = anomaly
        if anomaly_params is not None:
            self.anomaly_params = anomaly_params
        if length_params is not None:
            self.length_params = length_params
        if label is not None:
            self.label = label
        if seed is not None:
            self.seed = seed

    def _check_anomaly_type(self, anomaly: str):
        """
        Check whether selected anomaly type exists as method '_anomaly_NAME()'

        :param anomaly: Anomaly type to be checked.
        :type anomaly: str
        :raises WrongParameterException: If method for anomaly type does not exist.
        """
        attributes = [func for func in dir(self) if callable(getattr(self, func))]
        anomalies = [
            attr.replace("_anomaly_", "")
            for attr in attributes
            if attr.startswith("_anomaly_")
        ]
        if anomaly not in anomalies:
            raise WrongParameterException(
                f"Unknown anomaly type '{anomaly}'.",
                f"Please select an implemented anomaly type. ({anomalies})",
                self
            )

    def _get_anomaly_positions(self, x: xr.DataArray, labels: xr.DataArray,
                               distribution: str = "uniform",
                               mean: float = 1, std: float = 0,
                               min: int = 1, max: int = 1):
        """
        Get the position and length of the anomalies

        :param x: Array into which anomalies should be inserted.
        :type x: xr.Dataset
        :param labels: Where to save the boolean anomalies labels (default None).
                       If the index already exists, anomalies are added without collisions.
        :type labels: xr.DataArray
        :param distribution: Selected length distribution ('uniform' or 'normal') (default 'uniform').
        :type distribution: str
        :param mean: Mean of the normal distribution (default 1).
        :type mean: float
        :param std: Standard deviation of the normal distribution (default 0).
        :type std: float
        :param min: Minimum value of the uniform distribution (default 1).
        :type min: int
        :param max: Maximum value of the uniform distribution (default 1).
        :type max: int
        :raises WrongParameterException: If no possible position for new anomalies is found
                                         or distribution type is unknown.
        :return: Indices where new anomalies start, lengths of the anomalies,
                 and boolean array containing positions of anomalies.
        """
        # transform count value to number of elements
        if isinstance(self.count, int):
            quantity = self.count
        else:
            quantity = int(self.count * len(x))

        # get anomaly lengths by distribution
        if distribution == "uniform":
            lengths = np.random.uniform(min, max, quantity)
        if distribution == "normal":
            lengths = np.random.normal(mean, std, quantity)
        lengths = np.round(lengths).astype(int)

        # get anomaly labels array
        if labels is None:
            labels = np.zeros(x.size, dtype=int)
        else:
            labels = labels.values

        # check if self.label is set to use different label than the next possible
        if self.label is None:
            anomaly_type = np.unique(labels).max() + 1
        else:
            anomaly_type = self.label

        # pick randomly free position and check whether anomaly fits into the corresponding interval
        indices = np.full(quantity, -1)
        for i, length in enumerate(lengths):
            free_positions = np.arange(x.size)[labels.flatten() == 0]
            found = False
            while not found and free_positions.size > 0:
                # NOTE: selection method only works for 1 dimensional arrays
                choice = np.random.randint(0, free_positions.size)
                position = free_positions[choice]
                if position + length > labels.size or labels[position:position + length].any():
                    # some anomaly already exists in interval  => find other position
                    free_positions = np.delete(free_positions, choice)
                else:
                    # no anomaly in interval => insert anomaly here
                    indices[i] = position
                    labels[position:position + length] = anomaly_type
                    found = True
            if found is False:
                raise WrongParameterException("No suitable position for anomaly found.",
                                              "Choose less or shorter anomalies.", self)

        return indices, lengths, labels

    def _anomaly_gap(self, target: xr.DataArray, indices: List, lengths: List):
        """
        Insert gap anomalies where values are replaced with NaN.

        :param target: Array into which anomalies should be inserted.
        :type target: xr.DataArray
        :param indices: List of positions where the anomalies should start.
        :type indices: List
        :param lengths: List of lengths of the anomalies and their positions (indices).
        :type lengths: List
        :return: Data with newly inserted anomalies.
        :rtype: xr.DataArray
        """
        # allow np.nan values in target array
        target = target.astype(np.float)
        for idx, length in zip(indices, lengths):
            target[idx:idx + length] = np.nan
        return target

    def _anomaly_constant(self, target: xr.DataArray, indices: List, lengths: List,
                          constant: Union[numbers.Number, str] = "last"):
        """
        Insert constant anomalies.

        :param target: Array into which anomalies should be inserted.
        :type target: xr.DataArray
        :param indices: List of positions where the anomalies should start.
        :type indices: List
        :param lengths: List of lengths of the anomalies and their positions (indices).
        :type lengths: List
        :param constant: Type or specific value of the anomalies (default 'last').
                         If 'last', the last valid value will be the constant.
                         If 'random', a random value in [min, max] will be used for each anomaly .
        :type constant: Union[numbers.Number, str]
        :raises WrongParameterException: If constant value is invalid or unknown.
        :return: Data with newly inserted anomalies.
        :rtype: xr.DataArray
        """
        for idx, length in zip(indices, lengths):
            if isinstance(constant, numbers.Number):
                target[idx:idx + length] = constant
            elif constant == "last":
                target[idx:idx + length] = target[idx]
            elif constant == "random":
                target[idx:idx + length] = np.random.randint(
                    target.min(), target.max()
                )
            else:
                raise WrongParameterException(
                    f"Invalid or unknown constant '{constant}'.",
                    "Please select a number or one of the constant types ('last' or 'random').",
                    self
                )

        return target

    def _anomaly_negate(self, target: xr.DataArray, indices: List, lengths: List):
        """
        Insert anomalies that negate target values.

        :param target: Array into which anomalies should be inserted.
        :type target: xr.DataArray
        :param indices: List of positions where the anomalies should start.
        :type indices: List
        :param lengths: List of lengths of the anomalies and their positions (indices).
        :type lengths: List
        :return: Data with newly inserted anomalies.
        :rtype: xr.DataArray
        """
        for idx, length in zip(indices, lengths):
            target[idx:idx + length] *= -1
        return target

    def _anomaly_outlier(self, target: xr.DataArray, indices: List, lengths: List,
                         outlier_type: str = 'std', outlier_sign: Optional[str] = None,
                         random: bool = True, limits: List = [2, 4]):
        """
        Insert anomalies that randomly multiply and, if desired, negate the target values.

        :param target: Array into which anomalies should be inserted.
        :type target: xr.DataArray
        :param indices: List of positions where the anomalies should start.
        :type indices: List
        :param lengths: List of lengths of the anomalies and their positions (indices).
        :type lengths: List
        :param outlier_type: Type of the outlier (default 'std').
                             If 'std', values of outliers are mean +- rand * std.
                             If 'mean', values of outliers are +- mean * rand.
                             If 'multiple', values of outliers are +- x[i] * rand.
        :type outlier_type: str
        :param outlier_sign: Sign of the outliers to have only 'positive' or 'negative' outliers.
                             If None, randomly choose positive or negative sign.
        :type outlier_sign: Optional[str]
        :param random: If True, shuffle new random value for every anomaly (default True).
        :type random: bool
        :param limits: Upper and lower limits of the random value (default [2, 4]).
        :type limits: List
        :raises WrongParameterException: If type of outlier is unknown.
        :return: Data with newly inserted anomalies.
        :rtype: xr.DataArray
        """
        # get target data stats needed for outlier setting
        mean = target.mean()
        std = np.std(target.values)
        random_limit = lambda: limits[0] + np.random.rand() * limits[1]

        # define outlier functions
        if outlier_type == 'std':
            # outlier definition depending on mean and std
            high_outlier = lambda x: mean + random_limit() * std
            low_outlier = lambda x: mean - random_limit() * std
        elif outlier_type == 'mean':
            # outlier definition depending only on mean (Class 3 anomaly from Moritz Weber's thesis)
            high_outlier = lambda x: mean * random_limit()
            low_outlier = lambda x: -1 * mean * random_limit()
        elif outlier_type == 'multiple':
            # outlier definition depending only on the number itself
            high_outlier = lambda x: x * random_limit()
            low_outlier = lambda x: -1 * x * random_limit()
        else:
            # NOTE: Change this if more types are added!
            raise WrongParameterException(
                f"Unknown outlier_type '{outlier_type}'.",
                "Please select an implemented outlier type ('std', 'mean', or 'multiple').",
                self
            )

        # check whether a new outlier should be calculated for each idx
        # or whether there should be a 'global' outlier variable
        if not random:
            high = high_outlier(target[indices[0]])
            high_outlier = lambda x: high
            low = low_outlier(target[indices[0]])
            low_outlier = lambda x: low

        # check whether only negative or positive outlier should be used
        if outlier_sign == 'negative':
            high_outlier = low_outlier
        elif outlier_sign == 'positive':
            low_outlier = high_outlier

        # generate outlier
        for idx, length in zip(indices, lengths):
            # NOTE: Outlier with length > 1 will be constant over the interval
            #       because of target[idx:idx + length] = single_value.
            #       Solution: length = 1 and more anomalies to generate
            if np.random.rand() > 0.5:
                target[idx:idx + length] = high_outlier(target[idx])
            else:
                target[idx:idx + length] = low_outlier(target[idx])

        return target

    def _anomaly_lag(self, target: xr.DataArray, indices: List, lengths: List):
        """
        Insert lag anomalies that drop the value to zero and accumulate the missing values of the given period
        and adds it to the end of the period.
        
        NOTE: The length of these anomalies must be longer than two steps.

        :param target: Array into which anomalies should be inserted.
        :type target: xr.DataArray
        :param indices: List of positions where the anomalies should start.
        :type indices: List
        :param lengths: List of lengths of the anomalies and their positions (indices).
        :type lengths: List
        :return: Data with newly inserted anomalies.
        :rtype: xr.DataArray
        """
        for idx, length in zip(indices, lengths):
            target[idx:idx + length - 1] = 0
        return target

    def transform(self, x: xr.DataArray, labels=None) -> Dict[str, xr.DataArray]:
        """
        Finally insert anomalies using the given parameters.

        :param x: Array to be transformed.
        :type x: xr.DataArray
        :param labels: Array of anomaly labels
        :type labels: xr.DataArray
        :return: Transformed array.
        :rtype: Dict[str, xr.DataArray]
        """
        np.random.seed(self.seed)
        indices, lengths, labels = self._get_anomaly_positions(x, labels, **self.length_params)

        x = getattr(self, f"_anomaly_{self.anomaly}")(
            x, indices, lengths, **self.anomaly_params
        )
        labels = labels.reshape(x.shape)
        labels = xr.DataArray(labels, coords=x.coords)

        return {
            self.name: x,
            "labels": labels
        }
