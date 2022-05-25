from enum import IntEnum
from typing import Dict

import numpy as np
import pandas as pd
import tensorflow.keras as keras
import xarray as xr
import cloudpickle
from sklearn.metrics import mean_absolute_error, f1_score, accuracy_score

from pywatts.core.base_summary import BaseSummary
from pywatts.core.filemanager import FileManager
from pywatts.core.summary_object import SummaryObjectTable


class TSTRTask(IntEnum):
    """
    The implemented TSTRTasks
    """
    Classification = 1
    Regression = 2


class TrainSyntheticTestReal(BaseSummary):
    """
    The Train Synthetic Test Real (TSTR) evaluation strategy for synthetic data. It trains a classification or
    regression model on the synthetic data and evaluates it on real data.
    :param name: Name of the module
    :type name: str
    :param train_test_split: The share of data that should be used for training.
    :type train_test_split: float
    :param repetitions: The number of repetitions a model should be trained and evaluated on each data set.
    :type repetitions: int
    :param fit_kwargs: Key Word arguments for fitting the model.
    :type fit_kwargs: Dict
    :param get_model: A function that returns a model that implements a fit and a predict method.
    :type get_model: Callable
    :param n_targets: The number of classes for the classifaction task or the horizon for the regression task.
    :type n_targets: int
    :param task: Specifies if the TSTR should solve a regression task or a classification task
    :type task: TSTRTask
    :param metrics: A list of strings specifying metrics. Currently implemented are 'mae', 'rmse' 'mape' 'accuracy',
                    and 'f1'
    :type metrics: List[str]
    """

    def __init__(self, name: str = "TSTR", train_test_split=0.66,
                 repetitions=5, task=TSTRTask.Regression, metrics=None,
                 fit_kwargs=None, get_model=None, n_targets=1):
        super().__init__(name)
        self.repetitions = repetitions
        self.task = task
        self.n_targets = n_targets
        if metrics is None and self.task == TSTRTask.Regression:
            self.metrics = ["rmse", "mae"]
        elif metrics is None and self.task == TSTRTask.Classification:
            self.metrics = ["f1"]
        else:
            self.metrics = metrics
        self.train_test_split = train_test_split
        if fit_kwargs is None:
            self.fit_kwargs = {"epochs": 100, "validation_split": 0.2}
        else:
            self.fit_kwargs = fit_kwargs
        if get_model is None:
            self._get_model = self._get_regressor if task == TSTRTask.Regression else self._get_classifier
        else:
            self._get_model = get_model

    @staticmethod
    def _get_regressor(horizon, pred_horizon):
        """
        Default regressor
        :param horizon: the length of the input horizon
        :type horizon: int
        :param pred_horizon: The length of the values to be predicted
        :type pred_horizon: int
        """
        input = keras.layers.Input((horizon))
        state = keras.layers.Dense(10, activation="relu")(input)
        output = keras.layers.Dense(pred_horizon, activation="linear")(state)
        model = keras.Model(input, output)
        model.compile(loss="mse")
        return model

    @staticmethod
    def _get_classifier(horizon, n_targets):
        """
        Default classifier
        :param horizon: the length of the input horizon
        :type horizon: int
        :param n_targets: The number of classes that should be predicted.
        :type n_targets: int
        """
        input = keras.layers.Input((horizon))
        state = keras.layers.Dense(10, activation="relu")(input)
        output = keras.layers.Dense(n_targets, activation="softmax")(state)
        model = keras.Model(input, output)
        model.compile(loss="binary-crossentropy")
        return model

    def get_params(self) -> Dict[str, object]:
        """
        Get the params of the TSTR Module
        
        :return: Dict containing all parameters
        """

        return {
            "repetitions": self.repetitions,
            "train_test_split": self.train_test_split,
            "task": self.task,
            "metrics": self.metrics,
            "fit_kwargs": self.fit_kwargs,
            "get_model": self._get_model,
            "n_targets": self.n_targets,
        }

    def set_params(self, train_test_split=None, repetitions=None, task=None, metrics=None,
                   fit_kwargs=None, get_model=None, n_targets=None):
        """
        Set the params of the TSTR Module.
        
        :param train_test_split: The share of data that should be used for training.
        :type train_test_split: float
        :param repetitions: The number of repetitions a model should be trained and evaluated on each data set.
        :type repetitions: int
        :param fit_kwargs: Key Word arguments for fitting the model.
        :type fit_kwargs: Dict
        :param get_model: A function that returns a model that implements a fit and a predict method.
        :type get_model: Callable
        :param n_targets: The number of classes for the classifaction task or the horizon for the regression task.
        :type n_targets: int
        :param task: Specifies if the TSTR should solve a regression task or a classification task
        :type task: TSTRTask
        :param metrics: A list of strings specifying metrics. Currently implemented are 'mae', 'rmse' 'mape' 'accuracy',
                        and 'f1'
        :type metrics: List[str]
        """
        if repetitions is not None:
            self.repetitions = repetitions
        if train_test_split is not None:
            self.train_test_split = train_test_split
        if task is not None:
            self.task = task
        if metrics is not None:
            self.metrics = metrics
        if fit_kwargs is not None:
            self.fit_kwargs = fit_kwargs
        if get_model is not None:
            self._get_model = get_model
        if n_targets is not None:
            self.n_targets = n_targets

    @staticmethod
    def _rmse(y_pred, y_true):
        return np.sqrt(np.mean(np.square(y_pred - y_true)))

    @staticmethod
    def _mae(y_pred, y_true):
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def _mape(y_pred, y_true):
        return np.mean((y_true - y_pred) / y_true)

    @staticmethod
    def _accuracy(y_pred, y_true):
        return accuracy_score(y_pred, y_true)

    @staticmethod
    def _f1(y_pred, y_true):
        return f1_score(y_pred, y_true)

    def transform(self, file_manager: FileManager, real: xr.DataArray = None,
                  **kwargs: xr.DataArray) -> SummaryObjectTable:
        """
        This method calculates the TSTR score for each dataset in kwargs.
        :param file_manager: The filemanager
        :type file_manager: FileManager
        :param real: The original time series
        :type real: xr.DataArray
        :param kwargs: The generated syntehtic time series
        :type kwargs: Dict[xr.DataArray]
        :return: A summary containing all scores.
        :rtype: SummaryObjectTable
        """

        cols = []
        for metric_name in self.metrics:
            cols.extend([f"{metric_name}_Model", f"{metric_name}_Sum", f"{metric_name}_Min",
                         f"{metric_name}_Max", f"{metric_name}_Mean"])
        df_result = pd.DataFrame(columns=cols)

        if self.task == TSTRTask.Regression:
            for name, data in kwargs.items():
                X, y, X_test, y_test = self._get_regression_data(real.values, data.values)
                res = self._evaluate(X, y, X_test, y_test, self.n_targets, name)
                df_result = df_result.append(res, ignore_index=True)
            X, y, X_test, y_test = self._get_regression_data(real.values, real.values)
            res = self._evaluate(X, y, X_test, y_test, self.n_targets, "real_data")
            df_result = df_result.append(res, ignore_index=True)

        else:
            for name, data in kwargs.items():
                if name.endswith("_target"):
                    continue
                X, y, X_test, y_test = self._get_classification_data(real.values, kwargs[name].values,
                                                                     kwargs["real_target"].values,
                                                                     kwargs[name + "_target"].values)
                res = self._evaluate(X, y, X_test, y_test, self.n_targets, name)
                df_result = df_result.append(res, ignore_index=True)
            X, y, X_test, y_test = self._get_classification_data(real.values, real.values, kwargs["real_target"].values,
                                                                 kwargs["real_target"].values, )
            res = self._evaluate(X, y, X_test, y_test, self.n_targets, "real_data")
            df_result = df_result.append(res, ignore_index=True)

        resulting_summary = SummaryObjectTable(self.name, additional_information=str(df_result.columns))
        resulting_summary.set_kv(self.name, df_result.values)

        return resulting_summary

    def get_metrics(self, metric_name: str):
        """
        Select the metric implementation from the given name
        :param metric_name:
        :return: the metric method
        """
        if metric_name == "rmse":
            return self._rmse
        elif metric_name == "mase":
            return self._mase
        elif metric_name == "mae":
            return self._mae
        elif metric_name == "mape":
            return self._mape
        elif metric_name == "f1":
            return self._f1
        elif metric_name == "accuracy":
            return self._accuracy

    def _get_regression_data(self, real, synthetic):
        """
        Get a data set for the regression task.
        :param real: The dataset containing the real data.
        :type param: np.array
        :param synthetic: The dataset containing the synthetic data.
        :type synthetic: np.array
        """
        split_index = int(self.train_test_split * len(real))
        X_train = synthetic[:split_index, :-self.n_targets]
        y_train = synthetic[:split_index, -self.n_targets:]
        X_test = real[split_index:, :-self.n_targets]
        y_test = real[split_index:, -self.n_targets:]
        return X_train, y_train, X_test, y_test

    def _get_classification_data(self, real, synthetic, real_label, synthetic_label):
        """
        Get a data set for the classification task.
        :param real: The dataset containing the real data.
        :type param: np.array
        :param synthetic: The dataset containing the synthetic data.
        :type synthetic: np.array
        """
        split_index = int(self.train_test_split * len(real))
        X_train = synthetic[:split_index]
        y_train = synthetic_label[:split_index]
        X_test = real[split_index:]
        y_test = real_label[split_index:]
        return X_train, y_train, X_test, y_test

    def _evaluate(self, train_x, train_y, test_x, test_y, n_targets, name):
        """
        Evaluate the synthetic data.
        :param train_x: The input training data.
        :type train_x: np.array
        :param train_y: The target training data.
        :type test_y: np.array
        :param test_x: The input test data.
        :type test_x: np.array
        :param test_y: The target test data
        :type test_y: np.array.
        :param n_targets: The number of values that should be predicted or the number of classes.
        :type n_targets: int
        :param name: The name of the currently evaluated data.
        :type name: str
        :return: The results of the evaluation.
        :rtype: Dict
        """
        r_temp = {}
        for metric_name in self.metrics:
            r_temp.update({f"{metric_name}_Model": name, f"{metric_name}_Sum": 0,
                           f"{metric_name}_Min": 1000000, f"{metric_name}_Max": 0})

        for i in range(self.repetitions):
            is_nan = True
            while (is_nan):
                model = self._get_model(train_x.shape[1], n_targets)
                model.fit(train_x, train_y, **self.fit_kwargs)
                result = model.predict(test_x)
                is_nan = np.any(np.isnan(result))
                del model

            for metric_name in self.metrics:
                metric = self.get_metrics(metric_name)
                value = metric(result, test_y)
                r_temp[f"{metric_name}_Sum"] += value
                if r_temp[f"{metric_name}_Min"] > value:
                    r_temp[f"{metric_name}_Min"] = value
                if r_temp[f"{metric_name}_Max"] < value:
                    r_temp[f"{metric_name}_Max"] = value
                keras.backend.clear_session()
        for metric_name in self.metrics:
            r_temp[f"{metric_name}_Mean"] = r_temp[f"{metric_name}_Sum"] / self.repetitions
        return r_temp

    def save(self, fm: FileManager) -> Dict:
        """
        Stores the TSTR Summary
        :param fm: The Filemanager, which contains the path where the model should be stored
        """
        json = {"name": self.name,
                "class": self.__class__.__name__,
                "module": self.__module__}
        model_path = fm.get_path(f"{self.name}_get_model.pickle")
        with open(model_path, "wb") as outfile:
            cloudpickle.dump(self._get_model, outfile)

        fit_kwargs_path = fm.get_path(f"{self.name}_fit_kwargs.pickle")
        with open(fit_kwargs_path, "wb") as outfile:
            cloudpickle.dump(self._get_model, outfile)
        json["params"] = {
            "repetitions": self.repetitions,
            "train_test_split": self.train_test_split,
            "task": self.task,
            "metrics": self.metrics,
            "n_targets": self.n_targets,
        }
        json["get_model"] = model_path
        json["fit_kwargs"] = fit_kwargs_path

        return json

    @classmethod
    def load(cls, load_information: Dict):
        """
        Load the TSTR Summary
        :param load_information:  The parameters which should be used for restoring the summary.
        :return: A TSTR Summary.
        """
        name = load_information["name"]
        params = load_information["params"]
        fit_kwargs_path = load_information["fit_kwargs"]
        with open(fit_kwargs_path, "rb") as infile:
            fit_kwargs = cloudpickle.load(infile)
        model_path = load_information["get_model"]
        with open(model_path, "rb") as infile:
            get_model = cloudpickle.load(infile)

        module = cls(get_model=get_model, fit_kwargs=fit_kwargs, name=name, **params)
        return module
