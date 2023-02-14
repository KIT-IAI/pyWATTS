from typing import Dict

import cloudpickle
import numpy as np
import pandas as pd
from tensorflow import keras
import xarray as xr
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.utils.np_utils import to_categorical

from pywatts_pipeline.core.summary.base_summary import BaseSummary
from pywatts_pipeline.core.util.filemanager import FileManager
from pywatts_pipeline.core.summary.summary_object import SummaryObjectTable


class DiscriminativeScore(BaseSummary):
    """
    The discriminative score aims to measure how similar the synthetic and real data are. Therefore, the discriminative
    score trains a classifier to distinguish both. The resulting discriminative score is the accuracy on the test data
    minus 0.5.
    :param name: Name of the discriminative score.
    :type name: str
    :param fit_kwargs: Kwargs that are passed to the classifier if fit is called.
    :type fit_kwargs: Dict
    :param repititions: The number of repeititions the discriminative score should be calculated.
    :type repititions: int
    :param get_model: A function that returns a classifier. Default: A simple FC network.
    :type get_model: Callable
    :param test_size: The share of data that is used for testing
    :type test_size: float
    """
    @staticmethod
    def _get_model(horizon):
        input = keras.layers.Input(((horizon)))
        states = keras.layers.Dense(5, activation="tanh")(input)
        out = keras.layers.Dense(1, activation="sigmoid")(states)
        model = keras.Model(input, out)
        model.compile(optimizer="Adam",
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        return model

    def __init__(self, name: str = "DiscriminativeScore", fit_kwargs=None, repetitions=3, get_model=None, test_size=0.3):
        super().__init__(name)
        self.test_size = test_size
        self.repetitions = repetitions
        if fit_kwargs is None:
            self.fit_kwargs = {"epochs": 10, "validation_split": 0.2}
        else:
            self.fit_kwargs = fit_kwargs
        if get_model is not None:
            self.get_model = get_model
        else:
            self.get_model = self._get_model

    def transform(self, file_manager: FileManager, gt: xr.DataArray = None, **kwargs) -> SummaryObjectTable:
        """
        Calculates the discriminative score.

        :param gt: A data set containing the real data.
        :type gt: xr.DataArray.
        :param kwargs: The data sets containing the synthetic data.
        :type kwargs: xr.DataArray
        :return: A summary containing all the discriminative scores.
        :rtype: SummaryObjectTable
        """

        real_data_x = gt.values
        horizon = real_data_x.shape[-1]
        df_result = pd.DataFrame(columns=["Model", "ScoreSum", "ScoreMin", "ScoreMax", "ScoreMean"])

        if "gt_mask" in kwargs:
            use_masks = True
            gt = real_data_x[kwargs["gt_mask"].values]
        else:
            use_masks = False
            gt = real_data_x

        for name in filter(lambda key: not key.endswith("mask"), kwargs.keys()):
            if use_masks:
                synth_data = kwargs[name].values[kwargs[name + "_mask"].values]
            else:
                synth_data = kwargs[name].values

            number_of_data_per_class = min(len(gt), len(synth_data))

            x_data = np.concatenate([gt[:number_of_data_per_class], synth_data[:number_of_data_per_class]])
            y = np.ones(len(x_data))
            y[len(gt):] = 0

            x_data = StandardScaler().fit_transform(x_data)
            x_data = x_data.reshape((-1, horizon, 1))
            x_data_train, x_data_test, y_train, y_test = train_test_split(x_data, y, test_size=self.test_size,
                                                                          shuffle=True)
            r_temp = {"Model": name, "ScoreSum": 0, "ScoreMin": 1, "ScoreMax": 0}
            for i in range(self.repetitions):
                model = self.get_model(horizon)
                model.fit(x_data_train.reshape((-1, horizon)), y_train, **self.fit_kwargs)
                prediction = model.predict(x_data_test.reshape((-1, horizon)))
                prediction[prediction > 0.5] = 1
                prediction[prediction <= 0.5] = 0
                value = np.abs(accuracy_score(prediction, y_test) - 0.5)
                r_temp["ScoreSum"] += value
                if r_temp["ScoreMin"] > value:
                    r_temp["ScoreMin"] = value
                if r_temp["ScoreMax"] < value:
                    r_temp["ScoreMax"] = value
                del model
                keras.backend.clear_session()
            r_temp["ScoreMean"] = r_temp["ScoreSum"] / self.repetitions
            df_result = df_result.append(r_temp, ignore_index=True)

        resulting_summary = SummaryObjectTable(self.name, additional_information=str(df_result.columns))
        resulting_summary.set_kv(self.name, df_result.values)
        return resulting_summary

    def save(self, fm: FileManager) -> Dict:
        """
        Stores the DiscriminativeScore Summary
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
            "test_size": self.test_size,
        }
        json["get_model"] = model_path
        json["fit_kwargs"] = fit_kwargs_path

        return json

    @classmethod
    def load(cls, load_information: Dict):
        """
        Load the DiscriminativeScore Summary
        :param load_information:  The parameters which should be used for restoring the summary.
        :return: A DiscriminativeScore Summary.
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