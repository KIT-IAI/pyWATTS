import os
import unittest
from time import sleep

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from pywatts.core.pipeline import Pipeline
from pywatts.modules import LinearInterpolater, SKLearnWrapper
from pywatts.summaries import RMSE

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "../..",
    'data',
)


class TestSimplePipeline(unittest.TestCase):

    def setUp(self) -> None:
        # TODO: Look for better solution... This fails since the tests are faster than one second. Consequently all directory have the same timestamp
        sleep(1)

    def test_create_and_run_simple_pipeline(self):
        pipeline = Pipeline()
        imputer_power_statistics = LinearInterpolater(method="nearest", dim="time",
                                                      name="imputer_power")(x=pipeline["load_power_statistics"])
        imputer_price = LinearInterpolater(method="nearest", dim="time",
                                           name="imputer_price")(x=pipeline["price_day_ahead"])
        scaler = SKLearnWrapper(StandardScaler())(x=imputer_price)
        lin_regression = SKLearnWrapper(LinearRegression())(x=scaler, target1=imputer_price, target2=imputer_power_statistics)

        RMSE(name="Load")(y=imputer_power_statistics, pred=lin_regression["target2"])
        RMSE(name="Price")(y=imputer_price, pred=lin_regression["target1"])
        data = pd.read_csv(f"{FIXTURE_DIR}/getting_started_data.csv", index_col="time", sep=",", parse_dates=["time"],
                           infer_datetime_format=True)
        train = data[6000:]
        test = data[:6000]
        pipeline.train(train)
        pipeline.test(test)

    def test_run_reloaded_simple_pipeline(self):
        pipeline = Pipeline()

        imputer_power_statistics = LinearInterpolater(method="nearest", dim="time",
                                                      name="imputer_power")(x=pipeline["load_power_statistics"])
        imputer_price = LinearInterpolater(method="nearest", dim="time",
                                           name="imputer_price")(x=pipeline["price_day_ahead"])
        scaler = SKLearnWrapper(StandardScaler())(x=imputer_price)
        SKLearnWrapper(LinearRegression())(x=scaler, target1=imputer_price, target2=imputer_power_statistics)

        pipeline.to_folder("./pipe1")
        sleep(1)

        pipeline2 = Pipeline.from_folder("./pipe1")

        data = pd.read_csv(f"{FIXTURE_DIR}/getting_started_data.csv", index_col="time", sep=",", parse_dates=["time"],
                           infer_datetime_format=True)
        train = data[6000:]
        test = data[:6000]
        pipeline2.train(train)
        pipeline2.test(test)
