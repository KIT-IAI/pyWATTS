# -----------------------------------------------------------
# This example presents the code used in the advanced example
# guide in the pyWATTS documentation.
# -----------------------------------------------------------

import time

import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import tensorflow as tf
import torch
from pywatts.wrapper.keras_wrapper import KerasWrapper

# From pyWATTS the pipeline is imported
from pywatts.core.computation_mode import ComputationMode
from pywatts.core.pipeline import Pipeline

# Import the pyWATTS pipeline and the required modules
from pywatts.modules.calendar_extraction import CalendarExtraction
from pywatts.modules.whitelister import WhiteLister
from pywatts.wrapper.sklearn_wrapper import SKLearnWrapper
from pywatts.modules.clock_shift import ClockShift
from pywatts.modules.linear_interpolation import LinearInterpolater
from pywatts.modules.root_mean_squared_error import RmseCalculator

from pywatts.wrapper.pytorch_wrapper import PyTorchWrapper


def get_sequential_model():
    # D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    D_in, H, D_out = 2, 10, 1

    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
    )

    return model


if __name__ == "__main__":
    pytorch_model = get_sequential_model()

    pipeline = Pipeline(path="results")

    # Extract dummy calender features, using holidays from Germany
    calendar_features = CalendarExtraction(encoding="numerical", continent="Europe", country="Germany")(pipeline)

    # Select individual time-series (columns) and generate plots in the results folder
    white_lister_power_statistics = WhiteLister(target="load_power_statistics", name="filter_power")(pipeline,
                                                                                                     plot=True)
    white_lister_transparency = WhiteLister(target="load_transparency", name="filter_transparency")(pipeline, plot=True)
    white_lister_price = WhiteLister(target="price_day_ahead", name="filter_price")(pipeline, plot=True)

    # Deal with missing values through linear interpolation
    imputer_power_statistics = LinearInterpolater(method="nearest", dim="time",
                                                  name="imputer_power")([white_lister_power_statistics])

    # Scale the data using a standard SKLearn scaler
    power_scaler = SKLearnWrapper(module=StandardScaler(), name="scaler_power")
    scale_power_statistics = power_scaler([imputer_power_statistics])

    # Create lagged time series to later be used in the regression
    shift_power_statistics = ClockShift(lag=1, name="ClockShift_Lag1")([scale_power_statistics])
    shift_power_statistics2 = ClockShift(lag=2, name="ClockShift_Lag2")([scale_power_statistics])

    pytorch_wrapper = PyTorchWrapper(get_sequential_model(),
                                     fit_kwargs={"batch_size": 8, "epochs": 1},
                                     compile_kwargs={"loss": "mse", "optimizer": "Adam", "metrics": ["mse"]})\
        ([shift_power_statistics,
          shift_power_statistics2],
         targets=[scale_power_statistics])

    inverse_power_scale_dl = power_scaler([pytorch_wrapper],
                                          computation_mode=ComputationMode.Transform,
                                          use_inverse_transform=True,
                                          plot=True)

    rmse_dl = RmseCalculator(target="load_power_statistics", predictions=["scaler_power"])(
        [inverse_power_scale_dl, white_lister_power_statistics], to_csv=True)

    # Now, the pipeline is complete
    # so we can load data and train the model
    data = pd.read_csv("data/getting_started_data.csv", parse_dates=["time"], infer_datetime_format=True,
                       index_col="time")

    pipeline.train(data)
    pipeline.to_folder("./pipe_pytorch")
