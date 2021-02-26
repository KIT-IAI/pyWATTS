# -----------------------------------------------------------
# This example presents the code used in the advanced example
# guide in the pyWATTS documentation.
# -----------------------------------------------------------

import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

# From pyWATTS the pipeline is imported
from pywatts.core.computation_mode import ComputationMode
from pywatts.core.pipeline import Pipeline
from pywatts.callbacks import CSVCallback, LinePlotCallback


# Import the pyWATTS pipeline and the required modules
from pywatts.modules.calendar_extraction import CalendarExtraction
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
    calendar_features = CalendarExtraction(encoding="numerical", continent="Europe", country="Germany")\
                        (
                            x=pipeline["load_power_statistics"]
                        )

    # Deal with missing values through linear interpolation
    imputer_power_statistics = LinearInterpolater(method="nearest", dim="time",
                                                  name="imputer_power")(x=pipeline["load_power_statistics"])

    # Scale the data using a standard SKLearn scaler
    power_scaler = SKLearnWrapper(module=StandardScaler(), name="scaler_power")
    scale_power_statistics = power_scaler(x=imputer_power_statistics)

    # Create lagged time series to later be used in the regression
    shift_power_statistics = ClockShift(lag=1, name="ClockShift_Lag1")(x=scale_power_statistics)
    shift_power_statistics2 = ClockShift(lag=2, name="ClockShift_Lag2")(x=scale_power_statistics)

    pytorch_wrapper = PyTorchWrapper(get_sequential_model(),
                                     fit_kwargs={"batch_size": 8, "epochs": 1},
                                     compile_kwargs={"loss": "mse", "optimizer": "Adam", "metrics": ["mse"]})\
                      (
                        power_lag1=shift_power_statistics,
                        power_lag2=shift_power_statistics2,
                        target=scale_power_statistics
                      )

    inverse_power_scale = power_scaler(x=pytorch_wrapper,
                                       computation_mode=ComputationMode.Transform,
                                       use_inverse_transform=True,
                                       callbacks=[LinePlotCallback('forecast')])

    rmse_dl = RmseCalculator()(y_hat=inverse_power_scale, y=pipeline["load_power_statistics"], callbacks=[CSVCallback('RMSE')])

    # Now, the pipeline is complete
    # so we can load data and train the model
    data = pd.read_csv("data/getting_started_data.csv", parse_dates=["time"], infer_datetime_format=True,
                       index_col="time")

    pipeline.train(data)
    pipeline.to_folder("./pipe_pytorch")
