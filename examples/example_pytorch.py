# -----------------------------------------------------------
# This example presents the code used in the advanced example
# guide in the pyWATTS documentation.
# -----------------------------------------------------------

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# Import the pyWATTS pipeline and the required modules

from pywatts.callbacks import LinePlotCallback
from pywatts_pipeline.core.util.computation_mode import ComputationMode
from pywatts_pipeline.core.pipeline import Pipeline
from pywatts.modules import LinearInterpolater, SKLearnWrapper, PyTorchWrapper, ClockShift
from pywatts.modules.preprocessing.select import Select
from pywatts.summaries import RMSE


def get_sequential_model():
    model = torch.nn.Sequential(
        torch.nn.Linear(24, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 24),
    )

    return model


if __name__ == "__main__":
    pytorch_model = get_sequential_model()

    pipeline = Pipeline(path="../results")

    # Deal with missing values through linear interpolation
    imputer_power_statistics = LinearInterpolater(method="nearest", dim="time",
                                                  name="imputer_power")(x=pipeline["load_power_statistics"])

    # Scale the data using a standard SKLearn scaler
    power_scaler = SKLearnWrapper(module=StandardScaler(), name="scaler_power")
    scale_power_statistics = power_scaler(x=imputer_power_statistics)

    # Create lagged time series to later be used in the regression
    lag_features = Select(start=-23, stop=1, step=1, name="lag_features")(x=scale_power_statistics)
    target = Select(start=1, stop=25, step=1, name="target")(x=scale_power_statistics)

    model = get_sequential_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    pytorch_wrapper = PyTorchWrapper(model,
                                     fit_kwargs={"batch_size": 8, "epochs": 1},
                                     optimizer=optimizer,
                                     loss_fn=torch.nn.MSELoss(reduction='sum'))\
                      (
                        lag_features=lag_features,
                        target=target
                      )

    inverse_power_scale = power_scaler(x=pytorch_wrapper,
                                       computation_mode=ComputationMode.Transform,
                                       method="inverse_transform",
                                       callbacks=[LinePlotCallback('forecast')])

    rmse_dl = RMSE()(y_hat=inverse_power_scale, y=target)

    # Now, the pipeline is complete
    # so we can load data and train the model
    data = pd.read_csv("../data/getting_started_data.csv",
                       index_col="time",
                       parse_dates=["time"],
                       infer_datetime_format=True,
                       sep=",")

    pipeline.train(data[:6000])
    pipeline.test(data[6000:])
    pipeline.to_folder("./pipe_pytorch")

    pipeline2 = Pipeline.from_folder("./pipe_pytorch")
    pipeline2.test(data[6000:])
