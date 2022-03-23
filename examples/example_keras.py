# -----------------------------------------------------------
# This example presents the code used in the advanced example
# guide in the pyWATTS documentation.
# -----------------------------------------------------------

import pandas as pd

from sklearn.preprocessing import StandardScaler

from tensorflow.keras import layers, Model

from pywatts.callbacks import LinePlotCallback

# From pyWATTS the pipeline is imported
from pywatts.core.computation_mode import ComputationMode
from pywatts.core.pipeline import Pipeline

# Import the pyWATTS pipeline and the required modules
from pywatts.modules import ClockShift, LinearInterpolater, SKLearnWrapper, KerasWrapper
from pywatts.summaries import RMSE
from tensorflow.keras import backend as K

def get_keras_model():
    # write the model with the Functional API, Sequential does not support multiple input tensors

    D_in, H, D_out = 2, 10, 1  # input dimension, hidden dimension, output dimension
    input_1 = layers.Input(shape=(1,),
                           name='ClockShift_Lag1')  # layer name must match time series name
    input_2 = layers.Input(shape=(1,),
                           name='ClockShift_Lag2')  # layer name must match time series name
    merged = layers.Concatenate(axis=1)([input_1, input_2])
    hidden = layers.Dense(H,
                          input_dim=D_in,
                          activation='tanh',
                          name='hidden')(merged)
    output = layers.Dense(D_out,
                          activation='linear',
                          name='target')(hidden)  # layer name must match time series name
    model = Model(inputs=[input_1, input_2], outputs=output)
    return model


if __name__ == "__main__":
    keras_model = get_keras_model()

    pipeline = Pipeline(path="../results")

    # Deal with missing values through linear interpolation
    imputer_power_statistics = LinearInterpolater(method="nearest", dim="time",
                                                  name="imputer_power")(x=pipeline["load_power_statistics"])

    # Scale the data using a standard SKLearn scaler
    power_scaler = SKLearnWrapper(module=StandardScaler(), name="scaler_power")
    scale_power_statistics = power_scaler(x=imputer_power_statistics)

    # Create lagged time series to later be used in the regression
    # sampler_module -> 2D-Zeitreihe
    shift_power_statistics = ClockShift(lag=1, name="ClockShift_Lag1")(x=scale_power_statistics)
    shift_power_statistics2 = ClockShift(lag=2, name="ClockShift_Lag2")(x=scale_power_statistics)

    keras_wrapper = KerasWrapper(keras_model,
                                 custom_objects={"<lambda>": lambda x, y: K.sqrt(K.mean(K.square(x - y)))},
                                 fit_kwargs={"batch_size": 8, "epochs": 1},
                                 compile_kwargs={"loss": lambda x, y: K.sqrt(K.mean(K.square(x - y))),
                                                 "optimizer": "Adam",
                                                 "metrics": ["mse"]}) \
        (ClockShift_Lag1=shift_power_statistics,
         ClockShift_Lag2=shift_power_statistics2,
         target=scale_power_statistics)

    inverse_power_scale_dl = power_scaler(x=keras_wrapper,
                                          computation_mode=ComputationMode.Transform,
                                          use_inverse_transform=True,
                                          callbacks=[LinePlotCallback("prediction")])

    rmse_dl = RMSE()(keras_model=inverse_power_scale_dl, y=pipeline["load_power_statistics"])

    # Now, the pipeline is complete
    # so we can load data and train the model
    data = pd.read_csv("../data/getting_started_data.csv",
                       index_col="time",
                       parse_dates=["time"],
                       infer_datetime_format=True,
                       sep=",")

    pipeline.train(data)
    pipeline.to_folder("../results/pipe_keras")

    pipeline = Pipeline.from_folder("../results/pipe_keras")
    pipeline.train(data)
