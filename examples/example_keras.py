# -----------------------------------------------------------
# This example presents the code used in the advanced example
# guide in the pyWATTS documentation.
# -----------------------------------------------------------

import pandas as pd

from sklearn.preprocessing import StandardScaler

from keras import layers, Model

from pywatts.callbacks import LinePlotCallback

# From pyWATTS the pipeline is imported
from pywatts_pipeline.core.util.computation_mode import ComputationMode
from pywatts_pipeline.core.pipeline import Pipeline

# Import the pyWATTS pipeline and the required modules
from pywatts.modules import LinearInterpolater, SKLearnWrapper, KerasWrapper
from pywatts.modules.preprocessing.select import Select
from pywatts.summaries import RMSE
from tensorflow.keras import backend as K

def get_keras_model():
    # write the model with the Functional API, Sequential does not support multiple input tensors

    input_1 = layers.Input(shape=(24,),
                           name='lag_features')  # layer name must match time series name
    hidden = layers.Dense(10,
                          activation='tanh',
                          name='hidden')(input_1)
    output = layers.Dense(24,
                          activation='linear',
                          name='target')(hidden)  # layer name must match time series name
    model = Model(inputs=[input_1], outputs=output)
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
    lag_features = Select(start=-23, stop=1, step=1, name="lag_features")(x=scale_power_statistics)
    target = Select(start=1, stop=25, step=1, name="target")(x=scale_power_statistics)


    keras_wrapper = KerasWrapper(keras_model,
                                 custom_objects={"<lambda>": lambda x, y: K.sqrt(K.mean(K.square(x - y)))},
                                 fit_kwargs={"batch_size": 8, "epochs": 1},
                                 compile_kwargs={"loss": lambda x, y: K.sqrt(K.mean(K.square(x - y))),
                                                 "optimizer": "Adam",
                                                 "metrics": ["mse"]}) \
        (lag_features=lag_features,
         target=target)

    inverse_power_scale_dl = power_scaler(x=keras_wrapper,
                                          computation_mode=ComputationMode.Transform,
                                          method="inverse_transform",
                                          callbacks=[LinePlotCallback("prediction")])

    rmse_dl = RMSE()(keras_model=inverse_power_scale_dl, y=target)

    # Now, the pipeline is complete
    # so we can load data and train the model
    data = pd.read_csv("../data/getting_started_data.csv",
                       index_col="time",
                       parse_dates=["time"],
                       infer_datetime_format=True,
                       sep=",")

    pipeline.train(data[:6000])
    pipeline.test(data[6000:])
    pipeline.to_folder("../results/pipe_keras")

    pipeline = Pipeline.from_folder("../results/pipe_keras")
    pipeline.test(data[6000:])
