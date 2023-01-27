# -----------------------------------------------------------
# This example presents the code used in the advanced example
# guide in the pyWATTS documentation.
# -----------------------------------------------------------

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Import the pyWATTS pipeline and the required modules
from pywatts_pipeline.core.util.computation_mode import ComputationMode
from pywatts_pipeline.core.pipeline import Pipeline
from pywatts.callbacks import CSVCallback, LinePlotCallback
from pywatts.modules import RollingRMSE, SKLearnWrapper, Select


# The condition function. It returns True during daytime.
# For simplification we say daytime is between 8am and 8pm.
def is_daytime(x, _):
    return 8 < x["lag_features"].indexes["time"][0].hour < 20


if __name__ == "__main__":
    # Read the data via pandas.
    data = pd.read_csv("../data/getting_started_data.csv", parse_dates=["time"], infer_datetime_format=True,
                       index_col="time")

    # Split the data into train and test data.
    train = data[:6000]
    test = data[8700:]

    # Create all modules which are used multiple times.
    regressor_lin_reg = SKLearnWrapper(module=LinearRegression(fit_intercept=True), name="Regression")
    regressor_svr = SKLearnWrapper(module=SVR(), name="Regression")
    power_scaler = SKLearnWrapper(module=StandardScaler(), name="scaler_power")

    # Build a train pipeline. In this pipeline, each step processes all data at once.
    pipeline = Pipeline(path="../results/day_night")

    # Create preprocessing pipeline for the preprocessing steps
    scale_power_statistics = power_scaler(x=pipeline["load_power_statistics"],
                                          callbacks=[LinePlotCallback("scaled")])

    # Create lagged time series to later be used in the regression
    lag_features = Select(start=-2, stop=0, step=1, name="lag_features")(x=scale_power_statistics)

    # Addd the regressors to the train pipeline
    lr_reg = regressor_lin_reg(lag_features=lag_features,
                               target=scale_power_statistics,
                               condition=lambda x, y: is_daytime(x, y),
                               callbacks=[LinePlotCallback('LinearRegression')])
    svr_reg = regressor_svr(lag_features=lag_features,
                            target=scale_power_statistics,
                            condition=lambda x, y: not is_daytime(x, y),
                            callbacks=[LinePlotCallback('SVR')])

    RollingRMSE(window_size=1, window_size_unit="d")(
        y_hat=(svr_reg, lr_reg), y=pipeline["load_power_statistics"],
        callbacks=[LinePlotCallback('RMSE'), CSVCallback('RMSE')])

    print("Start training")
    pipeline.train(data)
    print("Training finished")

    print("Start testing")
    result = []
    pipeline._reset()
    for i in range(len(test)):
        result.append(pipeline.test(test.iloc[[i]], reset=False, summary=False))
    print("Testing finished")
    summary = pipeline.create_summary()
    pipeline.to_folder("stored_day_and_night")

    pipeline = Pipeline.from_folder("stored_day_and_night")
    print("Testing finished")
    result2 = []
    for i in range(len(test)):
        result2.append(pipeline.test(test.iloc[[i]], reset=False, summary=False))
    print("Testing finished")
    summary = pipeline.create_summary()

    # TODO add some assertions

    print("FINISHED")
