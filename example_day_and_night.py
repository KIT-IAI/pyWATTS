# -----------------------------------------------------------
# This example presents the code used in the advanced example
# guide in the pyWATTS documentation.
# -----------------------------------------------------------

import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Import the pyWATTS pipeline and the required modules
from pywatts.core.pipeline import Pipeline
from pywatts.core.computation_mode import ComputationMode
from pywatts.modules.clock_shift import ClockShift
from pywatts.modules.linear_interpolation import LinearInterpolater
from pywatts.modules.root_mean_squared_error import RmseCalculator
from pywatts.modules.whitelister import WhiteLister
from pywatts.wrapper.sklearn_wrapper import SKLearnWrapper


# The condition function. It returns True during daytime.
# For simplification we say daytime is between 8am and 8pm.
def is_daytime(x, _):
    return 8 < x.indexes["time"][0].hour < 20


# This function creates and returns the preprocessing pipeline
def create_preprocessing_pipeline(power_scaler):
    pipeline = Pipeline("preprocessing")

    # Deal with missing values through linear interpolation
    imputer_power_statistics = LinearInterpolater(method="nearest", dim="time",
                                                  name="imputer_power")(pipeline)
    # Scale the data using a standard SKLearn scaler
    scale_power_statistics = power_scaler([imputer_power_statistics])

    # Create lagged time series to later be used in the regression
    ClockShift(lag=1)([scale_power_statistics])
    ClockShift(lag=2)([scale_power_statistics])
    return pipeline


# This function creates the pipeline which we use for testing.
# The test pipeline works on batches with one hour
def create_test_pipeline(modules, whitelister):
    regressor_svr, regressor_lin_reg = modules

    # Create test pipeline which works on a batch size of one hour.
    pipeline = Pipeline("test_pipeline", batch=pd.Timedelta("1h"))
    target = whitelister(pipeline)

    clock_shift1 = WhiteLister(target="ClockShift")(pipeline)
    clock_shift2 = WhiteLister(target="ClockShift_0")(pipeline)

    # Add the svr regressor to the pipeline. This regressor should be called if it is not daytime
    regressor_svr_power_statistics = regressor_svr([clock_shift1, clock_shift2],
                                                   condition=lambda x, y: not is_daytime(x, y),
                                                   computation_mode=ComputationMode.Transform,
                                                   plot=True)

    # Add the linear regressor to the pipeline. This regressor should be called if it is daytime
    regressor_lin_reg_power_statistics = regressor_lin_reg([clock_shift1, clock_shift2],
                                                           condition=lambda x, y: is_daytime(x, y),
                                                           computation_mode=ComputationMode.Transform,
                                                           plot=True)

    # Calculate the root mean squared error (RMSE) between the linear regression and the true values, save it as csv file
    RmseCalculator(target="load_power_statistics", predictions=["Regression"])(
        [(regressor_svr_power_statistics, regressor_lin_reg_power_statistics), target], plot=True, to_csv=True)

    return pipeline


if __name__ == "__main__":
    # Read the data via pandas.
    data = pd.read_csv("data/getting_started_data.csv", parse_dates=["time"], infer_datetime_format=True,
                       index_col="time")

    # Split the data into train and test data.
    train = data[:6000]
    test = data[6000:]

    # Create all modules which are used multiple times.
    regressor_lin_reg = SKLearnWrapper(module=LinearRegression(fit_intercept=True), name="Regression")
    regressor_svr = SKLearnWrapper(module=SVR(), name="Regression")
    power_scaler = SKLearnWrapper(module=StandardScaler(), name="scaler_power")
    white_lister = WhiteLister(target="load_power_statistics",
                               name="filter_power")

    # Build a train pipeline. In this pipeline, each step processes all data at once.
    train_pipeline = Pipeline(path="train")

    white_lister_power_statistics = white_lister(train_pipeline, plot=True)

    # Create preprocessing pipeline for the preprocessing steps
    preprocessing_pipeline = create_preprocessing_pipeline(power_scaler)
    preprocessing_pipeline = preprocessing_pipeline(white_lister_power_statistics)

    # Addd the regressors to the train pipeline
    regressor_lin_reg(preprocessing_pipeline,
                      targets=[white_lister_power_statistics],
                      plot=True)
    regressor_svr(preprocessing_pipeline,
                  targets=[white_lister_power_statistics],
                  plot=True)

    print("Start training")
    train_pipeline.train(data)
    print("Training finished")


    # Create a second pipeline. Necessary, since this pipeline has additional steps in contrast to the train pipeline.
    pipeline = Pipeline(path="results")

    # Select individual time-series (columns) and generate plots in the results folder
    white_lister_power_statistics = white_lister(pipeline, plot=True)

    # Get preprocessing pipeline
    preprocessing_pipeline = create_preprocessing_pipeline(power_scaler)
    preprocessing_pipeline = preprocessing_pipeline(white_lister_power_statistics)

    # Get the test pipeline, the arguments are the modules, from the training pipeline, which should be reused
    test_pipeline = create_test_pipeline([regressor_lin_reg, regressor_svr], white_lister)
    test_pipeline([preprocessing_pipeline, white_lister_power_statistics])

    # Now, the pipeline is complete so we can run it and explore the results
    # Start the pipeline
    print("Start testing")
    pipeline.test(test)
    print("Testing finished")

    print("FINISHED")