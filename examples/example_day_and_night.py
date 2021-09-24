# -----------------------------------------------------------
# This example presents the code used in the advanced example
# guide in the pyWATTS documentation.
# -----------------------------------------------------------

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR



# Import the pyWATTS pipeline and the required modules
from pywatts.core.computation_mode import ComputationMode
from pywatts.core.pipeline import Pipeline
from pywatts.callbacks import CSVCallback, LinePlotCallback
from pywatts.modules import ClockShift, LinearInterpolater, RollingRMSE, SKLearnWrapper


# The condition function. It returns True during daytime.
# For simplification we say daytime is between 8am and 8pm.
def is_daytime(x, _):
    return 8 < x["ClockShift"].indexes["time"][0].hour < 20


# This function creates and returns the preprocessing pipeline
def create_preprocessing_pipeline(power_scaler):
    pipeline = Pipeline(path="../results/preprocessing")

    # Deal with missing values through linear interpolation
    imputer_power_statistics = LinearInterpolater(method="nearest", dim="time",
                                                  name="imputer_power")(x=pipeline["scaler_power"])
    # Scale the data using a standard SKLearn scaler
    scale_power_statistics = power_scaler(x=imputer_power_statistics)

    # Create lagged time series to later be used in the regression
    ClockShift(lag=1)(x=scale_power_statistics)
    ClockShift(lag=2)(x=scale_power_statistics)
    return pipeline


# This function creates the pipeline which we use for testing.
# The test pipeline works on batches with one hour
def create_test_pipeline(modules):
    regressor_svr, regressor_lin_reg = modules

    # Create test pipeline which works on a batch size of one hour.
    pipeline = Pipeline("../results/test_pipeline", batch=pd.Timedelta("1h"))

    # Add the svr regressor to the pipeline. This regressor should be called if it is not daytime
    regressor_svr_power_statistics = regressor_svr(ClockShift=pipeline["ClockShift"],
                                                   ClockShift_1=pipeline["ClockShift_1"],
                                                   condition=lambda x, y: not is_daytime(x, y),
                                                   computation_mode=ComputationMode.Transform,
                                                   callbacks=[LinePlotCallback('SVR')])

    # Add the linear regressor to the pipeline. This regressor should be called if it is daytime
    regressor_lin_reg_power_statistics = regressor_lin_reg(ClockShift=pipeline["ClockShift"],
                                                           ClockShift_1=pipeline["ClockShift_1"],
                                                           condition=lambda x, y: is_daytime(x, y),
                                                           computation_mode=ComputationMode.Transform,
                                                           callbacks=[LinePlotCallback('LinearRegression')])

    # TODO what kind of RMSE has to be used here?
    #   * Rolling would not work, since the complete RMSE should be calculated for each Time Point
    #   * Summary do not work, since summaries are only executed once
    #   Is the current solution useful?
    #   Possible Solution: window_size=-1 means that the window is from the start until the current point in time.
    #                      In that case, the online learning has to be built in that way, that module only calculate
    #                      data for the desired/requested time steps.

    # Calculate the root mean squared error (RMSE) between the linear regression and the true values, save it as csv file
    RollingRMSE(window_size=1, window_size_unit="d")(
        y_hat=(regressor_svr_power_statistics, regressor_lin_reg_power_statistics), y=pipeline["load_power_statistics"],
        callbacks=[LinePlotCallback('RMSE'), CSVCallback('RMSE')])

    return pipeline


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
    train_pipeline = Pipeline(path="../results/train")

    # Create preprocessing pipeline for the preprocessing steps
    preprocessing_pipeline = create_preprocessing_pipeline(power_scaler)
    preprocessing_pipeline = preprocessing_pipeline(scaler_power=train_pipeline["load_power_statistics"])

    # Addd the regressors to the train pipeline
    regressor_lin_reg(ClockShift=preprocessing_pipeline["ClockShift"],
                      ClockShift_1=preprocessing_pipeline["ClockShift_1"],
                      target=train_pipeline["load_power_statistics"],
                      callbacks=[LinePlotCallback('LinearRegression')])
    regressor_svr(ClockShift=preprocessing_pipeline["ClockShift"],
                  ClockShift_1=preprocessing_pipeline["ClockShift_1"],
                  target=train_pipeline["load_power_statistics"],
                  callbacks=[LinePlotCallback('SVR')])

    print("Start training")
    train_pipeline.train(data)
    print("Training finished")

    # Create a second pipeline. Necessary, since this pipeline has additional steps in contrast to the train pipeline.
    pipeline = Pipeline(path="../results")

    # Get preprocessing pipeline
    preprocessing_pipeline = create_preprocessing_pipeline(power_scaler)
    preprocessing_pipeline = preprocessing_pipeline(scaler_power=pipeline["load_power_statistics"])

    # Get the test pipeline, the arguments are the modules, from the training pipeline, which should be reused
    test_pipeline = create_test_pipeline([regressor_lin_reg, regressor_svr])

    test_pipeline(ClockShift=preprocessing_pipeline["ClockShift"],
                  ClockShift_1=preprocessing_pipeline["ClockShift_1"],
                  load_power_statistics=pipeline["load_power_statistics"],
                  callbacks=[LinePlotCallback('Pipeline'), CSVCallback('Pipeline')])

    # Now, the pipeline is complete so we can run it and explore the results
    # Start the pipeline
    print("Start testing")
    result = pipeline.test(test)

    pipeline.to_folder("stored_day_and_night")
    pipeline = Pipeline.from_folder("stored_day_and_night")
    print("Testing finished")
    result2 = pipeline.test(test)

    print("FINISHED")
