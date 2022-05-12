# -----------------------------------------------------------
# This example presents the code for the batch/online execution
# of the pipeline.
# Additionally, the regressor in this pipeline is retrained
# if the datapoint corresponds to midnight.
# -----------------------------------------------------------

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Import the pyWATTS pipeline and the required modules
from pywatts.callbacks import CSVCallback, LinePlotCallback
from pywatts.conditions.cd_condition import RiverDriftDetectionCondition
from pywatts.conditions.periodic_condition import PeriodicCondition
from pywatts.core.computation_mode import ComputationMode
from pywatts.core.pipeline import Pipeline
from pywatts.modules import ClockShift, LinearInterpolater, RollingRMSE, SKLearnWrapper, Sampler, FunctionModule


# This function creates and returns the preprocessing pipeline
from pywatts.utils._xarray_time_series_utils import numpy_to_xarray


def create_preprocessing_pipeline(power_scaler):
    pipeline = Pipeline(path="../results/preprocessing")

    # Deal with missing values through linear interpolation
    imputer_power_statistics = LinearInterpolater(method="nearest", dim="time",
                                                  name="imputed_power")(x=pipeline["scaler_power"])
    # Scale the data using a standard SKLearn scaler
    scale_power_statistics = power_scaler(x=imputer_power_statistics)

    # Create lagged time series to later be used in the regression
    historical_data = ClockShift(lag=24)(x=scale_power_statistics)
    Sampler(24, name="sampled_data")(x=historical_data)
    return pipeline


# This function creates the pipeline which we use for testing.
# The test pipeline works on batches with one hour
def create_test_pipeline(regressor_svr):
    # Create test pipeline which works on a batch size of one hour.
    pipeline = Pipeline("../results/test_pipeline", batch=pd.Timedelta("1h"))
    periodic_condition = PeriodicCondition(21)
    detection_condition = RiverDriftDetectionCondition()
    check_if_midnight = lambda x, _: len(x["historical_input"].indexes["time"]) > 0 and \
                                     x["historical_input"].indexes["time"][0].hour == 0
    # Add the svr regressor to the pipeline. This regressor should be called if it is not daytime
    regressor_svr_power_statistics = regressor_svr(historical_input=pipeline["historical_input"],
                                                   target=pipeline["load_power_statistics"],
                                                   computation_mode=ComputationMode.Refit,
                                                   callbacks=[LinePlotCallback('SVR')],
                                                   lag=pd.Timedelta(hours=24),
                                                   refit_conditions=[periodic_condition, check_if_midnight,
                                                                     detection_condition])
    detection_condition(y_hat=regressor_svr_power_statistics, y=pipeline["load_power_statistics"])

    RollingRMSE(window_size=1, window_size_unit="d")(
        y_hat=regressor_svr_power_statistics, y=pipeline["load_power_statistics"],
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
    regressor_svr = SKLearnWrapper(module=SVR(), name="regression")
    power_scaler = SKLearnWrapper(module=StandardScaler(), name="scaled_power")

    # Build a train pipeline. In this pipeline, each step processes all data at once.
    train_pipeline = Pipeline(path="../results/train")

    # Create preprocessing pipeline for the preprocessing steps
    preprocessing_pipeline = create_preprocessing_pipeline(power_scaler)
    preprocessing_pipeline = preprocessing_pipeline(scaler_power=train_pipeline["load_power_statistics"])

    target = FunctionModule(lambda x: numpy_to_xarray(
        x.values.reshape((-1,)), x
    ), name="target")(x=train_pipeline["load_power_statistics"])

    # Addd the regressors to the train pipeline
    regressor_svr(hist_input=preprocessing_pipeline["sampled_data"],
                  target=target,
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
    test_pipeline = create_test_pipeline(regressor_svr)
    test_pipeline(historical_input=preprocessing_pipeline["sampled_data"],
                  load_power_statistics=pipeline["load_power_statistics"],
                  callbacks=[LinePlotCallback('Pipeline'), CSVCallback('Pipeline')])

    # Now, the pipeline is complete so we can run it and explore the results
    # Start the pipeline
    print("Start testing")
    result = pipeline.test(test, online_start=pd.to_datetime("2018-12-30"))
    print("Testing finished")
