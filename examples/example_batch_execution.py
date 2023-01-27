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
from pywatts_pipeline.core.util.computation_mode import ComputationMode
from pywatts_pipeline.core.pipeline import Pipeline
from pywatts.modules import LinearInterpolater, RollingRMSE, SKLearnWrapper, FunctionModule, Select


# This function creates and returns the preprocessing pipeline
from pywatts_pipeline.utils._xarray_time_series_utils import numpy_to_xarray

from pywatts.summaries import MASE


def create_preprocessing_pipeline(power_scaler):
    pipeline = Pipeline(path="../results/preprocessing")

    # Deal with missing values through linear interpolation
    imputer_power_statistics = LinearInterpolater(method="nearest", dim="time",
                                                  name="imputed_power")(x=pipeline["scaler_power"])
    # Scale the data using a standard SKLearn scaler
    scale_power_statistics = power_scaler(x=imputer_power_statistics)

    # Create lagged time series to later be used in the regression
    Select(start=-24, stop=0, step=1, name="sampled_data")(x=scale_power_statistics)
    return pipeline


# This function creates the pipeline which we use for testing.
# The test pipeline works on batches with one hour


if __name__ == "__main__":
    # Read the data via pandas.
    data = pd.read_csv("../data/getting_started_data.csv", parse_dates=["time"], infer_datetime_format=True,
                       index_col="time")

    # Split the data into train and test data.
    train = data[:6000]
    test = data[8000:]

    # Create all modules which are used multiple times.
    regressor_svr = SKLearnWrapper(module=SVR(), name="regression")
    power_scaler = SKLearnWrapper(module=StandardScaler(), name="scaled_power")

    # Build a train pipeline. In this pipeline, each step processes all data at once.
    pipeline = Pipeline(path="../results/batch_pipeline")

    # Create preprocessing pipeline for the preprocessing steps
    preprocessing_pipeline = create_preprocessing_pipeline(power_scaler)
    preprocessing_pipeline = preprocessing_pipeline(scaler_power=pipeline["load_power_statistics"])

    target = FunctionModule(lambda x: numpy_to_xarray(
        x.values.reshape((-1,)), x
    ), name="target")(x=pipeline["load_power_statistics"])


    periodic_condition = PeriodicCondition(21)
    detection_condition = RiverDriftDetectionCondition()
    check_if_midnight = lambda x, _: len(x["sampled_data"].indexes["time"]) > 0 and \
                                     x["sampled_data"].indexes["time"][0].hour == 0
    # Add the svr regressor to the pipeline. This regressor should be called if it is not daytime
    # TODO improve error message if select is not named sampled_data
    regressor_svr_power_statistics = regressor_svr(historical_input=preprocessing_pipeline["sampled_data"],
                                                   target=target,
                                                   computation_mode=ComputationMode.Refit,
                                                   callbacks=[LinePlotCallback('SVR')],
                                                   lag=pd.Timedelta(hours=24),
                                                   refit_conditions=[periodic_condition, #check_if_midnight,
                                                                     detection_condition])


    print("Start training")
    pipeline.train(train)
    print("Training finished")

    detection_condition(y_hat=regressor_svr_power_statistics, y=target)

    RollingRMSE(window_size=1, window_size_unit="d")(
        y_hat=regressor_svr_power_statistics, y=target,
        callbacks=[LinePlotCallback('RMSE'), CSVCallback('RMSE')])
    MASE()(y_hat=regressor_svr_power_statistics, y=target)

    # Now, the pipeline is complete so we can run it and explore the results
    # Start the pipeline
    print("Start testing")
    result = []
    for i in range(len(test)):
        print(test.index[i])
        result.append(pipeline.test(test.iloc[[i]], reset=False, summary=False, refit=True))

    print("Testing finished")
    summary = pipeline.create_summary()
    assert pipeline.steps["RollingRMSE_5"].buffer["RollingRMSE"].shape, (len(test) - 24, 1)

