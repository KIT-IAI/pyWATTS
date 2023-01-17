# -----------------------------------------------------------
# This example presents the code used in the getting started
# guide in the pyWATTS documentation.
# -----------------------------------------------------------

# Other modules required for the pipeline are imported
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

from pywatts.callbacks import CSVCallback, LinePlotCallback
from pywatts_pipeline.core.util.computation_mode import ComputationMode
from pywatts_pipeline.core.pipeline import Pipeline

# All modules required for the pipeline are imported
from pywatts.modules import CalendarExtraction, CalendarFeature, ClockShift, LinearInterpolater, \
    SKLearnWrapper, SmTimeSeriesModelWrapper
from pywatts.summaries import RMSE

if __name__ == "__main__":
    # Create a pipeline
    pipeline = Pipeline(path="../results/statsmodel")

    # Extract dummy calender features, using holidays from Germany
    cal_features = CalendarExtraction(features=[CalendarFeature.hour, CalendarFeature.weekday, CalendarFeature.month],
                                      continent="Europe", country="Germany"
                                      )(x=pipeline["load_power_statistics"])

    # Deal with missing values through linear interpolation
    imputer_power_statistics = LinearInterpolater(
        method="nearest", dim="time", name="imputer_power"
    )(x=pipeline["load_power_statistics"])

    # Scale the data using a standard SKLearn scaler
    power_scaler = SKLearnWrapper(module=StandardScaler(), name="scaler_power")
    scale_power_statistics = power_scaler(x=imputer_power_statistics)

    # Create lagged time series to later be used in the regression
    lag_features = Select(start=-2, stop=0, step=1, name="lag_features"
                                        )(x=scale_power_statistics)

    # Create a statsmodel that uses the lagged values to predict the current value
    regressor_power_statistics = SmTimeSeriesModelWrapper(
        module=ARIMA,
        module_kwargs={
            "order": (2, 0, 0)
        }
    )(
        lag_features=lag_features,
        calendar=cal_features,
        target=scale_power_statistics, callbacks=[LinePlotCallback('ARIMA')],
    )

    # Rescale the predictions to be on the original time scale
    inverse_power_scale = power_scaler(
        x=regressor_power_statistics, computation_mode=ComputationMode.Transform,
        method="inverse_transform", callbacks=[LinePlotCallback('rescale')]
    )

    # Calculate the root mean squared error (RMSE) between the linear regression and the true values, save it as csv file
    rmse = RMSE()(y_hat=inverse_power_scale, y=pipeline["load_power_statistics"])

    # Now, the pipeline is complete so we can run it and explore the results
    # Start the pipeline
    data = pd.read_csv("../data/getting_started_data.csv",
                       index_col="time",
                       parse_dates=["time"],
                       infer_datetime_format=True,
                       sep=",")
    train = data.iloc[:6000, :]
    pipeline.train(data=train)

    test = data.iloc[6000:, :]
    data = pipeline.test(data=test)

    # Save the pipeline to a folder
    pipeline.to_folder("./pipe_statsmodel")

    print("Execute second pipeline")
    # Load the pipeline as a new instance
    pipeline2 = Pipeline.from_folder("./pipe_statsmodel", file_manager_path="../pipeline2_results/statsmodel")
    #       WARNING
    #       Sometimes from_folder use unpickle for loading modules. Note that this is not safe.
    #       Consequently, load only pipelines you trust with from_folder.
    #       For more details about pickling see https://docs.python.org/3/library/pickle.html
    result = pipeline2.test(test)
    print("Finished")
