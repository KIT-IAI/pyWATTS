# -----------------------------------------------------------
# This example presents the code used in the getting started
# guide in the pyWATTS documentation.
# -----------------------------------------------------------

# Other modules required for the pipeline are imported
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

# From pyWATTS the pipeline is imported
from pywatts.callbacks import LinePlotCallback
from pywatts.core.computation_mode import ComputationMode
from pywatts.core.pipeline import Pipeline
# All modules required for the pipeline are imported
from pywatts.modules import CalendarExtraction, CalendarFeature, ClockShift, LinearInterpolater, SKLearnWrapper, Sampler
from pywatts.summaries import RMSE

# The main function is where the pipeline is created and run
if __name__ == "__main__":
    # Create a pipeline
    pipeline = Pipeline(path="../results")

    # Extract dummy calendar features, using holidays from Germany
    # NOTE: CalendarExtraction can't return multiple features.
    calendar = CalendarExtraction(continent="Europe", country="Germany", features=[CalendarFeature.month,
                                                                                   CalendarFeature.weekday,
                                                                                   CalendarFeature.weekend]
                                  )(x=pipeline["load_power_statistics"])

    # Deal with missing values through linear interpolation
    imputer_power_statistics = LinearInterpolater(
        method="nearest", dim="time", name="imputer_power"
    )(x=pipeline["load_power_statistics"])

    # Scale the data using a standard SKLearn scaler
    power_scaler = SKLearnWrapper(module=StandardScaler(), name="scaler_power")
    scale_power_statistics = power_scaler(x=imputer_power_statistics)

    # Create lagged time series to later be used in the regression
    shift_power_statistics = ClockShift(lag=1, name="ClockShift_Lag1"
                                        )(x=scale_power_statistics)
    shift_power_statistics2 = ClockShift(lag=2, name="ClockShift_Lag2"
                                         )(x=scale_power_statistics)

    target_multiple_output = Sampler(24, name="sampled_data")(x=scale_power_statistics)

    # Select features based on F-statistic
    selected_features = SKLearnWrapper(
        module=SelectKBest(score_func=f_regression, k=2)
    )(
        power_lag1=shift_power_statistics,
        power_lag2=shift_power_statistics2,
        calendar=calendar,
        target=scale_power_statistics,
    )

    # Create a linear regression that uses the lagged values to predict the current value
    # NOTE: SKLearnWrapper has to collect all **kwargs itself and fit it against target.
    #       It is also possible to implement a join/collect class
    regressor_power_statistics = SKLearnWrapper(
        module=LinearRegression(fit_intercept=True)
    )(
        features=selected_features,
        target=target_multiple_output,
        callbacks=[LinePlotCallback("linear_regression")],
    )

    # Rescale the predictions to be on the original time scale
    inverse_power_scale = power_scaler(
        x=regressor_power_statistics, computation_mode=ComputationMode.Transform,
        use_inverse_transform=True, callbacks=[LinePlotCallback("rescale")]
    )

    # Calculate the root mean squared error (RMSE) between the linear regression and the true values
    # save it as csv file
    rmse = RMSE()(y_hat=inverse_power_scale, y=target_multiple_output)

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
    pipeline.to_folder("./pipe_getting_started")

    print("Execute second pipeline")
    # Load the pipeline as a new instance
    pipeline2 = Pipeline.from_folder("./pipe_getting_started", file_manager_path="../pipeline2_results")
    #       WARNING
    #       Sometimes from_folder use unpickle for loading modules. Note that this is not safe.
    #       Consequently, load only pipelines you trust with from_folder.
    #       For more details about pickling see https://docs.python.org/3/library/pickle.html
    result = pipeline2.test(test)
    print("Finished")
