# -----------------------------------------------------------
# This example presents the code used in the getting started
# guide in the pyWATTS documentation.
# -----------------------------------------------------------

import matplotlib.pyplot as plt
# Other modules required for the pipeline are imported
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# From pyWATTS the pipeline is imported
from pywatts.core.computation_mode import ComputationMode
from pywatts.core.pipeline import Pipeline
# All modules required for the pipeline are imported
from pywatts.modules.calendar_extraction import CalendarExtraction
from pywatts.modules.clock_shift import ClockShift
from pywatts.modules.linear_interpolation import LinearInterpolater
from pywatts.modules.root_mean_squared_error import RmseCalculator
from pywatts.wrapper.sklearn_wrapper import SKLearnWrapper

# The main function is where the pipeline is created and run
if __name__ == "__main__":
    # Create a pipeline
    pipeline = Pipeline(path="results")

    # Extract dummy calender features, using holidays from Germany
    # NOTE: Will it also work without x=pipeline["time"]? So, pipeline["time"]?
    # NOTE: CalendarExtraction can't return multiple features.
    # TODO how to handle this stuff with time? Since the load is not interessting only the name is of interest for this module..
    calendar_month = CalendarExtraction(
        encoding="numerical", continent="Europe", country="Germany"
    )(x=pipeline["load_power_statistics"])
    calendar_weekday = CalendarExtraction(
        encoding="numerical", continent="Europe", country="Germany"
    )(x=pipeline["load_transparency"])
    calendar_weekend = CalendarExtraction(
        encoding="numerical", continent="Europe", country="Germany"
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

    # Create a linear regression that uses the lagged values to predict the current value
    # NOTE: SKLearnWrapper has to collect all **kwargs itself and fit it against target.
    #       It is also possible to implement a join/collect class
    regressor_power_statistics = SKLearnWrapper(
        module=LinearRegression(fit_intercept=True)
    )(
        power_lag1=shift_power_statistics,
        power_lag2=shift_power_statistics2,
        cal_month=calendar_month,
        cal_weekday=calendar_weekday,
        call_weekend=calendar_weekend,
        target=scale_power_statistics,
        plot=True,
    )

    # Rescale the predictions to be on the original time scale
    inverse_power_scale = power_scaler(
        x=regressor_power_statistics, computation_mode=ComputationMode.Transform,
        use_inverse_transform=True, plot=True
    )

    # Calculate the root mean squared error (RMSE) between the linear regression and the true values, save it as csv file
    rmse = RmseCalculator()(y_hat=inverse_power_scale, y=pipeline["load_power_statistics"], to_csv=True)

    # Now, the pipeline is complete so we can run it and explore the results
    # Start the pipeline
    data = pd.read_csv("data/getting_started_data.csv",
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

    # Load the pipeline as a new instance
    pipeline2 = Pipeline("pipeline2_results")
    pipeline2.from_folder("./pipe_getting_started")
#       WARNING
#       Sometimes from_folder use unpickle for loading modules. Note that this is not safe.
#       Consequently, load only pipelines you trust with from_folder.
#       For more details about pickling see https://docs.python.org/3/library/pickle.html
    pipeline2.test(test)


