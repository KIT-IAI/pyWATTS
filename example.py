# -----------------------------------------------------------
# This example presents the code used in the getting started
# guide in the pyWATTS documentation.
# -----------------------------------------------------------

# From pyWATTS the pipeline is imported
from pywatts.core.computation_mode import ComputationMode
from pywatts.core.pipeline import Pipeline

# All modules required for the pipeline are imported
from pywatts.modules.calendar_extraction import CalendarExtraction
from pywatts.modules.whitelister import WhiteLister
from pywatts.wrapper.sklearn_wrapper import SKLearnWrapper
from pywatts.modules.clock_shift import ClockShift
from pywatts.modules.linear_interpolation import LinearInterpolater
from pywatts.modules.root_mean_squared_error import RmseCalculator

# Other modules required for the pipeline are imported
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# The main function is where the pipeline is created and run
if __name__ == "__main__":
    # Create a pipeline
    pipeline = Pipeline(path="results")

    # Extract dummy calender features, using holidays from Germany
    calendar_features = CalendarExtraction(encoding="numerical", continent="Europe", country="Germany")(pipeline)

    # Select individual time-series (columns) and generate plots in the results folder
    white_lister_power_statistics = WhiteLister(target="load_power_statistics", name="filter_power")(pipeline,
                                                                                                     plot=True)
    white_lister_transparency = WhiteLister(target="load_transparency", name="filter_transparency")(pipeline, plot=True)
    white_lister_price = WhiteLister(target="price_day_ahead", name="filter_price")(pipeline, plot=True)
    calendar_month = WhiteLister(target="month", name="filter_month")([calendar_features])
    calendar_weekday = WhiteLister(target="weekday", name="filter_weekday")([calendar_features])
    calendar_weekend = WhiteLister(target="weekend", name="filter_weekend")([calendar_features])

    # Deal with missing values through linear interpolation
    imputer_power_statistics = LinearInterpolater(method="nearest", dim="time",
                                                  name="imputer_power")([white_lister_power_statistics])

    # Scale the data using a standard SKLearn scaler
    power_scaler = SKLearnWrapper(module=StandardScaler(), name="scaler_power")
    scale_power_statistics = power_scaler([imputer_power_statistics])

    # Create lagged time series to later be used in the regression
    shift_power_statistics = ClockShift(lag=1, name="ClockShift_Lag1")([scale_power_statistics])
    shift_power_statistics2 = ClockShift(lag=2, name="ClockShift_Lag2")([scale_power_statistics])

    # Create a linear regression that uses the lagged values to predict the current value
    regressor_power_statistics = SKLearnWrapper(module=LinearRegression(fit_intercept=True))([shift_power_statistics,
                                                                                              shift_power_statistics2,
                                                                                              calendar_month,
                                                                                              calendar_weekday,
                                                                                              calendar_weekend],
                                                                                             targets=[
                                                                                                 scale_power_statistics]
                                                                                             )

    # Rescale the predictions to be on the original time scale
    inverse_power_scale = power_scaler([regressor_power_statistics],
                                       computation_mode=ComputationMode.Transform, use_inverse_transform=True,
                                       plot=True)

    # Calculate the root mean squared error (RMSE) between the linear regression and the true values, save it as csv file
    rmse = RmseCalculator(target="load_power_statistics", predictions=["scaler_power"])(
        [inverse_power_scale, white_lister_power_statistics], to_csv=True)

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

    # Generate a plot of the pipeline showing the flow of data through different modules
    figure = pipeline.draw()
    plt.show()

    # Save the pipeline to a folder
    pipeline.to_folder("./pipe_getting_started")

    # Load the pipeline as a new instance
    pipeline2 = Pipeline("pipeline2_results")
    pipeline2.from_folder("./pipe_getting_started")
    #       WARNING
    #       Sometimes from_folder use unpickle for loading modules. Note that this is not safe.
    #       Consequently, load only pipelines you trust with from_folder.
    #       For more details about pickling see https://docs.python.org/3/library/pickle.html

    # Set a new folder for the second pipeline
    pipeline2.to_folder("./pipe_gs_copy")

    # Generate a plot of the new pipeline
    pipeline2.draw()
    plt.show()