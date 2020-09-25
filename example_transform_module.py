# Other modules required for the pipeline are imported
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd

# From pyWATTS the pipeline is imported
from pywatts.core.pipeline import Pipeline
# All modules required for the pipeline are imported
from pywatts.modules.whitelister import WhiteLister
from pywatts.wrapper.function_module import FunctionModule


def custom_multiplication(x: xr.Dataset):
    # Multiply the given dataset with 100.
    return x * 42


# The main function is where the pipeline is created and run
if __name__ == "__main__":
    # Create a pipeline
    pipeline = Pipeline(path="results")

    # Select individual time-series (columns) and generate plots in the results folder
    white_lister_power_statistics = WhiteLister(target="load_power_statistics", name="filter_power")(pipeline,
                                                                                                     plot=True)
    # Add a custom function to the FunctionModule and add the module to the pipeline
    function_module = FunctionModule(custom_multiplication, name="Multiplication")([white_lister_power_statistics],
                                                                                   plot=True)

    # Now, the pipeline is complete so we can run it and explore the results
    # Start the pipeline
    df = pd.read_csv("data/getting_started_data.csv", parse_dates=["time"], infer_datetime_format=True,
                     index_col="time")

    pipeline.train(df)

    # Generate a plot of the pipeline showing the flow of data through different modules
    figure = pipeline.draw()
    plt.show()
