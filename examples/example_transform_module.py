# Other modules required for the pipeline are imported
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from pywatts.callbacks import CSVCallback, LinePlotCallback
# From pyWATTS the pipeline is imported
from pywatts.core.pipeline import Pipeline
# All modules required for the pipeline are imported
from pywatts.modules import FunctionModule


def custom_multiplication(x: xr.Dataset):
    # Multiply the given dataset with 100.
    return x * 1000


# The main function is where the pipeline is created and run
if __name__ == "__main__":
    # Create a pipeline
    pipeline = Pipeline(path="../results")

    # Add a custom function to the FunctionModule and add the module to the pipeline
    function_module = FunctionModule(custom_multiplication, name="Multiplication")(x=pipeline["load_power_statistics"],
                                                                                   callbacks=[CSVCallback("Mul"),
                                                                                              LinePlotCallback("Mul")])

    # Now, the pipeline is complete so we can run it and explore the results
    # Start the pipeline
    df = pd.read_csv("../data/getting_started_data.csv", parse_dates=["time"], infer_datetime_format=True,
                     index_col="time")

    pipeline.train(df)

    # Generate a plot of the pipeline showing the flow of data through different modules
    pipeline.draw()
    plt.show()
