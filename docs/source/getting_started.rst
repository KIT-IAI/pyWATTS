.. _gettingstarted:

Getting Started
===============

Note that pyWATTS is no longer actively maintained, however the main graph pipeline functionality of pyWATTS has been
integrated into the open source python package [sktime](https://www.sktime.net/en/stable/).**

Further Information
*******************
The purpose of this guide is to introduce some of the main features of `pyWATTS`.
It assumes a basic knowledge of data science and machine learning principles.

We will work through the steps of creating a pipeline, adding in modules,
running the pipeline and finding the results all based on the ``example.py``
pipeline. The data used in this guide is available through the
`Open Power System Data Portal <https://open-power-system-data.org/>`_.
We use load time series for Germany from various sources for the year 2018.

Initial Imports
***************

Before we start creating the pipeline, we need to import the pipeline module
from the `pyWATTS` core.

.. code-block:: python

   from pywatts.core.pipeline import Pipeline

We also need to import all the modules we plan on adding into our pipeline as well
as any external Scikit-Learn modules we will be using.

.. code-block:: python

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
    from pywatts.modules import CalendarExtraction, CalendarFeature, Select, LinearInterpolater, SKLearnWrapper,
    from pywatts.summaries import RMSE

With the modules imported, we can now work on building the pipeline.

Creating The Pipeline
*********************

We create the pipeline in the ``main`` function of ``example.py``. The very first step
is to create the pipeline and specify the path.

.. code-block:: python

   pipeline = Pipeline(path="results")

It is essential to specify the path since a time-stamped folder with all the outputs
from the pipeline will be generated and saved in this location every time we run the pipeline.

Now that the pipeline exists, we can add in modules.

**Dummy Calendrical Features**

Often we require dummy calendrical features, such as month, weekday, hour and whether or not the day is a weekend,
for forecasting problems. ``CalendarExtraction`` modules are able to extract these features.
Since this is the first module in our pipeline, we do not have to worry about defining
the proceeding module. However, we must specify the column of the dataset which should be used as input for that module.
Therefore, we use round brackets with the pipeline name inside and square brackets to to achieve this:
``(x=pipeline["load_power_statistics"])``.

.. code-block:: python

    calendar = CalendarExtraction(continent="Europe",
                                  country="Germany",
                                  features=[CalendarFeature.month, CalendarFeature.weekday, CalendarFeature.weekend]
                                 )(x=pipeline["load_power_statistics"])

When we define a ``CalendarExtraction`` module, we need to choose what encoding to use. In the case, we choose the
continent and the country that is used to calculate public holidays. This is particularly important for public holidays
that only exist in certain parts of the world (e.g. Thanksgiving). The extracted features are the numerical extracted
month, the weekday, and the weekend.


**Linear Interpolation**

The next model we include deals with missing values by filling them through linear interpolation.

.. code-block:: python

    imputer_power_statistics = LinearInterpolater(method="nearest",
                                                  dim="time",
                                                  name="imputer_power")(x=pipeline["load_power_statistics"])

The parameters here (method and dim) are related to the *scipy* ``interpolate`` method which is used
inside the module. As before, we need to correctly place the linear interpolator in the pipeline. This example
takes the column ''load_power_statistics'' from the input data. Consequently, we specify the input by
``(x=pipeline["load_power_statistics"])`` again.

**Scaling**

It is also possible to integrate SciKit-Learn modules directly into the pipeline. We achieve this by using
the ``SKLearnWrapper``:

.. code-block:: python

    power_scaler = SKLearnWrapper(module=StandardScaler(), name="scaler_power")
    scale_power_statistics = power_scaler(x=imputer_power_statistics)

Here we use the wrapper to import a SciKit-Learn ``StandardScaler`` in the pipeline. In the second line
we apply the ``StandardScaler`` on the imputed load time series, resulting in a normalised time series.

**Creating Lags**

Often in time-series analysis, we want to consider time-lags, i.e. shifting the time series back by
one or more values. In `pyWATTS`, we use the ``ClockShift`` module to perform this task.

.. code-block:: python

    lag_features = Select(start=-1, stop=1, step=1)(x=scale_power_statistics)

In the above example, we create a sampled time series with the two values for each time step
(past value and current value). The input for this module is the same scaled time series from above. When we modules
of the same type (here two ``ClockShift`` modules, it is highly advisable to name them. Without a user defined name
there will be a conflict in the pipeline. `pyWATTS` automatically changes the name to avoid this conflict and you
receive a warning message, but we advise avoiding this.

**Creating multiple targets**

For every hour, we want to predict the values for the next 24 hours.
We use the Select to create windows containing 24 values.

.. code-block:: python

    target_multiple_output = Select(start=1, stop=25, step=1 name="sampled_data")(x=scale_power_statistics)


**Selecting features**

We use the SciKit-learn wrapper around the module ``SelectKBest`` to automatically select useful features.

.. code-block:: python

    selected_features = SKLearnWrapper(
        module=SelectKBest(score_func=f_regression, k=2)
    )(
        power_lag1=shift_power_statistics,
        power_lag2=shift_power_statistics2,
        calendar=calendar,
        target=scale_power_statistics,
    )


**Linear Regression**

We also use the SciKit-learn wrapper for linear regression. The implementation is, however, slightly different.

.. code-block:: python

    regressor_power_statistics = SKLearnWrapper(
        module=LinearRegression(fit_intercept=True)
    )(
        features=selected_features,
        target=target_multiple_output,
        callbacks=[LinePlotCallback("linear_regression")]
    )

First we see that standard SciKit-learn parameters can be adjusted directly inside the SciKit-learn constructor.
Here, for example, we have set the ``fit_intercept`` parameter to true. Furthermore,
a linear regression can have more than one input and also requires a target for fitting. Therefore, we include
the inputs by keyword-arguments. Additional features could be added by using additional keywords.
Note that all keyword-arguments that start with *target* are considered as target
variables by pyWATTS.

**Rescaling**

Before we performed the linear regression, we normalised the time-series with a SciKit-learn module. To transform
the predictions from the linear regression back to the original scale, we need to call the scaler
a second time, and ensure we use the inverse transformation.

.. code-block:: python

   inverse_power_scale = power_scaler(x=regressor_power_statistics,
                                       computation_mode=ComputationMode.Transform, use_inverse_transform=True,
                                        callbacks=[LinePlotCallback('rescale')])


We also set ``computation_mode=ComputationMode.Transform`` for this inverse transformation to work. If
this is not set, then the scaler will automatically fit itself to the new scaled dataset, and the inverse transformation
will be useless. Moreover, we can use callbacks for visualizing or writing the results into files.

**Root Mean Squared Error**

To measure the accuracy of our regression model, we can calculate the root mean squared error (RMSE).

.. code-block:: python

    rmse = RMSE()(y_hat=inverse_power_scale, y=target_multiple_output)

The target variable is determined by the key-word ``y``. All other keyword arguments are considered as predictions.

Executing, Saving and Loading the Pipeline
******************************************

With the desired modules added to the pipeline, we can now train and test it.
We do this by calling the ``train`` method or ``test`` method. Both methods require some input data. Therefore,
we read some data with [pandas](https://pandas.pydata.org/) or [xarray](http://xarray.pydata.org/en/stable/index.html)
and split it into a train and a test set.

.. code-block:: python

    data = pd.read_csv("../data/getting_started_data.csv",
                index_col="time",
                parse_dates=["time"],
                infer_datetime_format=True,
                sep=",")
    train = data.iloc[:6000, :]
    pipeline.train(data=train)

    test = data.iloc[6000:, :]
    pipeline.test(data=test)

The above code snipped not only starts the pipeline and hereby
saves the results in the ``results`` folder, but also generates a graphical
representation of the pipeline. This enables us to see how the data flows
through the pipeline and to control if everything is set up as planned.

We can now save the pipeline to a folder:

.. code-block:: python

    pipeline.to_folder("./pipe_getting_started")

Saving the pipeline generates a series of *json* and *pickle* files
so that the same pipeline can be reloaded at any point in time in
the future to check results. We see below an example:

.. code-block:: python

    pipeline2 = Pipeline()
    pipeline2.from_folder("./pipe_getting_started")

Here, we create a new pipeline and use it to load the information from
the original pipeline.

.. warning::
    Sometimes from_folder use unpickle for loading modules. Note that this is not safe.
    Consequently, load only pipelines you trust with `from_folder`.
    For more details about pickling see https://docs.python.org/3/library/pickle.html

Results
*******
All results are saved in the ``results`` folder specified when creating the pipeline.
Here another folder with a time-stamp indicating when the pipeline was executed
will be automatically generated when the pipeline is run. In this folder, we find
the following items:

- *linear_regression_target.png*: A plot of the 24 training targets against time.
- *linear_regression_target_2..png*: A plot of the 24 test targets against time.
- *rescale_scaler_power.png*: A plot of the 24 rescaled predictions on the training set against time.
- *rescale_scaler_power_2..png*: A plot of the 24 rescaled predictions on the test set against time.
- *summary.md*: A summary of the training run, including the RMSE and runtimes.
- *summary_2..md*: A summary of the test run, including the RMSE and runtimes.

Furthermore, *pickle* and *json* files containing information about the pipeline can be found in the
folder ``pipe_getting_started``.

Summary
*******
This guide has provided an elementary introduction into `pyWATTS`. For more information,
consider working through the other examples provided or reading the documentation.

For further information on how to use pyWATTS, please have a look at (:ref:`howtouse`).
