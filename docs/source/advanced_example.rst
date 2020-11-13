Advanced Example
================

In the following, we will examine a more advanced example than the getting_started one.
This example shows you,

* how you can add conditions to single steps of the pipeline,
* how you can apply online learning, and
* how you can use subpipelines.



Online learning, Subpipelining, and Conditions in pyWATTS
----------------------------------------------------------
When it comes to testing time-series analysis tools, it is often essential to simulate a production environment.
E.g., how evolves the accuracy of a predictive maintenance method through time.

pyWATTS aims to support researchers to answer this question by providing the online learning feature.

Moreover, in pyWATTS, we can add **conditions** to steps. These conditions determine if a step should be executed or not.
Use-cases of these features might be

* do not perform a processing step if the resulting information already exists. E.g., if an output file already exists, or
* only perform the step if the data meets certain criteria. E.g., a timestamp is within a certain interval.

In the following, we implement one step ahead forecast, based on the last two values. Moreover, we use a support vector regressor during daytime and during the night a linear regression.

.. Note::
   Currently, if online learning is used in a subpipeline, the outer pipeline must not use online learning.


1. We need to import packages, which we use in the pipeline.

.. code-block::

    # For reading the data
    import pandas as pd

    # Sklearn regressors and preprocessing modules
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR

    # Import the pyWATTS pipeline and the required modules
    from pywatts.core.pipeline import Pipeline
    from pywatts.core.computation_mode import ComputationMode
    from pywatts.modules.clock_shift import ClockShift
    from pywatts.modules.linear_interpolation import LinearInterpolater
    from pywatts.modules.root_mean_squared_error import RmseCalculator
    from pywatts.modules.whitelister import WhiteLister
    from pywatts.wrapper.sklearn_wrapper import SKLearnWrapper

2. We create a condition function for distinguishing between day time and night time.

.. code-block::

    # The condition function. It returns True during daytime.
    # For simplification we say daytime is between 8am and 8pm.
    def is_daytime(x, _):
        return 8 < x.indexes["time"][0].hour < 20

3. We create a subpipeline for preprocessing, this method takes as argument
   a sklearn module for scaling the data.

.. code-block::

    def create_preprocessing_pipeline(power_scaler):
        pipeline = Pipeline("preprocessing")

        # Deal with missing values through linear interpolation
        imputer_power_statistics = LinearInterpolater(method="nearest", dim="time",
                                                      name="imputer_power")(pipeline)
        # Scale the data using a standard SKLearn scaler
        scale_power_statistics = power_scaler([imputer_power_statistics])

        # Create lagged time series to later be used in the regression
        ClockShift(lag=1)([scale_power_statistics])
        ClockShift(lag=2)([scale_power_statistics])
        return pipeline

4. For testing, we create a further subpipeline. This pipeline should work online. Consequently, we use the batch argument of the Pipeline constructor for specifying time intervals. This interval determines the temporal length of one batch. Since in our dataset, the data are recorded each hour, each batch consists of one element.

   Moreover, while adding the linear regression as well as the SVR module to the pipeline, we use the condition keyword and pass the daytime function specified above. This condition will be checked before executing the current step for each batch.

   Finally, when we add the power_scaler to the pipeline. We provide as input a list containing one tuple. This tuple consists
   of the step information from the SVR regressor and the linear regressor.
   The meaning of this tuple is that either the SVR or the linear regressor has to provide an output. So, in contrast,
   to a simple list, only one of the inputs has to be provided and is passed to the power scaler.

.. code-block::

    # This function creates the pipeline which we use for testing.
    # The test pipeline works on batches with one hour
    def create_test_pipeline(modules, whitelister):
        regressor_svr, regressor_lin_reg = modules

        # Create test pipeline which works on a batch size of one hour.
        pipeline = Pipeline("test_pipeline", batch=pd.Timedelta("1h"))
        target = whitelister(pipeline)

        clock_shift1 = WhiteLister(target="ClockShift")(pipeline)
        clock_shift2 = WhiteLister(target="ClockShift_0")(pipeline)

        # Add the svr regressor to the pipeline. This regressor should be called if it is not daytime
        regressor_svr_power_statistics = regressor_svr([clock_shift1, clock_shift2],
                                                       condition=lambda x, y: not is_daytime(x, y),
                                                       computation_mode=ComputationMode.Transform,
                                                       plot=True)

        # Add the linear regressor to the pipeline. This regressor should be called if it is daytime
        regressor_lin_reg_power_statistics = regressor_lin_reg([clock_shift1, clock_shift2],
                                                               condition=lambda x, y: is_daytime(x, y),
                                                               computation_mode=ComputationMode.Transform,
                                                               plot=True)

        # Calculate the root mean squared error (RMSE) between the linear regression and the true values, save it as csv file
        RmseCalculator(target="load_power_statistics", predictions=["Regression"])(
            [(regressor_svr_power_statistics, regressor_lin_reg_power_statistics), target], plot=True, to_csv=True)

        return pipeline

5. We have to read the data and create the modules which are shared by multiple pipelines.

.. code-block::

    data = pd.read_csv("data/getting_started_data.csv", parse_dates=["time"], infer_datetime_format=True,
                       index_col="time")

    # Split the data into train and test data.
    train = data[:6000]
    test = data[6000:]

    # Create all modules which are used multiple times.
    regressor_lin_reg = SKLearnWrapper(module=LinearRegression(fit_intercept=True), name="Regression")
    regressor_svr = SKLearnWrapper(module=SVR(), name="Regression")
    power_scaler = SKLearnWrapper(module=StandardScaler(), name="scaler_power")
    white_lister = WhiteLister(target="load_power_statistics",
                               name="filter_power")

6. We create and run the train pipeline. Here we use the ```create_preprocessing_pipeline`` function for getting
   the preprocessing pipeline, which we add to the train pipeline, like a normal module.

.. code-block::

    train_pipeline = Pipeline(path="train")

    white_lister_power_statistics = white_lister(train_pipeline, plot=True)

    # Create preprocessing pipeline for the preprocessing steps
    preprocessing_pipeline = create_preprocessing_pipeline(power_scaler)
    preprocessing_pipeline = preprocessing_pipeline(white_lister_power_statistics)

    # Addd the regressors to the train pipeline
    regressor_lin_reg(preprocessing_pipeline,
                      targets=[white_lister_power_statistics],
                      plot=True)
    regressor_svr(preprocessing_pipeline,
                  targets=[white_lister_power_statistics],
                  plot=True)

    print("Start training")
    train_pipeline.train(data)
    print("Training finished")

7. We create and test the test pipeline. To this pipeline, we add the preprocessing pipeline again and the pipeline which we receive from ```create_test_pipeline``.

.. code-block::

    pipeline = Pipeline(path="results")

    # Select individual time-series (columns) and generate plots in the results folder
    white_lister_power_statistics = white_lister(pipeline, plot=True)

    # Get preprocessing pipeline
    preprocessing_pipeline = create_preprocessing_pipeline(power_scaler)
    preprocessing_pipeline = preprocessing_pipeline(white_lister_power_statistics)

    # Get the test pipeline, the arguments are the modules, from the training pipeline, which should be reused
    test_pipeline = create_test_pipeline([regressor_lin_reg, regressor_svr], white_lister)
    test_pipeline([preprocessing_pipeline, white_lister_power_statistics])

    # Now, the pipeline is complete so we can run it and explore the results
    # Start the pipeline
    print("Start testing")
    pipeline.test(test)
    print("Testing finished")

    print("FINISHED")