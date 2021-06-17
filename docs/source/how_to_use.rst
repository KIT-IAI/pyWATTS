.. _howtouse:

How to use?
===========

Where should pyWATTS be used?
-----------------------------
pyWATTS is not a machine learning framework or library. Instead, pyWATTS' aim is to support users in performing data analysis tasks and hereby to avoid a script nightmare.
For this reason, pyWATTS is designed to automate frequently occurring tasks and
allows to define the sequence of necessary step in one pipeline.


How to use the modules?
-------------------------
Modules in pyWATTS contain the functionality to perform data analysis tasks with pyWATTS.

How to use pyWATTS?
-------------------

1. Split your workflow in separate steps.
   For example, filtering of columns, scaling of values, aggregating multiple timesteps, using a SKLearn estimator, and
   evaluating the result with the RMSE.
2. Create a pipeline object. Thereby, you can specify the

   * path: The path where the plots and csvs of the pipeline should be saved.
   * batch: The timedelta that specifies the interval at which data should be processed. Note that this is currently only possible for
     subpipelines.

3. For each step, create a module that contains the functionality. This functionality is either

   * implemented in a predefined module,
   * implemented in your own module (see below)
   * a wrapped functions in a FunctionModule,
   * a wrapped SKLearn model in a SKLearnWrapper,
   * etc.

   All of these approaches have their own strengths and weaknesses.
   The following table gives you an overview of which approach should be used when.


   ===============================  ======================================================
    Approach                          Use Case
   ===============================  ======================================================
    Predefined Module               If there exists already a predefined module for the specified step

    Own Module                      If there is no module that meets your requirements and you will need this functionality frequently.

    Function Module                 If there is no module that meets your requirements
                                    and you will probably need this functionality only once,
                                    the functionality is very simple, or you simply want to try out the functionality.

    SKLearnWrapper                  If you want to use an existing SKLearn module

   KerasWrapper or pyTorchWrapper   If you want to use a Keras or pyTorch model
                                    Note that you can write an own module for wrapping Deep Learning Models.


   ===============================  ======================================================

4. Add each module to the pipeline by calling the module object with:

   * inputs: Inputs can be the following:

     * the pipeline, if there is no previous step
     * StepInformation or a list of StepInformation if the module has a previous step, and the previous step has to provide a result
     * a tuple of StepInformation or a list of tuples of StepInformation if only one step in a tuple should has to provide a result.
       If multiple modules provide an output, the first one is chosen.

   * targets: The step against which the module should be fitted
   * computation_mode: Train, Transform, FitTransform
   * to_csv: To create a csv file with the used data
   * plot: To plot the used data
   * batch_size: Chunk of past data that should be used for fitting. [Necessary for Online Learning]
   * condition: Condition that describes to execute this module only if the condition is True
   * train_if: Train only if the condition is True
   * use_prob_transform
   * use_inverse_transform

   Calling a module will return a StepInformation, which can be used as input for the next step. If the added module has no successors, you
   can ignore the returned StepInformation.

5. Call pipeline.run(). This starts the pipeline and executes all added modules in the defined order.
6. Optionally, load and save your whole pipeline by calling from_folder or to_folder. Both methods take as argument the path where the pipeline should be saved or from which the pipeline should be loaded.

   The to_folder method will create a JSON containing the structure of the pipeline. Moreover, this JSON contains
   the parameters of all modules and information about the steps.
   Furthermore, in the folder specified by the path argument, modules might create further data. For example, the SKLearnWrapper creates pickle files and the KerasWrapper creates h5 files.


How to write an own modules?
-----------------------------

The simplest way of writing own modules is to make use of the template.py. Nevertheless, you normally have to perform the following steps:

1. Decide whether your module is a Transformer or Estimator. Accordingly, your module has to inherit from BaseTransformer or BaseEstimator.
   The difference between both is that the Estimator has a fit method in contrast to the Transformer.
2. Implement the following methods:

   * __init__: method to create the module and its specific parameters
   * transform: executes the transformation defined in the module. Note that the first dimension of the datavariable has to be a time dimension since we work with time series.
   * get_params: method to get the module specific parameters
   * set_params: method set the module specific parameters
   * fit: method to fit the model if you module is an estimator
   * save and load: method to save and load if your module has to save and load more than the parameters that are returned by get_params.
     Moreover, if you save your module, you have to use the file_manager to get a path where you can save your module.
     The pipeline will pass a filemanager to the save function.

   Please note that some util methods exist in pywatts.utils. These methods are often used by modules to handle
   xarray datasets.

After completing these two steps, you can use your created module in your pipeline.


# How can I get results?
Currently, there are four possibilities to get the results of a pyWATTS pipeline.

1. Pipeline.train and pipeline.test returns a xarray Dataset and a summary string:

   * The dataset contains the results of all steps without successing steps.
   * The summary string contains all information of summary modules and the training

2. If you have a step information, you get its buffer with:

    step_information.step.buffer

3. With callbacks you can perform operations on the output of steps including, writing them into a file.
4. You can use summary modules for example for calculating metrics. The results are the saved in summardy.md which is
   placed in the pipelines results folder.