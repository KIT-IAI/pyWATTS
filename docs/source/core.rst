Core
=====

Basic Idea
----------
The main goal of pyWATTS is to automate the time series analysis workflow.
Therefore, it aims to provide an end-to-end solution for executing experiments on time series.
To achieve this goal, the main Python object of pyWATTS is the pipeline.
The pipeline manages different steps and ensures the correct order of execution.


Architecture
------------

The three Python objects of pyWATTS are

* the pipeline,
* the step, and
* the module.

Module
.......
.. important::
   In pyWATTS, a module provides the transform operation for time series analysis.
   More specifically, modules transform the input time series into an output time series.

For example, the CalendarExtraction module uses a time series as input and outputs a new time series that contains information about the weekday, public holiday, and so on for each time-step in the input time series.

PyWATTS has the following requirements for modules:

1. The modules have to inherit either from `BaseEstimator`` or BaseTransformer.
2. The modules have have to implement `fit(...)`, `transform(...)`, `set_params(...)`, and `get_params(...)`.
  Note that if a module does not need to be fitted, it can inherit from `BaseTransformer` and does not have to implement `fit(...)` by itself.
3. The first dimension of each data-array in the output time-series has to be a time index.

Steps
.....

.. important::

    In pyWATTS, a step manages the execution of a module.

More specifically, steps

* fetch the input data from the previous steps and handle the output.
* are responsible for calling the correct transform method. In most cases, this is `transform`.
  However, if a module provides `prob_transform` or `inverse_transform` then it is also possible to call them instead of `transform`.
* are responsible for providing plots or CSV data of the corresponding modules.

Moreover, using steps makes it possible that the same module instance is added multiple times to the pipeline.

pyWATTS contains four kinds of steps:

* The `StartStep` is the first step in the pipeline.
* A `Step` wraps the modules and calls the transform method.
* The `CollectStep`'s task is to collect the output of all previous steps and merge them into one dataset.
  This step is necessary if the input of a module consists of multiple previous steps' output.
* The `EitherOrStep` is necessary if only one of the previous steps has to provide an output.
  This can occur after a condition in the pipeline.

Note that the user of the pipeline does not have to care about the steps.
Inserting and creating the correct steps is done by the `StepFactory` that works in the background.

Pipeline
........

.. important::
    The pipeline object in pyWATTS is the main Python object. It is responsible for executing the steps in the correct order.

Moreover, the pipeline object is responsible for the interaction between the user and the pipeline.
Therefore, the user interacts with this object for starting, storing, and reloading the pipeline.

Control Flow for adding Modules to a Pipeline
---------------------------------------------

To add a module to a pipeline, the user has to call the module with a `Pipeline` or a list of `StepInformation`.
Afterward, the module calls the `StepFactory` that determines what kind of steps are needed.
For example, if this is the first module in the pipeline, then a `StartStep` has to be added before adding the step that wraps the corresponding module.
Moreover, the `StepFactory` adds the dependencies to the newly created step. The dependencies are stored in the `StepInformation`.
Finally, the `StepFactory` adds the step to the pipeline and returns a new `StepInformation` to the module that in turn returns it to the user.
