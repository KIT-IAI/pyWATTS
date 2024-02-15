Core
=====

Note that pyWATTS is no longer actively maintained, however the main graph pipeline functionality of pyWATTS has been
integrated into the open source python pacakge [sktime](https://www.sktime.net/en/stable/).**


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
* the step,and
* the module.

Module
.......
.. important::
   In pyWATTS, a module provides the transform operation for time series analysis.
   More specifically, modules transform the input time series into an output time series.

For example, the CalendarExtraction module uses a time series as input and outputs a new time series that contains information about the weekday, public holiday, and so on for each time-step in the input time series.

PyWATTS has the following requirements for modules:

1. The modules have to inherit either from `BaseEstimator` or `BaseTransformer`.
2. The modules have have to implement `fit(...)`, `transform(...)`, `set_params(...)`, and `get_params(...)`.
   Note that if a module does not need to be fitted, it can inherit from `BaseTransformer` and does not have to implement `fit(...)` by itself.
3. The modules has either to output a xarray DataArray or Dict that contains DataArrays. Note in the case of Dict,
   the desired xarray has to specified if this module is passed as input to another module via square brackets. E.g.
   ``input=keras_wrapper["target_one"]``. Moreover, each datarray needs a time dimension which should be the first
   dimension of the DataArray.


Steps
.....

.. important::

    In pyWATTS, a step manages the execution of a module.

More specifically, steps

* fetch the input data from the previous steps and handle the output.
* are responsible for calling the correct transform method. In most cases, this is `transform`.
  However, if a module provides `prob_transform` or `inverse_transform` then it is also possible to call them instead of
  `transform`.
* are responsible for executing the callbacks.

Moreover, using steps makes it possible that the same module instance is added multiple times to the pipeline.

pyWATTS contains four kinds of steps:

* The `StartStep` is the first step in the pipeline. For each column, of the input data one start step is created that
  contains one column
* A `Step` wraps the modules and calls the transform method. Similiar `ProbabilisticStep` and `InverseStep` call the
  `probabilistic_transform` or `inverse_transform`.
* The `EitherOrStep` is necessary if only one of the previous steps has to provide an output.
  This can occur after a condition in the pipeline.
* `ResultStep` is needed if the previous step provides a dict with multiple keys as output. It selects in the background,
  the correct result for the successing step.

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

To add a module to a pipeline, the user has to call the module with the input the module needs. Then in the background
the `StepFactory` is called and creates the needed steps.
For example, if a module is called with ``x=pipeline["input"]``, then a `StartStep` is added before adding the
step that wraps the corresponding module. This `StartStep` selects the column "input" from the input data.
Moreover, the `StepFactory` adds the dependencies to the newly created step.
Finally, the `StepFactory` adds the step to the pipeline and returns a new `StepInformation` to the module that in turn returns it to the user.
