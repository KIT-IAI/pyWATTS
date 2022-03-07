![Pipeline status](https://github.com/KIT-IAI/pyWATTS/workflows/Python%20application/badge.svg?branch=master)
[![DOI](https://zenodo.org/badge/297579218.svg)](https://zenodo.org/badge/latestdoi/297579218)
[![Documentation](https://readthedocs.org/projects/pywatts/badge/)](https://pywatts.readthedocs.io/en/latest/)


# pyWATTS

## Installation

To install this project, perform the following steps.

1. Clone the project
2. Open a terminal of the virtual environment where you want to use the project
2. cd pywatts
3. ``pip install .`` or ``pip install -e .``
   if you want to install the project editable. If you aim to develop code for pyWATTS, you should
   use:  ``pip install -e .[dev]``

---
**NOTE**
If you want to use torch, you have to install it by yourself, since it is not possible to install torch via pypi on
windows. To install torch take a look at
[install Pytorch](https://pytorch.org/get-started/locally/).

---

## How to use

A simple example is given in example.py. If you like a more detailed explanation of this example, look at the
getting_started page in our documentations.

1. You need to create a pipeline.
2. You can add modules to the pipeline by calling them with either the previous step or with the pipeline, if it is a
   start step (i.e. Functional API).
3. Run the pipeline with pipeline.run()

An extract of the example in example.py is given in the following:

```python
from pywatts.core.pipeline import Pipeline
from pywatts.modules import SKLearnWrapper
from pywatts.modules import LinearInterpolater, CalendarFeature, CalendarExtraction
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline()

# Add modules
calendar = CalendarExtraction(continent="Europe", country="Germany", features=[CalendarFeature.month,
                                                                               CalendarFeature.weekday,
                                                                               CalendarFeature.weekend]
                              )(x=pipeline["load_power_statistics"])

# Deal with missing values through linear interpolation
imputer_power_statistics = LinearInterpolater(
    method="nearest", dim="time", name="imputer_power"
)(x=pipeline["load_power_statistics"])

power_scaler = SKLearnWrapper(module=StandardScaler(), name="scaler_power")(x=imputer_power_statistics)

# Train the pipeline
pipeline.train("data/getting_started_data.csv")
```

## Goals

The goals of pyWATTS (Python Workflow Automation Tool for Time-Series) are

* to support researchers in conducting automated time series experiments independent of the execution environment and
* to make methods developed during the research easily reusable for other researchers.

Therefore, pyWATTS is an automation tool for time series analysis that implements three core ideas:

* pyWATTS provides a pipeline to support the execution of experiments. This way, the execution of simple and often
  recurring tasks is simplified. For example, a defined preprocessing pipeline could be reused in other experiments.
  Furthermore, the execution of defined pipelines is independent of the execution environment. Consequently, for the
  repetition or reuse of a third-party experiment or pipeline, it should be sufficient to install pyWATTS and clone the
  third-party repository.
* pyWATTS allows to define end-to-end pipelines for experiments. Therefore, experiments can be easily executed that
  comprise the preprocessing, models and benchmark training, evaluation, and comparison of the models with the
  benchmark.
* pyWATTS defines an API that forces the different methods (called modules) to have the same interface in order to make
  newly developed methods more reusable.

## Features

* Reuseable modules
* Plug-and-play architecture to insert modules into the pipeline
* End-to-end pipeline for experiments such that pipeline performs all necessary steps from preprocessing to evaluation
* Conditions within the pipeline
* Saving and loading of the entire pipeline including the pipeline modules
* Adapters and wrappers for existing machine learning libraries

## Programming Guidelines

* Implement new features on a new, separate branch (see "Module Implementation Worflow" below). If everything works,
  open a pull request to merge the branch with the master. Note that you should name your branch with the following
  convention: <feature|docs|bugfix>/<issue_number>_<descriptive_name>
* Provide tests for your module (see "Tests" below).
* Provide proper logging information (see "Logging" below).
* Use a linter, follow pep8, and add docstrings to your classes and methods<br>
  To do so in PyCharm, activate in Settings -> Editor -> Inspections -> Python:
    * "PEP8 coding style violation"
    * "missing or empty docstring"
    * "missing type hinting for function parameter"
    * "package requirement"
* Use typing (see https://docs.python.org/3/library/typing.html).

## Module Implementation Workflow

* Create an issue or choose an existing one and assign yourself to the issue. This way, everyone knows whether an issue
  is under development or not.
* Create a new branch for your module/feature. For naming your branch use the following naming convention <
  feature|docs|bugfix>/<issue_number>_<descriptive_name>.
* Decide whether your module is a Transformer or an Estimator. Inherit of the BaseEstimator or BaseTransformer
  accordingly.

  | Estimator   | Transformer   |
    |-----|--------|
  | An Estimator is everything that is fitted before the prediction | A transformer converts an input to an output, it does not need to be fitted before.

* Implement the abstract methods of either BaseTransformer or BaseEstimator. The folllowing abstract methods exist:

  | Method          | Base Estimator | BaseTransformer |
    |-----------------|----------------|-----------------|
  | getParam        | yes            | yes             |
  | setParam        | yes            | yes             |
  | fit             | yes            | no              |
  | transform       | yes            | yes             |

* Add test cases for your module/feature. For example, if you implement a module, add the tests in the folder
  tests/unit/modules.
* Add a documentation for your module/feature
    * Use a docstring for your class (i.e. the module you developed) to describe what your class does and to describe
      the parameters
    * Use a docstring for each method
* Commit and push your changes
* Open a merge request and assign a maintainer
* If all tests are green, your code works, and your code is documented, then the assigned maintainer will merge your
  module/feature.

## Tests

### Writing tests

To write tests, see the test template where the different methods are explained. When writing tests, please consider the
following basic rules for testing:

* In general, tests should not only cover the normal case but also edge cases such as exceptions when wrong inputs are
  used.
* After writing tests, please also check if the tests sufficiently cover the code of your module. However, be aware that
  the line-coverage metric provided will only help you to identify uncovered source code. Please also note that 100%
  line coverage does not mean that your code is free of bugs.
* If you fix a bug, implement a test case which repeats the bug.

If you are looking for some test guidelines, have a look at https://docs.python-guide.org/writing/tests/

### Test libraries

* **[unittest](https://docs.python.org/3/library/unittest.html)** We use unittests to define our test cases.
    * **[unittest.mock](https://docs.python.org/3/library/unittest.mock.html)** We use this mock object library to mock
      calls of other methods. In mocks, it is possible to check whether a method is called correctly.
    * **[unittest.mock.patch](https://docs.python.org/3/library/unittest.mock.html#the-patchers)** We use this library
      to replace objects imported in the module under test with mock objects.
* **[pytest](https://docs.pytest.org/en/latest/)** We use pytest to run our tests.
* **[pytest-cov](https://pytest-cov.readthedocs.io/en/latest/)** We use pytest-coverage to calculate the test coverage
  of our source code.

### Run tests

To run tests, use your preferred editor, IDE, or the following CLI command

``
pytest tests
``

This command executes all tests and prints the results. It also provides the code coverage of the project.

## Documentation

To build the documentation, we use Sphinx. Perhaps, you first need to install it via pip.

### Documentation of the API

We use Sphinx-apidoc to automatically generate the documentation from the Python docstrings. Therefore, it is necessary
that each module, class, and method is documented with such a string. The source files of the API documentation are
located in docs/source/api. Do not change the files in this directory because they are automatically generated and
consequently the generation process would override any changes.

### Extending the documentation

To extend the documentation, you have to create a new restructured text (RST) file in docs/source. This file has to be
included in the index.rst or in another part of the documentation.

### Building the documentation

There are mainly two steps to build the documentation.

1. Create the documentation of the API. This is done via
   `sphinx-apidoc -o docs/source/api pywatts`. This command creates the rst files from the docstrings.
2. Build the documentation as html. For this, you have to change to the docs directory `cd docs` and execute `make html`
   . This command generates the static html website out of the rst files.

To view the documentation website locally, open the index.html in docs/build/html in your webbrowser.

### Further information on restructured text

* [RST Cheatsheet](https://github.com/ralsina/rst-cheatsheet/blob/master/rst-cheatsheet.rst)

## Logging

We use logging to obtain information from the pipeline about

* Modules added to the pipeline and
* Calls of fit and transform on modules.

Therefore, use the logging library provided by the Python standard library. Currently, the log information is logged
into pywatts.log.

**Note** Try to avoid print statements for messages. Instead, use logging with an appropriate logging level.

### Further information on logging

* [Basic logging tutorial](https://docs.python.org/3/howto/logging.html#basic-logging-tutorial)
* [Logging Cookbook](https://docs.python.org/3/howto/logging-cookbook.html#logging-cookbook)

## Glossary

For a common understanding of the various terms used in this framework, see the following table:

| Term            | Explanation                      |
  |-----------------|----------------------------------|
| Pipeline        | A pipeline implements the workflow. It executes the fit and transform methods of the individual modules in the pipeline.|
| Module          | A module is an element of the pipeline. Each module implements only one specific task, such as the detection of missing values or a clock shift.|
| BaseEstimator   | A module that has to be fitted must inherit from BaseEstimator. Such a module is also called Estimator.|
| BaseTransformer | A module that only transforms data must inherit from BaseTransformer. Such a module is also called Transformer.|
| Wrapper         | A wrapper is a special type of module that wraps models and methods of external libraries, such as sklearn.|  
| Summary         | A summary is an element of the pipeline. In contrast to the module it calculates one value for summarizing the time series instead of transforming it into a new.
| Step            | A step manages the execution of a single module, such as fetching the input, checking condition, and providing outputs.

## Current Development Status

### Standing assumptions

* The graph representing the pipeline is a directed acyclic graph (DAG).
* The first coordinate is always the time coordinate.

### Known issues* Currently, no edge cases are tested.

* Currently, no pipeline checks are implemented.
* Currently, the biclustering mixin of sklearn is not supported.
* Currently, subpipelining is only possible if the outer pipeline does use online learning.

### Outlook on further features

* Grid search to find the best combination of preprocessing and models
* Online/Batch Learning (partially, see known issues)
* Multiprocessing

# Funding

This project is supported by the Helmholtz Association under the Program “Energy System Design”, by the Helmholtz Association’s Initiative and Networking Fund through Helmholtz AI, by the Helmholtz Association under the Joint Initiative "Energy System 2050 - A Contribution of the Research Field Energy", and by the German Research Foundation (DFG) Research Training Group 2153 "Energy Status Data: Informatics Methods for its Collection, Analysis and Exploitation".

# Citation
If you use this framework in a scientific publication please cite the corresponding paper:

>Benedikt Heidrich, Andreas Bartschat, Marian Turowski, Oliver Neumann, Kaleb Phipps, Stefan Meisenbacher, Kai Schmieder, Nicole Ludwig, Ralf Mikut, Veit Hagenmeyer. “pyWATTS: Python Workflow Automation Tool for Time Series.” (2021). ). arXiv:2106.10157. http://arxiv.org/abs/2106.10157

# Contact
If you have any questions and want to talk to the pyWATTS Team directly, feel free to [contact us](mailto:pywatts-team@iai.kit.edu).
For more information on pyWATTSvisit the [project website](https://www.iai.kit.edu/english/1266_4162.php).
