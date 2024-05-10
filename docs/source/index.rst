
Welcome to the pyWATTS documentation!
=====================================

Please note that pyWATTS is no longer actively maintained. The graph pipeline functionality of pyWATTS has now been integrated into the open source python package sktime!

We would like to thank everybody who has helped develop, test, and use pyWATTS in the last few years.


Legacy Information
------------------

The goals of pyWATTS (Python Workflow Automation Tool for Time-Series) are

* to support researchers in conducting automated time series experiments independent of the execution environment and
* to make methods developed during the research easily reusable for other researchers.

Therefore, pyWATTS is an automation tool for time series analysis that implements three core ideas:

* pyWATTS provides a pipeline to support the execution of experiments. This way, the execution of simple and often recurring tasks is simplified. For example, a defined preprocessing pipeline could be reused in other experiments. Furthermore, the execution of defined pipelines is independent of the execution environment. Consequently, for the repetition or reuse of a third-party experiment or pipeline, it should be sufficient to install pyWATTS and clone the third-party repository.
* pyWATTS allows to define end-to-end pipelines for experiments. Therefore, experiments can be easily executed that comprise the preprocessing, models and benchmark training, evaluation, and comparison of the models with the benchmark.
* pyWATTS defines an API that forces the different methods (called modules) to have the same interface in order to make newly developed methods more reusable.

Features of pyWATTS:

* Reuseable modules
* Plug-and-play architecture to insert modules into the pipeline
* End-to-end pipeline for experiments such that pipeline performs all necessary steps from preprocessing to evaluation
* Conditions within the pipeline
* Saving and loading of the entire pipeline including the pipeline modules
* Adapters and wrappers for existing machine learning libraries


Table of Content
----------------

.. toctree::
   :maxdepth: 2

   install
   getting_started
   how_to_use
   core
   advanced_example
   neural_network_example
   about_us
   contribution


   
pyWATTS API
===========

.. toctree::
   :maxdepth: 5

   api/pywatts



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
