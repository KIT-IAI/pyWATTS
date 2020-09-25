.. _contribution:

How to Contribute
=================

Great to have you here! The best way to get involved is by starting to use pyWATTS.
We also happily welcome you if you're interested in contributing with new feature ideas,
filing bug reports or contributing your own modules.

Report a bug or request a feature
---------------------------------
You have found a bug or you are missing a great feature? Great!
We track our bugs and feature ideas with GitHub issues; feel free to open an issue if you encounter a bug or have a
feature that you would like to see implemented in the future.

Before you open a new issue, please check that it is not already addresses by any other issues or pull requests.

If you are *submitting a bug report*, please make sure you include the following information

- ideally, a short reproducible code snippet,
- if you can not provide a code snippet, please provide detailed and precise information about the problem,
  so that we can recreate it ourselves (what are you trying to do, what functions are involved, what type of data are
  you using, what are the dimension of this data, etc.)
- your operating system type and version number, as well as your Python,
  scikit-learn, numpy, and scipy versions
- the full traceback, if an exception is raised.

If you are *requesting a feature*, it would help us a lot if you include the following information

- a detailed description of the feature (ideally with an example)
- how you expect the input and output of the module to look like (optional)

If you encounter any issues when using pyWATTS feel free to contact us.

Work on known issues
--------------------

To start working on an issue, please assign the issue to yourself and
create a new branch. We have three categories

- docs
- bugfix
- feature

and request that you follow our naming convention with your branch name:

.. code-block::

  category/[issue number]_[short name of issue]

For example:

.. code-block::

  docs/102_naming_conventions

Before working on the issue, read carefully through our programming guidelines, tests, and documentation information
(:ref:`gettingstarted` and :ref:`howtouse`). Please follow our guidelines when coding, as this helps us with the review process and
 allows us to integrate your work into pyWATTS much faster. Moreover, you should update the CHANGELOG.md. Therefore,
add at the top the name of your pull request as a title. Afterwards, list the features you have implemented.

Write your own modules
----------------------

We always welcome new contributions to pyWATTS and therefore, if you want to write your own module or have already done
so, feel free to share it with the pyWATTS community. Before starting, make sure this isn't already an open issue, if
in doubt about duplicated work feel free to contact us. To avoid double ups, please let us know what your plans are by
opening an issue.

Before working on the module, read carefully through our programming guidelines, tests, and documentation information
(:ref:`gettingstarted` and :ref:`howtouse`).  Please follow our guidelines when coding, as this helps us with the review process and
 allows us to integrate your work into pyWATTS much faster. Moreover, you should update the CHANGELOG.md. Therefore,
add at the top the name of your pull request as a title. Afterwards, list the features you have implemented.

Once you are happy with your module, feel free to issue a pull request. Every line of pyWATTS code (including our own)
is reviewed by a second person. If we find any issues then we will contact you with feedback, otherwise your module
will be incorporated into our repository after approval.
