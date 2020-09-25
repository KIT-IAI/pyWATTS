Installation
===============

To install this project, perform the following steps.

1. Clone the project

2. Open a terminal of the virtual environment where you want to use the project

3. Go to the directory, where you cloned the project.

4. ``pip install .`` or ``pip install -e .`` if you want to install the project editable.
   If you want to develop some code for pyWATTS, then you should use
   ``pip install -e .[dev]``. This installs sphinx, pytest, and pylint, which are required for
   generating the documentation and testing the code.

.. Note::
   If you want to use torch, you have to install it by yourself, since it is not possible to install
   torch via pypi on windows. For installing torch take a look at
   `install Pytorch: <https://pytorch.org/get-started/locally/>`_.


