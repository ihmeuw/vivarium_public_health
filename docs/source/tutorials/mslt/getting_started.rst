Getting started
===============

- You need Python 3.6 or later installed. If you don't already have a suitable
  version of Python installed, the easiest option is to use
  `Anaconda <https://www.anaconda.com/distribution/>`__.

- You need to install the ``vivarium_public_health`` Python package. Do this
  by running the following command in a command prompt or terminal:

  .. Note that `pip` is the simplest way to do this when using Anaconda.

     https://www.anaconda.com/using-pip-in-a-conda-environment/

  .. code-block:: sh

     pip install vivarium_public_health

- Download the input data and model specification files
  `zip archive <https://github.com/collijk/mslt_port/archive/master.zip>`__,
  and unzip the contents; this will create a new directory called
  **mslt_port-master**.

  .. note:: This archive contains all of the files you will need in order to
     follow the tutorials.

  .. note:: We will probably need to change this link (and the directory name)
     to a canonical location.

Once you have completed these steps, you will be able to run all of the
simulations described in these tutorials. For each simulation there will be a
model specification file, whose file name ends in ``.yaml``. These are
plain text files, that you can edit in any text editor. To run the simulation
described in one of these files, run the following command in a command prompt
or terminal, from within the **mslt_port-master** directory:

.. code-block:: sh

   simulate run model_file.yaml

.. note:: Each simulation will produce one or more output CSV files. You can
   then extract relevant subsets from these data files and plot them using
   your normal plotting tools. This allows you to easily examine outcomes of
   interest for specific cohorts and/or over specific time intervals.

   The figures shown in these tutorials were created using external tools, not
   included in the Vivarium Public Health package and not documented here. Any
   plotting software could be used to produce similar figures.
