Uncertainty analyses
====================

.. todo:: Document how we're doing this

   I've implemented this by providing a module, ``mslt_port.parallel``, that
   runs multiple simulations for a single model specification, where each
   simulation uses a different draw number.
   It also provides support for running these simulations in parallel across
   an arbitrary number of cores.

   **This is a bespoke process** because you need to decide which draws are
   correlated (e.g., the incidence rate for a single disease, across all age
   groups, sex, and ethnicity).

   First, you need to generate data artifacts that contain the 2000 draws for
   each rate/value, using the ``build_simulation_files.py`` script in the
   **uncertainty_analysis** directory.
   Note that this is currently a **very slow process** and generates data
   artifacts that are around **3 GB** in size.

   A convenience script is provided in the **uncertainty_analysis** directory,
   which can be used as follows:

   .. code:: console

      run_uncertainty.py --draws 2000 --spawn 16 file1.yaml file2.yaml [...]

   This will run 2000 simulations for each of the model specifications
   (file1.yaml, file2.yaml, etc) and distribute these simulations across 16
   cores.

   Note that this means the user has to install the ``mslt_port`` package:

   .. code-block:: sh

      pip install .
