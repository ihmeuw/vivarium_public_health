The business-as-usual (BAU) scenario
====================================

The business-as-usual (BAU) scenario characterises what we expect to occur in
the absence of any intervention.
It comprises a population that are subject to BAU morbidity and mortality
rates, and is the baseline against which we quantitatively evaluate the impact
of different interventions.

Defining a simulation
---------------------

Model simulations are defined by text files that describe all of the
simulation components and configuration settings.
These details are written in the YAML markup language, and the file names
typically have a ``.yaml`` extension.

In the intervention example presented below, we provide a step-by-step
description of the contents of these YAML files.

In brief, these files will contain three sections:

+ The ``plugins`` section, where we load the plugins that allow us to make use
  of data artefacts;

+ The ``components`` section, where we list the simulation components that
  define the population demographics, the BAU scenario, and the intervention;
  and

+ The ``configuration`` section, where we identify the relevant data artefact,
  and define component-specific configuration settings and other simulation
  details.

.. literalinclude:: /_static/mslt_reduce_acmr.yaml
   :language: yaml
   :caption: An example simulation definition.

Data artefacts
--------------

Data artefacts collect all of the required
:ref:`input data tables <mslt_input_data>` into a single file.
The input data files that were used to generate the data artefacts for this
tutorial are stored in the ``mslt_port-master/data`` directory.
If you modify any of the input data files, you can rebuild these artefacts by
running the provided script:

.. code:: console

   ./build_minimal_artifacts.py

This will update the follow data artefacts:

+ ``mslt_tobacco_maori_20-years.hdf``: data for the Maori population, where
  cessation of smoking results in gradual recovery over the next 20 years.
+ ``mslt_tobacco_non-maori_20-years.hdf``: data for the non-Maori population,
  where cessation of smoking results in gradual recovery over the next 20
  years.
+ ``mslt_tobacco_maori_0-years.hdf``: data for the Maori population, where
  cessation of smoking results in immediate recovery.
+ ``mslt_tobacco_non-maori_0-years.hdf``: data for the non-Maori population,
  where cessation of smoking results in immediate recovery.

.. _mslt_reduce_acmr:

Intervention: a reduction in mortality rate
-------------------------------------------

In this section, we describe how to use the MSLT components to define a model
simulation that will evaluate the impact of reducing the all-cause mortality
rate.
We then show how to run this simulation and interpret the results.

.. note:: All of the MSLT components are contained within the
   ``vivarium_public_health.mslt`` module.
   This module is divided into several sub-modules; we will use the
   :mod:`.population`, :mod:`.intervention`, and :mod:`.observer` modules in
   this example.

Defining the model simulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:currentmodule:: vivarium_public_health.mslt.population

Because we are reading all of the necessary input data tables from a
preexisting data artifact, we need to load two Vivarium plugins:

.. literalinclude:: /_static/mslt_reduce_acmr.yaml
   :language: yaml
   :lines: 1-5
   :caption: Load the necessary Vivarium plugins.

We then need to specify the location of the data artifact in the configuration
settings:

.. literalinclude:: /_static/mslt_reduce_acmr.yaml
   :language: yaml
   :lines: 19-22
   :caption: Define the input data artifact.

The core components of the simulation are the population demographics
(:class:`~BasePopulation`), the mortality rate (:class:`~Mortality`), and the
years lost due to disability (YLD) rate (:class:`Disability`).
These components are located in the
:mod:`.population` module, and so we identify them as follows:

.. literalinclude:: /_static/mslt_reduce_acmr.yaml
   :language: yaml
   :lines: 7-13
   :caption: The core population components.

We define the number of population cohorts, and the simulation time period, in
the configuration settings:

.. literalinclude:: /_static/mslt_reduce_acmr.yaml
   :language: yaml
   :lines: 19,23-30
   :caption: Define the number of cohorts and the simulation time period.

.. py:currentmodule:: vivarium_public_health.mslt.intervention

We also add a component that will reduce the all-cause mortality rate
(:class:`ModifyAllCauseMortality`, which is located in the
:mod:`.intervention` module)
and give this intervention a name (``reduce_acmr``).
We define the reduction in all-cause mortality rate in the configuration
settings, identifying the intervention by name (``reduce_acmr``) and defining
the mortality rate scaling factor (``scale``):

.. literalinclude:: /_static/mslt_reduce_acmr.yaml
   :language: yaml
   :lines: 7-9,14-15,18-19,31-34
   :caption: The core population components.

.. py:currentmodule:: vivarium_public_health.mslt.observer

Finally, we need to record the core life table quantities (as shown in the
:ref:`example table <example_mslt_table>`) at each year of the simulation, by
using the :class:`MorbidityMortality` observer (located in the
:mod:`.observer` module) and specifying the prefix for output
files (``mslt_reduce_acmr``):

.. literalinclude:: /_static/mslt_reduce_acmr.yaml
   :language: yaml
   :lines: 7-9,16-17,18-19,35-36
   :caption: The core population components.

Putting all of these pieces together, we obtain the following simulation
definition:

.. literalinclude:: /_static/mslt_reduce_acmr.yaml
   :language: yaml
   :caption: The simulation definition for the BAU scenario and the
      intervention.

Running the model simulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The above simulation is already defined in ``mslt_reduce_acmr.yaml``. Run this
simulation with the following command:

.. code-block:: console

   simulate run mslt_reduce_acmr.yaml

When this has completed, the output recorded by the
:class:`MorbidityMortality` observer will be saved in the file
``mslt_reduce_acmr_mm.csv``.
The contents of this file will contain the following results:

.. csv-table:: An extract of the simulation results, showing a subset of rows
   for the cohort of males aged 50-54 in 2010.
   :file: ../../_static/table_mslt_reduce_acmr_mm.csv
   :header-rows: 1

We can now plot the survival of this cohort in both the BAU and intervention
scenarios, relative to the starting population, and see how the survival rate
has increased as a result of this intervention.

.. _mslt_reduce_acmr_fig:

.. figure:: /_static/mslt_reduce_acmr_survival.png
   :alt: The survival rates in the BAU and intervention scenarios, and the
      difference between these two rates.

   The impact of reducing the all-cause mortality rate by 5% on survival rate.
   Results are shown for the cohort of males aged 50-54 in 2010.
