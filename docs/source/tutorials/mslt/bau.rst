The business-as-usual (BAU) scenario
====================================

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
The contents of this file will look like:

.. csv-table:: An extract of the simulation results, showing a subset of rows
   for the cohort of males aged 50-54 in 2010.

   Year of birth,Sex,Age,Year,Population,BAU population,ACMR,BAU ACMR,Probability of death,BAU Probability of death,Deaths,BAU deaths,YLD rate,BAU YLD rate,Person years,BAU person years,HALYs,BAU HALYs
   ...
   1958,male,53,2011,129455.99013360904,129435.28592207265,0.0030389592575188165,0.003198904481598754,0.003034346294886081,0.0031937934380235067,394.00986639095765,414.7140779273523,0.112234968131794,0.112234968131794,129652.99506680452,129642.64296103633,115101.39529729007,115092.20505978286
   1958,male,54,2012,129037.13158471332,128994.49027457157,0.0032407741660891607,0.0034113412274622747,0.0032355285256666644,0.003405529213776126,418.8585488957143,440.79564750108415,0.112234968131794,0.112234968131794,129246.56085916118,129214.8880983221,114740.57721998925,114712.4592504536
   1958,male,55,2013,128579.02514898572,128512.47741357243,0.0035565076694606492,0.003743692283642789,0.003550190786958507,0.0037366934042930566,458.1064357276018,482.0128609991364,0.130096022623328,0.130096022623328,128808.07836684951,128753.48384407199,112050.65968956845,112003.1676970613
   ...
   1958,male,107,2065,304.4733203155968,221.37768288264857,0.45691667748159753,0.4809649236648395,0.3667668995586699,0.38181339895695254,176.35012385274337,136.73050404525435,0.35784212863237896,0.35784212863237896,392.64838224196853,289.7429349052758,252.14224933644252,186.06070632257908
   1958,male,108,2066,192.8025846251116,136.85271732801016,0.45691667748159753,0.4809649236648395,0.3667668995586699,0.38181339895695254,111.67073569048523,84.52496555463843,0.35784212863237896,0.35784212863237896,248.6379524703542,179.11520010532936,159.66481829956638,115.02023562922379
   1958,male,109,2067,122.08897843526132,84.60051616850757,0.45691667748159753,0.4809649236648395,0.3667668995586699,0.38181339895695254,70.71360618985027,52.2522011595026,0.35784212863237896,0.35784212863237896,157.44578153018645,110.72661674825886,101.10504792323603,71.10396851480029
   ...

We can now plot the survival of this cohort in both the BAU and intervention
scenarios, relative to the starting population, and see how the survival rate
has increased as a result of this intervention.

.. _mslt_reduce_acmr_fig:

.. figure:: /_static/mslt_reduce_acmr_survival.png
   :alt: The survival rates in the BAU and intervention scenarios, and the
      difference between these two rates.

   The impact of reducing the all-cause mortality rate by 5% on survival rate.
   Results are shown for the cohort of males aged 50-54 in 2010.
