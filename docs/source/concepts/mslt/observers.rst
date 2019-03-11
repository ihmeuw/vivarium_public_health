.. _concept_observer:

Recording life table outputs
============================

The multi-state life table contains a vast amount of information for each
population cohort at each time-step of a model simulation.
Since the primary objective of MSLT models is to predict the impact of
preventative interventions on population morbidity and mortality, only some
these data are relevant and worth recording.

The MSLT framework provides a number of "observers" that record tailored
summary statistics during a model simulation.
We now introduce each of the provided observers in turn.

.. note:: Typically, each observer will record summary statistics for the
   "business-as-usual" (BAU) scenario **and** for the intervention scenario.

Population morbidity and mortality
----------------------------------

This observer records the core life table quantities (as shown in the
:ref:`example table <example_mslt_table>`) at each year of the simulation.

.. todo:: Show an example table.

Chronic disease incidence, prevalence, and mortality
----------------------------------------------------

This observer records the chronic disease incidence and prevalence, and the
number of deaths caused by this disease, at each year of the simulation.

.. todo:: Show an example table.

Risk factor prevalence
----------------------

This observer records the prevalence of each exposure category for a specific
risk factor, at each year of the simulation.

.. todo:: Show an example table.
