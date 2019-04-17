.. _mslt_advanced:

Multi-State Life Tables
=======================

Here we describe how the multi-state life tables (MSLT) components are
implemented.

Note that when you run a simulation using the simulate command:

.. code-block:: console

   simulate run reduce_acmr.yaml

The following sequence of operations will be performed:

1. The model specification will be read (in this case, from the file
   ``reduce_acmr.yaml``) and a simulation object will be created.

2. The ``simulation.setup()`` method will call the ``setup()`` method for each
   of the MSLT components defined in the model specification.

   .. note:: This is where components will load data tables, register event
      handlers, etc.

3. The initial population is created (typically by the
   :class:`~vivarium_public_health.mslt.population.BasePopulation` component).

4. The time-steps will be simulated, with each time-step triggering the
   following events in turn:

   1. ``"time_step__prepare"``: The
      :class:`~vivarium_public_health.mslt.delay.DelayedRisk` component uses
      this event to account for transitions between exposure categories (i.e.,
      uptake, cessation, and transitions between tunnel states).
      The :class:`~vivarium_public_health.mslt.disease.Disease` component uses
      this event to update disease prevalence and mortality for both the BAU
      and intervention scenarios, so that mortality and morbidity adjustments
      can be calculated.
      The
      :class:`~vivarium_public_health.mslt.intervention.TobaccoEradication`
      component uses this event to move current smokers to the **0 years
      post-cessation** exposure category when tobacco is eradicated.

   2. ``"time_step"``: The
      :class:`~vivarium_public_health.mslt.population.BasePopulation`
      component uses this event to remove cohorts once they've reached the
      maximum age (110 years).
      The :class:`~vivarium_public_health.mslt.population.Mortality` component
      uses this event to calculate the number of deaths and survivors at each
      time-step.
      The :class:`~vivarium_public_health.mslt.population.Disability`
      component uses this event to calculate the HALYs for each cohort for
      both the BAU and intervention scenarios.

   3. ``"time_step__cleanup"``: no MSLT components respond to this event.

   4. ``"collect_metrics"``: the observer components will record relevant
      population details at the end of each time-step.

5. The simulation will trigger the ``"simulation_end"`` event and finish.
   The observer components use this event to write output tables to disk.

.. toctree::
   :maxdepth: 2
   :caption: Model analysis

   uncertainty

.. toctree::
   :maxdepth: 2
   :caption: Writing your own MSLT components

   custom_intervention
   custom_observer
   custom_risk_factor
