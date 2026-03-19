.. _population_mortality_concept:

=========
Mortality
=========

.. contents::
   :depth: 2
   :local:
   :backlinks: none

The :class:`~vivarium_public_health.population.mortality.Mortality` component
models all-cause mortality with optional cause-specific contributions from both
explicitly modeled disease components and an aggregated set of unmodeled causes.
It is instantiated as a sub-component of
:class:`~vivarium_public_health.population.base_population.BasePopulation` and
participates in the ``time_step`` :ref:`event <event_concept>` to determine
which simulants die, record their cause of death, and calculate years of life
lost.

The component manages three columns in the
:ref:`state table <population_concept>`:

- ``is_alive`` — whether the simulant is still living
- ``cause_of_death`` — the cause assigned to a dying simulant
- ``years_of_life_lost`` — the residual life expectancy at death

Cause-Deleted Mortality
-----------------------

The effective mortality rate is constructed using a cause-deleted pattern.
At each time step the component computes:

.. math::

   \text{mortality\_rate} = \text{ACMR} - \text{modeled CSMR}
   - \text{unmodeled CSMR (raw)} + \text{unmodeled CSMR (modified)}

where:

- **ACMR** is all-cause mortality from the configured data source.
- **Modeled CSMR** is contributed by disease
  :ref:`components <components_concept>` that register as modifiers on the
  mortality rate :ref:`pipeline <values_concept>`.
- **Unmodeled CSMR** captures causes not explicitly modeled but still
  potentially affected by modeled risk factors.

The mortality rate pipeline is registered with a
:func:`~vivarium.framework.values.combiners.list_combiner` and
:func:`~vivarium.framework.values.post_processors.union_post_processor` so that
independent cause-specific contributions compose correctly.

During each time step, the combined mortality hazard is converted into death
events using random draws. For simulants that die, a cause is selected
probabilistically from the hazard-weighted causes.

Unmodeled Causes
----------------

The ``mortality.unmodeled_causes`` configuration key identifies causes that are
not explicitly modeled in the simulation but should still be represented in
mortality accounting. At setup, the component loads the cause-specific mortality
rate for each named cause and aggregates them into a single unmodeled CSMR term.
Risk :ref:`components <components_concept>` can then modify this term through
the ``affected_unmodeled`` rate pipeline, allowing risks to influence mortality
from causes that are not themselves full disease models.

Configuration
-------------

Mortality behavior is controlled by keys under ``mortality``.

.. list-table::
   :widths: 35 20 45
   :header-rows: 1

   * - Key
     - Default
     - Effect
   * - ``mortality.data_sources.all_cause_mortality_rate``
     - ``cause.all_causes.cause_specific_mortality_rate``
     - Artifact key or method name for background all-cause mortality.
   * - ``mortality.data_sources.unmodeled_cause_specific_mortality_rate``
     - ``load_unmodeled_csmr``
     - Artifact key or method name for the aggregated unmodeled CSMR.
   * - ``mortality.data_sources.life_expectancy``
     - ``population.theoretical_minimum_risk_life_expectancy``
     - Source for residual life expectancy used in YLL calculations.
   * - ``mortality.unmodeled_causes``
     - ``[]``
     - List of cause names to include when computing unmodeled CSMR.

See Also
--------

- :ref:`population_dynamics_concept`
- :ref:`population_fertility_concept`
- :mod:`vivarium_public_health.population.mortality`
