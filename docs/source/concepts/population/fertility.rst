.. _population_fertility_concept:

====================
Population Fertility
====================

.. contents::
   :depth: 2
   :local:
   :backlinks: none

Overview
--------

Fertility components add newborn simulants during simulation runtime. The
population package supports three fertility models with different assumptions
and levels of detail.

Fertility Model Options
-----------------------

.. list-table:: **Fertility Component Comparison**
   :widths: 30 20 50
   :header-rows: 1

   * - Component
     - Stochasticity
     - Typical Use
   * - ``FertilityDeterministic``
     - Deterministic
     - Fixed, known number of new simulants per year.
   * - ``FertilityCrudeBirthRate``
     - Stochastic (Poisson)
     - Population-level birth modeling from crude birth data.
   * - ``FertilityAgeSpecificRates``
     - Stochastic (per-simulant hazard)
     - Individual-level fertility with age-specific rates and parent tracking.

Crude Birth Rate Model
----------------------

``FertilityCrudeBirthRate`` computes births from live-birth covariates and the
simulation scaling relationship between modeled and true population size.
It assumes births follow a Poisson process at each time step.

Age-Specific Fertility Model
----------------------------

``FertilityAgeSpecificRates`` applies fertility hazards to living females and
creates newborn simulants with parent links. Eligibility includes a gestational
spacing assumption based on ``last_birth_time``.

Configuration
-------------

Key fertility configuration includes:

- ``fertility.number_of_new_simulants_each_year``
- ``fertility.time_dependent_live_births``
- ``fertility.time_dependent_population_fraction``
- ``fertility_age_specific_rates.data_sources.age_specific_fertility_rate``

See :ref:`population_configuration_concept` for the combined population
configuration reference.

See Also
--------

- :mod:`vivarium_public_health.population.add_new_birth_cohorts`
- :ref:`population_dynamics_concept`
- :ref:`population_mortality_concept`
