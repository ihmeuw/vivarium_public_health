.. _population_fertility_concept:

=========
Fertility
=========

.. contents::
   :depth: 2
   :local:
   :backlinks: none

Fertility components introduce new :term:`simulants <Simulant>` into the
simulation during runtime, modeling the arrival of newborns. The population
package ships three fertility implementations that span a range of complexity
and data requirements. Models choose the one that best matches their needs by
including the appropriate component.

Unlike
:class:`~vivarium_public_health.population.base_population.BasePopulation`,
which is always required, fertility components are optional — a model that does
not need births simply omits them.

Fertility Model Options
-----------------------

.. list-table::
   :widths: 30 20 50
   :header-rows: 1

   * - Component
     - Stochasticity
     - Typical Use
   * - ``FertilityDeterministic``
     - Deterministic
     - Adds a fixed, known number of new simulants per year. Useful for
       controlled experiments or simple projections.
   * - ``FertilityCrudeBirthRate``
     - Stochastic (Poisson)
     - Uses population-level live-birth covariate data and the scaling
       relationship between the modeled and true population to determine
       births each time step.
   * - ``FertilityAgeSpecificRates``
     - Stochastic (per-simulant hazard)
     - Applies age-specific fertility hazards to individual living females,
       creates newborn simulants with parent links, and enforces a gestational
       spacing assumption via ``last_birth_time``.

Crude Birth Rate Model
----------------------

:class:`~vivarium_public_health.population.add_new_birth_cohorts.FertilityCrudeBirthRate`
computes the expected number of births from live-birth covariate data and the
ratio of the simulation's ``population.population_size`` to the true population
count in the artifact. On each time step it draws from a Poisson distribution
to decide how many new simulants to create.

The component respects two time-dependence toggles
(``fertility.time_dependent_live_births`` and
``fertility.time_dependent_population_fraction``) that control whether birth
rates and population fractions are held constant at the simulation start year
or allowed to vary.

Age-Specific Fertility Model
----------------------------

:class:`~vivarium_public_health.population.add_new_birth_cohorts.FertilityAgeSpecificRates`
operates at the individual simulant level. It registers a fertility rate
:ref:`pipeline <values_concept>` and, on each time step, evaluates whether each
eligible female simulant gives birth based on a hazard draw. Eligibility
includes an age window and a gestational spacing assumption tied to the
``last_birth_time`` column.

Newborn simulants are created through the framework's simulant creator and
receive a ``parent_id`` linking them back to their mother.

Configuration
-------------

Different fertility components expose different configuration keys.

.. list-table::
   :widths: 35 20 45
   :header-rows: 1

   * - Key
     - Default
     - Effect
   * - ``fertility.number_of_new_simulants_each_year``
     - ``1000``
     - Fixed annual births used by ``FertilityDeterministic``.
   * - ``fertility.time_dependent_live_births``
     - ``True``
     - Whether ``FertilityCrudeBirthRate`` uses year-varying birth data.
   * - ``fertility.time_dependent_population_fraction``
     - ``False``
     - Whether ``FertilityCrudeBirthRate`` uses year-varying population
       fractions.
   * - ``fertility_age_specific_rates.data_sources.age_specific_fertility_rate``
     - ``load_age_specific_fertility_rate_data``
     - Data source for ``FertilityAgeSpecificRates``.

See Also
--------

- :ref:`population_dynamics_concept`
- :ref:`population_mortality_concept`
- :mod:`vivarium_public_health.population.add_new_birth_cohorts`
