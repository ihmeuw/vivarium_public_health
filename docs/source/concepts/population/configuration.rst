.. _population_configuration_concept:

========================
Population Configuration
========================

.. contents::
   :depth: 2
   :local:
   :backlinks: none

Population Keys
---------------

The base population behavior is configured under ``population``.

.. list-table:: **Population Configuration**
   :widths: 35 20 45
   :header-rows: 1

   * - Key
     - Default
     - Effect
   * - ``population.initialization_age_min``
     - ``0``
     - Minimum age used for initial population creation.
   * - ``population.initialization_age_max``
     - ``125``
     - Maximum age used for initial population creation.
   * - ``population.include_sex``
     - ``Both``
     - Restrict initialization to ``Female``, ``Male``, or ``Both``.
   * - ``population.untracking_age``
     - ``None``
     - If set, simulants are untracked once age is at or above this threshold.

Deprecated keys ``age_start``, ``age_end``, and ``exit_age`` are still
recognized for compatibility but should be replaced by the keys above.

Mortality Keys
--------------

Mortality behavior is configured under ``mortality``.

.. list-table:: **Mortality Configuration**
   :widths: 35 20 45
   :header-rows: 1

   * - Key
     - Default
     - Effect
   * - ``mortality.data_sources.all_cause_mortality_rate``
     - ``cause.all_causes.cause_specific_mortality_rate``
     - Source for background all-cause mortality.
   * - ``mortality.data_sources.unmodeled_cause_specific_mortality_rate``
     - ``load_unmodeled_csmr``
     - Source for aggregated unmodeled CSMR.
   * - ``mortality.data_sources.life_expectancy``
     - ``population.theoretical_minimum_risk_life_expectancy``
     - Source for years of life lost calculations.
   * - ``mortality.unmodeled_causes``
     - ``[]``
     - Cause names to include in unmodeled CSMR aggregation.

Fertility Keys
--------------

Different fertility components expose different keys.

.. list-table:: **Fertility Configuration**
   :widths: 35 20 45
   :header-rows: 1

   * - Key
     - Default
     - Effect
   * - ``fertility.number_of_new_simulants_each_year``
     - ``1000``
     - Used by ``FertilityDeterministic``.
   * - ``fertility.time_dependent_live_births``
     - ``True``
     - Used by ``FertilityCrudeBirthRate``.
   * - ``fertility.time_dependent_population_fraction``
     - ``False``
     - Used by ``FertilityCrudeBirthRate``.
   * - ``fertility_age_specific_rates.data_sources.age_specific_fertility_rate``
     - ``load_age_specific_fertility_rate_data``
     - Used by ``FertilityAgeSpecificRates``.

Data Expectations
-----------------

Population components expect artifact data for at least:

- ``population.structure``
- ``cause.all_causes.cause_specific_mortality_rate``
- ``population.theoretical_minimum_risk_life_expectancy``
- ``covariate.live_births_by_sex.estimate`` (for crude-birth-rate fertility)
- ``covariate.age_specific_fertility_rate.estimate`` (for age-specific fertility)

See Also
--------

- :mod:`vivarium_public_health.population`
- :mod:`vivarium_public_health.population.mortality`
- :mod:`vivarium_public_health.population.add_new_birth_cohorts`
