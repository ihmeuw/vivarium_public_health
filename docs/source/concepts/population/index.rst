.. _population_concept:

==========
Population
==========

.. contents::
   :depth: 2
   :local:
   :backlinks: none

.. toctree::
   :hidden:

   base_population
   mortality
   fertility

The ``vivarium_public_health`` population package provides the
:ref:`components <components_concept>` that create, evolve, and retire
:term:`simulants <Simulant>` over the course of a simulation. Where the core
:mod:`vivarium` framework supplies the
:ref:`population management <population_concept>` machinery — the state table,
population views, and simulant creation — the public health population package
builds on that machinery to model real-world demographics: sampling age, sex,
and location from empirical data, aging simulants forward through time, removing
them via mortality, and optionally introducing new simulants through fertility.

The package is organized around three cooperating concerns:

1. **Base population** — initialization of demographic attributes from artifact
   data, deterministic aging on each time step, and age-based untracking.
2. **Mortality** — cause-deleted all-cause mortality with support for modeled
   and unmodeled cause-specific contributions.
3. **Fertility** — optional introduction of newborn simulants via deterministic,
   crude-birth-rate, or age-specific-rate models.

Each concern has a dedicated concept page linked above. Configuration keys and
data expectations are documented alongside the components they govern.

Data Expectations
-----------------

Population components expect the following artifact data to be available:

- ``population.structure`` — demographic composition by age, sex, and location
- ``cause.all_causes.cause_specific_mortality_rate`` — background mortality
- ``population.theoretical_minimum_risk_life_expectancy`` — for years-of-life-lost calculations
- ``covariate.live_births_by_sex.estimate`` — required by ``FertilityCrudeBirthRate``
- ``covariate.age_specific_fertility_rate.estimate`` — required by ``FertilityAgeSpecificRates``