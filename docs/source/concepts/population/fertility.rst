.. _fertility_concept:

=========
Fertility
=========

.. contents::
   :depth: 2
   :local:
   :backlinks: none

Fertility components introduce new :term:`simulants <Simulant>` into the simulation 
during runtime, e.g. to model the arrival of newborns or perhaps immigration. Three 
fertility implementations that span a range of complexity and data requirements currently exist.

Note that fertility components are optional — a model that does not need births simply omits them.

.. list-table::
   :widths: 30 20 50
   :header-rows: 1

   * - Component
     - Stochasticity
     - Typical Use
   * - :class:`~vivarium_public_health.population.add_new_birth_cohorts.FertilityDeterministic`
     - Deterministic
     - Adds a fixed, known number of new simulants per year. Useful for
       controlled experiments or simple projections.
   * - :class:`~vivarium_public_health.population.add_new_birth_cohorts.FertilityCrudeBirthRate`
     - Stochastic (Poisson)
     - Uses population-level live-birth covariate data and the scaling
       relationship between the modeled and true population to determine
       births each time step.
   * - :class:`~vivarium_public_health.population.add_new_birth_cohorts.FertilityAgeSpecificRates`
     - Stochastic (per-simulant hazard)
     - Applies age-specific fertility hazards to individual living females,
       creates newborn simulants with parent links, and enforces a gestational
       spacing assumption via ``last_birth_time``.

Deterministic Fertility Model
-----------------------------

:class:`~vivarium_public_health.population.add_new_birth_cohorts.FertilityDeterministic`
is the simplest fertility model. It adds a fixed number of new simulants per
year, controlled by the ``fertility.number_of_new_simulants_each_year``
configuration key. On each time step the component scales that
annual count by the step size and accumulates any fractional remainder across
steps so that no births are lost to rounding.

All newborns enter the simulation at age zero.

:term:`Crude Birth Rate` Model
------------------------------

:class:`~vivarium_public_health.population.add_new_birth_cohorts.FertilityCrudeBirthRate`
computes the expected number of births from live-birth covariate data and the
ratio of the simulation's ``population.population_size`` to the true population
count in the artifact. On each time step it draws from a Poisson distribution
to decide how many new simulants to create.

The component respects two time-dependence toggles (``fertility.time_dependent_live_births`` 
and ``fertility.time_dependent_population_fraction``) that control whether birth
rates and population fractions are held constant at the simulation start year
or allowed to vary.

:term:`Age-Specific Fertility Rate` Model
-----------------------------------------

:class:`~vivarium_public_health.population.add_new_birth_cohorts.FertilityAgeSpecificRates`
operates at the individual simulant level. It registers a :term:`fertility rate <Age-Specific Fertility Rate>`
:ref:`attribute pipeline <values_concept>` and, on each time step, evaluates whether each
eligible female simulant gives birth based on a hazard draw. Eligibility
includes an age window and a gestational spacing assumption tied to the
``last_birth_time`` column.

Newborn simulants are created through the framework's simulant creator and
receive a ``parent_id`` linking them back to their mother.

See Also
--------

- :ref:`base_population_concept`
- :ref:`mortality_concept`
- :mod:`vivarium_public_health.population.add_new_birth_cohorts`
