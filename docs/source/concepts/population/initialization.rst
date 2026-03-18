.. _population_initialization_concept:

=========================
Population Initialization
=========================

.. contents::
   :depth: 2
   :local:
   :backlinks: none

What Gets Initialized
---------------------

When new simulants are created, the population package assigns:

- ``age``
- ``sex``
- ``location``
- ``entrance_time``
- ``exit_time``

This initialization is performed by :class:`vivarium_public_health.population.base_population.BasePopulation`
(or :class:`vivarium_public_health.population.base_population.ScaledPopulation`).

Demographic Sampling
--------------------

The initializer loads ``population.structure`` data from the artifact and
builds demographic probabilities with
:func:`vivarium_public_health.population.data_transformations.assign_demographic_proportions`.
At a high level, it uses three related probability views:

.. list-table:: **Demographic Probability Views**
   :widths: 35 65
   :header-rows: 1

   * - Quantity
     - Meaning
   * - ``P(sex, location, age| year)``
     - Joint sampling distribution used to draw simulant demographics for a year.
   * - ``P(sex, location | age, year)``
     - Conditional distribution used for fixed-age initialization.
   * - ``P(age | year, sex, location)``
     - Conditional distribution used for age smoothing within demographic strata.

The initialization process uses the closest available reference year less than
or equal to the creation year. This allows long simulations to reuse available
artifact years without requiring a value for every simulation year.

Fixed Age vs Age Range
----------------------

Initialization supports two modes:

- **Fixed age**: ``age_start == age_end``. Simulants are assigned to the fixed
  age bin and then fuzzed smoothly inside the bin.
- **Age range**: ``age_start != age_end``. The age distribution is clipped to
  the requested range, bins are selected probabilistically, and ages are
  smoothed within selected bins.

Both modes use the common random number framework through dedicated randomness
streams so initialization remains reproducible.

Scaled Population
-----------------

:class:`vivarium_public_health.population.base_population.ScaledPopulation`
extends base initialization by multiplying the population structure by a scaling
factor before sampling. This is useful when simulants represent a known fraction
of the true population.

The scaling factor can be:

- A :class:`pandas.DataFrame` provided directly.
- A string artifact key loaded at runtime.

See Also
--------

- :mod:`vivarium_public_health.population.base_population`
- :mod:`vivarium_public_health.population.data_transformations`
