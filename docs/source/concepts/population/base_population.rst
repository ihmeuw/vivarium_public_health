.. _population_dynamics_concept:

===============
Base Population
===============

.. contents::
   :depth: 2
   :local:
   :backlinks: none

The :class:`~vivarium_public_health.population.base_population.BasePopulation`
component is the foundation of the public health population package. It is
responsible for two distinct jobs: *initializing* new simulants with
demographically consistent attributes, and *aging* them forward on each time
step. A companion component,
:class:`~vivarium_public_health.population.base_population.AgeOutSimulants`,
handles removing simulants that exceed a configured age threshold.

Because :mod:`vivarium` itself is agnostic to the meaning of the columns in the
:ref:`state table <population_concept>`, it is this component that gives
simulants their demographic identity.

.. _population_initialization_concept:

Initialization
--------------

When the framework's simulant creator function fires, ``BasePopulation``'s
initializer assigns each new simulant the following columns:

- ``age``
- ``sex``
- ``location``
- ``entrance_time``
- ``exit_time``

The ``entrance_time`` marks when the simulant enters the simulation. The
``exit_time`` is initially set to :data:`pandas.NaT` and is later updated by
other components (most commonly
:class:`~vivarium_public_health.population.mortality.Mortality`) when the
simulant leaves the simulation.

Demographic Sampling
++++++++++++++++++++

The initializer loads ``population.structure`` data from the artifact and
computes conditional sampling distributions with
:func:`~vivarium_public_health.population.data_transformations.assign_demographic_proportions`.
Three probability views are produced:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Quantity
     - Meaning
   * - ``P(sex, location, age| year)``
     - Joint distribution used to draw simulant demographics for a given year.
   * - ``P(sex, location | age, year)``
     - Conditional distribution used when a fixed initial age is specified.
   * - ``P(age | year, sex, location)``
     - Conditional distribution used for smoothing ages within demographic
       strata.

The initialization process selects the closest available reference year less
than or equal to the simulant creation year.  This allows long-running
simulations to reuse available artifact years without requiring data for every
simulation year.

Fixed Age vs. Age Range
+++++++++++++++++++++++

Initialization supports two modes, determined by the values of
``age_start`` and ``age_end`` passed through the ``user_data`` on the
simulant data object (or falling back to
the ``population.initialization_age_min`` and
``population.initialization_age_max`` configuration keys):

- **Fixed age** (``age_start == age_end``): All simulants are placed in the
  single age bin containing that age, and then smoothed uniformly within the
  bin.  Sex and location are drawn from
  ``P(sex, location | age, year)``.
- **Age range** (``age_start != age_end``): The age distribution is clipped to
  the requested range, bins are selected probabilistically from
  ``P(sex, location, age| year)``, and ages are smoothed within the
  selected bins.

Both paths use the :ref:`common random number <crn_concept>` framework through
dedicated randomness streams so that initialization remains reproducible across
simulation branches.

Scaled Population
+++++++++++++++++

:class:`~vivarium_public_health.population.base_population.ScaledPopulation`
extends the base initialization flow by multiplying the population structure by
an external scaling factor before sampling. This is useful when the number of
simulants in the model represents a known fraction of the true population size
and downstream components (like crude-birth-rate fertility) need to reason about
that relationship.

The scaling factor can be provided as a :class:`pandas.DataFrame` directly or
as a string artifact key that is loaded at runtime.

Time Step Behavior
------------------

On each ``time_step`` :ref:`event <event_concept>`, ``BasePopulation`` advances
the ``age`` of every living simulant by the event's step size. This update is
deterministic and ensures all downstream components see a consistent age state
within a single time step.

Aging Out and Untracking
------------------------

:class:`~vivarium_public_health.population.base_population.AgeOutSimulants`
runs during ``time_step__cleanup``. When ``population.untracking_age`` is
configured, any simulant whose age meets or exceeds that threshold is marked
``is_aged_out = True`` and subsequently untracked by the framework. This
provides a clean way to bound the active population for models focused on
specific age windows.

.. _population_configuration_concept:

Configuration
-------------

The base population behavior is controlled by keys under ``population``.

.. list-table::
   :widths: 35 20 45
   :header-rows: 1

   * - Key
     - Default
     - Effect
   * - ``population.initialization_age_min``
     - ``0``
     - Lower bound of the age range used when creating the initial population.
   * - ``population.initialization_age_max``
     - ``125``
     - Upper bound of the age range used when creating the initial population.
   * - ``population.include_sex``
     - ``Both``
     - Restrict initialization to ``Female``, ``Male``, or ``Both``.
   * - ``population.untracking_age``
     - ``None``
     - If set, simulants are untracked once their age reaches this value.

The deprecated keys ``age_start``, ``age_end``, and ``exit_age`` are still
recognized for backward compatibility but should be replaced by the keys above.

See Also
--------

- :ref:`population_mortality_concept`
- :ref:`population_fertility_concept`
- :mod:`vivarium_public_health.population.base_population`
- :mod:`vivarium_public_health.population.data_transformations`
