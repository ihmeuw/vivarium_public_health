===================================
Population Structures and Fertility
===================================

:mod:`vivarium_public_health` provides several components for creating and
managing simulated populations. This tutorial demonstrates the minimal
configuration required for each approach.

.. contents::
   :local:
   :depth: 2

.. testsetup:: *

   import numpy as np
   import pandas as pd
   from vivarium import InteractiveContext
   from vivarium_public_health.population import *
   from vivarium_public_health._example_data import *
   base_plugins = BASE_PLUGINS


Overview
--------

There are two categories of population components:

**Initial population** components create the starting set of simulants when the
simulation begins:

- :class:`~vivarium_public_health.population.base_population.BasePopulation` - the standard
  component that samples simulants from demographic data.
- :class:`~vivarium_public_health.population.base_population.ScaledPopulation` - a variant
  that rescales the demographic data before sampling.

**Fertility** components add new simulants during the simulation:

- :class:`~vivarium_public_health.population.add_new_birth_cohorts.FertilityDeterministic` - adds a
  fixed number of births per year.
- :class:`~vivarium_public_health.population.add_new_birth_cohorts.FertilityCrudeBirthRate` - adds
  births based on a population-level crude birth rate without accounting for
  the age or sex composition of the population.
- :class:`~vivarium_public_health.population.add_new_birth_cohorts.FertilityAgeSpecificRates` -
  adds births at the individual level based on age-specific fertility rates.

.. note::

   :class:`~vivarium_public_health.population.base_population.BasePopulation`
   includes three sub-components:
   :class:`~vivarium_public_health.population.mortality.Mortality`,
   ``AgeOutSimulants``, and ``Disability``. You do not need to add these.


Common Setup
------------

In a vivarium simulation, data can be supplied through a **data artifact** -
an HDF file that you build with all the input data your model needs. To
keep the code blocks in this tutorial simple, we use an example artifact for
keys that must come from the artifact, and supply the rest through
configuration overrides (see `Data sources`_). The `Artifact Data Format`_
section shows the expected key names and column layouts for every data key
so that you know exactly what to put in your own artifact.

Every code example in this tutorial uses two helpers imported from
:mod:`vivarium_public_health._example_data`:

.. testcode::

   from vivarium_public_health._example_data import BASE_PLUGINS, make_base_config

   # BASE_PLUGINS overrides the data plugin to use ExampleArtifactManager,
   # which serves example data from memory instead of requiring a real HDF file.
   # Pass it as plugin_configuration to InteractiveContext.
   base_plugins = BASE_PLUGINS

   # make_base_config() returns a configuration with sensible defaults for
   # time range, step size, and randomness key columns.
   config = make_base_config()


Artifact Data Format
--------------------

This section documents the **key name** and **column layout** that each
population component expects. Some components also support a
``data_sources`` configuration pattern that lets you override individual
keys with a scalar, DataFrame, or callable without rebuilding the artifact
(see `Data sources`_).


Data keys
^^^^^^^^^

The table below lists every data key used by the population components.
Keys marked **artifact-required** must be present in the artifact - the
component loads them directly and they cannot be replaced via configuration.
Keys marked **configurable** can be overridden in the ``data_sources``
section of the configuration (see `Data sources`_); the artifact key shown
is simply the default.

.. list-table::
   :header-rows: 1

   * - Key
     - Index columns
     - Value columns
     - Used by
     - Configurable?
   * - ``population.structure``
     - age, sex, year, location
     - ``value`` (population count)
     - :class:`~vivarium_public_health.population.base_population.BasePopulation`,
       :class:`~vivarium_public_health.population.base_population.ScaledPopulation`
     - No (artifact-required)
   * - ``population.age_bins``
     - One row per age group
     - ``age_start``, ``age_end``, ``age_group_name``
     - :class:`~vivarium_public_health.population.base_population.BasePopulation`
     - No (artifact-required)
   * - ``population.location``
     - *(scalar)*
     - A string (e.g. ``"Kenya"``)
     - :class:`~vivarium_public_health.population.base_population.BasePopulation`
     - No (artifact-required)
   * - ``cause.all_causes.cause_specific_mortality_rate``
     - age, sex, year
     - ``value`` (rate)
     - :class:`~vivarium_public_health.population.mortality.Mortality`
     - Yes - ``mortality.data_sources.all_cause_mortality_rate``
   * - ``population.theoretical_minimum_risk_life_expectancy``
     - age
     - ``value`` (years of remaining life)
     - :class:`~vivarium_public_health.population.mortality.Mortality`
     - Yes - ``mortality.data_sources.life_expectancy``
   * - ``covariate.live_births_by_sex.estimate``
     - year, sex, ``parameter``
     - ``value``
     - :class:`~vivarium_public_health.population.add_new_birth_cohorts.FertilityCrudeBirthRate`
     - No (artifact-required)
   * - ``covariate.age_specific_fertility_rate.estimate``
     - age, sex, year, ``parameter``
     - ``value``
     - :class:`~vivarium_public_health.population.add_new_birth_cohorts.FertilityAgeSpecificRates`
     - Yes - ``fertility_age_specific_rates.data_sources.age_specific_fertility_rate``


Data sources
^^^^^^^^^^^^

Some components support a ``data_sources`` configuration pattern that lets
you override individual data keys without rebuilding the artifact. This is
especially useful during development or for simple tutorial examples like
the ones in this page. Components that support it declare their data needs
in ``configuration_defaults``; by default each key points to the
corresponding artifact key. You can override any of them with:

- **Scalar** (int or float) - broadcast a constant value to all simulants.
- **DataFrame** - use the DataFrame directly.
- **Callable** - call the function at setup time to produce the data.
- **Artifact key** (string) - load a different key from the artifact.

For example, :class:`~vivarium_public_health.population.mortality.Mortality` declares
three configurable data sources:

.. code-block:: yaml

   # Default configuration (loaded from the artifact):
   mortality:
     data_sources:
       all_cause_mortality_rate: "cause.all_causes.cause_specific_mortality_rate"
       life_expectancy: "population.theoretical_minimum_risk_life_expectancy"
       unmodeled_cause_specific_mortality_rate: <internal method>

.. note::

   The ``unmodeled_cause_specific_mortality_rate`` default is shown as
   ``<internal method>`` because it is a bound Python method that cannot be
   expressed in YAML.

Any of these can be overridden in the simulation configuration:

.. code-block:: yaml

   # Override with a scalar - no artifact needed for this key:
   configuration:
     mortality:
       data_sources:
         all_cause_mortality_rate: 0.01
         life_expectancy: 80.0

The component sections below show the first few rows of the data each
component expects, so you can see the concrete layout.


BasePopulation
--------------

:class:`~vivarium_public_health.population.base_population.BasePopulation` is the standard way
to create an initial population. It loads a population structure from the data
artifact and samples simulants whose age, sex, and location distributions match
the source data.

``BasePopulation`` itself requires ``population.structure``,
``population.age_bins``, and ``population.location`` to be present in the
artifact (these are artifact-required keys). Its
:class:`~vivarium_public_health.population.mortality.Mortality` sub-component supports
the ``data_sources`` configuration, so mortality rates and life expectancy
can be overridden with scalars or DataFrames - which is what we do in the
tutorial examples below.


Artifact data consumed by BasePopulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``BasePopulation`` and its sub-components load the following keys from the
artifact. The examples below use the data builders from the
:mod:`~vivarium_public_health._example_data` module; a production artifact has
the same column layout but with real GBD values.

.. testcode::

   from vivarium_public_health._example_data import (
       population_structure,
       age_bins,
       theoretical_minimum_risk_life_expectancy,
   )

   # population.structure - population counts per demographic cell.
   pop_structure = population_structure()
   print(pop_structure.query("year_start == 1990").head(6).to_string(index=False))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

    age_start  age_end    sex  year_start  year_end location     value
     0.000000 0.019178   Male        1990      1991    Kenya  1.917808
     0.000000 0.019178 Female        1990      1991    Kenya  1.917808
     0.019178 0.076712   Male        1990      1991    Kenya  5.753425
     0.019178 0.076712 Female        1990      1991    Kenya  5.753425
     0.076712 1.000000   Male        1990      1991    Kenya 92.328767
     0.076712 1.000000 Female        1990      1991    Kenya 92.328767

.. testcode::

   # population.age_bins - defines the age groups used by the demographic data.
   print(age_bins().head(5).to_string(index=False))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

    age_start   age_end age_group_name
     0.000000  0.019178 Early Neonatal
     0.019178  0.076712  Late Neonatal
     0.076712  1.000000  Post Neonatal
     1.000000  5.000000         1 to 4
     5.000000 10.000000         5 to 9

.. testcode::

   # population.location - a scalar string identifying the simulated location.
   # In the example data this is the string "Kenya".

   # population.theoretical_minimum_risk_life_expectancy - remaining life
   # expectancy by age, used by the Mortality sub-component to compute years
   # of life lost. Indexed only by age (no sex, year, or location).
   tmrle = theoretical_minimum_risk_life_expectancy()
   print(tmrle.head(5).to_string(index=False))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

    age_start  age_end  value
          0.0      1.0   98.0
          1.0      2.0   98.0
          2.0      3.0   98.0
          3.0      4.0   98.0
          4.0      5.0   98.0


Default configuration
^^^^^^^^^^^^^^^^^^^^^

The absolute minimum is a ``population_size``. Everything else has sensible
defaults (ages 0–125, both sexes, no age-out):

.. testcode::

   from vivarium import InteractiveContext
   from vivarium_public_health.population import BasePopulation

   config = make_base_config()
   config.update(
       {
           "population": {
               "population_size": 10_000,
           },
           # Override mortality to zero so simulants don't die during
           # this demonstration.
           "mortality": {"data_sources": {"all_cause_mortality_rate": 0}},
       },
       layer="override",
   )

   sim = InteractiveContext(
       components=[BasePopulation()],
       configuration=config,
       plugin_configuration=base_plugins,
   )

   pop = sim.get_population(["age", "sex", "location"])
   assert len(pop) == 10_000
   assert pop["age"].min() >= 0
   assert pop["age"].max() <= 125
   assert set(pop["sex"].unique()) == {"Male", "Female"}
   print(f"Population: {len(pop)}")

.. testoutput::

   Population: 10000


Custom age range
^^^^^^^^^^^^^^^^

Use ``initialization_age_min`` and ``initialization_age_max`` to restrict the
age range of the initial population. This is the most common customization:

.. testcode::

   config = make_base_config()
   config.update(
       {
           "population": {
               "population_size": 10_000,
               "initialization_age_min": 0,
               "initialization_age_max": 5,
           },
           "mortality": {"data_sources": {"all_cause_mortality_rate": 0}},
       },
       layer="override",
   )

   sim = InteractiveContext(
       components=[BasePopulation()],
       configuration=config,
       plugin_configuration=base_plugins,
   )

   pop = sim.get_population(["age"])
   assert pop["age"].min() >= 0
   assert pop["age"].max() < 5
   print(f"All ages in [0, 5): {pop['age'].min() >= 0 and pop['age'].max() < 5}")

.. testoutput::

   All ages in [0, 5): True


Single-age initialization (newborns)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When ``initialization_age_min`` equals ``initialization_age_max``, all
simulants start at the same age. This can be used with fertility
components to represent a cohort of newborns:

.. testcode::

   config = make_base_config()
   config.update(
       {
           "population": {
               "population_size": 1_000,
               "initialization_age_min": 0,
               "initialization_age_max": 0,
           },
           "mortality": {"data_sources": {"all_cause_mortality_rate": 0}},
       },
       layer="override",
   )

   sim = InteractiveContext(
       components=[BasePopulation()],
       configuration=config,
       plugin_configuration=base_plugins,
   )

   pop = sim.get_population(["age"])
   # All simulants are newborns; ages are smoothed within the first time step.
   assert pop["age"].max() < 1.0
   print(f"All simulants under 1 year old: {pop['age'].max() < 1.0}")

.. testoutput::

   All simulants under 1 year old: True


Filtering by sex
^^^^^^^^^^^^^^^^^

The ``include_sex`` option restricts the population to a single sex. Valid
values are ``"Male"``, ``"Female"``, or ``"Both"`` (the default):

.. testcode::

   config = make_base_config()
   config.update(
       {
           "population": {
               "population_size": 10_000,
               "include_sex": "Female",
           },
           "mortality": {"data_sources": {"all_cause_mortality_rate": 0}},
       },
       layer="override",
   )

   sim = InteractiveContext(
       components=[BasePopulation()],
       configuration=config,
       plugin_configuration=base_plugins,
   )

   pop = sim.get_population(["sex"])
   assert (pop["sex"] == "Female").all()
   print(f"All Female: {len(pop)}")

.. testoutput::

   All Female: 10000


Aging out of a simulation
^^^^^^^^^^^^^^^^^^^^^^^^^

Setting ``untracking_age`` causes simulants to be removed from the tracked
population once they reach that age (see the
`vivarium population concepts <https://vivarium.readthedocs.io/en/latest/concepts/population.html>`_
documentation for more on untracking). This is useful when a model only
cares about a specific age window. The ``is_aged_out`` column is populated
by the ``AgeOutSimulants`` sub-component when ``untracking_age`` is set:

.. testcode::

   config = make_base_config()
   config.update(
       {
           "population": {
               "population_size": 10_000,
               "initialization_age_min": 4,
               "initialization_age_max": 4,
               "untracking_age": 5,
           },
           "time": {"step_size": 100},
           "mortality": {"data_sources": {"all_cause_mortality_rate": 0}},
       },
       layer="override",
   )

   sim = InteractiveContext(
       components=[BasePopulation()],
       configuration=config,
       plugin_configuration=base_plugins,
   )

   # All 4-year-olds at the start
   print(f"Tracked: {len(sim.get_population(['age']))}")

.. testoutput::

   Tracked: 10000

.. testcode::

   # After taking 6 steps of 100 days (~1.6 years), everyone has aged past 5
   sim.take_steps(number_of_steps=6)
   pop = sim.get_population(["is_aged_out", "exit_time"], include_untracked=True)
   print(f"Aged out: {pop['is_aged_out'].sum()}")

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   Aged out: 10000


Configuration summary for BasePopulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Key
     - Default
     - Description
   * - ``population.population_size``
     - 10000
     - Number of simulants to create.
   * - ``population.initialization_age_min``
     - 0
     - Minimum age (years) for the initial population.
   * - ``population.initialization_age_max``
     - 125
     - Maximum age (years) for the initial population.
   * - ``population.include_sex``
     - ``"Both"``
     - ``"Male"``, ``"Female"``, or ``"Both"``.
   * - ``population.untracking_age``
     - ``None``
     - Age at which simulants are removed from the tracked population.
       ``None`` means no age-out.
   * - ``mortality.data_sources.all_cause_mortality_rate``
     - ``"cause.all_causes.cause_specific_mortality_rate"``
     - All-cause mortality rate. Accepts a scalar, DataFrame, callable,
       or artifact key.
   * - ``mortality.data_sources.life_expectancy``
     - ``"population.theoretical_minimum_risk_life_expectancy"``
     - Remaining life expectancy by age. Accepts a scalar, DataFrame,
       callable, or artifact key.
   * - ``mortality.data_sources.unmodeled_cause_specific_mortality_rate``
     - internal method
     - CSMR for unmodeled causes. Accepts a scalar, DataFrame, callable,
       or artifact key.


ScaledPopulation
----------------

:class:`~vivarium_public_health.population.base_population.ScaledPopulation` works like
:class:`~vivarium_public_health.population.base_population.BasePopulation` but multiplies
the population structure by a scaling factor before sampling. This is useful
when simulants represent a subset of the real population (for example, only
the population eligible for an intervention).

The scaling factor can be either a :class:`pandas.DataFrame` with the same
demographic index as the population structure, or a string artifact key that
resolves to such a DataFrame.


``ScaledPopulation`` uses the same artifact keys as ``BasePopulation`` (see
`Artifact data consumed by BasePopulation`_) plus a user-supplied scaling
factor. The scaling factor can be passed as a :class:`pandas.DataFrame`
directly or as a string artifact key. Since we already have the data as a
DataFrame, we pass it directly to the constructor - no artifact write needed:

.. testcode::

   import numpy as np
   import pandas as pd
   from vivarium import InteractiveContext
   from vivarium_public_health.population import ScaledPopulation
   from vivarium_public_health._example_data import population_structure

   config = make_base_config()
   config.update(
       {
           "population": {
               "population_size": 100_000,
               "include_sex": "Both",
           },
           "mortality": {"data_sources": {"all_cause_mortality_rate": 0}},
       },
       layer="override",
   )

   # Build a scaling factor DataFrame - same demographic index as
   # population.structure, with a ``value`` per cell. Each cell's
   # population count is multiplied by its scaling value.
   scalar_data = (
       population_structure()
       .query("year_start == 1990")
       .drop(columns=["location"])
       .copy()
   )
   scalar_data["value"] = np.linspace(0.5, 2.0, len(scalar_data))
   print(scalar_data.head(6).to_string(index=False))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

    age_start  age_end    sex  year_start  year_end    value
     0.000000 0.019178   Male        1990      1991 0.500000
     0.000000 0.019178 Female        1990      1991 0.533333
     0.019178 0.076712   Male        1990      1991 0.566667
     0.019178 0.076712 Female        1990      1991 0.600000
     0.076712 1.000000   Male        1990      1991 0.633333
     0.076712 1.000000 Female        1990      1991 0.666667

.. testcode::

   # Pass the DataFrame directly - no need to write to the artifact.
   sim = InteractiveContext(
       components=[ScaledPopulation(scalar_data)],
       configuration=config,
       plugin_configuration=base_plugins,
   )

   pop = sim.get_population(["age", "sex"])
   assert len(pop) == 100_000
   assert set(pop["sex"].unique()) == {"Male", "Female"}
   print(f"Population: {len(pop)}, sexes: {sorted(pop['sex'].unique())}")

.. testoutput::

   Population: 100000, sexes: ['Female', 'Male']


Fertility Components
--------------------

Fertility components add new simulants during the simulation to model births.
They are paired with a population component such as
:class:`~vivarium_public_health.population.base_population.BasePopulation`.

.. note::

   All three fertility components create newborns with ``age_start=0`` and
   ``age_end=0``, meaning new simulants enter the simulation as newborns.


FertilityDeterministic
^^^^^^^^^^^^^^^^^^^^^^

:class:`~vivarium_public_health.population.add_new_birth_cohorts.FertilityDeterministic` adds a
fixed number of new simulants each year. This is the simplest fertility model
and does not require any artifact data.

.. testcode::

   from vivarium import InteractiveContext
   from vivarium_public_health.population import BasePopulation, FertilityDeterministic

   config = make_base_config()
   config.update(
       {
           "population": {
               "population_size": 1_000,
               "initialization_age_min": 0,
               "initialization_age_max": 100,
           },
           "time": {"step_size": 10},
           "mortality": {"data_sources": {"all_cause_mortality_rate": 0}},
           "fertility": {"number_of_new_simulants_each_year": 500},
       },
       layer="override",
   )

   sim = InteractiveContext(
       components=[BasePopulation(), FertilityDeterministic()],
       configuration=config,
       plugin_configuration=base_plugins,
   )

   sim.take_steps(number_of_steps=10)
   pop = sim.get_population(["age"])
   # Population grew from 1000 by ~500 * (100/365) ≈ 137 new simulants.
   assert len(pop) > 1_000
   print(f"Population grew: {len(pop) > 1_000}")

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   Population grew: True


FertilityCrudeBirthRate
^^^^^^^^^^^^^^^^^^^^^^^

:class:`~vivarium_public_health.population.add_new_birth_cohorts.FertilityCrudeBirthRate` models
births at the population level using a crude birth rate - the number of live
births per unit of population, regardless of age or sex structure. Because it
does not consider the demographic composition of the population, the number of
births depends only on the total population size and the overall birth rate.
This contrasts with
:class:`~vivarium_public_health.population.add_new_birth_cohorts.FertilityAgeSpecificRates`, which
models births at the individual level using rates that vary by age.

It requires ``initialization_age_min`` to be 0 and needs
``covariate.live_births_by_sex.estimate`` data in the artifact.

The artifact key ``covariate.live_births_by_sex.estimate`` should contain
a row for each year × sex combination:

.. testcode::

   from vivarium_public_health._example_data import live_births_by_sex

   # covariate.live_births_by_sex.estimate - each row gives the number of
   # live births for a year × sex combination.
   print(live_births_by_sex().head(6).to_string(index=False))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

    year_start  year_end    sex  parameter  value
          1990      1991 Female mean_value  500.0
          1990      1991   Male mean_value  500.0
          1991      1992 Female mean_value  500.0
          1991      1992   Male mean_value  500.0
          1992      1993 Female mean_value  500.0
          1992      1993   Male mean_value  500.0

This component's artifact key is artifact-required (it does not support
``data_sources`` overrides). The example artifact provides this data
automatically:

.. testcode::

   from vivarium import InteractiveContext
   from vivarium_public_health.population import BasePopulation, FertilityCrudeBirthRate

   config = make_base_config()
   config.update(
       {
           "population": {
               "population_size": 10_000,
               "initialization_age_min": 0,
               "initialization_age_max": 125,
           },
           "time": {"step_size": 10},
           "mortality": {"data_sources": {"all_cause_mortality_rate": 0}},
       },
       layer="override",
   )

   sim = InteractiveContext(
       components=[BasePopulation(), FertilityCrudeBirthRate()],
       configuration=config,
       plugin_configuration=base_plugins,
   )

   sim.take_steps(number_of_steps=10)
   pop = sim.get_population(["age"])
   assert len(pop) > 10_000
   print(f"Population grew: {len(pop) > 10_000}")

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   Population grew: True

.. important::

   ``FertilityCrudeBirthRate`` requires ``initialization_age_min`` to be 0.
   It will raise a ``ValueError`` if this is not the case.


FertilityAgeSpecificRates
^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~vivarium_public_health.population.add_new_birth_cohorts.FertilityAgeSpecificRates` models
fertility at the individual level. Each living female simulant who has not
given birth in the last nine months has a chance of giving birth determined
by age-specific fertility rates. Newborns are linked to their parent via a
``parent_id`` column.

By default this component loads ``covariate.age_specific_fertility_rate.estimate``
from the artifact. It also supports the ``data_sources`` configuration
pattern (see `Data sources`_), so you can override it with a scalar,
DataFrame, callable, or alternative artifact key. The expected data shape
is one row per age × year × sex × parameter combination:

.. testcode::

   from vivarium_public_health._example_data import age_specific_fertility_rate

   # covariate.age_specific_fertility_rate.estimate - each row gives a
   # fertility rate for an age × year × sex × parameter cell.
   asfr_data = age_specific_fertility_rate(rate=0.05)
   print(asfr_data.head(6).to_string(index=False))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

    year_start  year_end  age_start  age_end    sex   parameter  value
          1990      1991        0.0 0.019178 Female  mean_value   0.05
          1990      1991        0.0 0.019178 Female lower_value   0.05
          1990      1991        0.0 0.019178 Female upper_value   0.05
          1990      1991        0.0 0.019178   Male  mean_value   0.05
          1990      1991        0.0 0.019178   Male lower_value   0.05
          1990      1991        0.0 0.019178   Male upper_value   0.05

Because this component supports the ``data_sources`` configuration, the
tutorial example below supplies a constant rate directly instead of loading
from the artifact:

.. testcode::

   from vivarium import InteractiveContext
   from vivarium_public_health.population import BasePopulation, FertilityAgeSpecificRates

   config = make_base_config()
   config.update(
       {
           "population": {
               "population_size": 1_000,
               "initialization_age_min": 0,
               "initialization_age_max": 125,
           },
           "time": {"step_size": 10},
           "mortality": {"data_sources": {"all_cause_mortality_rate": 0}},
           # Override the fertility rate via data_sources configuration.
           "fertility_age_specific_rates": {
               "data_sources": {
                   "age_specific_fertility_rate": 0.05,
               },
           },
       },
       layer="override",
   )

   sim = InteractiveContext(
       components=[BasePopulation(), FertilityAgeSpecificRates()],
       configuration=config,
       plugin_configuration=base_plugins,
   )

   sim.take_steps(number_of_steps=100)
   pop = sim.get_population(["age", "parent_id", "last_birth_time"])

   # Newborns have a parent_id pointing to their mother
   newborns = pop[pop["parent_id"] >= 0]
   assert len(newborns) > 0
   print(f"Births occurred: {len(newborns) > 0}")

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

   Births occurred: True


Fertility configuration summary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Component
     - Configuration key
     - Artifact data required
     - Notes
   * - ``FertilityDeterministic``
     - ``fertility.number_of_new_simulants_each_year``
     - None
     - Simplest model; fixed birth count. Pure configuration.
   * - ``FertilityCrudeBirthRate``
     - ``fertility.time_dependent_live_births``,
       ``fertility.time_dependent_population_fraction``
     - ``covariate.live_births_by_sex.estimate``
     - Requires ``initialization_age_min == 0``. Artifact-required key
       (no ``data_sources`` support).
   * - ``FertilityAgeSpecificRates``
     - ``fertility_age_specific_rates.data_sources.age_specific_fertility_rate``
     - ``covariate.age_specific_fertility_rate.estimate`` (default)
     - Supports ``data_sources`` overrides (scalar, DataFrame, callable).
       Tracks parent–child relationships.
