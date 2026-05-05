==================================
Disease Models and State Machines
==================================

:mod:`vivarium_public_health` provides a flexible framework for modelling
diseases as state machines. This tutorial demonstrates how to build disease
models from states and transitions, and how to use the pre-built models for
common disease progressions.

The disease components in this package extend the base
:class:`~vivarium.framework.state_machine.State` and
:class:`~vivarium.framework.state_machine.Transition` classes from
:mod:`vivarium.framework.state_machine`.

.. contents::
   :local:
   :depth: 2

.. testsetup:: *

   import numpy as np
   import pandas as pd
   from vivarium import InteractiveContext
   from vivarium_public_health.disease import *
   from vivarium_public_health.population import BasePopulation
   from vivarium_public_health._example_data import *
   base_plugins = BASE_PLUGINS


Overview
--------

A disease model in ``vivarium_public_health`` is a state machine. Each
simulant occupies exactly one disease state at any time within a given model,
and moves between states according to transition rules. A simulation may
contain multiple independent disease models, each tracking its own state
column.

For a detailed explanation of states, transitions, and pre-built models, see
the :ref:`disease model concept documentation <disease_model_concept>`.

.. |rarr| unicode:: U+2192
.. |harr| unicode:: U+2194


Common Setup
------------

In a vivarium simulation, data is normally supplied through a **data
artifact** - an HDF file containing all the input data your model needs.
This tutorial uses an in-memory example artifact (via ``BASE_PLUGINS``) that
serves simple data without requiring a real HDF file. Some examples also
pass data directly to constructors or override keys via the ``data_sources``
configuration (see `Data sources`_).

Every code example in this tutorial uses imports and helpers shown below.
To run any example in a standalone script, include all of these at the top:

.. testcode::

   from vivarium import InteractiveContext
   from vivarium_public_health.disease import *
   from vivarium_public_health.population import BasePopulation
   from vivarium_public_health._example_data import BASE_PLUGINS, make_base_config

   # BASE_PLUGINS overrides the data plugin to use ExampleArtifactManager,
   # which serves example data from memory instead of requiring a real HDF file.
   # Pass it as plugin_configuration to InteractiveContext.
   base_plugins = BASE_PLUGINS

   # make_base_config() returns a configuration with sensible defaults for
   # time range, step size, and randomness key columns.
   config = make_base_config()

The `Artifact Data Format`_ section shows the expected key names and column
layouts for every data key so that you know exactly what to put in your own
artifact.


Artifact Data Format
--------------------

This section documents the **key name** and **column layout** that each
disease component expects. Some components also support a ``data_sources``
configuration pattern that lets you override individual keys with a scalar,
DataFrame, or callable without rebuilding the artifact (see `Data sources`_).


Data keys
^^^^^^^^^

The table below lists every data key used by the disease components.
Keys marked **configurable** can be overridden in the ``data_sources``
section of the configuration; the artifact key shown is simply the default.

.. list-table::
   :header-rows: 1

   * - Key
     - Index columns
     - Value columns
     - Used by
     - Configurable?
   * - ``cause.{cause}.prevalence``
     - age, sex, year
     - ``value`` (fraction)
     - :class:`~vivarium_public_health.disease.state.DiseaseState`
     - Yes - ``{state}.data_sources.prevalence``
   * - ``cause.{cause}.birth_prevalence``
     - age, sex, year
     - ``value`` (fraction)
     - :class:`~vivarium_public_health.disease.state.DiseaseState` (neonatal models)
     - Yes - ``{state}.data_sources.birth_prevalence``
   * - ``cause.{cause}.disability_weight``
     - age, sex, year (or single row)
     - ``value`` (weight)
     - :class:`~vivarium_public_health.disease.state.DiseaseState`
     - Yes - ``{state}.data_sources.disability_weight``
   * - ``cause.{cause}.excess_mortality_rate``
     - age, sex, year
     - ``value`` (rate)
     - :class:`~vivarium_public_health.disease.state.DiseaseState`
     - Yes - ``{state}.data_sources.excess_mortality_rate``
   * - ``cause.{cause}.incidence_rate``
     - age, sex, year
     - ``value`` (rate)
     - :class:`~vivarium_public_health.disease.transition.RateTransition` (from
       susceptible state)
     - Yes - ``{transition}.data_sources.transition_rate``
   * - ``cause.{cause}.remission_rate``
     - age, sex, year
     - ``value`` (rate)
     - :class:`~vivarium_public_health.disease.transition.RateTransition` (from
       infected state)
     - Yes - ``{transition}.data_sources.transition_rate``
   * - ``cause.{cause}.cause_specific_mortality_rate``
     - age, sex, year
     - ``value`` (rate)
     - :class:`~vivarium_public_health.disease.model.DiseaseModel`
     - Yes - ``{cause}.data_sources.cause_specific_mortality_rate``


Artifact data shapes
^^^^^^^^^^^^^^^^^^^^

Most cause-level measures share the same column layout: one row per
age × sex × year combination with a ``value`` column. The
examples below use the data builders from the
:mod:`~vivarium_public_health._example_data` module; a production artifact
has the same column layout but with real GBD values.

.. testcode::

   from vivarium_public_health._example_data import (
       build_cause_table,
       disease_disability_weight,
       disease_restrictions,
   )

   # cause.{cause}.prevalence - fraction of population in the disease state.
   prevalence = build_cause_table(0.05)
   print(prevalence.query("year_start == 1990").head(6).to_string(index=False))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

    age_start   age_end    sex  year_start  year_end  value
     0.000000  0.019178   Male        1990      1991   0.05
     0.000000  0.019178 Female        1990      1991   0.05
     0.019178  0.076712   Male        1990      1991   0.05
     0.019178  0.076712 Female        1990      1991   0.05
     0.076712  1.000000   Male        1990      1991   0.05
     0.076712  1.000000 Female        1990      1991   0.05

.. testcode::

   # cause.{cause}.incidence_rate - rate of new infections per person-year.
   # Same column layout as prevalence.
   incidence = build_cause_table(0.001)
   print(incidence.query("year_start == 1990").head(2).to_string(index=False))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

    age_start  age_end    sex  year_start  year_end  value
          0.0 0.019178   Male        1990      1991  0.001
          0.0 0.019178 Female        1990      1991  0.001

.. testcode::

   # cause.{cause}.remission_rate - same layout as incidence_rate.
   # cause.{cause}.excess_mortality_rate - same layout as incidence_rate.
   # cause.{cause}.cause_specific_mortality_rate - same layout as above.

   # cause.{cause}.disability_weight - can be a single-row DataFrame.
   dw = disease_disability_weight(0.1)
   print(dw.to_string(index=False))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

    value
      0.1

.. testcode::

   # cause.{cause}.restrictions - a dict.
   restrictions = disease_restrictions()
   print(restrictions)

.. testoutput::

   {'yld_only': False}


Data sources
^^^^^^^^^^^^

Disease components support a ``data_sources`` configuration pattern that lets
you override individual data keys without rebuilding the artifact. This is
especially useful during development or for simple tutorial examples like the
ones in this page. Components that support it declare their data needs in
``configuration_defaults``; by default each key points to the corresponding
artifact key. You can override any of them with:

- **Scalar** (int or float) - broadcast a constant value to all simulants.
- **DataFrame** - use the DataFrame directly.
- **Callable** - call the function at setup time to produce the data.
- **Artifact key** (string) - load a different key from the artifact.

For example, :class:`~vivarium_public_health.disease.state.DiseaseState` declares
five configurable data sources:

.. code-block:: yaml

   # Default configuration (loaded from the artifact):
   {state_id}:
     data_sources:
       prevalence: "cause.{state_id}.prevalence"
       birth_prevalence: 0.0
       dwell_time: 0.0
       disability_weight: "cause.{state_id}.disability_weight"
       excess_mortality_rate: "cause.{state_id}.excess_mortality_rate"

Any of these can be overridden in the simulation configuration or passed
directly to the constructor:

.. code-block:: yaml

   # Override with scalars - no artifact needed for these keys:
   configuration:
     my_disease:
       data_sources:
         prevalence: 0.1
         disability_weight: 0.05
         excess_mortality_rate: 0.0

:class:`~vivarium_public_health.disease.transition.RateTransition` has a single
configurable data source:

.. code-block:: yaml

   # Default configuration:
   {transition_name}:
     data_sources:
       transition_rate: "cause.{cause}.incidence_rate"  # or remission_rate
     rate_conversion_type: "linear"  # or "exponential"

The component sections below show the first few rows of the data each
component expects, so you can see the concrete layout.


DiseaseModel
------------

:class:`~vivarium_public_health.disease.model.DiseaseModel` is the state machine
driver that ties states and transitions together. It initializes simulants
into disease states based on prevalence data and steps them through
transitions each time step.

``DiseaseModel`` adds the cause-specific mortality rate (CSMR) to the
simulation's overall mortality rate. The CSMR can be loaded from the
artifact or overridden via configuration or the constructor.


Default configuration
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   {cause}:
     data_sources:
       cause_specific_mortality_rate: <internal method>

.. note::

   The ``cause_specific_mortality_rate`` default is shown as
   ``<internal method>`` because it is a bound Python method that cannot be
   expressed in YAML.

The default loads from the artifact at
``cause.{cause}.cause_specific_mortality_rate``. Override with a scalar,
DataFrame, callable, or artifact key.


Building a model from scratch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most explicit way to create a disease model is to instantiate states,
wire up transitions, and wrap them in a
:class:`~vivarium_public_health.disease.model.DiseaseModel`.

The following example builds an SIS (Susceptible |harr| Infected |harr|
Susceptible) model, passing data directly to constructors instead of
reading from the artifact:

.. testcode::

   config = make_base_config()
   config.update(
       {
           "population": {"population_size": 10_000},
           "mortality": {"data_sources": {"all_cause_mortality_rate": 0}},
       },
       layer="override",
   )

   # 1. Create the states.
   healthy = SusceptibleState("diarrheal_diseases")
   infected = DiseaseState(
       "diarrheal_diseases",
       prevalence=0.1,
       disability_weight=0.0,
       excess_mortality_rate=0.0,
   )

   # 2. Add transitions.
   # From susceptible to infected: uses incidence rate.
   healthy.add_rate_transition(infected, transition_rate=0.5)
   # From infected back to susceptible: uses remission rate.
   infected.add_rate_transition(healthy, transition_rate=1.0)

   # 3. Wrap in a DiseaseModel.
   model = DiseaseModel(
       "diarrheal_diseases",
       states=[healthy, infected],
       cause_specific_mortality_rate=0.0,
   )

   # 4. Run.
   sim = InteractiveContext(
       components=[BasePopulation(), model],
       configuration=config,
       plugin_configuration=base_plugins,
   )

   pop = sim.get_population(["diarrheal_diseases"])
   disease_col = pop["diarrheal_diseases"]
   # ~10% of the population should be infected (prevalence = 0.1).
   print(f"States: {sorted(disease_col.unique())}")

   # Step the simulation forward and observe transitions.
   sim.step()
   pop = sim.get_population(["diarrheal_diseases"])
   expected_states = {"susceptible_to_diarrheal_diseases", "diarrheal_diseases"}
   print(f"Transitions occurred: {set(pop['diarrheal_diseases'].unique()) == expected_states}")

.. testoutput::
   :options: +ELLIPSIS

   ...
   States: ['diarrheal_diseases', 'susceptible_to_diarrheal_diseases']
   ...
   Transitions occurred: True

.. note::

   When ``prevalence`` is set on a ``DiseaseState``, the
   :class:`~vivarium_public_health.disease.model.DiseaseModel` uses it to assign
   simulants to that state at initialization. The ``SusceptibleState`` gets
   the residual (1 minus the sum of all other state prevalences).


Providing custom transition rates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can pass rate data directly to the transition constructor instead of
relying on the artifact or configuration:

.. testcode::

   healthy = SusceptibleState("measles")
   infected = DiseaseState(
       "measles",
       prevalence=0.05,
       disability_weight=0.0,
       excess_mortality_rate=0.0,
   )

   # Pass a constant incidence rate of 0.01 per person-year.
   healthy.add_rate_transition(infected, transition_rate=0.01)

   # Pass a constant remission rate.
   infected.add_rate_transition(healthy, transition_rate=0.5)

   model = DiseaseModel(
       "measles",
       states=[healthy, infected],
       cause_specific_mortality_rate=0.0,
   )

   config = make_base_config()
   config.update(
       {
           "population": {"population_size": 10_000},
           "mortality": {"data_sources": {"all_cause_mortality_rate": 0}},
       },
       layer="override",
   )

   sim = InteractiveContext(
       components=[BasePopulation(), model],
       configuration=config,
       plugin_configuration=base_plugins,
   )

   pop = sim.get_population(["measles"])
   # ~5% of the population should be infected (prevalence = 0.05).
   print(f"States: {sorted(pop['measles'].unique())}")

.. testoutput::
   :options: +ELLIPSIS

   ...
   States: ['measles', 'susceptible_to_measles']


Pre-Built Models
-----------------

For common disease progressions,
:mod:`vivarium_public_health.disease.models` provides convenience functions
that create fully wired models in a single call. When using these, data is
typically supplied via the ``data_sources`` configuration or from the
artifact.


SI model (Susceptible |rarr| Infected)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest model: once infected, a simulant never recovers.

**Artifact keys used:**

- ``cause.{cause}.incidence_rate`` - susceptible |rarr| infected
- ``cause.{cause}.prevalence`` - initialization into disease state
- ``cause.{cause}.disability_weight`` - YLD calculation
- ``cause.{cause}.excess_mortality_rate`` - mortality
- ``cause.{cause}.cause_specific_mortality_rate`` - CSMR

.. testcode::

   config = make_base_config()
   config.update(
       {
           "population": {"population_size": 10_000},
           "mortality": {"data_sources": {"all_cause_mortality_rate": 0}},
       },
       layer="override",
   )

   # SI("test_cause") loads artifact data defined in _example_data.py:
   #   cause.test_cause.incidence_rate  -> _ARTIFACT_DATA (rate=0.5)
   #   cause.test_cause.prevalence      -> _CAUSE_DEFAULTS (value=0.0)
   #   cause.test_cause.disability_weight -> _CAUSE_DEFAULTS (value=0.0)
   #   cause.test_cause.excess_mortality_rate -> _CAUSE_DEFAULTS (value=0.0)
   #   cause.test_cause.cause_specific_mortality_rate -> _CAUSE_DEFAULTS (value=0.0)
   model = SI("test_cause")

   sim = InteractiveContext(
       components=[BasePopulation(), model],
       configuration=config,
       plugin_configuration=base_plugins,
   )

   # Initially everyone is susceptible (prevalence = 0).
   pop = sim.get_population(["test_cause"])
   print(f"All susceptible: {(pop['test_cause'] == 'susceptible_to_test_cause').all()}")

   # After several steps, some simulants become infected.
   for _ in range(5):
       sim.step()
   pop = sim.get_population(["test_cause"])
   n_infected = (pop["test_cause"] == "test_cause").sum()
   print(f"Infections occurred: {n_infected > 100}")

.. testoutput::
   :options: +ELLIPSIS

   ...
   All susceptible: True
   ...
   Infections occurred: True


SIS model (Susceptible |harr| Infected)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Simulants can recover and become susceptible again.

**Additional artifact keys used** (beyond SI):

- ``cause.{cause}.remission_rate`` - infected |rarr| susceptible

.. testcode::

   config = make_base_config()
   config.update(
       {
           "population": {"population_size": 10_000},
           "mortality": {"data_sources": {"all_cause_mortality_rate": 0}},
       },
       layer="override",
   )

   # SIS("test_cause") additionally loads:
   #   cause.test_cause.remission_rate -> _ARTIFACT_DATA (rate=5.0)
   model = SIS("test_cause")

   sim = InteractiveContext(
       components=[BasePopulation(), model],
       configuration=config,
       plugin_configuration=base_plugins,
   )

   for _ in range(10):
       sim.step()
   pop = sim.get_population(["test_cause"])
   # Both states should be populated (infections and recoveries).
   infected = (pop["test_cause"] == "test_cause").sum() > 0
   susceptible = (pop["test_cause"] == "susceptible_to_test_cause").sum() > 0
   print(f"Both states populated: {infected and susceptible}")

.. testoutput::
   :options: +ELLIPSIS

   ...
   Both states populated: True


SIR model (Susceptible |rarr| Infected |rarr| Recovered)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Simulants move from susceptible to infected to recovered, with no return
to susceptibility.

**Artifact keys used:**

- ``cause.{cause}.incidence_rate`` - susceptible |rarr| infected
- ``cause.{cause}.remission_rate`` - infected |rarr| recovered
- ``cause.{cause}.prevalence`` - initialization into disease state
- ``cause.{cause}.disability_weight`` - YLD calculation
- ``cause.{cause}.excess_mortality_rate`` - mortality
- ``cause.{cause}.cause_specific_mortality_rate`` - CSMR

.. testcode::

   config = make_base_config()
   config.update(
       {
           "population": {"population_size": 10_000},
           "mortality": {"data_sources": {"all_cause_mortality_rate": 0}},
       },
       layer="override",
   )

   # SIR("test_cause") loads the same keys as SIS (incidence + remission).
   model = SIR("test_cause")

   sim = InteractiveContext(
       components=[BasePopulation(), model],
       configuration=config,
       plugin_configuration=base_plugins,
   )

   for _ in range(10):
       sim.step()
   pop = sim.get_population(["test_cause"])
   states = set(pop["test_cause"].unique())
   # All three states should be present.
   expected = {"susceptible_to_test_cause", "test_cause", "recovered_from_test_cause"}
   print(f"All three states present: {expected.issubset(states)}")

.. testoutput::
   :options: +ELLIPSIS

   ...
   All three states present: True


SIS with fixed duration
^^^^^^^^^^^^^^^^^^^^^^^^

An SIS model where the infection lasts for a fixed number of days instead
of using a remission rate. Simulants cannot transition out of the infected
state until the dwell time has elapsed.

**Artifact keys used:**

- ``cause.{cause}.incidence_rate`` - susceptible |rarr| infected

No remission rate is needed; the dwell time is passed to the constructor.

.. testcode::

   config = make_base_config()
   config.update(
       {
           "population": {"population_size": 10_000},
           "time": {"step_size": 5},
           "mortality": {"data_sources": {"all_cause_mortality_rate": 0}},
       },
       layer="override",
   )

   # Infection lasts exactly 14 days.
   # cause.test_cause.incidence_rate -> _ARTIFACT_DATA (rate=0.5)
   # No remission_rate needed - dwell time handles the return transition.
   model = SIS_fixed_duration("test_cause", duration="14")

   sim = InteractiveContext(
       components=[BasePopulation(), model],
       configuration=config,
       plugin_configuration=base_plugins,
   )

   for _ in range(10):
       sim.step()
   pop = sim.get_population(["test_cause"])
   # Both states should be populated.
   infected = (pop["test_cause"] == "test_cause").sum() > 0
   susceptible = (pop["test_cause"] == "susceptible_to_test_cause").sum() > 0
   print(f"Both states populated: {infected and susceptible}")

.. testoutput::
   :options: +ELLIPSIS

   ...
   Both states populated: True


SIR with fixed duration
^^^^^^^^^^^^^^^^^^^^^^^^

Same as SIR, but the infection has a fixed duration before the simulant
moves to the recovered state.

.. testcode::

   config = make_base_config()
   config.update(
       {
           "population": {"population_size": 10_000},
           "time": {"step_size": 5},
           "mortality": {"data_sources": {"all_cause_mortality_rate": 0}},
       },
       layer="override",
   )

   # Infection lasts exactly 21 days before recovery.
   # cause.test_cause.incidence_rate -> _ARTIFACT_DATA (rate=0.5)
   model = SIR_fixed_duration("test_cause", duration="21")

   sim = InteractiveContext(
       components=[BasePopulation(), model],
       configuration=config,
       plugin_configuration=base_plugins,
   )

   for _ in range(10):
       sim.step()
   pop = sim.get_population(["test_cause"])
   states = set(pop["test_cause"].unique())
   expected = {"susceptible_to_test_cause", "test_cause", "recovered_from_test_cause"}
   print(f"All three states present: {expected.issubset(states)}")

.. testoutput::
   :options: +ELLIPSIS

   ...
   All three states present: True


Neonatal Models
----------------

Neonatal disease models assign a condition at birth based on birth
prevalence. They are designed for conditions that are present from the
start of life. The name ``NeonatalSWC`` stands for "Neonatal - Susceptible
With Condition."

**Artifact keys used:**

- ``cause.{cause}.birth_prevalence`` - for assigning condition at birth

NeonatalSWC without incidence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A model where the condition is assigned at birth and no new cases arise
afterward:

.. testcode::

   config = make_base_config()
   config.update(
       {
           "population": {
               "population_size": 10_000,
               "initialization_age_min": 0,
               "initialization_age_max": 0,
           },
           "mortality": {"data_sources": {"all_cause_mortality_rate": 0}},
       },
       layer="override",
   )

   # cause.neonatal_cause.birth_prevalence -> _ARTIFACT_DATA (rate=0.05)
   model = NeonatalSWC_without_incidence("neonatal_cause")

   sim = InteractiveContext(
       components=[BasePopulation(), model],
       configuration=config,
       plugin_configuration=base_plugins,
   )

   # Some newborns are born with the condition (based on birth prevalence).
   pop = sim.get_population(["neonatal_cause"])
   initial_infected = (pop["neonatal_cause"] == "neonatal_cause").sum()
   print(f"Born with condition: {initial_infected > 0}")

   # After stepping, no new cases appear because there are no transitions.
   for _ in range(5):
       sim.step()
   pop = sim.get_population(["neonatal_cause"])
   after_infected = (pop["neonatal_cause"] == "neonatal_cause").sum()
   print(f"No new cases: {after_infected == initial_infected}")

.. testoutput::
   :options: +ELLIPSIS

   ...
   Born with condition: True
   ...
   No new cases: True


NeonatalSWC with incidence
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A model where the condition is assigned at birth *and* new cases can arise
via an incidence rate.

**Additional artifact keys used:**

- ``cause.{cause}.incidence_rate`` - for ongoing incidence after birth

.. testcode::

   config = make_base_config()
   config.update(
       {
           "population": {
               "population_size": 10_000,
               "initialization_age_min": 0,
               "initialization_age_max": 0,
           },
           "mortality": {"data_sources": {"all_cause_mortality_rate": 0}},
       },
       layer="override",
   )

   # cause.neonatal_cause.birth_prevalence -> _ARTIFACT_DATA (rate=0.05)
   # cause.neonatal_cause.incidence_rate   -> _ARTIFACT_DATA (rate=0.5)
   model = NeonatalSWC_with_incidence("neonatal_cause")

   sim = InteractiveContext(
       components=[BasePopulation(), model],
       configuration=config,
       plugin_configuration=base_plugins,
   )

   pop = sim.get_population(["neonatal_cause"])
   initial_infected = (pop["neonatal_cause"] == "neonatal_cause").sum()
   print(f"Initially infected: {initial_infected > 0}")

   # After stepping, new cases arise via the incidence rate.
   for _ in range(5):
       sim.step()
   pop = sim.get_population(["neonatal_cause"])
   new_infected = (pop["neonatal_cause"] == "neonatal_cause").sum()
   print(f"New cases arose: {new_infected > initial_infected}")

.. testoutput::
   :options: +ELLIPSIS

   ...
   Initially infected: True
   ...
   New cases arose: True


Advanced Topics
----------------


Dwell time
^^^^^^^^^^

A **dwell time** forces simulants to remain in a state for a minimum
duration before they can transition out. This is useful for modelling
conditions with a known minimum duration (e.g., a 14-day infection).

Dwell time can be specified as a :class:`pandas.Timedelta`, a numeric
value (days), or directly in the :class:`~vivarium_public_health.disease.state.DiseaseState`
constructor:

.. testcode::

   config = make_base_config()
   config.update(
       {
           "population": {"population_size": 100},
           "time": {"step_size": 10},
           "mortality": {"data_sources": {"all_cause_mortality_rate": 0}},
       },
       layer="override",
   )

   healthy = BaseDiseaseState("healthy")
   acute = DiseaseState("acute_event", dwell_time=28, disability_weight=0.0, excess_mortality_rate=0.0)
   chronic = BaseDiseaseState("chronic")

   # Everyone starts healthy and transitions to acute immediately.
   healthy.add_dwell_time_transition(acute)
   # After 28 days in the acute state, simulants move to chronic.
   acute.add_dwell_time_transition(chronic)

   model = DiseaseModel(
       "dwell_demo",
       residual_state=healthy,
       states=[healthy, acute, chronic],
       cause_specific_mortality_rate=0.0,
   )

   sim = InteractiveContext(
       components=[BasePopulation(), model],
       configuration=config,
       plugin_configuration=base_plugins,
   )

   # Step 1: everyone moves from healthy to acute.
   sim.step()
   pop = sim.get_population(["dwell_demo"])
   print(f"All in acute: {(pop['dwell_demo'] == 'acute_event').all()}")

   # Steps 2-3: still in acute (only 20 days have passed, < 28 day dwell).
   sim.step()
   sim.step()
   pop = sim.get_population(["dwell_demo"])
   print(f"Still in acute: {(pop['dwell_demo'] == 'acute_event').all()}")

   # Step 4: 40 days have passed (> 28 day dwell), simulants move to chronic.
   sim.step()
   pop = sim.get_population(["dwell_demo"])
   print(f"All in chronic: {(pop['dwell_demo'] == 'chronic').all()}")

.. testoutput::
   :options: +ELLIPSIS

   ...
   All in acute: True
   ...
   Still in acute: True
   ...
   All in chronic: True


Excess mortality
^^^^^^^^^^^^^^^^^

A :class:`~vivarium_public_health.disease.state.DiseaseState` can carry an
**excess mortality rate** - an additional hazard of death for simulants in
that state. This is added on top of the all-cause mortality rate.

.. testcode::

   config = make_base_config()
   config.update(
       {
           "population": {"population_size": 1_000},
           "time": {"step_size": 10},
           "mortality": {"data_sources": {"all_cause_mortality_rate": 0}},
       },
       layer="override",
   )

   healthy = BaseDiseaseState("healthy")
   severe = DiseaseState("severe_event", dwell_time=14, disability_weight=0.0, excess_mortality_rate=0.7)
   recovered = BaseDiseaseState("recovered")

   healthy.add_dwell_time_transition(severe)
   severe.add_dwell_time_transition(recovered)

   model = DiseaseModel(
       "emr_demo",
       residual_state=healthy,
       states=[healthy, severe, recovered],
       cause_specific_mortality_rate=0.0,
   )

   sim = InteractiveContext(
       components=[BasePopulation(), model],
       configuration=config,
       plugin_configuration=base_plugins,
   )

   # Before any steps, all simulants are alive - background mortality is zero.
   assert sim.get_population(["is_alive"])["is_alive"].all()

   sim.step()  # everyone moves to severe state
   sim.step()  # excess mortality applies while in the severe state

   alive_after = sim.get_population(["is_alive"])["is_alive"].sum()
   # All-cause mortality is zero, so deaths are solely from the EMR.
   print(f"Deaths solely from EMR: {alive_after < 1_000}")

.. testoutput::
   :options: +ELLIPSIS

   ...
   Deaths solely from EMR: True


Proportion transitions
^^^^^^^^^^^^^^^^^^^^^^^

A :class:`~vivarium_public_health.disease.transition.ProportionTransition` moves a
fixed fraction of eligible simulants to a new state each time step, rather
than converting a rate to a probability:

.. testcode::

   config = make_base_config()
   config.update(
       {
           "population": {"population_size": 10_000},
           "mortality": {"data_sources": {"all_cause_mortality_rate": 0}},
       },
       layer="override",
   )

   stage_1 = BaseDiseaseState("stage_1")
   stage_2 = DiseaseState(
       "stage_2",
       prevalence=0.0,
       disability_weight=0.0,
       excess_mortality_rate=0.0,
   )

   # 20% of simulants in stage_1 move to stage_2 each time step.
   stage_1.add_proportion_transition(stage_2, proportion=0.2)

   model = DiseaseModel(
       "proportion_demo",
       residual_state=stage_1,
       states=[stage_1, stage_2],
       cause_specific_mortality_rate=0.0,
   )

   sim = InteractiveContext(
       components=[BasePopulation(), model],
       configuration=config,
       plugin_configuration=base_plugins,
   )

   sim.step()
   pop = sim.get_population(["proportion_demo"])
   n_stage_2 = (pop["proportion_demo"] == "stage_2").sum()
   actual_proportion = n_stage_2 / len(pop)
   # With proportion=0.2, approximately 20% should transition in one step.
   print(f"Proportion near 0.2: {0.15 < actual_proportion < 0.25}")

.. testoutput::
   :options: +ELLIPSIS

   ...
   Proportion near 0.2: True


Transient states
^^^^^^^^^^^^^^^^^

A :class:`~vivarium_public_health.disease.state.TransientDiseaseState` is a
pass-through state: simulants enter it and immediately transition onward
in the same time step. This is useful for routing logic where different
fractions of simulants should end up in different destination states:

.. testcode::

   config = make_base_config()
   config.update(
       {
           "population": {"population_size": 10_000},
           "mortality": {"data_sources": {"all_cause_mortality_rate": 0}},
       },
       layer="override",
   )

   start = BaseDiseaseState("start")
   router = TransientDiseaseState("router")
   outcome_a = DiseaseState(
       "outcome_a",
       prevalence=0.0,
       disability_weight=0.0,
       excess_mortality_rate=0.0,
   )
   outcome_b = DiseaseState(
       "outcome_b",
       prevalence=0.0,
       disability_weight=0.0,
       excess_mortality_rate=0.0,
   )

   # Everyone moves from start to the transient router state.
   start.add_dwell_time_transition(router)
   # From the router, 70% go to outcome_a, 30% go to outcome_b.
   router.add_proportion_transition(outcome_a, proportion=0.7)
   router.add_proportion_transition(outcome_b, proportion=0.3)

   model = DiseaseModel(
       "transient_demo",
       residual_state=start,
       states=[start, router, outcome_a, outcome_b],
       cause_specific_mortality_rate=0.0,
   )

   sim = InteractiveContext(
       components=[BasePopulation(), model],
       configuration=config,
       plugin_configuration=base_plugins,
   )

   sim.step()
   pop = sim.get_population(["transient_demo"])
   # No simulants remain in the "router" state.
   print(f"No simulants in router: {'router' not in pop['transient_demo'].values}")
   print(f"Both outcomes populated: {(pop['transient_demo'] == 'outcome_a').sum() > 0 and (pop['transient_demo'] == 'outcome_b').sum() > 0}")

.. testoutput::
   :options: +ELLIPSIS

   ...
   No simulants in router: True
   Both outcomes populated: True


Multiple disease states (sequelae)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A single disease can have multiple sequelae, each with its own prevalence,
disability weight, and transitions. The
:class:`~vivarium_public_health.disease.model.DiseaseModel` assigns simulants to
states at initialization based on relative prevalences:

.. testcode::

   config = make_base_config()
   config.update(
       {
           "population": {"population_size": 50_000},
           "mortality": {"data_sources": {"all_cause_mortality_rate": 0}},
       },
       layer="override",
   )

   healthy = BaseDiseaseState("healthy")
   mild = DiseaseState(
       "mild",
       prevalence=0.15,
       disability_weight=0.0,
       excess_mortality_rate=0.0,
   )
   moderate = DiseaseState(
       "moderate",
       prevalence=0.05,
       disability_weight=0.0,
       excess_mortality_rate=0.0,
   )
   severe = DiseaseState(
       "severe",
       prevalence=0.02,
       disability_weight=0.0,
       excess_mortality_rate=0.0,
   )

   model = DiseaseModel(
       "multi_state_demo",
       residual_state=healthy,
       states=[healthy, mild, moderate, severe],
       cause_specific_mortality_rate=0.0,
   )

   sim = InteractiveContext(
       components=[BasePopulation(), model],
       configuration=config,
       plugin_configuration=base_plugins,
   )

   pop = sim.get_population(["multi_state_demo"])
   states = set(pop["multi_state_demo"].unique())
   # All four states should be present based on the prevalences.
   print(f"All states present: {states == {'healthy', 'mild', 'moderate', 'severe'}}")
   # Residual state (healthy) should have the largest count.
   healthy_count = (pop["multi_state_demo"] == "healthy").sum()
   mild_count = (pop["multi_state_demo"] == "mild").sum()
   print(f"Residual state largest: {healthy_count > mild_count}")

.. testoutput::
   :options: +ELLIPSIS

   ...
   All states present: True
   Residual state largest: True


Overriding data via configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All data sources can be overridden through the simulation configuration
without changing the code that builds the model. This is useful for
sensitivity analyses or testing:

.. testcode::

   config = make_base_config()
   config.update(
       {
           "population": {"population_size": 10_000},
           "mortality": {"data_sources": {"all_cause_mortality_rate": 0}},
           # Override prevalence for the disease state via configuration.
           "disease_state.test_cause": {
               "data_sources": {
                   "prevalence": 0.3,
               },
           },
       },
       layer="override",
   )

   model = SI("test_cause")

   sim = InteractiveContext(
       components=[BasePopulation(), model],
       configuration=config,
       plugin_configuration=base_plugins,
   )

   # ~30% should start infected due to the prevalence override.
   pop = sim.get_population(["test_cause"])
   n_infected = (pop["test_cause"] == "test_cause").sum()
   print(f"High initial prevalence: {n_infected > 2000}")

.. testoutput::
   :options: +ELLIPSIS

   ...
   High initial prevalence: True


Event tracking columns
^^^^^^^^^^^^^^^^^^^^^^^

Each :class:`~vivarium_public_health.disease.state.DiseaseState` and
:class:`~vivarium_public_health.disease.state.BaseDiseaseState` automatically
adds two columns to the simulation state table:

- ``{state_id}_event_time`` - the timestamp of the last transition *into*
  this state.
- ``{state_id}_event_count`` - how many times the simulant has entered
  this state.

These are useful for tracking disease history:

.. testcode::

   config = make_base_config()
   config.update(
       {
           "population": {"population_size": 10_000},
           "mortality": {"data_sources": {"all_cause_mortality_rate": 0}},
       },
       layer="override",
   )

   model = SIS("test_cause")

   sim = InteractiveContext(
       components=[BasePopulation(), model],
       configuration=config,
       plugin_configuration=base_plugins,
   )

   for _ in range(20):
       sim.step()

   pop = sim.get_population(
       ["test_cause", "test_cause_event_time", "test_cause_event_count"]
   )
   # Show simulants who have been infected at least once.
   ever_infected = pop[pop["test_cause_event_count"] > 0]
   print(f"Simulants ever infected: {len(ever_infected) > 0}")
   print(f"Event columns present: {'test_cause_event_time' in ever_infected.columns and 'test_cause_event_count' in ever_infected.columns}")

.. testoutput::
   :options: +ELLIPSIS

   ...
   Simulants ever infected: True
   Event columns present: True


Configuration Summary
---------------------

.. list-table::
   :header-rows: 1

   * - Component
     - Key configuration options
     - Artifact data required
   * - ``DiseaseModel``
     - ``{cause}.data_sources.cause_specific_mortality_rate``
     - ``cause.{cause}.cause_specific_mortality_rate``
   * - ``DiseaseState``
     - ``{state}.data_sources.prevalence``,
       ``{state}.data_sources.birth_prevalence``,
       ``{state}.data_sources.dwell_time``,
       ``{state}.data_sources.disability_weight``,
       ``{state}.data_sources.excess_mortality_rate``
     - Artifact keys matching the pattern
       ``cause.{state_id}.{measure}``
   * - ``RateTransition``
     - ``{transition}.data_sources.transition_rate``,
       ``{transition}.rate_conversion_type``
     - Artifact key for the rate (e.g.,
       ``cause.{cause}.incidence_rate``)
   * - ``ProportionTransition``
     - ``{transition}.data_sources.proportion``
     - None (proportion usually provided directly)
   * - ``SI``
     - -
     - incidence rate, prevalence, disability weight,
       excess mortality rate, CSMR
   * - ``SIS``
     - -
     - incidence rate, remission rate, prevalence,
       disability weight, excess mortality rate, CSMR
   * - ``SIR``
     - -
     - incidence rate, remission rate, prevalence,
       disability weight, excess mortality rate, CSMR
   * - ``SIS_fixed_duration``
     - ``duration`` (days, passed to constructor)
     - incidence rate, prevalence, disability weight,
       excess mortality rate, CSMR
   * - ``SIR_fixed_duration``
     - ``duration`` (days, passed to constructor)
     - incidence rate, prevalence, disability weight,
       excess mortality rate, CSMR
   * - ``NeonatalSWC_without_incidence``
     - -
     - birth prevalence, prevalence, disability weight,
       excess mortality rate, CSMR
   * - ``NeonatalSWC_with_incidence``
     - -
     - birth prevalence, incidence rate, prevalence,
       disability weight, excess mortality rate, CSMR
