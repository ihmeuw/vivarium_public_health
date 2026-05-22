=========
Treatment
=========

:mod:`vivarium_public_health` provides components for modeling treatment
interventions in public health simulations. This tutorial covers the
treatment module - how to model intervention coverage, how interventions
reduce disease rates, and how to apply direct shifts or scale-ups to
epidemiological measures.

For how risk factor *exposures* modify disease outcomes, see the
:doc:`risk` and :doc:`risk_effect` tutorials.

.. contents::
   :local:
   :depth: 2

.. testsetup:: *

   import numpy as np
   import pandas as pd
   from loguru import logger
   logger.disable("vivarium")
   from vivarium import InteractiveContext
   from vivarium_public_health.treatment import (
       Intervention, InterventionEffect, AbsoluteShift,
       LinearScaleUp, TherapeuticInertia,
   )
   from vivarium_public_health.disease import *
   from vivarium_public_health.population import BasePopulation
   from vivarium_public_health._example_data import *
   base_plugins = BASE_PLUGINS


Overview
--------

The treatment module provides several components for modeling interventions:

**Intervention** - a dichotomous coverage model that assigns each simulant
a covered or uncovered status.
:class:`~vivarium_public_health.treatment.intervention.Intervention` is the
treatment analogue of
:class:`~vivarium_public_health.risks.base_risk.Risk`.

**InterventionEffect** - how intervention coverage modifies a target rate
or measure (via a relative risk).
:class:`~vivarium_public_health.treatment.intervention.InterventionEffect`
is the treatment analogue of
:class:`~vivarium_public_health.risks.effect.RiskEffect`.

**AbsoluteShift** - a simple component that directly replaces a target
measure with a configured value for simulants in a specified age range.
:class:`~vivarium_public_health.treatment.magic_wand.AbsoluteShift`

**LinearScaleUp** - linearly interpolates intervention coverage between
a start and end value over a configured date range.
:class:`~vivarium_public_health.treatment.scale_up.LinearScaleUp`

**TherapeuticInertia** - draws a population-level therapeutic inertia
value from a triangular distribution, representing the probability that
treatment is *not* escalated during a healthcare visit.
:class:`~vivarium_public_health.treatment.therapeutic_inertia.TherapeuticInertia`


Common Setup
------------

Every code example in this tutorial uses the imports and helpers shown below.
To run any example in a standalone script, include all of these at the top:

.. testcode::

   from vivarium import InteractiveContext
   from vivarium_public_health.treatment import (
       Intervention, InterventionEffect, AbsoluteShift,
       LinearScaleUp, TherapeuticInertia,
   )
   from vivarium_public_health.disease import SI, SIS
   from vivarium_public_health.population import BasePopulation
   from vivarium_public_health._example_data import BASE_PLUGINS, make_base_config

   # BASE_PLUGINS overrides the data plugin to use ExampleArtifactManager,
   # which serves example data from memory instead of requiring a real HDF file.
   base_plugins = BASE_PLUGINS

   # make_base_config() returns a configuration with sensible defaults for
   # time range, step size, and randomness key columns.
   config = make_base_config()


Intervention
------------

An :class:`~vivarium_public_health.treatment.intervention.Intervention`
component assigns each simulant a coverage status - either ``"covered"``
or ``"uncovered"``. The proportion covered is determined by the exposure
data source. Each simulant's propensity (a random value drawn at
initialization) determines whether they receive coverage.

``Intervention`` is a specialization of
:class:`~vivarium_public_health.causal_factor.exposure.CausalFactor`
restricted to the ``"intervention"`` entity type.

The configuration key for an intervention is its full entity string
(e.g., ``intervention.my_treatment``):

.. code-block:: yaml

   configuration:
     intervention.my_treatment:
       distribution_type: "dichotomous"
       data_sources:
         exposure: 0.5  # scalar, DataFrame, callable, or artifact key


Basic example
^^^^^^^^^^^^^

The simplest intervention model sets everyone to covered:

.. testcode::

   config = make_base_config()
   config.update(
       {
           "population": {"population_size": 1_000},
           "mortality": {"data_sources": {"all_cause_mortality_rate": 0}},
           "intervention.my_treatment": {
               "distribution_type": "dichotomous",
               "data_sources": {"exposure": 1.0},
           },
       },
       layer="override",
   )

   sim = InteractiveContext(
       components=[
           BasePopulation(),
           Intervention("intervention.my_treatment"),
       ],
       configuration=config,
       plugin_configuration=base_plugins,
   )

   pop = sim.get_population(["my_treatment.exposure"])
   # With exposure=1.0, all simulants are covered by the intervention.
   print(f"All covered: {(pop['my_treatment.exposure'] == 'covered').all()}")

.. testoutput::

   All covered: True


Partial coverage
^^^^^^^^^^^^^^^^

When the coverage proportion is less than 1, some simulants will be covered
and others will not. The split is determined by each simulant's propensity
draw:

.. testcode::

   config = make_base_config()
   config.update(
       {
           "population": {"population_size": 10_000},
           "mortality": {"data_sources": {"all_cause_mortality_rate": 0}},
           "intervention.my_treatment": {
               "distribution_type": "dichotomous",
               "data_sources": {"exposure": 0.4},
           },
       },
       layer="override",
   )

   sim = InteractiveContext(
       components=[
           BasePopulation(),
           Intervention("intervention.my_treatment"),
       ],
       configuration=config,
       plugin_configuration=base_plugins,
   )

   pop = sim.get_population(["my_treatment.exposure"])
   n_covered = (pop["my_treatment.exposure"] == "covered").sum()
   proportion = n_covered / len(pop)
   # Approximately 40% should be covered.
   print(f"Proportion covered near 0.4: {np.isclose(proportion, 0.4, atol=0.02)}")

.. testoutput::

   Proportion covered near 0.4: True


InterventionEffect
------------------

An :class:`~vivarium_public_health.treatment.intervention.InterventionEffect`
modifies disease dynamics based on intervention coverage. Unlike a risk
factor (where exposed simulants typically have a *higher* rate), an
intervention typically *reduces* the target rate for covered simulants
(relative risk < 1).

``InterventionEffect`` is a specialization of
:class:`~vivarium_public_health.causal_factor.effect.CausalFactorEffect`
for interventions. Its configuration key combines the intervention name and
the target:

.. code-block:: yaml

   configuration:
     intervention_effect.{intervention_name}_on_{target_entity}.{target_name}.{target_measure}:
       data_sources:
         relative_risk: 0.5           # scalar, DataFrame, callable, or artifact key
         population_attributable_fraction: 0  # typically 0 for interventions


Reducing disease incidence
^^^^^^^^^^^^^^^^^^^^^^^^^^

The following example demonstrates that covered simulants become infected
at a lower rate than uncovered simulants:

.. testcode::

   config = make_base_config()
   config.update(
       {
           "population": {"population_size": 10_000},
           "mortality": {"data_sources": {"all_cause_mortality_rate": 0}},
           "intervention.my_treatment": {
               "distribution_type": "dichotomous",
               "data_sources": {"exposure": 0.5},
           },
           "intervention_effect.my_treatment_on_cause.test_cause.incidence_rate": {
               "data_sources": {
                   "relative_risk": 0.2,
                   "population_attributable_fraction": 0,
               },
           },
       },
       layer="override",
   )

   sim = InteractiveContext(
       components=[
           BasePopulation(),
           Intervention("intervention.my_treatment"),
           InterventionEffect(
               "intervention.my_treatment", "cause.test_cause.incidence_rate"
           ),
           SI("test_cause"),
       ],
       configuration=config,
       plugin_configuration=base_plugins,
   )

   # Step forward to allow infections to occur.
   for _ in range(3):
       sim.step()

   pop = sim.get_population(["test_cause", "my_treatment.exposure"])
   covered = pop[pop["my_treatment.exposure"] == "covered"]
   uncovered = pop[pop["my_treatment.exposure"] == "uncovered"]

   covered_infection_rate = (covered["test_cause"] == "test_cause").sum() / len(
       covered
   )
   uncovered_infection_rate = (uncovered["test_cause"] == "test_cause").sum() / len(
       uncovered
   )

   # With RR=0.2, covered simulants should have ~1/5 the infection rate.
   ratio = covered_infection_rate / uncovered_infection_rate
   print(f"Rate ratio near 0.2: {np.isclose(ratio, 0.2, rtol=0.15)}")

.. testoutput::

   Rate ratio near 0.2: True


AbsoluteShift
-------------

An :class:`~vivarium_public_health.treatment.magic_wand.AbsoluteShift`
provides a direct override of a target epidemiological measure. When
``target_value`` is set to a numeric value, the component replaces the
target pipeline's value for all simulants within the configured age range.
When ``target_value`` is ``"baseline"``, no modification is applied - this
lets you use the same model specification for baseline and intervention
scenarios by switching a single config value.

The target is specified at instantiation as a string in the form
``"entity_type.entity_name.measure"`` (e.g.,
``"cause.test_cause.incidence_rate"``).


Configuration
^^^^^^^^^^^^^

.. code-block:: yaml

   configuration:
     # Config key is intervention_on_{entity_name}
     intervention_on_test_cause:
       target_value: 0.0       # numeric value or "baseline"
       age_start: 0            # minimum age for effect
       age_end: 125            # maximum age for effect


Eliminating disease incidence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following example sets the incidence rate of a disease to zero,
preventing all new infections:

.. testcode::

   config = make_base_config()
   config.update(
       {
           "population": {"population_size": 10_000},
           "mortality": {"data_sources": {"all_cause_mortality_rate": 0}},
           "intervention_on_test_cause": {
               "target_value": 0.0,
               "age_start": 0,
               "age_end": 125,
           },
       },
       layer="override",
   )

   sim = InteractiveContext(
       components=[
           BasePopulation(),
           AbsoluteShift("cause.test_cause.incidence_rate"),
           SI("test_cause"),
       ],
       configuration=config,
       plugin_configuration=base_plugins,
   )

   # Step forward - with incidence set to 0, nobody should get infected.
   for _ in range(5):
       sim.step()

   pop = sim.get_population(["test_cause"])
   infected = (pop["test_cause"] == "test_cause").sum()
   print(f"Nobody infected: {infected == 0}")

.. testoutput::

   Nobody infected: True


Age-targeted intervention
^^^^^^^^^^^^^^^^^^^^^^^^^

``AbsoluteShift`` supports targeting specific age ranges. The following
example only eliminates incidence for simulants aged 15-50, while others
remain at risk:

.. testcode::

   config = make_base_config()
   config.update(
       {
           "population": {"population_size": 10_000},
           "mortality": {"data_sources": {"all_cause_mortality_rate": 0}},
           "intervention_on_test_cause": {
               "target_value": 0.0,
               "age_start": 15,
               "age_end": 50,
           },
       },
       layer="override",
   )

   sim = InteractiveContext(
       components=[
           BasePopulation(),
           AbsoluteShift("cause.test_cause.incidence_rate"),
           SI("test_cause"),
       ],
       configuration=config,
       plugin_configuration=base_plugins,
   )

   for _ in range(5):
       sim.step()

   pop = sim.get_population(["test_cause", "age"])
   # Only simulants outside the 15-50 age range should get infected.
   infected = pop[pop["test_cause"] == "test_cause"]
   in_range = infected[(infected["age"] >= 15) & (infected["age"] <= 50)]
   outside_range = infected[(infected["age"] < 15) | (infected["age"] > 50)]
   print(f"No infections in target range: {len(in_range) == 0}")
   print(f"Some infections outside range: {len(outside_range) > 0}")

.. testoutput::

   No infections in target range: True
   Some infections outside range: True


LinearScaleUp
-------------

A :class:`~vivarium_public_health.treatment.scale_up.LinearScaleUp`
linearly interpolates an intervention's coverage between a start value and
an end value over a configured date range. Before the start date, the start
value applies; after the end date, the end value applies. It works by
modifying the ``exposure_parameters`` pipeline of an
:class:`~vivarium_public_health.treatment.intervention.Intervention`
component.

The ``LinearScaleUp`` component checks whether the simulation is running
an intervention scenario (``configuration.intervention.scenario != "baseline"``).
If the scenario is ``"baseline"``, no scale-up is applied.


Configuration
^^^^^^^^^^^^^

The configuration specifies dates and endpoint values:

.. code-block:: yaml

   configuration:
     # Enable intervention scenario
     intervention:
       scenario: "treatment"

     # Intervention coverage (initial value before scale-up)
     intervention.my_treatment:
       distribution_type: "dichotomous"
       data_sources:
         exposure: 0.2

     # Scale-up configuration
     my_treatment_scale_up:
       date:
         start: "1993-01-01"
         end: "1997-01-01"
       value:
         start: 0.2   # matches initial coverage
         end: 0.8     # target coverage after scale-up

The ``value.start`` and ``value.end`` can be numeric scalars or the string
``"data"`` to load endpoint values from the artifact.


Scale-up behavior
^^^^^^^^^^^^^^^^^

The scale-up modifies the intervention's ``exposure_parameters`` pipeline
by adding:

.. math::

   \text{adjustment} = \text{progress} \times (\text{end\_value} - \text{start\_value})

where :math:`\text{progress}` is 0 before the start date, 1 after the end
date, and linearly interpolated between them.

The following example demonstrates coverage increasing from 0% to 100%
over the scale-up period:

.. testcode::

   config = make_base_config()
   config.update(
       {
           "population": {"population_size": 10_000},
           "mortality": {"data_sources": {"all_cause_mortality_rate": 0}},
           "intervention": {"scenario": "treatment"},
           "intervention.my_treatment": {
               "distribution_type": "dichotomous",
               "data_sources": {"exposure": 0.0},
           },
           "my_treatment_scale_up": {
               "date": {
                   "start": "1990-07-01",
                   "end": "1995-07-01",
               },
               "value": {
                   "start": 0.0,
                   "end": 1.0,
               },
           },
       },
       layer="override",
   )

   sim = InteractiveContext(
       components=[
           BasePopulation(),
           Intervention("intervention.my_treatment"),
           LinearScaleUp("intervention.my_treatment"),
       ],
       configuration=config,
       plugin_configuration=base_plugins,
   )

   # At the start (before scale-up midpoint), coverage should be low.
   pop_early = sim.get_population(["my_treatment.exposure"])
   coverage_early = (pop_early["my_treatment.exposure"] == "covered").sum() / len(
       pop_early
   )

   # Step to the end of the scale-up period (each step ~30.5 days).
   # ~65 steps = ~5.4 years, past the end date of 1995-07-01.
   for _ in range(65):
       sim.step()

   pop_late = sim.get_population(["my_treatment.exposure"])
   coverage_late = (pop_late["my_treatment.exposure"] == "covered").sum() / len(
       pop_late
   )

   # Coverage should have increased substantially.
   print(f"Coverage increased: {coverage_late > coverage_early}")
   # After the end date, coverage should be near 100%.
   print(f"Full coverage achieved: {np.isclose(coverage_late, 1.0, atol=0.01)}")

.. testoutput::

   Coverage increased: True
   Full coverage achieved: True


TherapeuticInertia
------------------

:class:`~vivarium_public_health.treatment.therapeutic_inertia.TherapeuticInertia`
models the variety of reasons why a treatment algorithm might deviate from
clinical guidelines. At setup, a single scalar value is drawn from a
triangular distribution and exposed via the ``therapeutic_inertia`` pipeline.
This value represents the probability that treatment is *not* escalated
during a healthcare visit.


Configuration
^^^^^^^^^^^^^

.. code-block:: yaml

   configuration:
     therapeutic_inertia:
       triangle_min: 0.65
       triangle_max: 0.9
       triangle_mode: 0.875


Basic usage
^^^^^^^^^^^

.. testcode::

   config = make_base_config()
   config.update(
       {
           "population": {"population_size": 100},
           "mortality": {"data_sources": {"all_cause_mortality_rate": 0}},
           "therapeutic_inertia": {
               "triangle_min": 0.65,
               "triangle_max": 0.9,
               "triangle_mode": 0.875,
           },
       },
       layer="override",
   )

   sim = InteractiveContext(
       components=[
           BasePopulation(),
           TherapeuticInertia(),
       ],
       configuration=config,
       plugin_configuration=base_plugins,
   )

   # The therapeutic inertia value is a single scalar applied to all simulants.
   pop = sim.get_population()
   ti_values = sim.get_value("therapeutic_inertia")(pop.index)
   # All simulants have the same inertia value (population-level draw).
   unique_values = ti_values.unique()
   print(f"Single population-level value: {len(unique_values) == 1}")
   # The value should be within the configured triangle bounds.
   ti = unique_values[0]
   print(f"Within bounds: {0.65 <= ti <= 0.9}")

.. testoutput::

   Single population-level value: True
   Within bounds: True


Configuration Summary
---------------------

.. list-table::
   :header-rows: 1

   * - Component
     - Key configuration options
     - Purpose
   * - ``Intervention``
     - ``intervention.{name}.distribution_type``,
       ``intervention.{name}.data_sources.exposure``
     - Assign dichotomous coverage to simulants
   * - ``InterventionEffect``
     - ``intervention_effect.{name}_on_{target}.data_sources.relative_risk``,
       ``intervention_effect.{name}_on_{target}.data_sources.population_attributable_fraction``
     - Modify target rate based on coverage
   * - ``AbsoluteShift``
     - ``intervention_on_{target_name}.target_value``,
       ``intervention_on_{target_name}.age_start``,
       ``intervention_on_{target_name}.age_end``
     - Replace a measure with a fixed value
   * - ``LinearScaleUp``
     - ``{name}_scale_up.date.start``,
       ``{name}_scale_up.date.end``,
       ``{name}_scale_up.value.start``,
       ``{name}_scale_up.value.end``
     - Linearly ramp coverage over time
   * - ``TherapeuticInertia``
     - ``therapeutic_inertia.triangle_min``,
       ``therapeutic_inertia.triangle_max``,
       ``therapeutic_inertia.triangle_mode``
     - Draw a population-level inertia scalar

.. note::

   ``Intervention`` and ``InterventionEffect`` are specializations of the
   more general :class:`~vivarium_public_health.causal_factor.exposure.CausalFactor`
   and :class:`~vivarium_public_health.causal_factor.effect.CausalFactorEffect`
   base classes. For interventions, the exposure categories are ``"covered"``
   and ``"uncovered"`` (rather than ``"exposed"`` / ``"unexposed"`` for risk
   factors).

   For modeling risk factor exposures and effects, see the :doc:`risk` and
   :doc:`risk_effect` tutorials.
