==============
Risk Exposure
==============

:mod:`vivarium_public_health` provides components for modeling how risk factor
exposures modify disease outcomes. This tutorial covers **risk exposure** -
how simulants are assigned exposure values. For how exposure modifies disease
rates, see the :doc:`risk_effect` tutorial.

.. contents::
   :local:
   :depth: 2

.. testsetup:: *

   import numpy as np
   import pandas as pd
   from vivarium import InteractiveContext
   from vivarium_public_health.risks import Risk, RiskEffect, NonLogLinearRiskEffect
   from vivarium_public_health.disease import *
   from vivarium_public_health.population import BasePopulation
   from vivarium_public_health._example_data import *
   base_plugins = BASE_PLUGINS


Overview
--------

A risk model in ``vivarium_public_health`` has two primary components:

**Exposure** - the risk factor and how simulants are exposed to it.
:class:`~vivarium_public_health.risks.base_risk.Risk` assigns each
simulant an exposure category (for dichotomous/polytomous risks) or a
continuous exposure value.

**Effect** - how exposure modifies a target rate or measure. A *target* is a
specific rate modified by the risk, identified by an entity and a measure
(e.g., ``cause.lung_cancer.incidence_rate``). See the :doc:`risk_effect`
tutorial for details on
:class:`~vivarium_public_health.risks.effect.RiskEffect` and
:class:`~vivarium_public_health.risks.effect.NonLogLinearRiskEffect`.

This tutorial focuses on the exposure side: how the
:class:`~vivarium_public_health.risks.base_risk.Risk` component determines
*who* is exposed.


Common Setup
------------

Every code example in this tutorial uses the imports and helpers shown below.
To run any example in a standalone script, include all of these at the top:

.. testcode::

   from vivarium import InteractiveContext
   from vivarium_public_health.risks import Risk, RiskEffect
   from vivarium_public_health.disease import SI, SIS
   from vivarium_public_health.population import BasePopulation
   from vivarium_public_health._example_data import BASE_PLUGINS, make_base_config

   # BASE_PLUGINS overrides the data plugin to use ExampleArtifactManager,
   # which serves example data from memory instead of requiring a real HDF file.
   base_plugins = BASE_PLUGINS

   # make_base_config() returns a configuration with sensible defaults for
   # time range, step size, and randomness key columns.
   config = make_base_config()

The `Artifact Data Format`_ section shows the expected key names and column
layouts for every data key so that you know exactly what to put in your own
artifact.


Artifact Data Format
--------------------

This section documents the **key name** and **column layout** that the
:class:`~vivarium_public_health.risks.base_risk.Risk` component expects.
Risk components support the ``data_sources`` configuration pattern that lets
you override individual keys with a scalar, DataFrame, or callable without
rebuilding the artifact (see `Data sources`_).


Data keys
^^^^^^^^^

The table below lists every data key used by the
:class:`~vivarium_public_health.risks.base_risk.Risk` component.

.. list-table::
   :header-rows: 1

   * - Key
     - Index columns
     - Value columns
     - Configurable?
   * - ``risk_factor.{name}.distribution``
     - *(scalar)*
     - A string (e.g., ``"dichotomous"``)
     - Yes - ``risk_factor.{name}.distribution_type``
   * - ``risk_factor.{name}.exposure``
     - age, sex, year, parameter
     - ``value`` (proportion per category)
     - Yes - ``risk_factor.{name}.data_sources.exposure``


Artifact data shapes
^^^^^^^^^^^^^^^^^^^^

The examples below use the data builders from the
:mod:`~vivarium_public_health._example_data` module; a production artifact
has the same column layout but with real GBD values.

.. testcode::

   from vivarium_public_health._example_data import risk_exposure_dichotomous

   # risk_factor.{name}.exposure - proportion per exposure category.
   exposure = risk_exposure_dichotomous(0.6)
   print(exposure.query("year_start == 1990 and parameter == 'exposed'").head(4).to_string(index=False))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

    age_start   age_end    sex  year_start  year_end parameter  value
     0.000000  0.019178   Male        1990      1991   exposed    0.6
     0.000000  0.019178 Female        1990      1991   exposed    0.6
     0.019178  0.076712   Male        1990      1991   exposed    0.6
     0.019178  0.076712 Female        1990      1991   exposed    0.6


Data sources
^^^^^^^^^^^^

Risk components support a ``data_sources`` configuration pattern that lets
you override individual data keys without rebuilding the artifact. You can
override any key with:

- **Scalar** (int or float) - broadcast a constant value to all simulants.
- **DataFrame** - use the DataFrame directly.
- **Callable** - call the function at setup time to produce the data.
- **Artifact key** (string) - load a different key from the artifact.

:class:`~vivarium_public_health.risks.base_risk.Risk` declares a
configurable exposure data source:

.. code-block:: yaml

   # Default configuration (loaded from the artifact):
   risk_factor.{name}:
     data_sources:
       exposure: "risk_factor.{name}.exposure"
     distribution_type: "risk_factor.{name}.distribution"

This can be overridden with a scalar in the simulation configuration:

.. code-block:: yaml

   configuration:
     risk_factor.{name}:
       distribution_type: "dichotomous"  # or load from artifact
       data_sources:
         exposure: 0.6  # scalar, DataFrame, callable, or artifact key


Risk
----

A :class:`~vivarium_public_health.risks.base_risk.Risk` component assigns
each simulant an exposure value. In the simplest case - a **dichotomous**
risk - each simulant falls into one of two categories:
exposed or unexposed. The proportion exposed is determined by the exposure
data source. Each simulant's propensity (a random value drawn at
initialization) determines whether they are in the exposed group.


Basic example
^^^^^^^^^^^^^

The simplest risk model sets everyone to exposed:

.. testcode::

   config = make_base_config()
   config.update(
       {
           "population": {"population_size": 1_000},
           "mortality": {"data_sources": {"all_cause_mortality_rate": 0}},
           "risk_factor.test_risk": {
               "distribution_type": "dichotomous",
               "data_sources": {"exposure": 1.0},
           },
       },
       layer="override",
   )

   sim = InteractiveContext(
       components=[
           BasePopulation(),
           Risk("risk_factor.test_risk"),
       ],
       configuration=config,
       plugin_configuration=base_plugins,
   )

   pop = sim.get_population(["test_risk.exposure"])
   # With exposure=1.0, all simulants are in the exposed category.
   print(f"All exposed: {(pop['test_risk.exposure'] == 'exposed').all()}")

.. testoutput::

   All exposed: True


Partial exposure
^^^^^^^^^^^^^^^^

When the exposure proportion is less than 1, some simulants will be exposed
and others unexposed. The split is determined by each simulant's propensity
draw:

.. testcode::

   config = make_base_config()
   config.update(
       {
           "population": {"population_size": 10_000},
           "mortality": {"data_sources": {"all_cause_mortality_rate": 0}},
           "risk_factor.test_risk": {
               "distribution_type": "dichotomous",
               "data_sources": {"exposure": 0.4},
           },
       },
       layer="override",
   )

   sim = InteractiveContext(
       components=[
           BasePopulation(),
           Risk("risk_factor.test_risk"),
       ],
       configuration=config,
       plugin_configuration=base_plugins,
   )

   pop = sim.get_population(["test_risk.exposure"])
   n_exposed = (pop["test_risk.exposure"] == "exposed").sum()
   proportion = n_exposed / len(pop)
   # Approximately 40% should be exposed.
   print(f"Proportion exposed near 0.4: {np.isclose(proportion, 0.4, atol=0.02)}")

.. testoutput::

   Proportion exposed near 0.4: True


Configuration Summary
---------------------

.. list-table::
   :header-rows: 1

   * - Component
     - Key configuration options
     - Artifact data required
   * - ``Risk``
     - ``risk_factor.{name}.distribution_type``,
       ``risk_factor.{name}.data_sources.exposure``
     - ``risk_factor.{name}.distribution``,
       ``risk_factor.{name}.exposure``

.. note::

   For how exposure modifies disease rates via relative risks and PAF, see
   the :doc:`risk_effect` tutorial.

   For more advanced use cases - including polytomous risks, coverage gaps,
   alternative risk factors, and parameterized effect distributions - see
   the :doc:`non_standard_risk` tutorial.

