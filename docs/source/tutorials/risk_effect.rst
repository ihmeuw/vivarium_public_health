============
Risk Effects
============

:mod:`vivarium_public_health` provides components for modeling how risk factor
exposures modify disease outcomes. This tutorial covers **risk effects** -
the components that translate exposure into changes in disease rates.

For how simulants are assigned exposure values, see the :doc:`risk` tutorial.

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

A risk effect modifies a *target* rate based on exposure values of that risk.
A *target* is identified by an entity type, entity name, and measure in dotted
form: ``{entity_type}.{entity_name}.{measure}`` (e.g.,
``cause.lung_cancer.incidence_rate``).

There are two effect components:

- :class:`~vivarium_public_health.risks.effect.RiskEffect` - multiplies a
  target rate (e.g., disease incidence) by a relative risk for exposed
  simulants.
- :class:`~vivarium_public_health.risks.effect.NonLogLinearRiskEffect` -
  a variant where relative risk is parameterized by exposure level, using
  piecewise linear interpolation.

A typical risk model pairs a :class:`~vivarium_public_health.risks.base_risk.Risk`
with one or more :class:`~vivarium_public_health.risks.effect.RiskEffect`
components to modify disease rates. The
:class:`~vivarium_public_health.risks.base_risk.Risk` determines *who* is
exposed, and the :class:`~vivarium_public_health.risks.effect.RiskEffect`
determines *how much* that exposure changes the outcome.


Common Setup
------------

Every code example in this tutorial uses imports and helpers shown below.
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


Artifact Data Format
--------------------

This section documents the **key name** and **column layout** that the
risk effect components expect. Risk effect components support the
``data_sources`` configuration pattern that lets you override individual
keys with a scalar, DataFrame, or callable without rebuilding the artifact
(see `Data sources`_).


Data keys
^^^^^^^^^

The table below lists every data key used by the risk effect components.

.. list-table::
   :header-rows: 1

   * - Key
     - Index columns
     - Value columns
     - Used by
     - Configurable?
   * - ``risk_factor.{name}.relative_risk``
     - age, sex, year, parameter, affected_entity, affected_measure
     - ``value`` (relative risk per category)
     - :class:`~vivarium_public_health.risks.effect.RiskEffect`
     - Yes - ``risk_effect.{name}_on_{target}.data_sources.relative_risk``
   * - ``risk_factor.{name}.population_attributable_fraction``
     - age, sex, year, affected_entity, affected_measure
     - ``value`` (fraction)
     - :class:`~vivarium_public_health.risks.effect.RiskEffect`
     - Yes - ``risk_effect.{name}_on_{target}.data_sources.population_attributable_fraction``


Artifact data shapes
^^^^^^^^^^^^^^^^^^^^

The examples below use the data builders from the
:mod:`~vivarium_public_health._example_data` module; a production artifact
has the same column layout but with real GBD values.

.. testcode::

   from vivarium_public_health._example_data import (
       risk_relative_risk_dichotomous,
       risk_paf,
   )

   # risk_factor.{name}.relative_risk - RR per exposure category and target.
   rr = risk_relative_risk_dichotomous(2.0, "test_cause", "incidence_rate")
   print(rr.query("year_start == 1990 and parameter == 'exposed'").head(2).to_string(index=False))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

    age_start  age_end    sex  year_start  year_end parameter  value affected_entity affected_measure
          0.0 0.019178   Male        1990      1991   exposed    2.0      test_cause   incidence_rate
          0.0 0.019178 Female        1990      1991   exposed    2.0      test_cause   incidence_rate

.. testcode::

   # risk_factor.{name}.population_attributable_fraction - PAF per target.
   paf = risk_paf(0.3, "test_cause", "incidence_rate")
   print(paf.query("year_start == 1990").head(2).to_string(index=False))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

    age_start  age_end    sex  year_start  year_end affected_entity affected_measure  value
          0.0 0.019178   Male        1990      1991      test_cause   incidence_rate    0.3
          0.0 0.019178 Female        1990      1991      test_cause   incidence_rate    0.3


Data sources
^^^^^^^^^^^^

Risk effect components support a ``data_sources`` configuration pattern that
lets you override individual data keys without rebuilding the artifact. You
can override any key with:

- **Scalar** (int or float) - broadcast a constant value to all simulants.
- **DataFrame** - use the DataFrame directly.
- **Callable** - call the function at setup time to produce the data.
- **Artifact key** (string) - load a different key from the artifact.

:class:`~vivarium_public_health.risks.effect.RiskEffect` declares:

.. code-block:: yaml

   # Default configuration (loaded from the artifact):
   risk_effect.{name}_on_{target}:
     data_sources:
       relative_risk: "risk_factor.{name}.relative_risk"
       population_attributable_fraction: "risk_factor.{name}.population_attributable_fraction"

Any of these can be overridden with scalars in the simulation configuration:

.. code-block:: yaml

   configuration:
     risk_effect.{name}_on_{target_entity}.{target_name}.{target_measure}:
       data_sources:
         relative_risk: 2.0  # scalar, DataFrame, callable, or artifact key
         population_attributable_fraction: 0  # scalar or artifact key

.. note::

   In configuration keys, ``{target}`` expands to the dotted form
   ``{target_entity}.{target_name}.{target_measure}`` (e.g.,
   ``risk_effect.smoking_on_cause.lung_cancer.incidence_rate``).

For dichotomous risks, all data sources can be overridden this way. For
continuous risks, some keys (e.g., ``tmred`` and ``relative_risk_scalar``)
are loaded directly from the artifact and cannot be overridden via
configuration - see the `NonLogLinearRiskEffect`_ section for the
``setup=False`` workaround.


RiskEffect
----------

A :class:`~vivarium_public_health.risks.effect.RiskEffect` modifies disease
dynamics based on exposure. In the standard pattern, exposed simulants have
a higher incidence rate (multiplied by the relative risk) than unexposed
simulants.


Observing the effect
^^^^^^^^^^^^^^^^^^^^

The following example demonstrates that exposed simulants become infected
at a higher rate than unexposed simulants:

.. testcode::

   config = make_base_config()
   config.update(
       {
           "population": {"population_size": 10_000},
           "mortality": {"data_sources": {"all_cause_mortality_rate": 0}},
           "risk_factor.test_risk": {
               "distribution_type": "dichotomous",
               "data_sources": {"exposure": 0.5},
           },
           "risk_effect.test_risk_on_cause.test_cause.incidence_rate": {
               "data_sources": {
                   "relative_risk": 5.0,
                   "population_attributable_fraction": 0,
               },
           },
       },
       layer="override",
   )

   sim = InteractiveContext(
       components=[
           BasePopulation(),
           Risk("risk_factor.test_risk"),
           RiskEffect("risk_factor.test_risk", "cause.test_cause.incidence_rate"),
           SI("test_cause"),
       ],
       configuration=config,
       plugin_configuration=base_plugins,
   )

   # Step forward to allow infections to occur.
   for _ in range(3):
       sim.step()

   pop = sim.get_population(["test_cause", "test_risk.exposure"])
   exposed = pop[pop["test_risk.exposure"] == "exposed"]
   unexposed = pop[pop["test_risk.exposure"] == "unexposed"]

   exposed_infection_rate = (exposed["test_cause"] == "test_cause").sum() / len(exposed)
   unexposed_infection_rate = (unexposed["test_cause"] == "test_cause").sum() / len(
       unexposed
   )

   # With RR=5, the ratio of infection rates should be approximately 5.
   ratio = exposed_infection_rate / unexposed_infection_rate
   print(f"Rate ratio near 5: {np.isclose(ratio, 5, rtol=0.15)}")

.. testoutput::

   Rate ratio near 5: True


Multiple risk effects
^^^^^^^^^^^^^^^^^^^^^

A single :class:`~vivarium_public_health.risks.base_risk.Risk` can have
effects on multiple targets, and multiple risks can target the same disease.
Each :class:`~vivarium_public_health.risks.effect.RiskEffect` multiplies the
target rate independently:

.. testcode::

   config = make_base_config()
   config.update(
       {
           "population": {"population_size": 20_000},
           "mortality": {"data_sources": {"all_cause_mortality_rate": 0}},
           # First risk: smoking
           "risk_factor.smoking": {
               "distribution_type": "dichotomous",
               "data_sources": {"exposure": 0.3},
           },
           "risk_effect.smoking_on_cause.test_cause.incidence_rate": {
               "data_sources": {
                   "relative_risk": 3.0,
                   "population_attributable_fraction": 0,
               },
           },
           # Second risk: air pollution
           "risk_factor.air_pollution": {
               "distribution_type": "dichotomous",
               "data_sources": {"exposure": 0.7},
           },
           "risk_effect.air_pollution_on_cause.test_cause.incidence_rate": {
               "data_sources": {
                   "relative_risk": 2.0,
                   "population_attributable_fraction": 0,
               },
           },
       },
       layer="override",
   )

   sim = InteractiveContext(
       components=[
           BasePopulation(),
           Risk("risk_factor.smoking"),
           Risk("risk_factor.air_pollution"),
           RiskEffect("risk_factor.smoking", "cause.test_cause.incidence_rate"),
           RiskEffect("risk_factor.air_pollution", "cause.test_cause.incidence_rate"),
           SI("test_cause"),
       ],
       configuration=config,
       plugin_configuration=base_plugins,
   )

   for _ in range(3):
       sim.step()

   pop = sim.get_population(
       ["test_cause", "smoking.exposure", "air_pollution.exposure"]
   )
   # Compute infection rates by exposure group.
   both_exposed = pop[
       (pop["smoking.exposure"] == "exposed")
       & (pop["air_pollution.exposure"] == "exposed")
   ]
   smoking_only = pop[
       (pop["smoking.exposure"] == "exposed")
       & (pop["air_pollution.exposure"] == "unexposed")
   ]
   pollution_only = pop[
       (pop["smoking.exposure"] == "unexposed")
       & (pop["air_pollution.exposure"] == "exposed")
   ]
   neither_exposed = pop[
       (pop["smoking.exposure"] == "unexposed")
       & (pop["air_pollution.exposure"] == "unexposed")
   ]
   both_rate = (both_exposed["test_cause"] == "test_cause").sum() / len(both_exposed)
   smoking_only_rate = (smoking_only["test_cause"] == "test_cause").sum() / len(
       smoking_only
   )
   pollution_only_rate = (pollution_only["test_cause"] == "test_cause").sum() / len(
       pollution_only
   )
   neither_rate = (neither_exposed["test_cause"] == "test_cause").sum() / len(
       neither_exposed
   )

   # Combined RR is 3*2=6, so both-exposed vs neither should have ratio near 6.
   both_ratio = both_rate / neither_rate
   print(f"Both-exposed ratio near 6: {np.isclose(both_ratio, 6, rtol=0.2)}")
   # Smoking-only RR is 3, so ratio should be near 3.
   smoking_ratio = smoking_only_rate / neither_rate
   print(f"Smoking-only ratio near 3: {np.isclose(smoking_ratio, 3, rtol=0.1)}")
   # Air-pollution-only RR is 2, so ratio should be near 2.
   pollution_ratio = pollution_only_rate / neither_rate
   print(f"Pollution-only ratio near 2: {np.isclose(pollution_ratio, 2, rtol=0.1)}")

.. testoutput::

   Both-exposed ratio near 6: True
   Smoking-only ratio near 3: True
   Pollution-only ratio near 2: True


Population Attributable Fraction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The **population attributable fraction** (PAF) adjusts the baseline rate so
that the population-level rate (after applying relative risks to exposed
simulants) matches the original data. When PAF is 0, the baseline rate is
used as-is; when PAF is greater than 0, the baseline is reduced so that the
population-average rate remains consistent with the input data.

Without PAF correction, adding a risk with RR > 1 inflates the
population-average rate above the input data. With the correct PAF, the
baseline is scaled down to compensate.

.. testcode::

   # Run two simulations: one with PAF=0 and one with PAF=0.3.
   # The PAF simulation should have a lower overall infection rate
   # because the baseline rate is multiplied by (1 - PAF).

   def run_paf_sim(paf_value):
       config = make_base_config()
       config.update(
           {
               "population": {"population_size": 10_000},
               "mortality": {"data_sources": {"all_cause_mortality_rate": 0}},
               "risk_factor.test_risk": {
                   "distribution_type": "dichotomous",
                   "data_sources": {"exposure": 0.5},
               },
               "risk_effect.test_risk_on_cause.test_cause.incidence_rate": {
                   "data_sources": {
                       "relative_risk": 2.0,
                       "population_attributable_fraction": paf_value,
                   },
               },
           },
           layer="override",
       )
       sim = InteractiveContext(
           components=[
               BasePopulation(),
               Risk("risk_factor.test_risk"),
               RiskEffect("risk_factor.test_risk", "cause.test_cause.incidence_rate"),
               SI("test_cause"),
           ],
           configuration=config,
           plugin_configuration=base_plugins,
       )
       for _ in range(3):
           sim.step()
       pop = sim.get_population(["test_cause"])
       return (pop["test_cause"] == "test_cause").sum() / len(pop)

   rate_no_paf = run_paf_sim(0.0)
   rate_with_paf = run_paf_sim(0.3)

   # PAF > 0 reduces the baseline, so the overall population infection rate
   # should be lower than without PAF.
   print(f"PAF reduces population rate: {rate_with_paf < rate_no_paf}")

   # The ratio of rates should be approximately (1 - PAF) = 0.7.
   ratio = rate_with_paf / rate_no_paf
   print(f"Rate ratio near (1 - PAF): {np.isclose(ratio, 0.7, atol=0.05)}")

.. testoutput::

   PAF reduces population rate: True
   Rate ratio near (1 - PAF): True


NonLogLinearRiskEffect
----------------------

A :class:`~vivarium_public_health.risks.effect.NonLogLinearRiskEffect` models
the relationship between a **continuous** exposure and a target rate using
piecewise linear interpolation. Unlike ``RiskEffect`` (which applies a single
RR to exposed simulants), this component assigns each simulant an
individually interpolated RR based on their actual exposure level.

The relative risk data must contain a numeric ``parameter`` column with
exposure thresholds (typically 1000 values spanning the plausible range)
and corresponding ``value`` entries. The component also requires TMRED
(Theoretical Minimum-Risk Exposure Distribution) data, which defines the
exposure level at which relative risk equals 1.


Building relative risk data
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The RR data for ``NonLogLinearRiskEffect`` has a numeric ``parameter``
column (exposure thresholds) rather than categorical labels:

.. testcode::

   # Build RR data: 1000 exposure thresholds from 1 to 9.
   # RR increases linearly from 1.0 (at exposure=1) to 5.0 (at exposure=9).
   from vivarium_public_health._example_data import risk_relative_risk_continuous

   rr_data = risk_relative_risk_continuous(
       exposure_min=1, exposure_max=9, rr_min=1.0, rr_max=5.0
   )

   print(f"RR data shape: {rr_data.shape}")
   print(f"Parameter range: {rr_data['parameter'].min():.1f} to {rr_data['parameter'].max():.1f}")
   print(f"RR range: {rr_data['value'].min():.1f} to {rr_data['value'].max():.1f}")

.. testoutput::

   RR data shape: (1000, 6)
   Parameter range: 1.0 to 9.0
   RR range: 1.0 to 5.0


Running with continuous exposure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``NonLogLinearRiskEffect`` requires a continuous exposure pipeline.
Because it uses piecewise linear interpolation over numeric exposure
values, we need a ``Risk`` subclass that produces numeric exposures
rather than categorical ones. The simulation is created with
``setup=False`` so that RR and TMRED data can be written to the artifact
before components initialize:

.. note::

   ``ContinuousExposureRisk`` is a minimal demo shortcut that bypasses the
   parent ``Risk`` machinery (propensity, distribution lookup, framework
   randomness). Production continuous risks should use ``Risk`` directly
   with a continuous ``distribution_type``.

.. note::

   ``sim._data`` is an internal API. The ``setup=False`` pattern followed
   by ``sim._data.write()`` is specific to interactive and tutorial
   contexts where data must be injected before component setup.

.. testcode::

   class ContinuousExposureRisk(Risk):
       """A Risk that assigns random numeric exposures between 1 and 9."""

       def setup(self, builder):
           self.distribution_type = None
           col = f"{self.causal_factor.name}_exposure_for_non_loglinear_riskeffect"
           self._col = col
           builder.value.register_attribute_producer(
               f"{self.causal_factor.name}.exposure", source=self._get_exposure
           )
           builder.population.register_initializer(
               initializer=self._init_exposure, columns=col
           )

       def _init_exposure(self, pop_data):
           rng = np.random.default_rng(12345)
           values = rng.uniform(1, 9, size=len(pop_data.index))
           self.population_view.initialize(
               pd.Series(values, index=pop_data.index, name=self._col)
           )

       def _get_exposure(self, index):
           return self.population_view.get(index, self._col)

       def on_time_step_prepare(self, event):
           pass

   # Build RR data using the helper from _example_data.
   from vivarium_public_health._example_data import risk_relative_risk_continuous
   rr_data = risk_relative_risk_continuous(
       exposure_min=1, exposure_max=9, rr_min=1.0, rr_max=5.0
   )

   # TMRED defines the exposure level where RR = 1.
   # With min=max=1, the TMREL is exactly 1.0.
   tmred = {"distribution": "uniform", "min": 1, "max": 1, "inverted": False}

   config = make_base_config()
   config.update(
       {
           "population": {"population_size": 10_000},
           "mortality": {"data_sources": {"all_cause_mortality_rate": 0}},
       },
       layer="override",
   )

   risk = ContinuousExposureRisk("risk_factor.test_risk")
   effect = NonLogLinearRiskEffect(risk.name, "cause.test_cause.incidence_rate")

   sim = InteractiveContext(
       components=[BasePopulation(), risk, effect, SI("test_cause")],
       configuration=config,
       plugin_configuration=base_plugins,
       setup=False,
   )

   # Write NonLogLinearRiskEffect data to the artifact before setup.
   sim._data.write("risk_factor.test_risk.relative_risk", rr_data)
   sim._data.write("risk_factor.test_risk.tmred", tmred)
   sim._data.write("risk_factor.test_risk.population_attributable_fraction", 0)
   sim._data.write("cause.test_cause.incidence_rate", 0.5)

   sim.setup()

   for _ in range(3):
       sim.step()

   # The cached exposure column used by NonLogLinearRiskEffect.
   # This name is derived from the risk factor name by the component internally.
   exposure_col = "test_risk_exposure_for_non_loglinear_riskeffect"
   pop = sim.get_population(["test_cause", exposure_col])

   # Verify RR is monotonically increasing: split into quartiles and check
   # that infection rate increases with each quartile.
   quartiles = pd.qcut(pop[exposure_col], 4, labels=["Q1", "Q2", "Q3", "Q4"])
   rates_by_quartile = pop.groupby(quartiles, observed=True).apply(
       lambda g: (g["test_cause"] == "test_cause").sum() / len(g)
   )
   print(f"Monotonically increasing: {all(rates_by_quartile.diff().dropna() > 0)}")

.. testoutput::

   Monotonically increasing: True


Configuration Summary
---------------------

.. list-table::
   :header-rows: 1

   * - Component
     - Key configuration options
     - Artifact data required
   * - ``RiskEffect``
     - ``risk_effect.{name}_on_{target}.data_sources.relative_risk``,
       ``risk_effect.{name}_on_{target}.data_sources.population_attributable_fraction``
     - ``risk_factor.{name}.relative_risk``,
       ``risk_factor.{name}.population_attributable_fraction``
   * - ``NonLogLinearRiskEffect``
     - ``non_log_linear_risk_effect.{name}_on_{target}.data_sources.relative_risk``
     - ``risk_factor.{name}.relative_risk``,
       ``risk_factor.{name}.tmred``,
       ``risk_factor.{name}.population_attributable_fraction``

.. note::

   For more advanced use cases - including polytomous risks, coverage gaps,
   alternative risk factors, and parameterized effect distributions - see
   the :doc:`non_standard_risk` tutorial.
