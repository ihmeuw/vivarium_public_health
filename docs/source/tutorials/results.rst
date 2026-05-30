=========
Observers
=========

This tutorial gives minimal working examples of each public health observer
and demonstrates three ways to configure stratification.

.. contents::
   :local:
   :depth: 2

.. testsetup:: *

   import numpy as np
   import pandas as pd
   from loguru import logger
   logger.disable("vivarium")
   from vivarium import Component, InteractiveContext
   from vivarium.framework.engine import Builder
   from vivarium_public_health.disease import (
       DiseaseModel, DiseaseState, SusceptibleState, SI, SIS,
   )
   from vivarium_public_health.population import BasePopulation
   from vivarium_public_health.results import (
       DiseaseObserver, MortalityObserver, DisabilityObserver,
       CategoricalRiskObserver, ResultsStratifier,
   )
   from vivarium_public_health.risks import Risk
   from vivarium_public_health._example_data import (
       BASE_PLUGINS, make_base_config, build_cause_table,
       disease_disability_weight,
   )
   base_plugins = BASE_PLUGINS


Common Setup
------------

.. testcode::

   from vivarium import Component, InteractiveContext
   from vivarium.framework.engine import Builder
   from vivarium_public_health.disease import (
       DiseaseModel, DiseaseState, SusceptibleState, SI, SIS,
   )
   from vivarium_public_health.population import BasePopulation
   from vivarium_public_health.results import (
       DiseaseObserver, MortalityObserver, DisabilityObserver,
       CategoricalRiskObserver, ResultsStratifier,
   )
   from vivarium_public_health.risks import Risk
   from vivarium_public_health._example_data import (
       BASE_PLUGINS, make_base_config, build_cause_table,
       disease_disability_weight,
   )

   base_plugins = BASE_PLUGINS
   config = make_base_config()


DiseaseObserver
---------------

A :class:`~vivarium_public_health.results.DiseaseObserver` registers two
observations for a disease model: ``person_time`` in each state and
``transition_count`` between states.

.. testcode::

   config = make_base_config()
   config.update({"population": {"population_size": 1000}}, layer="model_override")

   sim = InteractiveContext(
       components=[
           BasePopulation(),
           SI("test_cause"),
           DiseaseObserver("test_cause"),
           ResultsStratifier(),
       ],
       configuration=config,
       plugin_configuration=base_plugins,
   )
   sim.step()
   results = sim.get_results()

   print(sorted(results.keys()))

.. testoutput::

   ['person_time_test_cause', 'transition_count_test_cause']

.. testcode::

   pt = results["person_time_test_cause"]
   print(pt.columns.tolist())
   print(pt["sub_entity"].tolist())

.. testoutput::

   ['measure', 'entity_type', 'entity', 'sub_entity', 'value']
   ['susceptible_to_test_cause', 'test_cause']

.. testcode::

   tc = results["transition_count_test_cause"]
   print(tc["measure"].iloc[0])
   print(tc["sub_entity"].iloc[0])
   assert tc["value"].iloc[0] > 0

.. testoutput::

   transition_count
   susceptible_to_test_cause_to_test_cause


MortalityObserver
-----------------

A :class:`~vivarium_public_health.results.MortalityObserver` registers
``deaths`` and ``ylls`` (years of life lost) observations, stratified by
cause. To produce non-zero values, the simulation needs a disease state
with non-zero ``excess_mortality_rate``.

.. testcode::

   healthy = SusceptibleState("test_cause")
   infected = DiseaseState("test_cause", excess_mortality_rate=build_cause_table(5.0))
   healthy.add_rate_transition(infected)
   fatal_model = DiseaseModel("test_cause", states=[healthy, infected])

   config = make_base_config()
   config.update({"population": {"population_size": 1000}}, layer="model_override")

   sim = InteractiveContext(
       components=[
           BasePopulation(),
           fatal_model,
           MortalityObserver(),
           ResultsStratifier(),
       ],
       configuration=config,
       plugin_configuration=base_plugins,
   )
   for _ in range(5):
       sim.step()

   results = sim.get_results()
   deaths = results["deaths"]
   print(deaths.columns.tolist())
   print(deaths["entity"].tolist())

.. testoutput::

   ['measure', 'entity_type', 'entity', 'sub_entity', 'value']
   ['test_cause', 'other_causes']

.. testcode::

   test_cause_deaths = deaths.loc[deaths["entity"] == "test_cause", "value"].iloc[0]
   assert test_cause_deaths > 0

   ylls = results["ylls"]
   test_cause_ylls = ylls.loc[ylls["entity"] == "test_cause", "value"].iloc[0]
   assert test_cause_ylls > 0


DisabilityObserver
------------------

A :class:`~vivarium_public_health.results.DisabilityObserver` registers a
``ylds`` (years lived with disability) observation. It requires at least one
disease state with a non-zero ``disability_weight``.

.. testcode::

   healthy = SusceptibleState("test_cause")
   infected = DiseaseState("test_cause", disability_weight=disease_disability_weight(0.3))
   healthy.add_rate_transition(infected)
   disability_model = DiseaseModel("test_cause", states=[healthy, infected])

   config = make_base_config()
   config.update({"population": {"population_size": 1000}}, layer="model_override")

   sim = InteractiveContext(
       components=[
           BasePopulation(),
           disability_model,
           DisabilityObserver(),
           ResultsStratifier(),
       ],
       configuration=config,
       plugin_configuration=base_plugins,
   )
   for _ in range(3):
       sim.step()

   results = sim.get_results()
   ylds = results["ylds"]
   print(ylds.columns.tolist())
   print(sorted(ylds["entity"].unique().tolist()))

.. testoutput::

   ['measure', 'entity_type', 'entity', 'sub_entity', 'stratification', 'value']
   ['all_causes', 'test_cause']

.. testcode::

   test_cause_ylds = ylds.loc[ylds["entity"] == "test_cause", "value"].iloc[0]
   assert test_cause_ylds > 0


CategoricalRiskObserver
-----------------------

A :class:`~vivarium_public_health.results.CategoricalRiskObserver` registers
a ``person_time`` observation stratified by exposure category for a
categorical risk factor.

.. testcode::

   config = make_base_config()
   config.update({"population": {"population_size": 1000}}, layer="model_override")

   sim = InteractiveContext(
       components=[
           BasePopulation(),
           Risk("risk_factor.test_risk"),
           CategoricalRiskObserver("test_risk"),
           ResultsStratifier(),
       ],
       configuration=config,
       plugin_configuration=base_plugins,
   )
   for _ in range(3):
       sim.step()

   results = sim.get_results()
   pt = results["person_time_test_risk"]
   print(pt.columns.tolist())
   print(sorted(pt["sub_entity"].tolist()))

.. testoutput::

   ['measure', 'entity_type', 'entity', 'sub_entity', 'value']
   ['exposed', 'unexposed']

.. testcode::

   assert all(pt["value"] > 0)
   exposed_pt = pt.loc[pt["sub_entity"] == "exposed", "value"].iloc[0]
   unexposed_pt = pt.loc[pt["sub_entity"] == "unexposed", "value"].iloc[0]
   assert exposed_pt > unexposed_pt


Stratification
--------------

Including a stratification
^^^^^^^^^^^^^^^^^^^^^^^^^^

Add a registered stratification to one observer via
``stratification.<observer_name>.include``. Here we include ``sex``, one of
the four stratifications registered by ``ResultsStratifier`` (``age_group``,
``current_year``, ``event_year``, and ``sex``):

.. testcode::

   config = make_base_config()
   config.update(
       {
           "population": {"population_size": 1000},
           "stratification": {
               "test_cause": {
                   "include": ["sex"],
                   "exclude": [],
               },
           },
       },
       layer="model_override",
   )

   sim = InteractiveContext(
       components=[
           BasePopulation(),
           SI("test_cause"),
           DiseaseObserver("test_cause"),
           ResultsStratifier(),
       ],
       configuration=config,
       plugin_configuration=base_plugins,
   )
   sim.step()

   pt = sim.get_results()["person_time_test_cause"]
   print(pt.columns.tolist())
   print(sorted(pt["sex"].unique().tolist()))
   print(len(pt))

.. testoutput::

   ['measure', 'entity_type', 'entity', 'sub_entity', 'sex', 'value']
   ['Female', 'Male']
   4


Excluding a default stratification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set global defaults with ``stratification.default``, then exclude specific
ones per observer with ``stratification.<observer_name>.exclude``:

.. testcode::

   config = make_base_config()
   config.update(
       {
           "population": {"population_size": 1000},
           "stratification": {
               "default": ["age_group", "sex"],
               "test_cause": {
                   "include": [],
                   "exclude": ["age_group"],
               },
           },
       },
       layer="model_override",
   )

   sim = InteractiveContext(
       components=[
           BasePopulation(),
           SI("test_cause"),
           DiseaseObserver("test_cause"),
           ResultsStratifier(),
       ],
       configuration=config,
       plugin_configuration=base_plugins,
   )
   sim.step()

   pt = sim.get_results()["person_time_test_cause"]
   # age_group excluded - only sex remains from defaults
   print(pt.columns.tolist())
   print(len(pt))

.. testoutput::

   ['measure', 'entity_type', 'entity', 'sub_entity', 'sex', 'value']
   4


Including a custom stratification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Register a custom stratification from any component, then reference it by
name in the observer's ``include`` list:

.. testcode::

   import pandas as pd

   class AgeCohortStratifier(Component):
       """Register a binary young/old stratification."""

       def setup(self, builder: Builder) -> None:
           builder.results.register_stratification(
               "age_cohort",
               ["young", "old"],
               mapper=self.map_age_cohort,
               is_vectorized=True,
               requires_attributes=["age"],
           )

       @staticmethod
       def map_age_cohort(pop: pd.DataFrame) -> pd.Series:
           age = pop.squeeze(axis=1)
           return age.apply(lambda a: "young" if a < 50 else "old")

.. testcode::

   config = make_base_config()
   config.update(
       {
           "population": {"population_size": 1000},
           "stratification": {
               "test_cause": {
                   "include": ["age_cohort"],
                   "exclude": [],
               },
           },
       },
       layer="model_override",
   )

   sim = InteractiveContext(
       components=[
           BasePopulation(),
           SI("test_cause"),
           DiseaseObserver("test_cause"),
           AgeCohortStratifier(),
           ResultsStratifier(),
       ],
       configuration=config,
       plugin_configuration=base_plugins,
   )
   sim.step()

   pt = sim.get_results()["person_time_test_cause"]
   print(pt.columns.tolist())
   print(sorted(pt["age_cohort"].unique().tolist()))

.. testoutput::

   ['measure', 'entity_type', 'entity', 'sub_entity', 'age_cohort', 'value']
   ['old', 'young']

.. testcode::

   young_total = pt.loc[pt["age_cohort"] == "young", "value"].sum()
   old_total = pt.loc[pt["age_cohort"] == "old", "value"].sum()
   assert young_total > 0
   assert old_total > 0
