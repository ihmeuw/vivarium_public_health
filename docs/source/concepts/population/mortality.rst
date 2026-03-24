.. _mortality_concept:

=========
Mortality
=========

.. contents::
   :depth: 2
   :local:
   :backlinks: none

The :class:`~vivarium_public_health.population.mortality.Mortality` component
models :term:`all-cause mortality <ACMR>` and allows for :term:`cause-specific <CSMR>` 
contributions from both explicitly modeled disease components and an aggregated set 
of :term:`unmodeled causes <Unmodeled Cause>`. It is instantiated as a sub-component of
:class:`~vivarium_public_health.population.base_population.BasePopulation` and
participates in the ``time_step`` :ref:`event <event_concept>` to determine
which simulants die, record their cause of death, and calculate
:term:`years of life lost <Years of Life Lost>`.

The component manages three :ref:`state table <population_concept>` attributes:

- ``is_alive`` — whether the simulant is still living
- ``cause_of_death`` — the cause assigned to a dying simulant
- ``years_of_life_lost`` — the residual life expectancy at death, computed
  from the :term:`TMRLE` table

:term:`Cause-Deleted Mortality`
-------------------------------

The effective :term:`mortality rate <Mortality Rate>` is constructed using a
:term:`cause-deleted <Cause-Deleted Mortality>` pattern.
At each time step the component computes:

.. math::

   \text{mortality\_rate} = \text{ACMR} - \text{modeled CSMR}
   - \text{unmodeled CSMR (raw)} + \text{unmodeled CSMR (modified)}

where:

- **:term:`ACMR`** is all-cause mortality from the configured data source.
- **Modeled :term:`CSMR`** is tracked via a separate ``cause_specific_mortality_rate``
  :ref:`attribute pipeline <values_concept>` that disease :ref:`components <components_concept>`
  contribute to.
- **:term:`Unmodeled <Unmodeled Cause>` CSMR** captures causes not explicitly
  modeled but still potentially affected by modeled risk factors.

During each time step, the combined mortality hazard is converted into death
events using random draws. For simulants that die, a cause is selected
probabilistically from the hazard-weighted causes.

Unmodeled Causes
~~~~~~~~~~~~~~~~

:term:`Unmodeled causes <Unmodeled Cause>` are those that are not explicitly
modeled in the simulation but should still be represented in mortality accounting.
At setup, the component loads the :term:`cause-specific mortality rate <CSMR>` for
each named cause and aggregates them into a single unmodeled CSMR term. 
:class:`~vivarium_public_health.risks.effect.RiskEffect`  components can then modify 
this term by registering modifiers on the  ``affected_unmodeled.cause_specific_mortality_rate`` 
:ref:`attribute pipeline <values_concept>`, thereby allowing risks to influence mortality 
from causes that are not themselves full disease models.

Omitting Mortality
------------------

``Mortality`` is always present as a sub-component of ``BasePopulation``, but
it can be made inert. If you set the :term:`all-cause mortality rate <ACMR>` to
zero in your configuration and do not register any :term:`cause-specific <CSMR>`
mortality contributions, no simulant will ever die:

.. code-block:: yaml

   configuration:
       mortality:
           data_sources:
               all_cause_mortality_rate: 0

Alternatively, you can subclass :class:`~vivarium_public_health.population.base_population.BasePopulation` 
and omit ``Mortality`` from its sub-components entirely.

See Also
--------

- :ref:`base_population_concept`
- :ref:`fertility_concept`
- :mod:`vivarium_public_health.population.mortality`
