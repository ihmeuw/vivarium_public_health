.. _population_mortality_concept:

====================
Population Mortality
====================

.. contents::
   :depth: 2
   :local:
   :backlinks: none

Overview
--------

:class:`vivarium_public_health.population.mortality.Mortality` models all-cause
mortality with optional modeled and unmodeled cause-specific contributions.
It is responsible for updating:

- ``is_alive``
- ``cause_of_death``
- ``years_of_life_lost``

Cause-Deleted Mortality
-----------------------

The effective mortality rate follows a cause-deleted pattern:

.. math::

   \text{mortality\_rate} = \text{ACMR} - \text{modeled CSMR}
   - \text{unmodeled CSMR (raw)} + \text{unmodeled CSMR (modified)}

where:

- ACMR is all-cause mortality from the configured data source.
- Modeled CSMR is contributed by disease components through pipeline modifiers.
- Unmodeled CSMR captures causes not explicitly modeled but still potentially
  affected by modeled risks.

During each time step, mortality hazards are converted into death events using
random draws. For simulants that die, a cause is selected probabilistically
from the hazard-weighted causes.

Unmodeled Causes
----------------

The ``mortality.unmodeled_causes`` configuration key identifies causes that are
not explicitly modeled but should still be represented in mortality accounting.
These causes are aggregated into the unmodeled CSMR term and can be modified by
risk components through the affected-unmodeled pipeline.

Configuration
-------------

Key configuration options include:

- ``mortality.data_sources.all_cause_mortality_rate``
- ``mortality.data_sources.unmodeled_cause_specific_mortality_rate``
- ``mortality.data_sources.life_expectancy``
- ``mortality.unmodeled_causes``

See :ref:`population_configuration_concept` for a broader configuration summary.

See Also
--------

- :mod:`vivarium_public_health.population.mortality`
- :ref:`population_dynamics_concept`
- :ref:`population_fertility_concept`
