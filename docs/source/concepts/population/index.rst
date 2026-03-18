.. _population_concept:

==========
Population
==========

The public health population package provides the components that initialize,
evolve, and retire simulants over time. It combines demographic initialization,
mortality, and fertility in a modular way so models can choose the level of
detail needed for a given use case.

.. contents::
   :depth: 2
   :local:
   :backlinks: none

Overview
--------

Population behavior in ``vivarium_public_health`` is organized into three
conceptual layers:

1. **Initialization**: Sample initial age, sex, and location from artifact
   demographic data and assign simulation metadata such as entrance and exit
   times.
2. **Dynamics**: Advance simulants through time by aging them, applying
   mortality, and optionally adding new birth cohorts.
3. **Configuration**: Control behavior through configuration keys and data
   source settings so models can move between deterministic and data-driven
   population assumptions.

These concepts map directly to the population API reference under
``api_reference/population``.

Concept Guides
--------------

.. toctree::
   :maxdepth: 1

   initialization
   dynamics
   mortality
   fertility
   configuration
