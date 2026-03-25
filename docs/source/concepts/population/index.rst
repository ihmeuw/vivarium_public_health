.. _vph_population_concept:

==========
Population
==========

.. contents::
   :depth: 2
   :local:
   :backlinks: none

.. toctree::
   :hidden:

   base_population
   mortality
   fertility

The ``vivarium_public_health`` population package provides :ref:`components <components_concept>` 
to create and maintain a population of :term:`simulants <Simulant>` over the course 
of a simulation. While the core :mod:`vivarium` framework supplies the
:ref:`population management <population_concept>` machinery (the state table,
population views, and simulant creation), the public health population package
is used to model simulant demographics by sampling age, sex, and location from empirical 
data, aging simulants forward through time, managing which are alive or dead, and 
introducing new simulants through fertility.

The package is organized around three cooperating concerns:

1. **Base population** — initialization of demographic attributes from artifact
   data, deterministic aging on each time step, and age-based untracking.
2. **Mortality** — :term:`mortality rates <Mortality Rate>` with support for modeled and 
   unmodeled cause-specific contributions.
3. **Fertility** — introduction of newborn simulants via deterministic,
   :term:`crude-birth-rate <Crude Birth Rate>`, or 
   :term:`age-specific-rate <Age-Specific Fertility Rate>` models.
