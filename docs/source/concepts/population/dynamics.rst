.. _population_dynamics_concept:

===================
Population Dynamics
===================

.. contents::
   :depth: 2
   :local:
   :backlinks: none

Core Time Step Behavior
-----------------------

After initialization, the population package updates simulants every time step:

1. Age living simulants.
2. Apply mortality hazards and deaths.
3. Optionally add newborn simulants through a fertility model.
4. Mark simulants as untracked when they exceed configured age limits.

This behavior is implemented by cooperating components rather than a single
monolithic component.

Aging
-----

During each ``time_step`` event, living simulants have their ``age`` advanced
by the event step size. This update is handled by
:class:`vivarium_public_health.population.base_population.BasePopulation`.

The age update is deterministic conditional on the simulation clock and ensures
all downstream components see a consistent age state within a time step.

Aging Out and Untracking
------------------------

:class:`vivarium_public_health.population.base_population.AgeOutSimulants`
marks simulants as aged out when their age reaches
``population.untracking_age``. This provides a clean way to bound the active
population for models focused on specific age windows.

Related Dynamics
----------------

Mortality and fertility each have dedicated concept pages:

- :ref:`population_mortality_concept`
- :ref:`population_fertility_concept`

See Also
--------

- :mod:`vivarium_public_health.population.base_population`
