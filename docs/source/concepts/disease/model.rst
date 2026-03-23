.. _disease_model_concept:

=============
Disease Model
=============

.. contents::
   :depth: 2
   :local:
   :backlinks: none

The :class:`~vivarium_public_health.disease.model.DiseaseModel` is the top-level
:term:`component <Component>` that orchestrates
:ref:`disease states <disease_state_concept>` and
:ref:`transitions <disease_transition_concept>` into a complete disease
simulation. It extends vivarium's
:class:`~vivarium.framework.state_machine.Machine` class.

A ``DiseaseModel`` is responsible for:

- Composing states and transitions into a coherent state machine
- Initializing simulants into disease states based on
  :term:`prevalence <Prevalence>` data
- Contributing :term:`cause-specific mortality <Excess Mortality Rate>` to the
  simulation's mortality pipeline

Building a Disease Model
------------------------

The typical workflow for constructing a disease model is:

1. **Create states** — instantiate the
   :ref:`disease states <disease_state_concept>` that simulants can occupy.
2. **Add transitions** — connect states with
   :ref:`transitions <disease_transition_concept>` that define how simulants
   move between them.
3. **Create the model** — pass the states to a ``DiseaseModel`` which
   registers them as a state machine.

.. code-block:: python

   from vivarium_public_health.disease import (
       DiseaseModel,
       DiseaseState,
       RecoveredState,
       SusceptibleState,
   )

   # 1. Create states
   susceptible = SusceptibleState("measles")
   infected = DiseaseState("measles")
   recovered = RecoveredState("measles")

   # 2. Add transitions
   susceptible.add_rate_transition(infected)   # incidence_rate
   infected.add_rate_transition(recovered)     # remission_rate

   # 3. Compose into a model
   model = DiseaseModel("measles", states=[susceptible, infected, recovered])

The ``cause`` parameter names the disease. The optional ``cause_type`` parameter
(default ``"cause"``) can be set to ``"sequela"`` when modeling a specific
sequela rather than a top-level cause.

Initialization
--------------

When the simulation creates a population,
:meth:`~vivarium_public_health.disease.model.DiseaseModel.initialize_state`
assigns each simulant to a disease state:

- For simulants with **age > 0**, each state's ``prevalence`` data source
  determines the probability of starting in that state.
- For simulants at **age 0** (newborns), each state's ``birth_prevalence`` data
  source is used instead (see :term:`Birth Prevalence`).

One state acts as the **residual state** — its prevalence is calculated as
``1 − Σ(other state prevalences)``. By default, this is the model's
:class:`~vivarium_public_health.disease.state.SusceptibleState`. A different
state can be designated by passing ``residual_state`` to the constructor.

Mortality Integration
---------------------

A ``DiseaseModel`` participates in the simulation's mortality accounting by
modifying the ``cause_specific_mortality_rate``
:ref:`attribute pipeline <values_concept>`. During setup, it loads
cause-specific mortality rate (CSMR) data from the artifact and registers a
modifier that adds the disease's CSMR to the total.

For YLD-only causes (those with no associated mortality), the CSMR defaults
to 0. This is detected automatically from the cause's ``restrictions`` metadata
in the artifact.

The ``cause_specific_mortality_rate`` data source is configurable:

.. code-block:: yaml

   measles:
       data_sources:
           cause_specific_mortality_rate: cause.measles.cause_specific_mortality_rate

See :mod:`vivarium_public_health.population.mortality` for details on how
cause-specific mortality rates are aggregated.

Pre-built Models
----------------

The :mod:`~vivarium_public_health.disease.models` module provides factory
functions for commonly used disease model parameterizations. These functions
create the appropriate states, add transitions, and return a configured
``DiseaseModel``:

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Function
     - States
     - Description
   * - :func:`~vivarium_public_health.disease.models.SI`
     - Susceptible → Infected
     - One-way infection with no recovery. Suitable for chronic or
       irreversible conditions.
   * - :func:`~vivarium_public_health.disease.models.SIR`
     - Susceptible → Infected → Recovered
     - Infection followed by permanent immunity.
   * - :func:`~vivarium_public_health.disease.models.SIS`
     - Susceptible ↔ Infected
     - Cyclic infection and recovery with no lasting immunity.
   * - :func:`~vivarium_public_health.disease.models.SIS_fixed_duration`
     - Susceptible ↔ Infected (dwell)
     - SIS variant where infection lasts a configurable number of days
       using :term:`dwell time <Dwell Time>`.
   * - :func:`~vivarium_public_health.disease.models.SIR_fixed_duration`
     - Susceptible → Infected (dwell) → Recovered
     - SIR variant where infection lasts a configurable number of days.
   * - :func:`~vivarium_public_health.disease.models.NeonatalSWC_without_incidence`
     - Susceptible, With Condition
     - Neonatal model with :term:`birth prevalence <Birth Prevalence>` only.
       No transitions — simulants remain in their initial state.
   * - :func:`~vivarium_public_health.disease.models.NeonatalSWC_with_incidence`
     - Susceptible → With Condition
     - Neonatal model with :term:`birth prevalence <Birth Prevalence>` and
       an :term:`incidence rate <Incidence Rate>` transition from susceptible
       to the condition.

Each factory function takes a ``cause`` string and returns a fully configured
``DiseaseModel``. For example:

.. code-block:: python

   from vivarium_public_health.disease.models import SIR

   measles_model = SIR("measles")

The fixed-duration variants also accept a ``duration`` parameter specifying the
infection duration in days.

See Also
--------

- :ref:`disease_state_concept`
- :ref:`disease_transition_concept`
- :ref:`special_disease_concept`
- :mod:`vivarium_public_health.population.mortality`
- :mod:`vivarium_public_health.disease.model`
- :mod:`vivarium_public_health.disease.models`
