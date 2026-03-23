.. _disease_state_concept:

==============
Disease States
==============

.. contents::
   :depth: 2
   :local:
   :backlinks: none

Disease states represent the distinct health conditions a simulant can occupy
within a :ref:`disease model <disease_model_concept>`. They extend vivarium's
:class:`~vivarium.framework.state_machine.State` class with public-health-specific
attributes such as :term:`prevalence <Prevalence>`,
:term:`disability weight <Disability Weight>`, and
:term:`excess mortality rate <Excess Mortality Rate>`.

States are not used in isolation. They are composed into a
:class:`~vivarium_public_health.disease.model.DiseaseModel` together with
:ref:`transitions <disease_transition_concept>` that define how simulants move
between them.

Base Disease State
------------------

:class:`~vivarium_public_health.disease.state.BaseDiseaseState` provides the
common foundation for all disease states. It manages:

- **Prevalence data** — used during :ref:`initialization <state_initialization>`
  to assign simulants to states.
- **Event tracking** — columns recording when a simulant entered the state
  (``{state_id}_event_time``) and how many times it has entered
  (``{state_id}_event_count``).
- **Dwell time** — an optional minimum duration a simulant must remain in the
  state before any outgoing transition can fire (see :term:`Dwell Time`).

``BaseDiseaseState`` also provides convenience methods for attaching transitions:

- :meth:`~vivarium_public_health.disease.state.BaseDiseaseState.add_rate_transition`
- :meth:`~vivarium_public_health.disease.state.BaseDiseaseState.add_proportion_transition`
- :meth:`~vivarium_public_health.disease.state.BaseDiseaseState.add_dwell_time_transition`

These are the primary way disease states are connected together
(see :ref:`disease_transition_concept`).

Susceptible State
-----------------

:class:`~vivarium_public_health.disease.state.SusceptibleState` represents the
absence of disease. It automatically prepends ``susceptible_to_`` to the
provided cause name to form the state ID. For example,
``SusceptibleState("measles")`` creates a state with ID
``susceptible_to_measles``.

The susceptible state serves as the **residual state** in a disease model: its
:term:`prevalence <Prevalence>` is calculated as
``1 − Σ(other state prevalences)`` rather than loaded from data directly. This
ensures the total initialization weights across all states always sum to 1.

When adding a transition from a ``SusceptibleState`` to a
:class:`~vivarium_public_health.disease.state.DiseaseState` without specifying
a rate, the default rate type is :term:`incidence rate <Incidence Rate>`:

.. code-block:: python

   healthy = SusceptibleState("measles")
   infected = DiseaseState("measles")
   healthy.add_rate_transition(infected)  # uses cause.measles.incidence_rate

Disease State
-------------

:class:`~vivarium_public_health.disease.state.DiseaseState` represents the
active presence of a disease. In addition to the base state attributes, it
provides:

- **Disability weight** — a :term:`Disability Weight` pipeline that contributes
  to the overall ``all_causes.disability_weight``
  :ref:`attribute pipeline <values_concept>`. The weight is only applied
  to simulants currently in this state.
- **Excess mortality rate** — an :term:`EMR` pipeline that adds disease-specific
  mortality on top of the background mortality rate. This is provided by the
  ``ExcessMortalityState`` mixin.
- **Dwell time** — a configurable minimum
  :term:`dwell time <Dwell Time>` before outgoing transitions become eligible.

When adding a transition from a ``DiseaseState`` without specifying a rate, the
default rate type is :term:`remission rate <Remission Rate>`:

.. code-block:: python

   infected = DiseaseState("measles")
   recovered = RecoveredState("measles")
   infected.add_rate_transition(recovered)  # uses cause.measles.remission_rate

Data sources for a ``DiseaseState`` are configurable through the simulation
configuration. The defaults load from the artifact:

.. code-block:: yaml

   measles:
       data_sources:
           prevalence: cause.measles.prevalence
           birth_prevalence: 0.0
           dwell_time: 0.0
           disability_weight: cause.measles.disability_weight
           excess_mortality_rate: cause.measles.excess_mortality_rate

Recovered State
---------------

:class:`~vivarium_public_health.disease.state.RecoveredState` represents
post-infection immunity or recovery. It automatically prepends
``recovered_from_`` to the provided cause name. For example,
``RecoveredState("measles")`` creates a state with ID
``recovered_from_measles``.

Like ``SusceptibleState``, this is a ``NonDiseasedState`` — it has no disability
weight or excess mortality. It is typically used as a terminal state in SIR
models.

Transient Disease State
-----------------------

:class:`~vivarium_public_health.disease.state.TransientDiseaseState` uses
vivarium's :class:`~vivarium.framework.state_machine.Transient` mixin to create
states that simulants pass through instantaneously within a single time step.
This is useful for intermediate states in multi-step disease progressions, e.g.
an "infection" state that immediately resolves to either "with_condition" or
"recovered" based on further transition logic.

.. _state_initialization:

Initialization
--------------

When a simulation starts, the
:class:`~vivarium_public_health.disease.model.DiseaseModel` assigns each
simulant to an initial disease state based on :term:`prevalence <Prevalence>`
data:

- For simulants initialized at **age > 0**, the model uses each state's
  ``prevalence`` data source.
- For simulants initialized at **age 0** (newborns), the model uses each
  state's ``birth_prevalence`` data source (see :term:`Birth Prevalence`).

The residual state (typically ``SusceptibleState``) absorbs any remaining
probability: its prevalence is ``1 − Σ(other state prevalences)``.

See Also
--------

- :ref:`disease_transition_concept`
- :ref:`disease_model_concept`
- :mod:`vivarium_public_health.disease.state`
