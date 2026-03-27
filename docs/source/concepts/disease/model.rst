.. _disease_model_concept:

=============
Disease Model
=============

.. contents::
   :depth: 3
   :local:
   :backlinks: none

The :class:`~vivarium_public_health.disease.model.DiseaseModel` is the top-level
:term:`component <Component>` that orchestrates disease states and transitions
into a complete disease model. It extends vivarium's
:class:`~vivarium.framework.state_machine.Machine` class.

A ``DiseaseModel`` is responsible for:

- Composing states and transitions into a coherent state machine
- Initializing simulants into disease states based on
  :term:`prevalence <Prevalence>` data
- Contributing :term:`cause-specific mortality <Excess Mortality Rate>` to the
  simulation's mortality pipeline

Building a Disease Model
------------------------

Disease models are built from two core building blocks — **states** and
**transitions**. States represent the distinct health conditions a simulant can
occupy, and transitions define the rules for moving between them. Together they
form a state machine that drives disease progression.

The typical workflow for constructing a disease model is:

1. **Create states** — instantiate the disease states that simulants can
   occupy (:ref:`see below <disease_state_concept>`).
2. **Add transitions** — connect states with transitions that define how
   simulants move between them (:ref:`see below <disease_transition_concept>`).
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

.. _disease_state_concept:

Disease States
--------------

Disease states represent the distinct health conditions a simulant can occupy
within a disease model. They extend vivarium's
:class:`~vivarium.framework.state_machine.State` class with public-health-specific
attributes such as :term:`prevalence <Prevalence>`,
:term:`disability weight <Disability Weight>`, and
:term:`excess mortality rate <Excess Mortality Rate>`.

Base Disease State
++++++++++++++++++

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

These are the primary way disease states are connected together.

Susceptible State
+++++++++++++++++

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
+++++++++++++

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
+++++++++++++++

:class:`~vivarium_public_health.disease.state.RecoveredState` represents
post-infection immunity or recovery. It automatically prepends
``recovered_from_`` to the provided cause name. For example,
``RecoveredState("measles")`` creates a state with ID
``recovered_from_measles``.

Like ``SusceptibleState``, this is a ``NonDiseasedState`` — it has no disability
weight or excess mortality. It is typically used as a terminal state in SIR
models.

Transient Disease State
+++++++++++++++++++++++

:class:`~vivarium_public_health.disease.state.TransientDiseaseState` uses
vivarium's :class:`~vivarium.framework.state_machine.Transient` mixin to create
states that simulants pass through instantaneously within a single time step.
This is useful for intermediate states in multi-step disease progressions, e.g.
an "infection" state that immediately resolves to either "with_condition" or
"recovered" based on further transition logic.

.. _disease_transition_concept:

Disease Transitions
-------------------

Disease transitions define the rules by which simulants move between disease
states. They extend vivarium's
:class:`~vivarium.framework.state_machine.Transition` with disease-specific
behavior — primarily the conversion of epidemiological rates into transition
probabilities and support for fixed-proportion and dwell-time-based transitions.

Rate Transition
+++++++++++++++

:class:`~vivarium_public_health.disease.transition.RateTransition` models
transitions governed by a time-varying rate. At each time step, the rate is
converted into a probability to determine which simulants transition.

Rate Type
~~~~~~~~~

Each ``RateTransition`` has a ``rate_type`` that determines how the transition
is named and what data it looks up by default:

.. list-table::
   :widths: 25 40 35
   :header-rows: 1

   * - Rate Type
     - Pipeline Name
     - Typical Use
   * - ``"incidence_rate"``
     - ``{output_state}.incidence_rate``
     - Susceptible → Diseased
   * - ``"remission_rate"``
     - ``{input_state}.remission_rate``
     - Diseased → Recovered
   * - ``"transition_rate"``
     - ``{input_state}_to_{output_state}.transition_rate``
     - Any other state-to-state transition

When using the convenience methods on
:class:`~vivarium_public_health.disease.state.BaseDiseaseState`, the rate type
is selected automatically based on the type of state:

- ``SusceptibleState.add_rate_transition()``
  defaults to ``"incidence_rate"``
- :meth:`DiseaseState.add_rate_transition()
  <vivarium_public_health.disease.state.DiseaseState.add_rate_transition>`
  defaults to ``"remission_rate"``
- :meth:`BaseDiseaseState.add_rate_transition()
  <vivarium_public_health.disease.state.BaseDiseaseState.add_rate_transition>`
  defaults to ``"transition_rate"``

Rate Conversion
~~~~~~~~~~~~~~~

Rates are converted to probabilities using one of two methods, controlled by the
``rate_conversion_type`` configuration option:

- **Linear** (default): :math:`p = r \cdot \Delta t`
- **Exponential**: :math:`p = 1 - e^{-r \cdot \Delta t}`

where :math:`r` is the rate and :math:`\Delta t` is the time step size.

All ``RateTransitions`` within a single
:class:`~vivarium_public_health.disease.model.DiseaseModel` must use the same
conversion type. The model validates this during ``on_post_setup``.

.. code-block:: yaml

   # Configuration to switch to exponential conversion
   susceptible_to_measles_TO_measles:
       rate_conversion_type: exponential

Risk Modification
~~~~~~~~~~~~~~~~~

Each ``RateTransition`` registers its rate as a risk-affected
:ref:`attribute pipeline <values_concept>`. This means risk factors and
interventions can modify the transition rate by registering modifiers on
the pipeline, without the disease model needing explicit knowledge of those
risks.

Proportion Transition
+++++++++++++++++++++

:class:`~vivarium_public_health.disease.transition.ProportionTransition` models
transitions where a fixed proportion of eligible simulants move to the output
state at each time step. The proportion is loaded from configuration or provided
directly:

.. code-block:: python

   infected.add_proportion_transition(recovered, proportion=0.05)

Unlike rate transitions, proportion transitions are **not** converted — the
configured value is used directly as the transition probability.

Dwell Time Transition
+++++++++++++++++++++

A dwell time transition is a plain
:class:`~vivarium.framework.state_machine.Transition` (with no rate or
proportion) that is gated by the :term:`dwell time <Dwell Time>` configured on
the source state. Simulants remain in the state for the specified duration, then
transition unconditionally.

Dwell time transitions are created using
:meth:`~vivarium_public_health.disease.state.BaseDiseaseState.add_dwell_time_transition`:

.. code-block:: python

   infected = DiseaseState("measles", dwell_time=pd.Timedelta(days=10))
   recovered = RecoveredState("measles")
   infected.add_dwell_time_transition(recovered)

In this example, simulants remain in the ``measles`` state for at least 10 days
before transitioning to ``recovered_from_measles``.

Adding Transitions to States
+++++++++++++++++++++++++++++

Transitions are typically added to states using the convenience methods on
:class:`~vivarium_public_health.disease.state.BaseDiseaseState`, rather than
being constructed directly:

.. code-block:: python

   healthy = SusceptibleState("measles")
   infected = DiseaseState("measles")
   recovered = RecoveredState("measles")

   # Rate-based transitions
   healthy.add_rate_transition(infected)     # incidence_rate (automatic)
   infected.add_rate_transition(recovered)   # remission_rate (automatic)

   # Or with explicit parameters
   healthy.add_rate_transition(
       infected,
       transition_rate="some.custom.rate",
       rate_type="transition_rate",
   )

   # Proportion-based transition
   infected.add_proportion_transition(recovered, proportion=0.1)

   # Dwell time transition
   infected.add_dwell_time_transition(recovered)

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

- :ref:`risk_attributable_disease_concept`
- :mod:`vivarium_public_health.population.mortality`
- :mod:`vivarium_public_health.disease.model`
- :mod:`vivarium_public_health.disease.models`
- :mod:`vivarium_public_health.disease.state`
- :mod:`vivarium_public_health.disease.transition`
