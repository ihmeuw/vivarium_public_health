.. _disease_transition_concept:

===================
Disease Transitions
===================

.. contents::
   :depth: 2
   :local:
   :backlinks: none

Disease transitions define the rules by which simulants move between
:ref:`disease states <disease_state_concept>`. They extend vivarium's
:class:`~vivarium.framework.state_machine.Transition` with disease-specific
behavior — primarily the conversion of epidemiological rates into transition
probabilities and support for fixed-proportion and dwell-time-based transitions.

Rate Transition
---------------

:class:`~vivarium_public_health.disease.transition.RateTransition` models
transitions governed by a time-varying rate. At each time step, the rate is
converted into a probability to determine which simulants transition.

Rate Type
+++++++++

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
+++++++++++++++

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
+++++++++++++++++

Each ``RateTransition`` registers its rate as a risk-affected
:ref:`attribute pipeline <values_concept>`. This means risk factors and
interventions can modify the transition rate by registering modifiers on
the pipeline, without the disease model needing explicit knowledge of those
risks.

Proportion Transition
---------------------

:class:`~vivarium_public_health.disease.transition.ProportionTransition` models
transitions where a fixed proportion of eligible simulants move to the output
state at each time step. The proportion is loaded from configuration or provided
directly:

.. code-block:: python

   infected.add_proportion_transition(recovered, proportion=0.05)

Unlike rate transitions, proportion transitions are **not** converted — the
configured value is used directly as the transition probability.

Dwell Time Transition
---------------------

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
-----------------------------

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

See Also
--------

- :ref:`disease_state_concept`
- :ref:`disease_model_concept`
- :mod:`vivarium_public_health.disease.transition`
