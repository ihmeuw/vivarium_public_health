.. _vph_disease_concept:

=======
Disease
=======

.. contents::
   :depth: 2
   :local:
   :backlinks: none

.. toctree::
   :hidden:

   state
   transition
   model
   special_disease

The ``vivarium_public_health`` disease package provides
:term:`components <Component>` for modeling disease progression in a
population of simulants. It builds on vivarium's
:class:`~vivarium.framework.state_machine.Machine` framework to represent
diseases as state machines where simulants move between health states according
to epidemiological rates.

The package is organized around four cooperating concerns:

1. **Disease states** — the health conditions a simulant can occupy
   (susceptible, infected, recovered, transient). See :ref:`disease_state_concept`.
2. **Transitions** — the rules governing movement between states
   (rate-based, proportion-based, or dwell-time-based). See
   :ref:`disease_transition_concept`.
3. **Disease models** — the orchestrator that composes states and transitions
   into a complete disease simulation component, integrating with
   mortality and the broader vivarium framework. See
   :ref:`disease_model_concept`.
4. **Special disease** — an alternative modeling approach where disease state is
   derived directly from risk factor exposure rather than explicit state
   transitions. See :ref:`special_disease_concept`.
