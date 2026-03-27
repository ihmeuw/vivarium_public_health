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

   model
   risk_attributable_disease

The ``vivarium_public_health`` disease package provides
:term:`components <Component>` for modeling disease progression in a
population of simulants. It builds on vivarium's
:class:`~vivarium.framework.state_machine.Machine` framework to represent
diseases as state machines where simulants move between health states according
to epidemiological rates.

The package is organized around three cooperating concerns:

1. **Disease models** — the orchestrator that composes states and transitions
   into a complete disease simulation component, integrating with
   mortality and the broader vivarium framework. States represent the health
   conditions a simulant can occupy (susceptible, infected, recovered,
   transient), while transitions define the rules governing movement between
   them (rate-based, proportion-based, or dwell-time-based). See
   :ref:`disease_model_concept`.
2. **Risk attributable disease** — an alternative modeling approach where
   disease state is derived directly from risk factor exposure rather than
   explicit state transitions. See :ref:`risk_attributable_disease_concept`.
