.. _vph_treatment_concept:

=========
Treatment
=========

.. contents::
   :depth: 2
   :local:
   :backlinks: none

.. toctree::
   :hidden:

   scale_up
   therapeutic_inertia

The ``vivarium_public_health`` treatment package provides
:ref:`components <components_concept>` for modeling health interventions and
their effects within a simulation. While :mod:`vivarium` supplies the value
:ref:`pipeline <values_concept>` framework for combining modifiers,
the treatment package uses those pipelines to model specific styles of
intervention.

The package is organized around four components:

1. **Absolute shift** — a simple intervention that directly sets the value of a
   target epidemiological measure for :term:`simulants <Simulant>` within a
   configured age range. See :class:`~vivarium_public_health.treatment.magic_wand.AbsoluteShift`.
2. **Linear scale-up** — a time-varying intervention that linearly interpolates
   exposure parameters between configured start and end values over a date
   range. See :ref:`scale_up_concept`.
3. **Therapeutic inertia** — a population-level scalar representing the
   probability that treatment is *not* escalated during a healthcare visit,
   drawn from a triangular distribution. See :ref:`therapeutic_inertia_concept`.
4. **Intervention and intervention effect** — components that wrap the
   :mod:`~vivarium_public_health.causal_factor` framework to model dichotomous
   intervention exposures and their relative-risk effects on target measures.
   See :class:`~vivarium_public_health.treatment.intervention.Intervention` and
   :class:`~vivarium_public_health.treatment.intervention.InterventionEffect`.

Absolute Shift
--------------

The :class:`~vivarium_public_health.treatment.magic_wand.AbsoluteShift` component
is the simplest intervention model. It registers an
:ref:`attribute modifier <values_concept>` on a target pipeline that replaces the
current value with a configured absolute value for all simulants within a
specified age range. When the ``target_value`` is set to ``"baseline"``, no
modification is applied.

.. code-block:: yaml

    configuration:
        intervention_on_my_cause:
            target_value: 0.01
            age_start: 15
            age_end: 65

Intervention and Intervention Effect
-------------------------------------

The :class:`~vivarium_public_health.treatment.intervention.Intervention` class
is a specialization of
:class:`~vivarium_public_health.causal_factor.exposure.CausalFactor` restricted
to ``"intervention"`` entity types. It models a dichotomous coverage exposure
(exposed vs. unexposed) and can source data from the artifact or the
configuration.

The :class:`~vivarium_public_health.treatment.intervention.InterventionEffect`
class models the effect of an ``Intervention`` on a target entity's measure
using relative risk data. It is a specialization of
:class:`~vivarium_public_health.causal_factor.effect.CausalFactorEffect`.

See Also
--------

- :mod:`vivarium_public_health.treatment`
- :mod:`vivarium_public_health.causal_factor`
