.. _therapeutic_inertia_concept:

====================
Therapeutic Inertia
====================

.. contents::
   :depth: 2
   :local:
   :backlinks: none

The :class:`~vivarium_public_health.treatment.therapeutic_inertia.TherapeuticInertia`
component models the phenomenon where treatment algorithms deviate from
clinical guidelines — for example, when a clinician does not escalate
treatment even though guidelines recommend it.

How It Works
------------

At setup, a single scalar value is drawn from a triangular distribution and
exposed as the ``therapeutic_inertia``
:ref:`attribute pipeline <values_concept>`. This value represents the
probability that treatment is *not* escalated during a healthcare visit, and
remains constant for the entire simulation.

The triangular distribution is parameterized by three values:

- ``triangle_min`` — lower bound of the distribution (default 0.65)
- ``triangle_max`` — upper bound of the distribution (default 0.9)
- ``triangle_mode`` — the peak (most likely value) of the distribution
  (default 0.875)

Configuration
-------------

.. code-block:: yaml

    configuration:
        therapeutic_inertia:
            triangle_min: 0.65
            triangle_max: 0.9
            triangle_mode: 0.875

Other components can consume the ``therapeutic_inertia`` pipeline to
incorporate this probability into their treatment decision logic.

See Also
--------

- :ref:`vph_treatment_concept`
- :ref:`scale_up_concept`
- :mod:`vivarium_public_health.treatment.therapeutic_inertia`
