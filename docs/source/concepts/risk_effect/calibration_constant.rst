.. _calibration_constant_concept:

====================
Calibration Constant
====================

.. contents::
   :depth: 2
   :local:
   :backlinks: none

Consider a rate drawn from input data, such as a disease incidence rate.
Multiplying this rate by each :term:`simulant's <Simulant>`
:term:`relative risk <Relative Risk>` shifts the population-level average
upward (assuming RR ≥ 1). We want the population-average rate to still match
the original input data after these relative risks are applied, so we first
adjust the baseline rate downward to compensate. This adjustment factor is the
:term:`calibration constant <Calibration Constant>`.

The calibration constant is typically derived from the :term:`population
attributable fraction <PAF>` — the proportion of the target rate attributable
to the risk factor. However, calibration constants are not always equal to
PAFs; the PAF is simply the most common source for this value.

How Calibration Works
---------------------

The ``_RiskAffectedPipeline``
class manages the interaction between the target rate pipeline and its
calibration constant. When a target rate pipeline is registered as
risk-affected (via
:func:`~vivarium_public_health.causal_factor.calibration_constant.register_risk_affected_rate_producer`),
the class:

1. Creates a companion ``{target}.calibration_constant`` pipeline whose value
   is computed from all risk effects that modify this target.
2. Registers a post-processor on the target rate pipeline that multiplies
   non-zero rate values by :math:`(1 - c)`, where :math:`c` is the joint
   calibration constant:

.. math::

   \text{adjusted\_rate} = \text{base\_rate} \times (1 - c)

After this adjustment, when individual simulants' relative risks are
multiplied in, the population-average rate matches the original input data.

Multiple Risk Effects
---------------------

When several risk effects target the same rate (e.g., two risk factors both
affecting ``cause.incidence_rate``), each effect registers its own PAF data as
a modifier on the shared calibration constant pipeline. The pipeline combines
these individual contributions into a single joint calibration constant using
the ``raw_union_post_processor``, so each risk effect's contribution is
composed automatically without any special handling by the modeler.

See Also
--------

- :ref:`relative_risk_concept`
- :ref:`vph_risk_exposure_concept`
- :mod:`vivarium_public_health.causal_factor.calibration_constant`
