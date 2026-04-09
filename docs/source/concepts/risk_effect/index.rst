.. _vph_risk_effect_concept:

===========
Risk Effect
===========

.. contents::
   :depth: 2
   :local:
   :backlinks: none

.. toctree::
   :hidden:

   relative_risk
   calibration_constant

The ``vivarium_public_health`` risk effect package provides
:ref:`components <components_concept>` that modify target rates (such as
disease incidence or mortality) based on a :term:`simulant's <Simulant>`
:ref:`exposure <vph_risk_exposure_concept>` to a risk factor. While the core
:mod:`vivarium` framework supplies the :ref:`values <values_concept>` pipeline
machinery, this package uses it to translate each simulant's exposure level into
a multiplicative adjustment on one or more target rates.

The package is organized around two key concepts:

1. **Relative Risk** — a measure of how much more likely an outcome is for a
   simulant at a given exposure level compared to a reference level. A relative
   risk of 1 means the simulant is no more or less likely to experience the
   outcome than at the reference level. The risk effect component computes a
   per-simulant relative risk and registers it as a modifier on the target rate
   pipeline.
2. **Calibration Constant** — a value, typically derived from the
   :term:`population attributable fraction <PAF>`, that adjusts the baseline
   target rate so that, after relative risk multiplication across the
   population, the overall rate remains consistent with input data.
