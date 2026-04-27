.. _vph_risk_concept:

====
Risk
====

.. contents::
   :depth: 2
   :local:
   :backlinks: none

.. toctree::
   :hidden:

   exposure
   distributions
   relative_risk
   calibration_constant

.. _vph_risk_exposure_concept:

Risk Exposure
=============

The ``vivarium_public_health`` risk exposure package provides
:ref:`components <components_concept>` to assign risk factor exposure values to
:term:`simulants <Simulant>` during a simulation. While the core :mod:`vivarium`
framework supplies the :ref:`values <values_concept>` and
:ref:`population <population_concept>` machinery, this package uses it to assign
each simulant a :term:`propensity <Propensity>` and translate that propensity
into an exposure level through a statistical distribution.

The package is organized around two key concepts:

1. **Exposure** — involves initialization of a fixed :term:`propensity <Propensity>` for
   each simulant, selection of a distribution type, and registration of an
   exposure :ref:`pipeline <values_concept>` that converts propensities into
   exposure values.
2. **Distributions** — the statistical models that translate propensities into
   exposure levels, ranging from simple :term:`dichotomous <Dichotomous Distribution>`
   (exposed/unexposed) categories to complex
   :term:`ensemble <Ensemble Distribution>` distributions.

.. _vph_risk_effect_concept:

Risk Effect
===========

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
