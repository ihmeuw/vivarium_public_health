.. _vph_risk_exposure_concept:

=============
Risk Exposure
=============

.. contents::
   :depth: 2
   :local:
   :backlinks: none

.. toctree::
   :hidden:

   exposure
   distributions

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
