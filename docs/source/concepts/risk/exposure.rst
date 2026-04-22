.. _risk_exposure_model_concept:

========
Exposure
========

.. contents::
   :depth: 2
   :local:
   :backlinks: none

The :class:`~vivarium_public_health.risks.base_risk.Risk` component (and its
parent class :class:`~vivarium_public_health.causal_factor.exposure.CausalFactor`)
is the entry point for modeling risk exposure. It has two responsibilities:
assigning each :term:`simulant <Simulant>` a stable :term:`propensity <Propensity>`
and registering an exposure :ref:`pipeline <values_concept>` that converts that
propensity into an exposure value using the configured
:ref:`distribution <risk_distributions_concept>`.

.. _propensity_concept:

Propensity
----------

When new simulants are created, the component draws a uniform random value in
[0, 1] for each simulant and stores it in a ``propensity`` column on the
:ref:`state table <population_concept>`. This value is fixed for the life of
the simulant and acts as the simulant's position in the cumulative distribution
of the risk factor. Because propensities are drawn from a dedicated
:ref:`randomness stream <crn_concept>`, they are reproducible across simulation
branches that share a random seed.

The propensity is used as the quantile input to the distribution's
:term:`percent-point function <PPF>`: given a propensity *q*, the distribution
returns the exposure value *x* such that *P(X ≤ x) = q*. This mechanism
ensures that the simulated exposure distribution matches the empirical
distribution sourced from artifact data, while also allowing pipeline modifiers
to shift the distribution over time without needing to re-draw random numbers.

.. _exposure_pipeline_concept:

Exposure Pipeline
-----------------

During setup, the component registers an ``{risk_name}.exposure``
:ref:`attribute pipeline <values_concept>`. The pipeline's source is the
distribution's PPF, which itself is registered as a separate
``{risk_name}.exposure_distribution.ppf`` pipeline. This two-layer design
allows other components to modify the distribution parameters (e.g., shifting
the mean) independently of the final exposure value.

Configuration
+++++++++++++

Risk exposure data is loaded from the simulation artifact by default, but can
be overridden with a scalar value or a covariate name in the configuration.
The distribution type is similarly configurable. See the
:class:`~vivarium_public_health.risks.base_risk.Risk` class documentation for
the full set of configuration keys and YAML examples.

.. _exposure_rebinning_concept:

Rebinning and Category Thresholds
---------------------------------

Two optional configuration blocks allow transforming exposure representations:

- **Rebinning** (``rebinned_exposed``) — collapses a polytomous risk into a
  :term:`dichotomous <Dichotomous Distribution>` one by merging specified
  categories into a single "exposed" group. All remaining categories become
  "unexposed". This is useful when a risk has many GBD categories but the
  model only needs an exposed/unexposed distinction.

- **Category thresholds** (``category_thresholds``) — bins a continuous
  distribution into ordered categories defined by the given thresholds. This
  is designed for alternative risk factors that need categorical
  representations. The two options are mutually exclusive.

See Also
--------

- :ref:`risk_distributions_concept`
- :mod:`vivarium_public_health.causal_factor.exposure`
- :mod:`vivarium_public_health.risks.base_risk`
