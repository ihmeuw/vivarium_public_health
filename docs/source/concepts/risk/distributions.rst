.. _risk_distributions_concept:

=============
Distributions
=============

.. contents::
   :depth: 2
   :local:
   :backlinks: none

Distribution components translate a :term:`simulant's <Simulant>`
:term:`propensity <Propensity>` into an exposure value. All distributions
inherit from
:class:`~vivarium_public_health.causal_factor.distributions.CausalFactorDistribution`
and implement an ``exposure_ppf`` method that evaluates the
:term:`percent-point function <PPF>` at the simulant's propensity.
The PPF is the inverse of a cumulative distribution function: given a
propensity *q* (a number between 0 and 1), it returns the exposure value *x*
such that exactly a fraction *q* of the population has an exposure at or below
*x*. In practical terms, each simulant's propensity selects a point on the
exposure distribution, and the PPF converts that point into a concrete exposure
value (e.g., a blood-pressure reading or a category label). The distribution
type is selected automatically from the risk's configuration or artifact data.

.. list-table::
   :widths: 25 20 55
   :header-rows: 1

   * - Distribution
     - Exposure Type
     - Description
   * - :class:`~vivarium_public_health.causal_factor.distributions.DichotomousDistribution`
     - Categorical (2)
     - Assigns simulants to "exposed" or "unexposed" based on a single
       probability threshold. Supports rebinning from polytomous data.
   * - :class:`~vivarium_public_health.causal_factor.distributions.PolytomousDistribution`
     - Categorical (N)
     - Assigns simulants to one of *N* ordered or unordered categories using
       cumulative exposure probabilities.
   * - :class:`~vivarium_public_health.causal_factor.distributions.ContinuousDistribution`
     - Continuous
     - Models exposure with a ``normal`` or ``lognormal`` parametric
       distribution.
   * - :class:`~vivarium_public_health.causal_factor.distributions.EnsembleDistribution`
     - Continuous
     - Combines multiple weighted parametric distributions to capture
       complex exposure shapes.

.. _dichotomous_distribution_concept:

:term:`Dichotomous Distribution`
--------------------------------

:class:`~vivarium_public_health.causal_factor.distributions.DichotomousDistribution`
models exposure as two mutually exclusive categories. When determining a
simulant's exposure, the component compares the simulant's
:term:`propensity <Propensity>` to the exposure probability. if the propensity
falls below the threshold the simulant is assigned to the "exposed" category;
otherwise, "unexposed".

When the underlying risk data is polytomous but the model needs a
dichotomous representation, the ``rebinned_exposed`` configuration collapses
selected categories into a single "exposed" group. See
:ref:`exposure_rebinning_concept`.

.. _polytomous_distribution_concept:

Polytomous Distribution
-----------------------

:class:`~vivarium_public_health.causal_factor.distributions.PolytomousDistribution`
handles ordered and unordered categorical risks with *N* categories. Exposure
probabilities for each category are loaded from the artifact, pivoted into a
wide-format lookup table, and their cumulative sum is compared against each
simulant's :term:`propensity <Propensity>` to select a category.

Because categories are sorted before the cumulative sum is computed, results
are reproducible and consistent with the
:ref:`common random number <crn_concept>` framework.

.. _continuous_distribution_concept:

:term:`Continuous Distribution`
-------------------------------

:class:`~vivarium_public_health.causal_factor.distributions.ContinuousDistribution`
supports ``normal`` and ``lognormal`` distribution types from the
``risk_distributions`` package. During setup, the component:

1. Loads mean exposure and standard deviation data from the artifact.
2. Computes the distribution's native parameters (e.g., *μ* and *σ* for
   log-normal) via ``risk_distributions.Normal.get_parameters`` or
   ``risk_distributions.LogNormal.get_parameters``.
3. Builds a lookup table of those parameters, keyed by demographic bins.
4. When determining exposure, looks up the parameters for each simulant and
   passes the simulant's :term:`propensity <Propensity>` through the
   distribution's :term:`PPF` to obtain a concrete exposure value (e.g., a
   systolic blood-pressure reading).

Propensity values are clipped to the range [0.0011, 0.998] before evaluation
to avoid numerical issues at the distribution tails.

.. _ensemble_distribution_concept:

:term:`Ensemble Distribution`
-----------------------------

:class:`~vivarium_public_health.causal_factor.distributions.EnsembleDistribution`
models exposure using a weighted combination of several parametric
distributions (for example, normal, log-normal, gamma, and others supported
by the ``risk_distributions`` package). The component:

1. Loads distribution weights and exposure data from the artifact.
2. Computes per-distribution parameters via
   ``risk_distributions.EnsembleDistribution.get_parameters``.
3. At initialization, draws a second propensity per simulant
   (``ensemble_propensity``) that selects which child distribution to use.
4. When determining exposure, the
   ``risk_distributions.EnsembleDistribution.ppf`` method uses both the
   simulant's :term:`propensity <Propensity>` (quantile) and ensemble
   propensity (distribution selection) to produce an exposure value.

This approach captures complex, potentially multi-modal exposure shapes that
no single parametric family can represent.

See Also
--------

- :ref:`risk_exposure_model_concept`
- :mod:`vivarium_public_health.causal_factor.distributions`
