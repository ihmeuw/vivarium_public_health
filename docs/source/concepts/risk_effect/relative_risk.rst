.. _relative_risk_concept:

=============
Relative Risk
=============

.. contents::
   :depth: 2
   :local:
   :backlinks: none

A :term:`relative risk <Relative Risk>` (RR) quantifies how much a
:term:`simulant's <Simulant>` :ref:`exposure <vph_risk_exposure_concept>` to a
risk factor increases (or decreases) the rate of some outcome compared to a
reference exposure level. A relative risk of 1 means no additional risk; values
greater than 1 indicate elevated risk; values between 0 and 1 indicate
protective effects.

The :class:`~vivarium_public_health.causal_factor.effect.CausalFactorEffect`
base class (and its subclass
:class:`~vivarium_public_health.risks.effect.RiskEffect`) registers a
``{risk_name}_on_{target_name}.relative_risk``
:ref:`pipeline <values_concept>` whose source computes RR values for each
simulant. This pipeline is then registered as a modifier on the target rate
pipeline (e.g., ``cause_name.incidence_rate``), so the target rate is
multiplied by each simulant's relative risk every time the rate is evaluated.
Relative risk data is loaded from the simulation artifact by default, but can
be overridden with a scalar value or a ``scipy.stats`` distribution name in the
configuration (see the
:class:`~vivarium_public_health.causal_factor.effect.CausalFactorEffect` class
documentation for details).

How relative risk is computed depends on whether the risk factor's exposure is
categorical or continuous.

Categorical Exposure
--------------------

For categorical risks (dichotomous or polytomous), relative risk data is loaded
from the simulation artifact as a table mapping each exposure category to an RR
value. When determining a simulant's relative risk, the component looks up the
simulant's current exposure category and returns the corresponding RR from the
table.

For :term:`dichotomous <Dichotomous Distribution>` risks, the "unexposed"
category always has an RR of 1 and the "exposed" category carries the loaded
RR value.

.. _log_linear_risk_effect_concept:

Log-Linear Model (Continuous Exposure)
--------------------------------------

For continuous risk factors, the default
:class:`~vivarium_public_health.risks.effect.RiskEffect` component uses a
:term:`log-linear model <Log-Linear Model>`. In this model, the logarithm of
the relative risk is proportional to the difference between the simulant's
exposure and the :term:`TMREL`:

.. math::

   RR_{\text{simulant}} = RR_{\text{per-unit}}^{\;(x - \text{TMREL})\,/\,\text{scale}}

where:

- :math:`x` is the simulant's current exposure value,
- :math:`\text{TMREL}` is the :term:`theoretical minimum-risk exposure level
  <TMREL>`, computed as the midpoint of the :term:`TMRED`,
- :math:`RR_{\text{per-unit}}` is the relative risk per unit of exposure loaded
  from the artifact, and
- :math:`\text{scale}` is a scalar loaded from the artifact
  (``{risk}.relative_risk_scalar``) that defines the exposure increment to
  which the per-unit RR corresponds.

The result is clipped to a minimum of 1, so the relative risk never falls
below the baseline.

.. _non_log_linear_risk_effect_concept:

Non-Log-Linear Model (Piecewise Interpolation)
-----------------------------------------------

When the dose–response relationship between exposure and relative risk is not
well described by a log-linear curve, the
:class:`~vivarium_public_health.risks.effect.NonLogLinearRiskEffect` component
can be used instead. This component:

1. Loads :term:`TMRED` data from the artifact and computes the :term:`TMREL`
   as a uniform random draw between the TMRED's minimum and maximum.
2. Interpolates the RR at the TMREL from the configured RR data points and
   divides all RR values by this quantity, so that the RR at the TMREL equals
   1. The result is clipped to a minimum of 1.
3. Constructs a lookup table of piecewise-linear intervals from the normalized
   RR data. Each interval has a left and right exposure boundary and
   corresponding left and right RR values.
4. When determining a simulant's relative risk, identifies which interval
   contains the simulant's exposure and linearly interpolates:

.. math::

   RR_{\text{simulant}} = RR_{\text{left}} + \frac{RR_{\text{right}} - RR_{\text{left}}}{x_{\text{right}} - x_{\text{left}}} \cdot (x - x_{\text{left}})

where :math:`x` is the simulant's exposure and the left/right values are the
boundaries of the enclosing bin.

To avoid recomputing the exposure pipeline every time the relative risk is
queried, the ``NonLogLinearRiskEffect`` caches each simulant's exposure value
in a column on the :ref:`state table <population_concept>` and reads from that
cached column when interpolating.

Unmodeled Cause Mortality
-------------------------

Risk effects can target :term:`unmodeled cause <Unmodeled Cause>` mortality
rates. In this case the risk effect component modifies the
:ref:`cause-deleted mortality <mortality_concept>` pipeline in the same way it
would modify any other target rate — by multiplying the rate by each simulant's
relative risk and applying a :ref:`calibration constant
<calibration_constant_concept>`.

See Also
--------

- :ref:`calibration_constant_concept`
- :ref:`vph_risk_exposure_concept`
- :mod:`vivarium_public_health.causal_factor.effect`
- :mod:`vivarium_public_health.risks.effect`
