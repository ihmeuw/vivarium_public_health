.. _vph_glossary:

========
Glossary
========

.. glossary::

    All-Cause Mortality Rate
    ACMR
        The total mortality rate from all causes combined for a given population. Used as the 
        starting point for the :ref:`cause-deleted mortality <mortality_concept>` calculation.
        See :ref:`mortality_concept`.

    Age-Specific Fertility Rate
        A per-age fertility hazard applied to individual female :term:`simulants <Simulant>` 
        to determine births at each time step. See :ref:`fertility_concept`.

    Cause-Specific Mortality Rate
    CSMR
        The mortality rate attributable to a single cause. See :ref:`mortality_concept`.

    Crude Birth Rate
        A population-level measure of births computed from live-birth covariate data and the 
        ratio of simulated to true population size. See :ref:`fertility_concept`.

    Cause-Deleted Mortality
    Mortality Rate
        The effective per-simulant mortality rate. It is calculated by starting
        from the :term:`ACMR` and subtracting the :term:`CSMR` for explicitly 
        modeled and unmodeled causes, then adding back any risk-modified unmodeled
        contributions. See :ref:`mortality_concept`.

    Theoretical Minimum Risk Life Expectancy
    TMRLE
        The maximum life expectancy achievable if all risk factors were at
        their theoretical minimum levels. Sourced from the Global Burden of
        Disease study and used as the reference table for computing
        :term:`YLL`. See :ref:`mortality_concept`.

    Unmodeled Cause
        A cause of death that is not represented by its own disease :term:`component <Component>` 
        in the simulation but whose :term:`CSMR` is still accounted for in the mortality calculation.
        See :ref:`mortality_concept`.

    Years of Life Lost
    YLL
        The residual life expectancy at the time of a :term:`simulant's <Simulant>` 
        death, computed from the :term:`TMRLE` table. Accumulated across 
        the population as a summary measure of premature mortality. See :ref:`mortality_concept`.

    Continuous Distribution
        A risk exposure distribution modeled using a standard statistical distribution such as
        ``normal`` or ``lognormal``. Exposure values are obtained by evaluating the
        distribution's :term:`PPF` at each :term:`simulant's <Simulant>`
        :term:`propensity <Propensity>`. See :ref:`continuous_distribution_concept`.

    Dichotomous Distribution
        A two-category exposure distribution that classifies :term:`simulants <Simulant>`
        as "exposed" or "unexposed" based on a single probability threshold.
        See :ref:`dichotomous_distribution_concept`.

    Ensemble Distribution
        A risk exposure distribution formed by combining multiple weighted parametric
        distributions to capture complex, potentially multi-modal exposure shapes.
        See :ref:`ensemble_distribution_concept`.

    Percent-Point Function
    PPF
        The inverse of the cumulative distribution function. Given a quantile
        *q*, the PPF returns the value *x* such that *P(X ≤ x) = q*. Used to convert a
        :term:`simulant's <Simulant>` :term:`propensity <Propensity>` into an exposure value.
        See :ref:`risk_exposure_model_concept`.

    Propensity
        A uniform random value in [0, 1] assigned to each :term:`simulant <Simulant>` at
        initialization and held constant for the duration of the simulation. It represents
        the simulant's position in the cumulative distribution of a risk factor and is used
        as input to the :term:`PPF`. See :ref:`propensity_concept`.


