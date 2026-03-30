.. _vph_glossary:

========
Glossary
========

.. glossary::

    Age-Specific Fertility Rate
        A per-age fertility hazard applied to individual female :term:`simulants <Simulant>`
        to determine births at each time step. See :ref:`fertility_concept`.

    All-Cause Mortality Rate
    ACMR
        The total mortality rate from all causes combined for a given population. Used as the
        starting point for the :ref:`cause-deleted mortality <mortality_concept>` calculation.
        See :ref:`mortality_concept`.

    Birth Prevalence
        The proportion of newborn simulants who are born with a given condition.
        Used to initialize disease state at birth. See :ref:`disease_state_concept`.

    Cause-Deleted Mortality
    Mortality Rate
        The effective per-simulant mortality rate. It is calculated by starting
        from the :term:`ACMR` and subtracting the :term:`CSMR` for explicitly
        modeled and unmodeled causes, then adding back any risk-modified unmodeled
        contributions. See :ref:`mortality_concept`.

    Cause-Specific Mortality Rate
    CSMR
        The mortality rate attributable to a single cause. See :ref:`mortality_concept`.

    Crude Birth Rate
        A population-level measure of births computed from live-birth covariate data and the
        ratio of simulated to true population size. See :ref:`fertility_concept`.

    Disability Weight
        A severity weight between 0 and 1 that represents the magnitude of health
        loss associated with a disease state. Used to compute years lived with
        disability (YLDs). See :ref:`disease_state_concept`.

    Dwell Time
        The minimum duration a simulant must remain in a disease state before any
        outgoing transition can fire. Specified in days. See :ref:`disease_state_concept`.

    Excess Mortality Rate
    EMR
        The additional mortality rate attributable to being in a particular disease
        state, above and beyond the background mortality rate. See
        :ref:`disease_state_concept`.

    Incidence Rate
        The rate at which simulants in a susceptible state acquire a disease and
        transition to a diseased state. See :ref:`disease_transition_concept`.

    Population Attributable Fraction
    PAF
        The fraction of disease burden in a population that is attributable to a
        particular risk factor exposure. A PAF of 1 means the disease is fully
        attributed to the risk. Typically used to compute the
        :term:`calibration constant <Calibration Constant>`.
        See :ref:`risk_attributable_disease_concept` and
        :ref:`calibration_constant_concept`.

    Prevalence
        The proportion of the population that occupies a given disease state at a
        point in time. Used during initialization to assign simulants to states.
        See :ref:`disease_state_concept`.

    Remission Rate
        The rate at which simulants in a diseased state recover and transition out
        of that state. See :ref:`disease_transition_concept`.

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

    Calibration Constant
        A value, typically derived from the :term:`population attributable fraction <PAF>`,
        that adjusts a target rate so that, after :term:`relative risk <Relative Risk>`
        multiplication, the population-level rate remains consistent with input data.
        See :ref:`calibration_constant_concept`.

    Log-Linear Model
        A dose–response model in which the logarithm of the
        :term:`relative risk <Relative Risk>` is proportional to the difference between a
        :term:`simulant's <Simulant>` exposure and the :term:`TMREL`.
        See :ref:`log_linear_risk_effect_concept`.

    Relative Risk
        A measure of how much more likely an outcome is for a :term:`simulant <Simulant>` at
        a given exposure level compared to a reference level. Used by risk effect components
        to modify target rates such as disease incidence or mortality.
        See :ref:`relative_risk_concept`.

    Theoretical Minimum-Risk Exposure Distribution
    TMRED
        The distribution of exposure levels at which the risk to health is at a theoretical
        minimum. The midpoint of this distribution defines the :term:`TMREL`.
        See :ref:`relative_risk_concept`.

    Theoretical Minimum-Risk Exposure Level
    TMREL
        The exposure level at which the risk to health is at a theoretical minimum, typically
        computed as the midpoint of the :term:`TMRED`. Relative risks are normalized so that
        the :term:`relative risk <Relative Risk>` at the TMREL equals 1.
        See :ref:`relative_risk_concept`.


