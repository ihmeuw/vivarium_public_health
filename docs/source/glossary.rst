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
        attributed to the risk. See :ref:`risk_attributable_disease_concept`.

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
    
    Therapeutic Inertia
        The tendency for treatment algorithms to deviate from clinical
        guidelines — for example, when treatment is not escalated during a
        healthcare visit despite guidelines recommending escalation. Modeled
        as a probability drawn from a triangular distribution. See
        :ref:`therapeutic_inertia_concept`.

    Linear Scale-Up
        A time-varying pattern in which an intervention's coverage is linearly
        interpolated between a start value and an end value over a configured
        date range. See :ref:`scale_up_concept`.