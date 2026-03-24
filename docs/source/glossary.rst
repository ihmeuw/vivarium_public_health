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
