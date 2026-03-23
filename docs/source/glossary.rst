.. _vph_glossary:

========
Glossary
========

.. glossary::

    Birth Prevalence
        The proportion of newborn simulants who are born with a given condition.
        Used to initialize disease state at birth. See :ref:`disease_state_concept`.

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
        attributed to the risk. See :ref:`special_disease_concept`.

    Prevalence
        The proportion of the population that occupies a given disease state at a
        point in time. Used during initialization to assign simulants to states.
        See :ref:`disease_state_concept`.

    Remission Rate
        The rate at which simulants in a diseased state recover and transition out
        of that state. See :ref:`disease_transition_concept`.
