.. _mslt_input_data:

Input data requirements
=======================

The data required for an MSLT model depend on the model components.
Here, we define the data requirements for each type of component.

In general, rates and values are stored in tables with the following columns:

.. note:: For convenience, all of these input data can be collected into a
   single data artifact.
   For each of the tables described below, we identify the name under which it
   should be stored in a data artifact.

   We will see how to use data artifacts in the
   :ref:`MSLT tutorials <mslt_tutorials>`.

Core MSLT
---------

The cohorts and their population sizes are defined in the
``population.structure`` table:

.. csv-table::

   **year**,**age**,**sex**,**population**,**bau_population**
   2011,2,female,108970.000,108970.000
   2011,2,male,114970.000,114970.000
   2011,7,female,105600.000,105600.000
   2011,7,male,110470.000,110470.000
   ...,...,...,...,...
   2011,102,female,1035.000,1035.000
   2011,102,male,433.125,433.125
   2011,107,female,207.000,207.000
   2011,107,male,86.625,86.625

The age-specific, sex-specific mortality rates are defined in the
``cause.all_causes.mortality`` table:

.. csv-table::

   **year_start**,**year_end**,**age_group_start**,**age_group_end**,**sex**,**rate**
   2011,2012,0,1,female,0.003586
   2011,2012,0,1,male,0.004390
   2011,2012,1,2,female,0.000330
   2011,2012,1,2,male,0.000340
   ...,...,...,...
   2120,2121,109,110,female,0.524922
   2120,2121,109,110,male,0.529281

.. note:: Rates and other values that apply to specific cohorts during the
   simulation (i.e., all input data except for the initial cohort population
   sizes and initial disease/risk factor prevalence) are indexed by time
   intervals and age intervals.

   In the mortality rate table shown above, the rate in each row applies:

   + From the time in **year_start** up to (but not including) the time in
     **year_end**; and
   + To cohorts whose age lies between **age_group_start** (inclusive) and
     **age_group_end** (exclusive).

Similarly, the age-specific, sex-specific disability rates are defined in the
``cause.all_causes.disability_rate`` table:

.. csv-table::

   **year_start**,**year_end**,**age_group_start**,**age_group_end**,**sex**,**rate**
   2011,2012,0,1,female,0.014837
   2011,2012,0,1,male,0.020674
   2011,2012,1,2,female,0.022379
   2011,2012,1,2,male,0.026409
   ...,...,...,...
   2120,2121,109,110,female,0.366114
   2120,2121,109,110,male,0.357842

Chronic diseases
----------------

For each chronic disease, the initial prevalence and disease-specific rates
are stored in the following tables (where the disease name is ``NAME``).

The incidence rate \(i\) is stored in ``chronic_disease.NAME.incidence``:

.. csv-table::

   **year_start**,**year_end**,**age_group_start**,**age_group_end**,**sex**,**NAME_i**
   2011,2012,0,1,female,0.0
   ...,...,...,...

The disability rate \(DR\) is stored in ``chronic_disease.NAME.morbidity``:

.. csv-table::

   **year_start**,**year_end**,**age_group_start**,**age_group_end**,**sex**,**NAME_DR**
   2011,2012,0,1,female,0.0
   ...,...,...,...

The mortality rate \(f\) is stored in ``chronic_disease.NAME.mortality``:

.. csv-table::

   **year_start**,**year_end**,**age_group_start**,**age_group_end**,**sex**,**NAME_f**
   2011,2012,0,1,female,0.0
   ...,...,...,...

The initial prevalence is stored in ``chronic_disease.NAME.prevalence``:

.. csv-table::

   **year**,**age**,**sex**,**NAME_prev**
   2011,0,female,0.0
   ...,...,...,...

The remission rate \(r\) is stored in ``chronic_disease.NAME.remission``:

.. csv-table::

   **year_start**,**year_end**,**age_group_start**,**age_group_end**,**sex**,**NAME_r**
   2011,2012,0,1,female,0.0
   ...,...,...,...

.. note:: Note that the column names are different in each table.

Acute diseases and other events
-------------------------------

For each acute disease/event, the morbidity and mortality rates are stored in
the following tables (where the disease/event names is ``NAME``).

The morbidity rate is stored in ``acute_disease.NAME.morbidity``:

.. csv-table::

   **year_start**,**year_end**,**age_group_start**,**age_group_end**,**sex**,**NAME_disability_rate**
   2011,2012,0,1,female,0.000301
   ...,...,...,...

The mortality rate is stored in ``acute_disease.NAME.mortality``:

.. csv-table::

   **year_start**,**year_end**,**age_group_start**,**age_group_end**,**sex**,**NAME_excess_mortality**
   2011,2012,0,1,female,0.000032
   ...,...,...,...

.. note:: Note that the column names are different in each table.

Risk factors
------------

The tobacco risk factor (as implemented by the
:class:`~vivarium_public_health.mslt.delay.DelayedRisk` component) requires
several data tables.

The incidence rate is stored in ``risk_factor.tobacco.incidence``:

.. csv-table::

   **year_start**,**year_end**,**age_group_start**,**age_group_end**,**sex**,**incidence**
   2011,2012,0,1,female,0.000301
   ...,...,...,...

The remission rate is stored in ``risk_factor.tobacco.remission``:

.. csv-table::

   **year_start**,**year_end**,**age_group_start**,**age_group_end**,**sex**,**remission**
   2011,2012,0,1,female,0.000301
   ...,...,...,...

The initial prevalence for each exposure category is stored in
``risk_factor.tobacco.prevalence``:

.. csv-table::

  **year**,**age**,**sex**,**tobacco.no**,**tobacco.yes**,**tobacco.0**,**tobacco.1**,...,**tobacco.20**,**tobacco.21**
   2011,0,female,1.0,0.0,0.0,0.0,...,0.0,0.0
   ...,...,...,...,...,...,...,...,...,...

The relative risk of mortality for each exposure category (defined separately
for the BAU and intervention scenarios) is stored in
``risk_factor.tobacco.mortality_relative_risk``:

.. csv-table::

  **year_start**,**year_end**,**age_group_start**,**age_group_end**,**sex**,**tobacco.no**,**tobacco.yes**,...,**tobacco.21**,**tobacco_intervention.no**,**tobacco_intervention.yes**,...,**tobacco_intervention.21**
   2011,2012,0,1,female,1.0,1.0,...,1.0,1.0,1.0,...,1.0
   ...,...,...,...,...,...,...,...,...,...,...

The relative risk of chronic disease incidence for each exposure category is
stored in ``risk_factor.tobacco.disease_relative_risk``, which contains
separate columns for each chronic disease.
Shown here is an example for two chronic diseases, called ``DiseaseA`` and
``DiseaseB``:

.. csv-table::

   **year_start**,**year_end**,**age_group_start**,**age_group_end**,**sex**,**DiseaseA_no**,**DiseaseA_yes**,...,**DiseaseA_21**,**DiseaseB_no**,**DiseaseB_yes**,...,**DiseaseB_21**
    2011,2012,0,1,female,1.0,1.0,...,1.0,1.0,1.0,...,1.0
    ...,...,...,...,...,...,...,...,...,...,...

Interventions
-------------

The :class:`~vivarium_public_health.mslt.intervention.TobaccoEradication`
and :class:`~vivarium_public_health.mslt.intervention.TobaccoFreeGeneration`
interventions don't have any data requirements.
The tobacco tax intervention, however, is characterised in terms of its effect
on the incidence (i.e., uptake) and remission (i.e., cessation) rates.

The incidence effect is stored in
``risk_factor.tobacco.tax_effect_incidence``:

.. csv-table::

   **year_start**,**year_end**,**age_group_start**,**age_group_end**,**sex**,**incidence_effect**
   2011,2012,0,1,female,1.0
   2011,2012,0,1,male,1.0
   2011,2012,1,2,female,1.0
   2011,2012,1,2,male,1.0
   ...,...,...,...
   2120,2121,108,109,female,0.866004
   2120,2121,108,109,male,0.866004
   2120,2121,109,110,female,0.866004
   2120,2121,109,110,male,0.866004

The remission effect is stored in
``risk_factor.tobacco.tax_effect_remission``:

.. csv-table::

   **year_start**,**year_end**,**age_group_start**,**age_group_end**,**sex**,**remission_effect**
   2011,2012,0,1,female,1.0
   2011,2012,0,1,male,1.0
   2011,2012,1,2,female,1.0
   2011,2012,1,2,male,1.0
   ...,...,...,...
   2031,2032,22,23,female,0.975724
   2031,2032,22,23,male,0.975724
   2031,2032,23,24,female,0.975724
   2031,2032,23,24,male,0.975724
   ...,...,...,...
   2120,2121,108,109,female,1.0
   2120,2121,108,109,male,1.0
   2120,2121,109,110,female,1.0
   2120,2121,109,110,male,1.0
