Multi-State Life Tables
=======================

Multi-state life tables (MSLT) are a tool that can be used to predict the
impact of preventative interventions on chronic disease morbidity and
mortality, by interventions acting through changes in risk factors that affect
multiple disease incidence rates (hence "multi-state" life tables).
Metrics such as health-adjusted life years (HALYs) and health-adjusted life
expectancy (HALE) can be used to quantify intervention impacts.

To demonstrate how a MSLT works, we begin by showing a life table can be used
to estimate HALYs and HALE before any intervention is applied, and then show
to simulate simple intervention effects.

.. _example_mslt_table:

.. csv-table:: A simple life table example, which shows how morbidity and
   mortality data are used to calculate life expectancy and life year
   statistics.
   Input data are shown in **bold text**, everything else is calculated within
   the life table.
   :header: **Year**,**Age**,**Sex**,**Population**,**Mortality rate**,**Probability of death**,**Number of deaths**,**Number of survivors**,**Person years lived**,**Life expectancy**,**YLD rate**,**HALYs**,**HALE**

   2011,**52**,**male**,"**129,850**",**0.0030**,0.0030,390,"129,460","129,655",33.12,**0.1122**,"115,103",26.00
   2012,53,male,"129,460",**0.0032**,0.0032,413,"129,047","129,254",32.23,**0.1122**,"114,747",25.18
   ...,...,...,...,...,...,...,...,...,...,...,...,...
   2067,108,male,221,**0.4811**,0.3819,84,136,179,1.62,**0.3578**,115,1.04
   2068,109,male,136,**0.4811**,0.3819,52,84,110,1.31,**0.3578**,71,0.84
   2069,110,male,84,**0.4812**,0.3820,32,52,68,0.81,**0.3578**,44,0.52

..  The above data were taken from the BAU results for the non-Maori
    population in the simulation where ACMR was reduced by 5%.

The above table shows a life table for the population cohort who were 52 years
old at the start of the year 2011.
The inputs for this life table (shown in bold, above) are:

1. The cohort age after the first time-step (52), sex (male), and initial
   population size (129,850);
2. The age-specific, sex-specific mortality rate; and
3. The age-specific, sex-specific years lost due to disability (YLD) rate.

For each future year, the following calculations are performed:

1. The (age-specific) mortality rate is converted into a mortality risk (i.e.,
   the probability that an individual will die in that year);
2. The risk is multiplied by the population size to calculate the number of
   deaths that occur in that year, which also determines the number of
   survivors;
3. The person-years lived are calculated under the assumption that the deaths
   occur at a constant rate, and so this the mean of the starting population
   and the surviving population;
4. The life expectancy is defined as the sum of all future life years, divided
   by the starting population size; and
5. The years lost due to disability (YLD) rate is used to discount the
   person-years lived and the life expectancy, which yields the
   health-adjusted life years (HALYs) and health-adjusted life expectancy
   (HALE) for this cohort.

The above life table simulated the lifespan of the 52 year old male cohort.
Within Vivarium, the same calculations are performed in parallel for multiple
cohorts.
In the simulations presented here we divide the population into five-year
age-group cohorts for each sex, under the assumption that, e.g., males aged
50-54 can be reasonably approximated as a single cohort aged 52 years.

The above examples is also called the "business as usual" (BAU) scenario, and
uses reference values for the mortality and YLD rates.
A simple intervention that lowers mortality rates by, say, 5% would generate
more LYs and HALYs, and longer LEs and HALEs, than those obtained in the BAU
scenario.
These difference between the BAU and intervention life tables comprise the
intervention effect.
However, in the MSLT model the intervention effect is typically not modelled
directly as a change in the all-cause mortality and morbidity rates.
Rather, we construct multiple disease-specific life tables and allow
interventions to affect disease incidence rates.
Changes to disease incidence will result in changes to disease-specific
mortality and morbidity rates.
The sum of these differences across all diseases is then subtracted from the
all-cause mortality and morbidity rates in the intervention life table.
We now address each of these concepts in turn.

.. toctree::
   :maxdepth: 2
   :caption: MSLT components

   chronic_disease
   acute_disease
   risk_factors
   interventions

.. toctree::
   :maxdepth: 2
   :caption: Data requirements

   input_data
   observers
   uncertainty
