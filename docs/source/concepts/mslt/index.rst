Multi-State Life Tables
=======================

Multi-state life tables (MSLT) are a tool that can be used to predict the
impact of preventative interventions on chronic disease morbidity and
mortality, in terms of standard metrics such as health-adjusted life years
(HALYs) and health-adjusted life expectancy (HALE).

.. _example_mslt_table:

.. csv-table:: A minimal MSLT example, which shows how morbidity and mortality
   data are used to calculate life expectancy and life year statistics.
   Input data are shown in **bold text**.

   **Year**,**Age**,**Sex**,**Population**,**Mortality rate**,**Probability of death**,**Number of deaths**,**Number of survivors**,**Person years lived**,**Life expectancy**,**YLD rate**,**HALYs**,**HALE**
   2011,**53**,**male**,"**129,850**",**0.0032**,0.0032,420,"129,430","129,640",32.31,**0.1122**,"115,090",25.25
   2012,54,male,"129,430",**0.0035**,0.0034,446,"128,984","129,207",31.41,**0.1122**,"114,706",24.44
   ...,...,...,...,...,...,...,...,...,...,...,...,...
   2066,108,male,221,**0.4810**,0.3819,85,137,179,2.12,**0.3578**,115,1.36
   2067,109,male,137,**0.4810**,0.3819,52,85,111,1.31,**0.3578**,71,0.84

..  The above data were taken from the BAU results for the non-Maori
    population in the simulation where ACMR was reduced by 5%.

The above table shows a life table for the population cohort who were 52 years
old at the start of the simulation (the year 2010).
The inputs for this life table (shown in bold, above) are:

1. The cohort age after the first time-step (53), sex (male), and initial
   population size (129,850);
2. The age-specific, sex-specific mortality rate; and
3. The age-specific, sex-specific years lost due to disability (YLD) rate.

For each year of the simulation, the following calculations are performed:

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

At each time-step, the life table contain the current state of each population
cohort.
For example, after the first time-step each cohort will have aged by one year
and the life table will look like:

.. csv-table:: An example of the MSLT after the first time-step.

   **Year**,**Age**,**Sex**,**Population**,**Mortality rate**,**Probability of death**,**Number of deaths**,**Number of survivors**,**Person years lived**,**Life expectancy**,**YLD rate**,**HALYs**,**HALE**
   ...,...,...,...,...,...,...,...,...,...,...,...,...
   2011,53,female,"135,320",0.0022,0.0022,292,"135,028","135,174",34.46,0.1276,"117,919",26.53
   2011,53,male,"129,850",0.0032,0.0032,420,"129,430","129,640",32.31,0.1122,"115,090",25.25
   2011,58,female,"118,100",0.0033,0.0033,392,"117,708","117,904",29.92,0.1451,"100,791",22.55
   2011,58,male,"114,260",0.0052,0.0051,588,"113,672","113,966",27.71,0.1301,"99,139",21.17
   ...,...,...,...,...,...,...,...,...,...,...,...,...

One time-step later, the rows for these same cohorts in the life table will
look like:

.. csv-table:: An example of the MSLT after the second time-step.

   **Year**,**Age**,**Sex**,**Population**,**Mortality rate**,**Probability of death**,**Number of deaths**,**Number of survivors**,**Person years lived**,**Life expectancy**,**YLD rate**,**HALYs**,**HALE**
   ...,...,...,...,...,...,...,...,...,...,...,...,...
   2012,54,female,"135,028",0.0023,0.0023,311,"134,717","134,872",33.53,0.1276,"117,656",25.71
   2012,54,male,"129,430",0.0035,0.0034,446,"128,984","129,207",31.41,0.1122,"114,706",24.44
   2012,59,female,"117,708",0.0036,0.0036,422,"117,285","117,497",29.02,0.1451,"100,443",21.77
   2012,59,male,"113,672",0.0056,0.0055,631,"113,041","113,356",26.85,0.1301,"98,609",20.41
   ...,...,...,...,...,...,...,...,...,...,...,...,...

..  The above data were taken from the BAU results for the non-Maori
    population in the simulation where ACMR was reduced by 5%.

.. note:: The above life table examples include values for the life expectancy
   and HALE. These can only be calculated at the end of the simulation,
   because they depend on the number of person-years at every time-step.


We start with reference values for the mortality and YLD rates, which define
the "business as usual" (BAU) scenario.
Interventions which affect these rates are then incorporated into the model,
in order to measure their impact on the population.
To implement these interventions we first need to separate the mortality and
YLD rates into cause-specific rates (e.g., chronic diseases), and to model
risk factors that may be acted upon by an intervention (e.g., smoking).
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
