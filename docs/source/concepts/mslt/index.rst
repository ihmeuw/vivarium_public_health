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

   **Age**,**Population**,**Mortality rate**,**Probability of death**,**Number of deaths**,**Number of survivors**,**Person years lived**,**Life expectancy**,**YLD rate**,**HALYs**,**HALE**
   **52**,"**129,850**",**0.0029**,0.0029,380,"129,470","129,660",33.23,**0.1122**,"115,107",26.08
   53,"129,470",**0.0031**,0.0031,404,"129,066","129,268",32.33,**0.1122**,"114,759",25.27
   ...,...,...,...,...,...,...,...,...,...,...
   108,221,**0.4811**,0.3819,84,136,179,2.12,**0.3578**,115,1.36
   109,136,**0.4811**,0.3819,52,84,110,1.31,**0.3578**,71,0.84

The above table shows a life table for the population cohort who were 52 years
old at the start of the simulation.
The inputs for this life table (shown in bold, above) are:

1. The initial cohort age (52) and population size (129,850);
2. The age-specific mortality rate; and
3. The age-specific years lost due to disability (YLD) rate.

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
