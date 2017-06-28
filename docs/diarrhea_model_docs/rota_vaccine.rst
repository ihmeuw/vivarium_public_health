Rotavirus Vaccine
=================

Purpose
*******
The purpose of the rota vaccine component is to allow the user to model interventions aimed at reducing the burden of diarrhea through distribution of the rotavirus vaccine. 

Entities, state variables, and scales
*************************************
All models in CEAM define simulants by demographic characteristics (e.g. age and sex). In addition to the general states associated with any given CEAM model, the rota vaccine component includes a state indicating how many doses of the vaccine a simulant has received, when a simulant received each dose, the time window within which the vaccine should have an effect on the simulant, and a state indicating whether or not the vaccine should confer any benefit to a simulant during the current time step.

Process overview and scheduling
*******************************
The rota vaccine component handles everything that we care about with regards to the rotavirus vaccine. 

Each time step:
        1) We determine who should receive the vaccine. We check to see if any of our simulants are at an age where they should receive a dose of the vaccine. Simulants that are at an age where they should receive a dose of the vaccine will be given the vaccine or not according to age/sex/year/location specific coverage estimates from GBD*. If we're running an intervention, we increase the probability of each individual being vaccinated by the probability specified by the intervention (that is, if the intervention specifies that coverage should increase by 50%, each individual's probability of being vaccinated increases by .5). We also have a special option that allows the user to decide whether they want to scale up the rota vaccine coverage to DTP3 coverage. Simulants can only receive the second dose of the vaccine if they received the first dose of the vaccine, just as they can only receive the third dose of the vaccine if they received the second dose.

        2) After determining who should receive the vaccine, we set up the vaccine immunity start/end time. The vaccine has a shelf life for how long it will last. Currently, we're saying that the vaccine will have an effect 14 days after the 3rd dose is administered and that the effect will last for 2 years after the vaccine's protective effect kicks in.**

        3) We then determine the protection that each simulant is receiving from the vaccine. The protection can currently be 0 (if a simulant has not been vaccinated or if the vaccine's effect has worn off) or full protection (we're currently saying a 39% reduction in diarrhea due to rota is the full protective benefit of the vaccine). If we include waning and onset of immunity, the protective effect can be somewhere in between 0 and full protection. The vaccine only affects the incidence of diarrhea due to rota***.

:sup:`*` An important caveat of our current approach is that the rota vaccine coverage estimate from GBD yields the proportion of people in a given age/sex/year/location that received the COMPLETE vaccine series. We're currently using the estimate from GBD to determine the probability that a simulant receives the first dose of the vaccine.

:sup:`**` We're well positioned to include onset of immunity (i.e. effect of first/second doses of the vaccine) but have not discussed what the effect (if any) of the first 2 doses should be. We're well positioned to include waning immunity, but haven't discussed what that should look like in the simulation.

:sup:`***` In truth, the vaccine affects incidence due to other pathogens, but we're not including effects on other pathogens presently.

Design concepts
***************
Basic Concept -- We model the rota vaccine state as a "finite state machnie". The finite state machines concept is related to discrete-time markhov models, but allows for added complexity.

Initialization
**************
No simulants have received a dose of the vaccine at the beginning of the simulation.

Input data
**********
Simulants that receive the vaccine experience a reduction in the probability that they will get diarrhea due to rota. We determine the probability that a simulant will get diarrhea due to rota by multiplying the incidence rate of diarrhea and the PAF of rota on diarrhea (both from the GBD) and converting the product to a probability.

The rest of the data that we need for the rota vaccine component comes from external sources. There are 14 parameters that we can change.

+---------------------------------------------------+---------------------------------------------------------------------------------------------------+-------------------+-----------------------------------------------------------------------+
| **Parameter**                                     | **Description**                                                                                   | **Current value** | **Source**                                                            |
+---------------------------------------------------+---------------------------------------------------------------------------------------------------+-------------------+-----------------------------------------------------------------------+
| RV5_dose_cost                                     | Unit cost of RotaTeq (in US dollars)                                                              | 3.5               | needs citation                                                        |
+---------------------------------------------------+---------------------------------------------------------------------------------------------------+-------------------+-----------------------------------------------------------------------+
| cost_to_administer_each_dose                      | Delivery cost of RotaTeq (in US dollars)                                                          | 0                 | needs citation                                                        |
+---------------------------------------------------+---------------------------------------------------------------------------------------------------+-------------------+-----------------------------------------------------------------------+
| first_dose_protection                             | Benefit conferred from 1 dose                                                                     | 0                 | needs citation                                                        |
+---------------------------------------------------+---------------------------------------------------------------------------------------------------+-------------------+-----------------------------------------------------------------------+
| second_dose_protection                            | Benefit conferred from 2 doses                                                                    | 0                 | needs citation                                                        |
+---------------------------------------------------+---------------------------------------------------------------------------------------------------+-------------------+-----------------------------------------------------------------------+
| third_dose_protection                             | Benefit conferred from 3 doses                                                                    | .39               | Lamberti -- A Systematic Review of the Effect of Rotavirus Vaccination|
+---------------------------------------------------+---------------------------------------------------------------------------------------------------+-------------------+-----------------------------------------------------------------------+
| vaccine_full_immunity_duration                    | Amount of time vaccine confers full benefit (in days)                                             | 730               | needs citation                                                        |
+---------------------------------------------------+---------------------------------------------------------------------------------------------------+-------------------+-----------------------------------------------------------------------+
| waning_immunity_time                              | Amount of time vaccine will confer partial benefit (in days)                                      | 0                 | needs citation                                                        |
+---------------------------------------------------+---------------------------------------------------------------------------------------------------+-------------------+-----------------------------------------------------------------------+
| age_at_first_dose                                 | age (in days) 1st dose should be administered                                                     | 61                | https://www.cdc.gov/rotavirus/vaccination.html                        |
+---------------------------------------------------+---------------------------------------------------------------------------------------------------+-------------------+-----------------------------------------------------------------------+
| age_at_second_dose                                | age (in days) 2nd dose should be administered                                                     | 122               | https://www.cdc.gov/rotavirus/vaccination.html                        |
+---------------------------------------------------+---------------------------------------------------------------------------------------------------+-------------------+-----------------------------------------------------------------------+
| age_at_third_dose                                 | age (in days) 3rd dose should be administered                                                     | 183               | https://www.cdc.gov/rotavirus/vaccination.html                        |
+---------------------------------------------------+---------------------------------------------------------------------------------------------------+-------------------+-----------------------------------------------------------------------+
| second_dose_retention                             | % simulants that will get 2nd dose given they got 1st dose and are alive at age_at_second_dose    | 100               | needs citation                                                        |
+---------------------------------------------------+---------------------------------------------------------------------------------------------------+-------------------+-----------------------------------------------------------------------+
| third_dose_retention                              | % simulants that will get 3rd dose given they got 2nd dose and are alive at age_at_third_dose     | 100               | needs citation                                                        |
+---------------------------------------------------+---------------------------------------------------------------------------------------------------+-------------------+-----------------------------------------------------------------------+
| vaccination_proportion_increase                   | % increase in the probability simulant will be vaccinated                                         | varies            | N/A                                                                   |
+---------------------------------------------------+---------------------------------------------------------------------------------------------------+-------------------+-----------------------------------------------------------------------+
| time_after_dose_at_which_immunity_is_conferred    | time after vaccine is administered at which it starts to confer a benefit (in days)               | 14                | needs citation                                                        |
+---------------------------------------------------+---------------------------------------------------------------------------------------------------+-------------------+-----------------------------------------------------------------------+

Questions
*********
1. Should different doses have different durations? Or time in between dosage and immunity being conferred?
2. Should vaccine lose effect 2 years after its administered? Or 2 years after it starts to take effect (i.e. 2 years and 2 weeks after its administered)?
4. Right now we've only modeled RotaTeq. Do Ethiopia, Bangladesh, and Nigeria all use RotaTeq?

Future Improvements
********************
1. We could potentially use the code I've written to make a more general vaccine component
2. Would be nice to have uncertainty estimate for the effect of the vaccine. Would also be nice get a risk-deleted incidence (e.g. the lack-of-rota-vaccine deleted incidence) and then multiply the benefit of the vaccine by the risk-deleted incidence.
3. The vaccine has a much greater impact on severe diarrhea than it does on moderate and mild diarrhea. We should figure out how to incorporate this.
