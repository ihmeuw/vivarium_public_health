ORS Intervention
================

Purpose
*******
The purpose of the ORS component is to model lack-of-ORS as a risk factor and to incorpoarate the effect of any interventions aimed at reducing the burden of diarrhea through increasing access to ORS.

Entities, state variables, and scales
*************************************
All models in CEAM define simulants by demographic characteristics (e.g. age and sex). In addition to the general states associated with any given CEAM model, the ors component includes a propensity score which determines how likely a simulant is to receive ORS and a state variable that denotes whether a simulant is currently receiving ORS or not.

Process overview and scheduling
*******************************
At the beginning of the simulation:
        1) We gather all relevant data (ORS exposure, PAF, rr, and weighted cost*)
        2) We manipulate the exposure if we are running an intervention (that is, we can increase probability that a simulant will get ORS)
        3) We assign an 'ORS propensity score' to each simulant. The propensity score for each simulant is a random number between 0 and 1. Each simulant will retain their propensity score for the duration of the simulation.**
Each time step:
        1) We filter down our population to people that got severe diarrhea in the CURRENT time step
        2) We then assign ORS based on the simulant's propensity score and the ORS exposure estimates from GBD. Each simulant that gets ORS receives the benefit for the entire duration of the bout of diarrhea***
        4) Each time step, we risk-delete the severe diarrhea excess mortality rate for all simulants that currently have diarrhea. For those that are not exposed to the risk (i.e. the simulants that are receiving ORS) we multiply the risk-deleted severe diarrhea excess mortality rate by the relative risk.
        5) Then we accrue costs and counts.
At the end of the simulation:
        1) We sum up and output costs and counts.

:sup:`*` This is subject to change.

:sup:`**` Setting the propensity score at the beginning of the simulation is potentially problematic. The ORS exposure from GBD is calculated from DHS surveys, which ask about the probability of a child receiving ORS, given that they have had diarrhea in the past 2 weeks. Since the propensity score is a random number between 0 and 1, it has nothing to do with whether or not a simulant has diarrhea. Therefore, we are not guaranteed that the exposure scores will be perfectly match in a simulation. Are we comfortable with this potential issue? I think it's less of an issue with larger sample sizes and/or for countries with higher diarrhea incidence rates. Should we put in some sort of in-simulation check to ensure that our coverage estimates are close to GBD?

:sup:`***` Currently, each simulant that receives ORS gets it on the day that their bout of diarrhea starts and takes ORS for the entirity of the bout. This doesn't seem realistic. Should we inject some uncertainty here (i.e. make it so that people can get diarrhea on any day during a bout of diarrhea)?

:sup:`****` Should we keep ORS exposure constant across severity levels? That is, do we want to use GBD or Paola's estimates?

Design concepts
***************
Basic Concept -- We model the ORS state as a "finite state machnie". The finite state machines concept is related to discrete-time markhov models, but allows for added complexity.

Initialization
**************
No simulants are receiving ORS (since no simulants have diarrhea) at the start of the simulation.

Input data
**********
The ABSENCE of ORS is a risk in GBD. Like all other risks in GBD, there is a PAF, relative risk, and exposure associated with the lack of ORS.

The PAF and relative risk affect the severe diarrhea excess mortality rate. We first calculate the risk-deleted severe diarrhea excess mortality rate (severe diarrhea excess mortality rate * (1 - PAF)). For those that are exposed to the risk (i.e. people that get diarrhea but not ORS), we multipy the risk-deleted excess mortality rate by the relative risk (severe diarrhea excess mortality rate * (1 - PAF) * rr).

Our costs come from Marcia and Mark's costing work.

