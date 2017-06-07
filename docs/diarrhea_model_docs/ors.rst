ORS Intervention
================
- The ORS supplementation component handles everything with regards to ORS that we care about
- How is ORS modeled in GBD?
    - The ABSENCE of ORS is a risk. Like all other risks in GBD, there is a PAF, relative risk, and exposure associated with the lack of ORS.
    - The PAF and relative risk affect the severe diarrhea excess mortality rate. We first calculate the risk-deleted severe diarrhea excess mortality rate (severe diarrhea excess mortality rate * (1 - PAF)). For those that are exposed to the risk (i.e. people that get diarrhea but not ORS), we multipy the risk-deleted excess mortality rate by the relative risk (severe diarrhea excess mortality rate * (1 - PAF) * rr).
- How is ORS handled in CEAM?
    - The ORS component is structured as follows
        - We gather all relevant data (ORS exposure, PAF, rr, and outpatient visit cost*)
        - We manipulate the exposure if we are running an intervention (that is, we can increase probability that a simulant will get ORS)
        - We assign an 'ORS propensity score' to each simulant at the beginning of the simulation. The propensity score for each simulant is a random number between 0 and 1. Each simulant will retain their propensity score for the duration of the simulation.**
        - Each time step, we filter down our population to people that got severe diarrhea in the CURRENT time step
        - We then assign ORS based on the simulant's propensity score and the ORS exposure estimates from GBD***
        - Each simulant that gets ORS receives the benefit for the entire duration of their current bout of diarrhea****
        - Each time step, we risk-delete the severe diarrhea excess mortality rate for all simulants that currently have diarrhea. For those that are exposed to the risk (i.e. are not receiving ORS) we multiply the risk-deleted severe diarrhea excess mortality rate by the relative risk.
        - Then we collect and output cost and count metrics.

Questions
=========
* Do we want to include a unit cost of ors on top of the outpatient visit cost?

** Setting the propensity score at the beginning of the simulation is potentially problematic. The ORS exposure from GBD is calculated from DHS surveys, which ask about the probability of a child receiving ORS, given that they have had diarrhea in the past 2 weeks. Since the propensity score is a random number between 0 and 1, it has nothing to do with whether or not a simulant has diarrhea. Therefore, we are not guaranteed that the exposure scores will be perfectly mathced in a simulation. Are we comfortable with this potential issue? I think it's less of an issue with larger sample sizes and/or for countries with higher diarrhea incidence rates. Should we put in some sort of in-simulation check to ensure that our coverage estimates are close to GBD?

*** Currently, each simulant that receives ORS gets it on the day that their bout of diarrhea starts. This doesn't seem realistic. Should we inject some uncertainty here (i.e. make it so that people can get diarrhea on any day during a bout of diarrhea)?

**** Do we want some uncertainty here as well?

***** Should we keep ORS exposure constant across severity levels? Aren't children with severe diarrhea more likely to get ORS than children with mild diarrhea?
